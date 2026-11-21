"""Residual policy/value network with a shared scorer over exact legal actions."""
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from .encoding import (BOARD_CHANNELS, ENCODING_VERSION, NO_SQUARE,
                       encode_actions, encode_board)
from .rules import PUBLIC, terminal_value

MODEL_VERSION = 1
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class NetworkConfig:
    blocks: int = 6
    channels: int = 96

    def __post_init__(self):
        if self.blocks < 0 or self.channels < 4:
            raise ValueError("blocks must be nonnegative and channels must be at least 4")


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = math.gcd(8, channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class PolicyValueNet(nn.Module):
    """Joint legal-action logits and W/D/L logits, all for the current player.

    Inputs are B,C,8,8 boards, B,A,6 action descriptors, and a B,A boolean
    legal/padding mask. Masked logits are -inf. Each row must contain at least
    one action; terminal states are handled by the evaluator/search directly.
    Group normalization permits batch-one inference without running statistics.
    """
    def __init__(self, config: NetworkConfig | None = None):
        super().__init__()
        self.config = config or NetworkConfig()
        channels = self.config.channels
        self.trunk = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, channels, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(8, channels), channels), nn.ReLU(),
            *(_ResidualBlock(channels) for _ in range(self.config.blocks)),
        )
        self.square_embedding = nn.Embedding(65, 16, padding_idx=NO_SQUARE)
        self.orientation_embedding = nn.Embedding(9, 8, padding_idx=8)
        self.kind_embedding = nn.Embedding(6, 8)
        self.deployment_embedding = nn.Embedding(8, 8, padding_idx=0)
        self.policy_head = nn.Sequential(nn.Linear(4 * channels + 72, 2 * channels),
                                         nn.ReLU(), nn.Linear(2 * channels, 1))
        self.value_head = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(),
                                        nn.Linear(channels, 3))

    def forward(self, boards, actions, legal_mask):
        if boards.ndim != 4 or tuple(boards.shape[1:]) != (BOARD_CHANNELS, 8, 8):
            raise ValueError(f"boards must have shape B,{BOARD_CHANNELS},8,8")
        if actions.ndim != 3 or actions.shape[0] != boards.shape[0] or actions.shape[2] != 6:
            raise ValueError("actions must have shape B,A,6")
        if tuple(legal_mask.shape) != tuple(actions.shape[:2]):
            raise ValueError("legal_mask must have shape B,A")
        legal_mask = legal_mask.bool()
        if not torch.all(legal_mask.any(dim=1)):
            raise ValueError("Each nonterminal position must have at least one legal action")
        # Padded descriptors need not be valid indices. They never participate
        # in gathers or embeddings, and their final logits remain masked.
        padding = actions.new_tensor([64, 64, 64, 8, 5, 0])
        actions = torch.where(legal_mask.unsqueeze(-1), actions, padding)
        spatial = self.trunk(boards)
        global_features = spatial.mean(dim=(2, 3))
        squares = spatial.flatten(2).transpose(1, 2)
        squares = torch.cat((squares, squares.new_zeros((len(boards), 1, self.config.channels))), dim=1)
        batch_index = torch.arange(len(boards), device=boards.device)[:, None]
        local = [squares[batch_index, actions[:, :, column]] for column in range(3)]
        features = torch.cat((*local, global_features[:, None].expand(-1, actions.shape[1], -1),
                              self.square_embedding(actions[:, :, :3]).flatten(2),
                              self.orientation_embedding(actions[:, :, 3]),
                              self.kind_embedding(actions[:, :, 4]),
                              self.deployment_embedding(actions[:, :, 5])), dim=-1)
        policy_logits = self.policy_head(features).squeeze(-1).masked_fill(~legal_mask, -torch.inf)
        return policy_logits, self.value_head(global_features)


class TorchEvaluator:
    """Inference adapter: normalized priors in legal_moves order and W minus L.

    predict_batch accepts parallel sequences of boards and legal-move lists.
    Exact terminal values bypass the model. The network is always in eval mode;
    callers training the same instance must restore train mode for each update.
    """
    def __init__(self, model: PolicyValueNet, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def predict(self, board, legal_moves):
        return self.predict_batch([board], [legal_moves])[0]

    def predict_batch(self, boards: Sequence, legal_moves: Sequence):
        if len(boards) != len(legal_moves):
            raise ValueError("boards and legal_moves must have equal lengths")
        results = [None] * len(boards)
        pending = []
        for index, (board, moves) in enumerate(zip(boards, legal_moves)):
            value = terminal_value(board)
            if value is not None:
                results[index] = ([1.0 / len(moves)] * len(moves) if moves else [], value)
            elif not moves:
                raise ValueError("A nonterminal position has no supplied legal actions")
            else:
                pending.append(index)
        if not pending:
            return results
        max_actions = max(len(legal_moves[index]) for index in pending)
        encoded = np.zeros((len(pending), max_actions, 6), dtype=np.int64)
        mask = np.zeros((len(pending), max_actions), dtype=np.bool_)
        for batch_index, index in enumerate(pending):
            count = len(legal_moves[index])
            encoded[batch_index, :count] = encode_actions(boards[index], legal_moves[index])
            mask[batch_index, :count] = True
        board_inputs = np.stack([encode_board(boards[index]) for index in pending])
        self.model.eval()
        with torch.inference_mode():
            policy, wdl = self.model(torch.from_numpy(board_inputs).to(self.device),
                                     torch.from_numpy(encoded).to(self.device),
                                     torch.from_numpy(mask).to(self.device))
            probabilities = policy.softmax(dim=-1).cpu().numpy()
            wdl = wdl.softmax(dim=-1).cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.isfinite(wdl).all():
            raise ValueError("Network produced nonfinite probabilities")
        for batch_index, index in enumerate(pending):
            results[index] = (probabilities[batch_index, :len(legal_moves[index])].tolist(),
                              float(wdl[batch_index, 0] - wdl[batch_index, 2]))
        return results


def rules_source_hash() -> str:
    """Rules identity (not the CPU heuristic), persisted with every checkpoint."""
    return hashlib.sha256((PUBLIC / "engine.py").read_bytes()).hexdigest()


def save_checkpoint(path, model: PolicyValueNet, optimizer=None, **metadata):
    """Atomically save model/config/versions/rules identity and training metadata."""
    checkpoint = {"checkpoint_version": CHECKPOINT_VERSION, "model_version": MODEL_VERSION,
                  "encoding_version": ENCODING_VERSION, "rules_sha256": rules_source_hash(),
                  "config": asdict(model.config), "model_state": model.state_dict(),
                  "metadata": metadata}
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            torch.save(checkpoint, output)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(path, device="cpu", *, strict_rules=True):
    """Return (model, payload); payload includes optional optimizer and metadata.

    Version/encoding mismatch is always rejected. Rules mismatch is rejected by
    default; strict_rules=False is an explicit escape hatch for reviewed rules
    edits. Only tensor/basic-data checkpoints are accepted by torch.load.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    for name, expected in (("checkpoint_version", CHECKPOINT_VERSION),
                           ("model_version", MODEL_VERSION), ("encoding_version", ENCODING_VERSION)):
        if checkpoint.get(name) != expected:
            raise ValueError(f"Unsupported {name}: {checkpoint.get(name)!r}; expected {expected}")
    if strict_rules and checkpoint.get("rules_sha256") != rules_source_hash():
        raise ValueError("Checkpoint rules hash differs from the current authoritative engine")
    model = PolicyValueNet(NetworkConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), checkpoint
