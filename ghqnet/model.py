"""Residual policy/value network with a shared scorer over exact legal actions."""
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn

from .encoding import (BOARD_CHANNELS, ENCODING_VERSION, NO_SQUARE,
                       encode_actions, encode_board)
from .rules import PUBLIC, terminal_value

MODEL_VERSION = 1
CHECKPOINT_VERSION = 1
_ACTION_MAXIMA = np.asarray([64, 64, 64, 8, 5, 7], dtype=np.int64)


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
    supports_inference_profile = True

    def __init__(self, model: PolicyValueNet, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def predict(self, board, legal_moves):
        return self.predict_batch([board], [legal_moves])[0]

    def predict_batch(self, boards: Sequence, legal_moves: Sequence, *, profile=None):
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
        tick = time.perf_counter() if profile is not None else 0
        board_inputs = [encode_board(boards[index]) for index in pending]
        action_inputs = [encode_actions(boards[index], legal_moves[index]) for index in pending]
        if profile is not None:
            profile["encode_seconds"] = profile.get("encode_seconds", 0.0) + time.perf_counter() - tick
        predictions = self.predict_encoded_batch(board_inputs, action_inputs, profile=profile)
        for index, prediction in zip(pending, predictions):
            results[index] = prediction
        return results

    def predict_encoded_batch(self, board_inputs: Sequence, action_inputs: Sequence, *, profile=None):
        """Score already canonicalized nonterminal inputs, without rules parsing.

        Each board is a finite float32 ndarray [C,8,8]; its aligned legal actions
        are an int64 ndarray [A,6], A>=1, in the original requested order.
        Callers must handle terminals before this entry point. The result has
        the same (probabilities, current-player W-minus-L) contract as predict.

        Optional ``profile`` accumulates calls, rows and phase *_seconds.
        Packing includes validation/padding; forward includes both softmaxes;
        d2h_postprocess includes output transfers, validation and Python lists.
        CPU phases are wall time. Explicit CUDA profiling synchronizes phase
        boundaries and reports synchronized wall time, not GPU event time.
        No extra CUDA synchronization is added when profiling is disabled.
        """
        if len(board_inputs) != len(action_inputs):
            raise ValueError("board_inputs and action_inputs must have equal lengths")
        if not len(board_inputs):
            return []

        def boundary():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return time.perf_counter()

        tick = boundary() if profile is not None else 0
        counts = []
        for board, actions in zip(board_inputs, action_inputs):
            if (not isinstance(board, np.ndarray) or board.dtype != np.float32 or
                    board.shape != (BOARD_CHANNELS, 8, 8)):
                raise ValueError(f"Each board input must be float32 [{BOARD_CHANNELS},8,8]")
            if (not isinstance(actions, np.ndarray) or actions.dtype != np.int64 or
                    actions.ndim != 2 or actions.shape[1] != 6 or not len(actions)):
                raise ValueError("Each action input must be int64 [A,6] with at least one action")
            if np.any(actions < 0) or np.any(actions > _ACTION_MAXIMA):
                raise ValueError("Encoded action descriptor out of range")
            counts.append(len(actions))
        boards = np.stack(board_inputs)
        if not np.isfinite(boards).all():
            raise ValueError("Board inputs must be finite")
        encoded = np.zeros((len(counts), max(counts), 6), dtype=np.int64)
        mask = np.zeros(encoded.shape[:2], dtype=np.bool_)
        for index, (actions, count) in enumerate(zip(action_inputs, counts)):
            encoded[index, :count] = actions
            mask[index, :count] = True
        packing_end = time.perf_counter() if profile is not None else 0
        self.model.eval()
        with torch.inference_mode():
            board_tensor = torch.from_numpy(boards).to(self.device)
            action_tensor = torch.from_numpy(encoded).to(self.device)
            mask_tensor = torch.from_numpy(mask).to(self.device)
            h2d_end = boundary() if profile is not None else 0
            policy, wdl = self.model(board_tensor, action_tensor, mask_tensor)
            policy, wdl = policy.softmax(dim=-1), wdl.softmax(dim=-1)
            forward_end = boundary() if profile is not None else 0
            probabilities, wdl = policy.cpu().numpy(), wdl.cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.isfinite(wdl).all():
            raise ValueError("Network produced nonfinite probabilities")
        results = [(probabilities[index, :count].tolist(), float(wdl[index, 0] - wdl[index, 2]))
                   for index, count in enumerate(counts)]
        if profile is not None:
            end = boundary()
            profile.update(device=str(self.device), timing_scope=("synchronized_wall_seconds"
                           if self.device.type == "cuda" else "wall_seconds"))
            profile["calls"] = profile.get("calls", 0) + 1
            profile["rows"] = profile.get("rows", 0) + len(counts)
            for name, duration in (("packing_seconds", packing_end - tick),
                                   ("h2d_seconds", h2d_end - packing_end),
                                   ("forward_seconds", forward_end - h2d_end),
                                   ("d2h_postprocess_seconds", end - forward_end)):
                profile[name] = profile.get(name, 0.0) + duration
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
