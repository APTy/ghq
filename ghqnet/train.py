"""Variable-action policy distillation and terminal-outcome WDL training."""
import random
import time

import numpy as np
import torch
from torch import nn

from .encoding import encode_actions, encode_board
from .rules import e


def device_name(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot see a GPU")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return requested


def prepare(samples):
    result = []
    for sample in samples:
        board = sample.validate()
        actions = [e.Move.from_uci(action) for action in sample.actions]
        result.append((encode_board(board), encode_actions(board, actions),
                       np.asarray(sample.policy, dtype=np.float32), sample.outcome))
    return result


def batch(records, device):
    count, width = len(records), max(len(row[1]) for row in records)
    actions = np.zeros((count, width, 6), dtype=np.int64)
    mask = np.zeros((count, width), dtype=np.bool_)
    targets = np.zeros((count, width), dtype=np.float32)
    outcomes = np.full(count, -1, dtype=np.int64)
    for index, (_, encoded, policy, outcome) in enumerate(records):
        actions[index, :len(encoded)] = encoded
        mask[index, :len(encoded)] = True
        targets[index, :len(encoded)] = policy
        if outcome is not None:
            outcomes[index] = outcome
    return tuple(torch.from_numpy(array).to(device) for array in
                 (np.stack([row[0] for row in records]), actions, mask, targets, outcomes))


def losses(model, tensors):
    boards, actions, mask, targets, outcomes = tensors
    policy, wdl = model(boards, actions, mask)
    log_policy = policy.log_softmax(-1).masked_fill(~mask, 0)
    policy_loss = -(targets * log_policy).sum(-1).mean()
    labeled = outcomes >= 0
    value_loss = (nn.functional.cross_entropy(wdl[labeled], outcomes[labeled])
                  if labeled.any() else wdl.sum() * 0)
    return policy_loss, value_loss


def fit(model, samples, *, epochs=2, batch_size=64, learning_rate=1e-3,
        device="cpu", seed=0, optimizer=None):
    started = time.perf_counter()
    if not samples or epochs < 1 or batch_size < 1:
        raise ValueError("Training requires samples, positive epochs and batch size")
    records = prepare(samples)
    model.to(device)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)

    def measure():
        model.eval()
        sums = [0.0, 0.0]
        labeled_total = sum(row[3] is not None for row in records)
        with torch.no_grad():
            for index in range(0, len(records), batch_size):
                rows = records[index:index + batch_size]
                values = losses(model, batch(rows, device))
                sums[0] += float(values[0]) * len(rows) / len(records)
                if labeled_total:
                    sums[1] += float(values[1]) * sum(row[3] is not None for row in rows) / labeled_total
        return {"policy": sums[0], "value": sums[1], "total": sum(sums)}

    before = measure()
    steps = 0
    for _ in range(epochs):
        rng.shuffle(records)
        model.train()
        for index in range(0, len(records), batch_size):
            optimizer.zero_grad(set_to_none=True)
            policy_loss, value_loss = losses(model, batch(records[index:index + batch_size], device))
            loss = policy_loss + value_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; refusing to save updated weights")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
    return optimizer, {"before": before, "after": measure(), "samples": len(samples),
                       "value_labels": sum(sample.outcome is not None for sample in samples),
                       "steps": steps, "seconds": time.perf_counter() - started}
