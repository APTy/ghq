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


class _PreparationDeadline(Exception):
    pass


def _value_mask(samples, value_mask, *, name="value_mask"):
    """Validate eligibility separately from the immutable recorded outcomes."""
    if value_mask is None:
        return None
    try:
        values = tuple(value_mask)
    except TypeError as error:
        raise ValueError(f"{name} must be a boolean sequence aligned with samples") from error
    if len(values) != len(samples) or any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{name} must be a boolean sequence aligned with samples")
    return values


def prepare(samples, deadline=None, value_mask=None):
    value_mask = _value_mask(samples, value_mask)
    result = []
    for index, sample in enumerate(samples):
        if deadline is not None and time.perf_counter() >= deadline:
            raise _PreparationDeadline()
        board = sample.validate()
        actions = [e.Move.from_uci(action) for action in sample.actions]
        result.append((encode_board(board), encode_actions(board, actions),
                       np.asarray(sample.policy, dtype=np.float32),
                       sample.outcome if value_mask is None or value_mask[index] else None))
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
        device="cpu", seed=0, optimizer=None, validation_samples=None, max_seconds=None,
        max_examples=None, value_mask=None, validation_value_mask=None):
    """Train a deterministic replay prefix, bounded by epochs, rows and time.

    ``max_examples`` counts actual row exposures, including repeat visits in
    later epochs. It does not count preparation or validation. The final batch
    is shortened to honor the cap exactly. Separate calls restart the seeded
    shuffle; optimizer continuation does not imply a restored data-order cursor.
    A cap smaller than the replay prepares and measures only that shuffled
    prefix; the returned measurement scope identifies this smaller panel.
    ``value_mask`` controls which recorded outcomes contribute to the value
    loss, without changing policy targets or raw replay records. Validation is
    unmasked unless its separate ``validation_value_mask`` is provided.
    """
    started = time.perf_counter()
    if max_examples is not None and (type(max_examples) is not int or max_examples < 0):
        raise ValueError("max_examples must be a nonnegative integer or None")
    if (not samples and max_examples != 0) or epochs < 1 or batch_size < 1:
        raise ValueError("Training requires samples, positive epochs and batch size")
    value_mask = _value_mask(samples, value_mask)
    validation_value_mask = _value_mask(validation_samples or [], validation_value_mask,
                                        name="validation_value_mask")
    deadline = None if max_seconds is None else started + max(0.0, max_seconds)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    epoch_limit = len(samples) * epochs
    planned = epoch_limit if max_examples is None else min(epoch_limit, max_examples)
    steps = trained_examples = epochs_completed = 0
    trained_value_examples = 0
    trained_value_examples_by_source = dict.fromkeys((sample.source for sample in samples), 0)
    raw_value_labels = sum(sample.outcome is not None for sample in samples)
    eligible_value_labels = sum(sample.outcome is not None and (value_mask is None or value_mask[index])
                                for index, sample in enumerate(samples))
    prefix_only = 0 < planned < len(samples)
    measurement_samples = min(len(samples), planned)

    def report(*, before=None, after=None, validation_before=None, validation_after=None,
               interrupted=False, preparation_interrupted=False):
        budget_exhausted = max_examples is not None and trained_examples >= max_examples
        reason = ("deadline" if interrupted else "example_budget"
                  if budget_exhausted and (max_examples < epoch_limit or max_examples == 0)
                  else "epochs_complete")
        return {"before": before, "after": after, "samples": len(samples),
                "value_labels": raw_value_labels, "raw_value_labels": raw_value_labels,
                "eligible_value_labels": int(eligible_value_labels),
                "trained_value_examples": trained_value_examples,
                "trained_value_examples_by_source": dict(trained_value_examples_by_source),
                "validation_before": validation_before, "validation_after": validation_after,
                "validation_samples": len(validation_samples or []), "interrupted": interrupted,
                "preparation_interrupted": preparation_interrupted, "steps": steps,
                "planned_examples": planned, "trained_examples": trained_examples,
                "epochs_completed": epochs_completed, "example_budget_exhausted": budget_exhausted,
                "stopped_reason": reason, "training_measurement_samples": measurement_samples,
                "training_measurement_scope": ("none" if planned == 0 else
                                               "shuffled_prefix" if prefix_only else "full_replay"),
                "seconds": time.perf_counter() - started}

    if planned == 0:
        return optimizer, report()
    if deadline is not None and time.perf_counter() >= deadline:
        return optimizer, report(interrupted=True, preparation_interrupted=True)
    model.to(device)
    rng = random.Random(seed)
    try:
        if prefix_only:
            # Shuffling integer indices has exactly the same permutation as
            # shuffling the fully prepared replay, without encoding unused rows.
            order = list(range(len(samples)))
            rng.shuffle(order)
            selected_samples = [samples[index] for index in order[:planned]]
            selected_mask = (None if value_mask is None else
                             [value_mask[index] for index in order[:planned]])
        else:
            selected_samples, selected_mask = samples, value_mask
        records = (prepare(selected_samples, deadline) if selected_mask is None else
                   prepare(selected_samples, deadline, value_mask=selected_mask))
        # Records carry their masked outcome through every shuffle. This map
        # tracks provenance without changing the public four-field record shape.
        record_sources = {id(record): sample.source for record, sample in zip(records, selected_samples)}
        validation = (prepare(validation_samples or [], deadline) if validation_value_mask is None else
                      prepare(validation_samples or [], deadline, value_mask=validation_value_mask))
    except _PreparationDeadline:
        return optimizer, report(interrupted=True, preparation_interrupted=True)

    def measure(rows_to_measure):
        model.eval()
        sums = [0.0, 0.0]
        labeled_total = sum(row[3] is not None for row in rows_to_measure)
        if not rows_to_measure:
            return None
        with torch.no_grad():
            for index in range(0, len(rows_to_measure), batch_size):
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                rows = rows_to_measure[index:index + batch_size]
                values = losses(model, batch(rows, device))
                sums[0] += float(values[0]) * len(rows) / len(rows_to_measure)
                if labeled_total:
                    sums[1] += float(values[1]) * sum(row[3] is not None for row in rows) / labeled_total
        return {"policy": sums[0], "value": sums[1], "total": sum(sums)}

    before = measure(records) if deadline is None else None
    validation_before = measure(validation) if deadline is None else None
    interrupted = False
    for _ in range(epochs):
        if trained_examples >= planned:
            break
        if not prefix_only:
            rng.shuffle(records)
        epoch_examples = min(len(records), planned - trained_examples)
        epoch_trained = 0
        model.train()
        for index in range(0, epoch_examples, batch_size):
            if deadline is not None and time.perf_counter() >= deadline:
                interrupted = True
                break
            rows = records[index:min(index + batch_size, epoch_examples)]
            optimizer.zero_grad(set_to_none=True)
            policy_loss, value_loss = losses(model, batch(rows, device))
            loss = policy_loss + value_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; refusing to save updated weights")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            trained_examples += len(rows)
            for row in rows:
                if row[3] is not None:
                    trained_value_examples += 1
                    trained_value_examples_by_source[record_sources[id(row)]] += 1
            epoch_trained += len(rows)
        epochs_completed += epoch_trained == len(samples)
        if interrupted:
            break
    # A bounded run prioritizes saving its model/optimizer over a full extra
    # replay pass. Independent validation is still measured when time remains.
    after = measure(records) if deadline is None else None
    validation_after = (measure(validation) if deadline is None or
                        time.perf_counter() < deadline else None)
    return optimizer, report(before=before, after=after, validation_before=validation_before,
                             validation_after=validation_after, interrupted=interrupted)
