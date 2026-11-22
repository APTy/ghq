"""Read-only diagnostics on an explicitly supplied held-out replay.

Teacher/search agreement measures imitation, not playing strength. A supplied
validation file does not by itself prove absence of historical training leakage.
No model or replay updates are performed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .curriculum import fingerprint
from .data import read_replay
from .model import load_checkpoint, rules_source_hash
from .rules import e
from .train import batch, device_name, prepare

REPORT_VERSION = 1
WDL_NAMES = ("win", "draw", "loss")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family(sample):
    return sample.family_id or sample.game_id or "unidentified"


def select_indices(samples, max_samples=None, seed=4173):
    """Deterministic family round-robin; a cap may retain only part of a family."""
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")
    if max_samples is None or max_samples >= len(samples):
        return list(range(len(samples)))

    def order(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).digest()

    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[_family(sample)].append(index)
    families = sorted(groups, key=order)
    for family, indices in groups.items():
        indices.sort(key=lambda index: order(f"{family}:{samples[index].state}:{index}"))
    selected, cursor = [], 0
    while len(selected) < max_samples:
        for family in families:
            if cursor < len(groups[family]):
                selected.append(groups[family][cursor])
                if len(selected) == max_samples:
                    break
        cursor += 1
    return sorted(selected)


def row_metrics(sample, log_policy, log_wdl):
    """Score one row. Lexical UCI order breaks target and prediction ties."""
    actions = sample.actions
    log_policy = np.asarray(log_policy, dtype=np.float64)
    log_wdl = np.asarray(log_wdl, dtype=np.float64)
    target = np.asarray(sample.policy, dtype=np.float64)
    if log_policy.shape != target.shape or log_wdl.shape != (3,) or not np.isfinite(log_policy).all() or not np.isfinite(log_wdl).all():
        raise ValueError("Nonfinite or incorrectly shaped inference output")
    if not np.isclose(np.exp(log_policy).sum(), 1) or not np.isclose(np.exp(log_wdl).sum(), 1):
        raise ValueError("Inference log probabilities must be normalized")
    ranked = sorted(range(len(actions)), key=lambda index: (-log_policy[index], actions[index]))
    target_best = min(range(len(actions)), key=lambda index: (-target[index], actions[index]))
    board = e.BaseBoard.deserialize(sample.state)
    moves = [e.Move.from_uci(action) for action in actions]
    automatic = all(move.name == "AutoCapture" for move in moves)
    if automatic:
        phase = "automatic_" + str(moves[0].auto_capture_type)
    else:
        phase = {0: "fresh_turn_start", 1: "second_action", 2: "third_action"}.get(board.turn_moves, "invalid_action_index")
    skip = actions.index("skip") if "skip" in actions else None
    predicted = ranked[0]
    row = {"source": sample.source, "family": _family(sample), "game_id": sample.game_id,
           "phase": phase, "turn_actions_used": board.turn_moves,
           "single_legal_action": len(actions) == 1, "automatic_capture": automatic,
           "non_forced": len(actions) > 1 and not automatic,
           "canonical_state_id": fingerprint(sample), "legal_actions": len(actions),
           "top1": float(predicted == target_best), "top5": float(target_best in ranked[:5]),
           "target_mass_at_top1": float(target[predicted]), "target_mass_at_top5": float(target[ranked[:5]].sum()),
           "policy_nll": float(-np.dot(target, log_policy)),
           "target_skip": float(skip == target_best), "predicted_skip": float(skip == predicted),
           "target_skip_mass": 0.0 if skip is None else float(target[skip]),
           "predicted_skip_mass": 0.0 if skip is None else float(np.exp(log_policy[skip])),
           "outcome": sample.outcome}
    if sample.outcome is not None:
        expected = np.eye(3)[sample.outcome]
        row.update(wdl_logloss=float(-log_wdl[sample.outcome]),
                   wdl_brier=float(np.square(np.exp(log_wdl) - expected).sum()),
                   wdl_accuracy=float(int(np.argmax(log_wdl)) == sample.outcome))
    return row


def aggregate(rows):
    def mean(key, subset=rows):
        return float(np.mean([row[key] for row in subset])) if subset else None

    labeled = [row for row in rows if row["outcome"] is not None]
    fresh = [row for row in rows if row["phase"] == "fresh_turn_start"]
    return {"samples": len(rows), "canonical_unique_states": len({row["canonical_state_id"] for row in rows}),
            "families": len({row["family"] for row in rows}), "mean_legal_actions": mean("legal_actions"),
            "policy": {"top1_agreement": mean("top1"), "top5_agreement": mean("top5"),
                       "target_mass_at_predicted_top1": mean("target_mass_at_top1"),
                       "target_mass_at_predicted_top5": mean("target_mass_at_top5"),
                       "soft_target_nll": mean("policy_nll")},
            "fresh_turn_skip": {"samples": len(fresh), "target_argmax_rate": mean("target_skip", fresh),
                                "predicted_argmax_rate": mean("predicted_skip", fresh),
                                "target_probability_mean": mean("target_skip_mass", fresh),
                                "predicted_probability_mean": mean("predicted_skip_mass", fresh)},
            "wdl": {"labeled_samples": len(labeled), "unlabeled_samples": len(rows) - len(labeled),
                    "class_counts": {name: sum(row["outcome"] == index for row in labeled)
                                     for index, name in enumerate(WDL_NAMES)},
                    "logloss": mean("wdl_logloss", labeled), "brier": mean("wdl_brier", labeled),
                    "accuracy": mean("wdl_accuracy", labeled)}}


def breakdown(rows):
    """Every aggregate includes an explicitly non-forced companion."""
    return {"all": aggregate(rows),
            "non_forced": aggregate([row for row in rows if row["non_forced"]]),
            "single_legal_action": aggregate([row for row in rows if row["single_legal_action"]]),
            "automatic_capture": aggregate([row for row in rows if row["automatic_capture"]])}


def evaluate_samples(model, samples, *, device="cpu", batch_size=16):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    rows = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            selected = samples[start:start + batch_size]
            boards, actions, mask, _, _ = batch(prepare(selected), device)
            policy, wdl = model(boards, actions, mask)
            log_policy = policy.log_softmax(-1).cpu().numpy()
            log_wdl = wdl.log_softmax(-1).cpu().numpy()
            for index, sample in enumerate(selected):
                rows.append(row_metrics(sample, log_policy[index, :len(sample.actions)], log_wdl[index]))
    sources = sorted({row["source"] for row in rows})
    phases = sorted({row["phase"] for row in rows})
    return {"overall": breakdown(rows),
            "by_source": {source: breakdown([row for row in rows if row["source"] == source]) for source in sources},
            "by_phase": {phase: breakdown([row for row in rows if row["phase"] == phase]) for phase in phases},
            "by_source_phase": {source: {phase: breakdown([row for row in rows if row["source"] == source and row["phase"] == phase])
                                         for phase in phases if any(row["source"] == source and row["phase"] == phase for row in rows)}
                                for source in sources}}


def evaluate_checkpoint(checkpoint, heldout_replay, *, device="cpu", batch_size=16, max_samples=None, seed=4173):
    started = time.perf_counter()
    checkpoint, heldout_replay = Path(checkpoint), Path(heldout_replay)
    checkpoint_sha = file_sha256(checkpoint)
    replay_sha = file_sha256(heldout_replay)
    with heldout_replay.open() as source:
        header = json.loads(next(source))
    if not isinstance(header, dict):
        raise ValueError("Replay header must be an object")
    replay_metadata = header.get("metadata", {})
    if not isinstance(replay_metadata, dict):
        raise ValueError("Replay metadata must be an object")
    if replay_metadata.get("kind") in ("training", "train"):
        raise ValueError("The supplied held-out replay is marked as training data")
    device = device_name(device)
    model, payload = load_checkpoint(checkpoint, device)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be an object")
    if metadata.get("replay_buffer_sha256") == replay_sha:
        raise ValueError("The supplied held-out replay matches the checkpoint training buffer")
    samples = read_replay(heldout_replay)
    if not samples:
        raise ValueError("Held-out replay contains no samples")
    # Sample.validate accepts bool as an int; reject malformed outcome types
    # here rather than quietly scoring a fabricated draw or fractional label.
    if any(sample.outcome is not None and type(sample.outcome) is not int for sample in samples):
        raise ValueError("Outcome labels must be integer W/D/L classes or absent")
    if any(not isinstance(sample.source, str) or not isinstance(sample.family_id, str) or not isinstance(sample.game_id, str)
           for sample in samples):
        raise ValueError("Replay source, family and game identities must be strings")
    indices = select_indices(samples, max_samples, seed)
    selected = [samples[index] for index in indices]
    metrics = evaluate_samples(model, selected, device=device, batch_size=batch_size)
    validation_sha = metadata.get("validation_buffer_sha256")
    matching_validation = validation_sha == replay_sha if isinstance(validation_sha, str) else None
    if file_sha256(checkpoint) != checkpoint_sha or file_sha256(heldout_replay) != replay_sha:
        raise ValueError("Checkpoint or replay changed during evaluation")
    return {"version": REPORT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
            "replay": str(heldout_replay.resolve()), "replay_sha256": replay_sha,
            "rules_sha256": rules_source_hash(), "checkpoint_rules_sha256": payload["rules_sha256"],
            "replay_rules_sha256": header["rules_sha256"], "checkpoint_iteration": metadata.get("iteration"),
            "holdout": {"caller_declared": True, "replay_kind": replay_metadata.get("kind"),
                        "matches_checkpoint_validation_buffer": matching_validation,
                        "historical_training_overlap_verified_absent": False,
                        "note": "Validation-buffer identity is checked when recorded, but historical game-family/state leakage is not independently audited."},
            "sampling": {"method": "full" if len(selected) == len(samples) else "deterministic_family_round_robin",
                         "seed": seed, "max_samples": max_samples, "available_samples": len(samples),
                         "evaluated_samples": len(selected), "available_canonical_unique_states": len({fingerprint(row) for row in samples}),
                         "evaluated_canonical_unique_states": metrics["overall"]["all"]["canonical_unique_states"],
                         "available_families": len({_family(row) for row in samples}),
                         "evaluated_families": len({_family(row) for row in selected}),
                         "unidentified_family_samples": sum(not row.family_id and not row.game_id for row in samples),
                         "note": "Counts are sample-weighted, not independent-game estimates; capped sampling may retain only part of each family."},
            "definitions": {"policy": "Top1/top5 compare the modal target action against model ranks; ties use lexical UCI order. NLL is full soft-target cross-entropy in nats.",
                            "forced": "Single-action and automatic-capture groups can overlap. Non-forced excludes both; all groups include a non-forced companion.",
                            "phase": "Fresh-turn start is the first chosen action (turn_moves=0), after any mandatory automatic captures; second/third use turn_moves=1/2.",
                            "wdl": "Only replay outcome labels are scored, for the player to move. Brier is the sum across W/D/L (range 0..2). No cutoff is assigned a draw.",
                            "strength": "Teacher/search-target agreement is imitation, not playing strength; use paired game evaluations separately."},
            "device": str(device), "batch_size": batch_size, "seconds": time.perf_counter() - started, **metrics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=4173)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("threads, batch-size and max-samples must be positive")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    torch.set_num_threads(args.threads)
    try:
        report = evaluate_checkpoint(args.checkpoint, args.heldout_replay, device=args.device,
                                     batch_size=args.batch_size, max_samples=args.max_samples, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            json.dump(report, target, indent=2, allow_nan=False)
            target.write("\n")
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, f"{error}\n")
    scores = report["overall"]["non_forced"]
    print(json.dumps({"output": str(args.output), "evaluated_samples": report["sampling"]["evaluated_samples"],
                      "non_forced_samples": scores["samples"], "non_forced_policy": scores["policy"],
                      "wdl": report["overall"]["all"]["wdl"], "note": report["definitions"]["strength"]}, indent=2))


if __name__ == "__main__":
    main()
