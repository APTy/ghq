"""Prepare, or explicitly execute, a bounded value-label selection comparison.

Both arms learn identical policy rows. A samples mixed outcome labels; B uses
only explicitly proven same-model self-play. Their labeled counts match in
every optimizer batch. This also changes the distribution of value states: it
does not prove that mixed-opponent labels are invalid. No games are collected,
no winner is selected, and diagnostic checkpoints cannot resume production.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time

import numpy as np
import torch

from .ablate_replay import (AblationDeadline, check_time, evaluate_bounded, file_sha,
                            make_optimizer, metric_deltas, optimizer_steps,
                            tree_sha, verify_partition, write_json)
from .data import read_replay
from .model import load_checkpoint, rules_source_hash, save_checkpoint
from .rules import PUBLIC
from .train import device_name, fit
from .value_targets import audit_value_targets, value_eligible


def _hash_indices(indices):
    return hashlib.sha256(b"".join(struct.pack("<Q", index) for index in indices)).hexdigest()


def _label_summary(samples, mask):
    chosen = [row for row, eligible in zip(samples, mask) if eligible and row.outcome is not None]
    return {"rows": len(chosen), "games": len({row.game_id for row in chosen}),
            "families": len({row.family_id for row in chosen}),
            "wdl": {name: sum(row.outcome == index for row in chosen)
                    for index, name in enumerate(("win", "draw", "loss"))},
            "by_source": dict(sorted(Counter(row.source for row in chosen).items()))}


def make_seed_plan(samples, eligible, *, seed, steps, batch_size):
    """Mirror fit's first seeded permutation; select equal value counts per batch."""
    count = steps * batch_size
    if len(samples) < count:
        raise ValueError("Training replay is too small for a no-repeat policy prefix")
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    order = order[:count]
    control_seed = int.from_bytes(hashlib.sha256(f"value-control:{seed}".encode()).digest(), "big")
    control = random.Random(control_seed)
    masks = {"A": [False] * len(samples), "B": [False] * len(samples)}
    batches = []
    for start in range(0, count, batch_size):
        indices = order[start:start + batch_size]
        candidates = [index for index in indices if samples[index].outcome is not None]
        selfplay = [index for index in indices if eligible[index]]
        selected = control.sample(candidates, len(selfplay))
        for index in selected:
            masks["A"][index] = True
        for index in selfplay:
            masks["B"][index] = True
        batches.append({"batch": start // batch_size, "policy_rows": len(indices),
                        "raw_value_labels": len(candidates), "A_value_labels": len(selected),
                        "B_value_labels": len(selfplay), "overlap": len(set(selected) & set(selfplay))})
    outside = sum(include and not eligible[index] for index, include in enumerate(masks["A"]))
    control_count = sum(masks["A"])
    return {"seed": seed, "policy_rows": count, "order": order,
            "order_sha256": _hash_indices(order), "control_seed": str(control_seed),
            "masks": masks, "mask_sha256": {key: hashlib.sha256(bytes(mask)).hexdigest()
                                               for key, mask in masks.items()},
            "labels": {key: _label_summary(samples, mask) for key, mask in masks.items()},
            "overlap": sum(left and right for left, right in zip(masks["A"], masks["B"])),
            "control_outside_selfplay_labels": outside,
            "control_outside_selfplay_fraction": outside / control_count if control_count else 0,
            "per_batch": batches}


def _requirements(summary, *, rows, games, families, label):
    if summary["rows"] < rows or summary["games"] < games or summary["families"] < families:
        raise ValueError(f"Insufficient eligible {label}: need {rows} rows, {games} terminal games "
                         f"and {families} families; found {summary['rows']} rows, "
                         f"{summary['games']} games and {summary['families']} families")


def pooled_policy_metrics(metrics):
    """Pool source metrics by non-forced row counts, never by model/source count."""
    groups = {"teacher": [], "neural": []}
    for source, measured in metrics["by_source"].items():
        group = "teacher" if source.startswith(("ghq-cpu", "teacher")) else "neural"
        groups[group].append(measured["non_forced"])
    result = {}
    for name, rows in groups.items():
        count = sum(row["samples"] for row in rows)
        result[name] = {"non_forced_samples": count}
        for metric in ("soft_target_nll", "top1_agreement", "top5_agreement"):
            result[name][metric] = (sum(row["policy"][metric] * row["samples"]
                                        for row in rows if row["samples"]) / count if count else None)
    return result


def _policy_deltas(before, after):
    return {group: {metric: None if before[group][metric] is None or after[group][metric] is None else
                   after[group][metric] - before[group][metric]
                   for metric in ("soft_target_nll", "top1_agreement", "top5_agreement")}
            for group in ("teacher", "neural")}


def _paired_comparison(arms, seeds):
    pairs = []
    for seed in seeds:
        a, b = (next(arm for arm in arms if arm["seed"] == seed and arm["arm"] == name)
                for name in ("A", "B"))
        pairs.append({"seed": seed, "primary_selfplay": metric_deltas(a["primary_selfplay"], b["primary_selfplay"]),
                      "validation_diagnostic": metric_deltas(a["validation_diagnostic"], b["validation_diagnostic"]),
                      "legacy_diagnostic": metric_deltas(a["legacy_diagnostic"], b["legacy_diagnostic"]),
                      "legacy_policy_retention": _policy_deltas(a["legacy_diagnostic"]["pooled_policy"],
                                                                  b["legacy_diagnostic"]["pooled_policy"])})
    means = {}
    for panel in ("primary_selfplay", "validation_diagnostic", "legacy_diagnostic", "legacy_policy_retention"):
        means[panel] = {}
        for group, values in pairs[0][panel].items():
            means[panel][group] = {}
            for metric in values:
                numbers = [pair[panel][group][metric] for pair in pairs]
                means[panel][group][metric] = (None if any(number is None for number in numbers)
                                              else sum(numbers) / len(numbers))
    return {"direction": "B minus A", "paired": pairs, "mean_deltas": means,
            "automatic_selection": False, "playing_strength_established": False}


def run_ablation(args):
    validate_args(args)
    if args.output.exists():
        raise ValueError("Ablation output must be a new directory")
    started = time.perf_counter()
    deadline = started + args.max_seconds
    work_deadline = deadline - min(10.0, args.max_seconds * .1)
    args.output.mkdir(parents=True)
    report = {"version": 1, "purpose": "value-label selection screen", "status": "preparing",
              "complete": False, "comparison_eligible": False, "training_performed": False,
              "promoted": False, "winner": None, "resumable": False, "arms": [], "comparison": None,
              "inputs_unchanged": None, "created_at": datetime.now(timezone.utc).isoformat(),
              "limitations": ["Selecting self-play labels also changes the distribution of value states.",
                              "This comparison cannot establish that mixed-opponent labels are invalid.",
                              "Prediction metrics alone do not establish playing strength."]}
    inputs = {}

    def publish(event, **details):
        report["seconds"] = time.perf_counter() - started
        write_json(args.output / "report.json", report)
        record = {"event": event, "elapsed_seconds": report["seconds"], **details}
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        torch.set_num_threads(args.threads)
        for path in (args.checkpoint, args.training, args.validation, args.partition):
            inputs[path] = file_sha(path, work_deadline)
        # Freeze the implementations whose behavior determines this comparison.
        for name in ("ablate_value_targets.py", "ablate_replay.py", "train.py", "model.py",
                     "data.py", "value_targets.py", "encoding.py", "evaluate_replay.py", "curriculum.py", "rules.py"):
            path = Path(__file__).parent / name
            inputs[path] = file_sha(path, work_deadline)
        for name in ("engine.py", "search.py"):
            path = PUBLIC / name
            inputs[path] = file_sha(path, work_deadline)
        training, validation = read_replay(args.training), read_replay(args.validation)
        check_time(work_deadline, "reading replay")
        if not training or not validation or len(validation) > 8192:
            raise ValueError("Require nonempty training and a fixed validation panel of at most 8192 rows")
        for row in training + validation:
            if (not isinstance(row.game_id, str) or not row.game_id.strip() or
                    not isinstance(row.family_id, str) or not row.family_id.strip()):
                raise ValueError("Experiment rows require nonempty game and family IDs")
            if row.outcome is not None and type(row.outcome) is not int:
                raise ValueError("Experiment outcomes must be integer W/D/L or absent")
            if not isinstance(row.source, str) or not row.source:
                raise ValueError("Experiment rows require nonempty source IDs")
        state = json.loads(args.partition.read_text())
        partition = verify_partition(training, validation, state, work_deadline)
        eligible = [value_eligible(row, "selfplay", args.model_id) for row in training]
        validation_mask = [value_eligible(row, "selfplay", args.model_id) for row in validation]
        primary = [row for row, include in zip(validation, validation_mask) if include]
        legacy = [row for row, include in zip(validation, validation_mask) if not include]
        if not legacy:
            raise ValueError("Require a nonempty separate legacy validation panel")
        primary_summary = _label_summary(validation, validation_mask)
        _requirements(primary_summary, rows=args.min_validation_rows,
                      games=args.min_validation_games, families=args.min_validation_families,
                      label="validation self-play")
        plans = [make_seed_plan(training, eligible, seed=seed, steps=args.steps,
                                batch_size=args.batch_size) for seed in args.seeds]
        for plan in plans:
            _requirements(plan["labels"]["B"], rows=args.min_training_rows,
                          games=args.min_training_games, families=args.min_training_families,
                          label=f"training prefix for seed {plan['seed']}")
            if plan["control_outside_selfplay_fraction"] < args.min_control_contrast:
                raise ValueError(f"Insufficient control contrast for seed {plan['seed']}: "
                                 f"require {args.min_control_contrast:.1%} of A labels outside eligible self-play")
        # CPU loading and hashing are allowed during preparation; no inference,
        # optimization, collection or device allocation is needed to freeze it.
        model, payload = load_checkpoint(args.checkpoint, "cpu")
        optimizer, restored = make_optimizer(model, payload.get("optimizer_state"), args.learning_rate)
        initial_model = tree_sha(model.state_dict())
        initial_optimizer = tree_sha(payload["optimizer_state"])
        if restored["restored_optimizer_sha256"] != initial_optimizer:
            raise ValueError("Initial optimizer restoration changed its state")
        initial_steps = optimizer_steps(optimizer)
        del optimizer
        manifest = {"version": 1, "configuration": {key: str(value) if isinstance(value, Path) else value
                                                   for key, value in vars(args).items()},
                    "rules_sha256": rules_source_hash(), "torch": str(torch.__version__),
                    "inputs": {str(path.resolve()): digest for path, digest in inputs.items()},
                    "partition": partition, "initial_model_sha256": initial_model,
                    "initial_optimizer_sha256": initial_optimizer, "initial_optimizer_steps": initial_steps,
                    "training_audit": audit_value_targets(training, "selfplay", args.model_id),
                    "validation_audit": audit_value_targets(validation, "selfplay", args.model_id),
                    "primary_validation": primary_summary, "seeds": plans,
                    "legacy_validation": {"rows": len(legacy), "games": len({row.game_id for row in legacy}),
                                          "families": len({row.family_id for row in legacy}),
                                          "by_source": dict(sorted(Counter(row.source for row in legacy).items()))},
                    "limitations": report["limitations"]}
        write_json(args.output / "manifest.json", manifest)
        report["manifest_sha256"] = file_sha(args.output / "manifest.json", work_deadline)
        report["manifest"] = "manifest.json"
        publish("manifest-frozen", execute=args.execute)
        if args.execute:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
            device = device_name(args.device)
            model.to(device)

            def evaluate(current):
                result = {"primary_selfplay": evaluate_bounded(current, primary, device=device,
                                                             batch_size=args.batch_size, deadline=work_deadline),
                        "validation_diagnostic": evaluate_bounded(current, validation, device=device,
                                                                   batch_size=args.batch_size, deadline=work_deadline),
                        "legacy_diagnostic": evaluate_bounded(current, legacy, device=device,
                                                               batch_size=args.batch_size, deadline=work_deadline)}
                result["legacy_diagnostic"]["pooled_policy"] = pooled_policy_metrics(result["legacy_diagnostic"])
                return result

            report["baseline"] = evaluate(model)
            del model, payload
            publish("baseline-evaluated")
            for plan in plans:
                for name in ("A", "B"):
                    check_time(work_deadline, "arm initialization")
                    model, payload = load_checkpoint(args.checkpoint, device)
                    if (tree_sha(model.state_dict()) != initial_model or
                            tree_sha(payload["optimizer_state"]) != initial_optimizer):
                        raise ValueError("Arm did not reload the exact initial model and Adam state")
                    optimizer, restored = make_optimizer(model, payload["optimizer_state"], args.learning_rate)
                    if restored["restored_optimizer_sha256"] != initial_optimizer:
                        raise ValueError("Arm optimizer state was not restored exactly")
                    before_steps = optimizer_steps(optimizer)
                    random.seed(plan["seed"])
                    np.random.seed(plan["seed"])
                    torch.manual_seed(plan["seed"])
                    if device == "cuda":
                        torch.cuda.manual_seed_all(plan["seed"])
                    arm = {"arm": name, "seed": plan["seed"], "complete": False,
                           "usable_for_comparison": False, "initial_model_sha256": initial_model,
                           "initial_optimizer_sha256": initial_optimizer,
                           "order_sha256": plan["order_sha256"], "mask_sha256": plan["mask_sha256"][name],
                           "initial_optimizer_steps": before_steps}
                    report["arms"].append(arm)
                    publish("arm-start", arm=name, seed=plan["seed"])
                    optimizer, metrics = fit(model, training, epochs=1, batch_size=args.batch_size,
                                             learning_rate=args.learning_rate, device=device, seed=plan["seed"],
                                             optimizer=optimizer, validation_samples=[], value_mask=plan["masks"][name],
                                             max_examples=plan["policy_rows"],
                                             max_seconds=max(0.0, work_deadline - time.perf_counter()))
                    arm["training"] = metrics
                    report["training_performed"] |= metrics.get("steps", 0) > 0
                    after_steps = optimizer_steps(optimizer)
                    arm["final_optimizer_steps"] = after_steps
                    if metrics.get("interrupted"):
                        raise AblationDeadline("An arm exceeded its training deadline")
                    if (metrics.get("steps") != args.steps or metrics.get("trained_examples") != plan["policy_rows"] or
                            metrics.get("trained_value_examples") != plan["labels"][name]["rows"]):
                        raise ValueError("Actual policy/value training exposure differs from frozen arm budget")
                    actual_sources = {key: value for key, value in metrics.get("trained_value_examples_by_source", {}).items() if value}
                    if actual_sources != plan["labels"][name]["by_source"]:
                        raise ValueError("Actual per-source value exposure differs from the frozen mask")
                    if any(after_steps.get(key) != value + args.steps for key, value in before_steps.items()):
                        raise ValueError("Actual Adam counters differ from the frozen step budget")
                    if any(value != args.steps for key, value in after_steps.items() if key not in before_steps):
                        raise ValueError("New Adam counters differ from the frozen step budget")
                    arm.update(evaluate(model))
                    path = args.output / f"seed-{plan['seed']}-arm-{name}.pt"
                    check_time(work_deadline, "saving diagnostic checkpoint")
                    save_checkpoint(path, model, purpose="diagnostic-value-target-ablation", resumable=False,
                                    promoted=False, input_checkpoint_sha256=inputs[args.checkpoint],
                                    manifest_sha256=report["manifest_sha256"], arm=name, seed=plan["seed"])
                    arm.update(complete=True, checkpoint=path.name, checkpoint_sha256=file_sha(path, work_deadline))
                    publish("arm-complete", arm=name, seed=plan["seed"])
                    del model, optimizer, payload
            report["comparison"] = _paired_comparison(report["arms"], args.seeds)
        for path, digest in inputs.items():
            if file_sha(path, deadline) != digest:
                raise ValueError(f"Input changed during ablation: {path}")
        report["inputs_unchanged"] = True
        if args.execute:
            for arm in report["arms"]:
                arm["usable_for_comparison"] = True
        report.update(complete=True, comparison_eligible=args.execute,
                      status="complete" if args.execute else "prepared")
        publish("comparison-complete" if args.execute else "preparation-complete")
    except Exception as error:
        report.update(status="incomplete-deadline" if isinstance(error, AblationDeadline) else "failed",
                      complete=False, comparison_eligible=False, comparison=None,
                      error=f"{type(error).__name__}: {error}")
        for arm in report["arms"]:
            arm["usable_for_comparison"] = False
        try:
            report["inputs_unchanged"] = all(file_sha(path, deadline) == digest for path, digest in inputs.items()) if inputs else None
        except (OSError, ValueError, AblationDeadline):
            report["inputs_unchanged"] = None
        publish("incomplete" if isinstance(error, AblationDeadline) else "failed", reason=report["error"])
        if not isinstance(error, AblationDeadline):
            raise
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old_cudnn
        if old_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training", "validation", "partition", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--model-id", action="append", required=True, help="Exact eligible self-play model ID; repeat for an allowlist")
    result.add_argument("--seeds", nargs="+", type=int, default=[20261125, 20261126, 20261127])
    result.add_argument("--steps", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=.001)
    result.add_argument("--max-seconds", type=float, default=900)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    result.add_argument("--min-training-games", type=int, default=16)
    result.add_argument("--min-validation-games", type=int, default=8)
    result.add_argument("--min-training-families", type=int, default=16)
    result.add_argument("--min-validation-families", type=int, default=8)
    result.add_argument("--min-training-rows", type=int, default=512)
    result.add_argument("--min-validation-rows", type=int, default=256)
    result.add_argument("--min-control-contrast", type=float, default=.2,
                        help="Minimum fraction of A labels outside B's self-play eligibility")
    result.add_argument("--execute", action="store_true", help="Run optimization and inference after freezing the manifest")
    return result


def validate_args(args):
    for name, maximum in (("steps", 4096), ("batch_size", 512), ("threads", 64),
                          ("min_training_games", 1000000), ("min_validation_games", 8192),
                          ("min_training_families", 1000000), ("min_validation_families", 8192),
                          ("min_training_rows", 1000000), ("min_validation_rows", 8192)):
        value = getattr(args, name)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    if (not 1 <= len(args.seeds) <= 8 or len(set(args.seeds)) != len(args.seeds) or
            any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in args.seeds)):
        raise ValueError("Provide 1–8 distinct seeds in [0,2**32)")
    if (not args.model_id or len(set(args.model_id)) != len(args.model_id) or
            any(not isinstance(value, str) or not value.strip() for value in args.model_id)):
        raise ValueError("Provide distinct nonempty exact model IDs")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1:
        raise ValueError("learning-rate must be finite and in (0,1]")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("max-seconds must be finite and positive")
    if args.min_training_rows > args.steps * args.batch_size:
        raise ValueError("Minimum eligible training rows exceed the policy prefix budget")
    if not math.isfinite(args.min_control_contrast) or not 0 <= args.min_control_contrast <= 1:
        raise ValueError("min-control-contrast must be finite and in [0,1]")
    return args


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        report = run_ablation(args)
    except (ValueError, OSError, RuntimeError) as error:
        command.exit(1, f"{error}\n")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
