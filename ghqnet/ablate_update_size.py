"""Compare gentler updates on fixed policy rows and verified self-play values.

Each rate/seed reloads the same model and saved Adam moments/counters. Only the
learning rate changes; the rows, order, value mask and step count stay paired.
Preparation freezes the manifest without inference. Execution is opt-in and
produces diagnostic weights, never a selected or promoted playing model.
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
import time

import numpy as np
import torch

from .ablate_replay import (AblationDeadline, check_time, evaluate_bounded, file_sha,
                            make_optimizer, metric_deltas, optimizer_steps,
                            tree_sha, verify_partition, write_json)
from .ablate_value_targets import (_hash_indices, _label_summary, _policy_deltas,
                                   _requirements, pooled_policy_metrics)
from .data import read_replay
from .model import load_checkpoint, rules_source_hash, save_checkpoint
from .rules import PUBLIC
from .train import device_name, fit
from .value_targets import audit_value_targets, value_eligible


def rate_name(rate):
    return f"lr-{rate:.12g}"


def make_seed_plan(samples, eligible, *, seed, steps, batch_size):
    """Freeze exactly fit's first epoch or no-repeat prefix, shared by all rates."""
    if len(eligible) != len(samples) or any(type(value) is not bool for value in eligible):
        raise ValueError("Eligibility mask must contain one boolean per training row")
    count = steps * batch_size
    if len(samples) < count:
        raise ValueError("Training replay is too small for a no-repeat policy prefix")
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    order = order[:count]
    mask = [False] * len(samples)
    for index in order:
        if eligible[index] and samples[index].outcome is None:
            raise ValueError("Eligible self-play row lacks a value label")
        mask[index] = eligible[index]
    return {"seed": seed, "policy_rows": count, "order": order,
            "order_sha256": _hash_indices(order), "mask": mask,
            "mask_sha256": hashlib.sha256(bytes(mask)).hexdigest(),
            "labels": _label_summary(samples, mask),
            "per_batch": [{"batch": start // batch_size, "policy_rows": batch_size,
                           "value_labels": sum(mask[index] for index in order[start:start + batch_size])}
                          for start in range(0, count, batch_size)]}


def require_finite(value, context):
    """Reject NaN/Inf metrics or tensors before publishing/saving a candidate."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Nonfinite {context}")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item, context)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item, context)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError(f"Nonfinite {context}")


def summarize(baseline, arms, rates):
    """Report paired baseline differences; selection belongs to a frozen outer plan."""
    groups = []
    for rate in rates:
        chosen = [arm for arm in arms if arm["learning_rate"] == rate]
        pairs = []
        for arm in chosen:
            item = {"seed": arm["seed"]}
            for panel in ("primary_selfplay", "validation_diagnostic", "legacy_diagnostic"):
                item[panel] = metric_deltas(baseline[panel], arm[panel])
            item["legacy_policy_retention"] = _policy_deltas(
                baseline["legacy_diagnostic"]["pooled_policy"], arm["legacy_diagnostic"]["pooled_policy"])
            pairs.append(item)
        means = {}
        for panel in ("primary_selfplay", "validation_diagnostic", "legacy_diagnostic", "legacy_policy_retention"):
            means[panel] = {}
            for group, metrics in pairs[0][panel].items():
                means[panel][group] = {}
                for metric in metrics:
                    numbers = [pair[panel][group][metric] for pair in pairs]
                    means[panel][group][metric] = (None if any(value is None for value in numbers)
                                                  else sum(numbers) / len(numbers))
        groups.append({"learning_rate": rate, "paired": pairs, "mean_deltas": means})
    return {"direction": "updated minus unchanged baseline", "rates": groups,
            "automatic_selection": False, "playing_strength_established": False}


def run_ablation(args):
    validate_args(args)
    if args.output.exists():
        raise ValueError("Ablation output must be a new directory")
    started = time.perf_counter()
    deadline = started + args.max_seconds
    work_deadline = deadline - min(10.0, args.max_seconds * .1)
    args.output.mkdir(parents=True)
    report = {"version": 1, "purpose": "fixed-data self-play update-size comparison",
              "status": "preparing", "complete": False, "comparison_eligible": False,
              "training_performed": False, "promoted": False, "winner": None,
              "resumable": False, "arms": [], "comparison": None, "inputs_unchanged": None,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "limitations": ["The development data and shuffle seeds were used in a previous experiment.",
                              "This calibrates an update on fixed data; it does not test adding more self-play.",
                              "Prediction metrics and teacher agreement do not establish playing strength."]}
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
        for name in ("ablate_update_size.py", "ablate_value_targets.py", "ablate_replay.py", "train.py", "model.py",
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
            if any(not isinstance(identity, str) or not identity.strip() for identity in (row.game_id, row.family_id, row.source)):
                raise ValueError("Experiment rows require nonempty game, family and source IDs")
            if row.outcome is not None and type(row.outcome) is not int:
                raise ValueError("Experiment outcomes must be integer W/D/L or absent")
        partition = verify_partition(training, validation, json.loads(args.partition.read_text()), work_deadline)
        eligible = [value_eligible(row, "selfplay", args.model_id) for row in training]
        validation_mask = [value_eligible(row, "selfplay", args.model_id) for row in validation]
        primary = [row for row, include in zip(validation, validation_mask) if include]
        legacy = [row for row, include in zip(validation, validation_mask) if not include]
        if not legacy:
            raise ValueError("Require a nonempty separate legacy validation panel")
        primary_summary = _label_summary(validation, validation_mask)
        _requirements(primary_summary, rows=args.min_validation_rows, games=args.min_validation_games,
                      families=args.min_validation_families, label="validation self-play")
        plans = [make_seed_plan(training, eligible, seed=seed, steps=args.steps,
                                batch_size=args.batch_size) for seed in args.seeds]
        for plan in plans:
            _requirements(plan["labels"], rows=args.min_training_rows, games=args.min_training_games,
                          families=args.min_training_families, label=f"training prefix for seed {plan['seed']}")
        model, payload = load_checkpoint(args.checkpoint, "cpu")
        require_finite(model.state_dict(), "initial model")
        optimizer, restored = make_optimizer(model, payload.get("optimizer_state"), args.learning_rates[0])
        require_finite(payload["optimizer_state"], "initial optimizer")
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
                result = {name: evaluate_bounded(current, rows, device=device, batch_size=args.batch_size,
                                                deadline=work_deadline)
                          for name, rows in (("primary_selfplay", primary), ("validation_diagnostic", validation),
                                             ("legacy_diagnostic", legacy))}
                result["legacy_diagnostic"]["pooled_policy"] = pooled_policy_metrics(result["legacy_diagnostic"])
                require_finite(result, "evaluation metrics")
                return result

            report["baseline"] = evaluate(model)
            del model, payload
            publish("baseline-evaluated")
            for plan in plans:
                for rate in args.learning_rates:
                    check_time(work_deadline, "arm initialization")
                    model, payload = load_checkpoint(args.checkpoint, device)
                    if (tree_sha(model.state_dict()) != initial_model or
                            tree_sha(payload["optimizer_state"]) != initial_optimizer):
                        raise ValueError("Arm did not reload the exact initial model and Adam state")
                    optimizer, restored = make_optimizer(model, payload["optimizer_state"], rate)
                    if restored["restored_optimizer_sha256"] != initial_optimizer:
                        raise ValueError("Arm optimizer state was not restored exactly")
                    before_steps = optimizer_steps(optimizer)
                    random.seed(plan["seed"])
                    np.random.seed(plan["seed"])
                    torch.manual_seed(plan["seed"])
                    if device == "cuda":
                        torch.cuda.manual_seed_all(plan["seed"])
                    arm = {"arm": rate_name(rate), "learning_rate": rate, "seed": plan["seed"], "complete": False,
                           "usable_for_comparison": False, "initial_model_sha256": initial_model,
                           "initial_optimizer_sha256": initial_optimizer, "optimizer_restoration": restored,
                           "order_sha256": plan["order_sha256"], "mask_sha256": plan["mask_sha256"],
                           "initial_optimizer_steps": before_steps}
                    report["arms"].append(arm)
                    publish("arm-start", learning_rate=rate, seed=plan["seed"])
                    optimizer, metrics = fit(model, training, epochs=1, batch_size=args.batch_size,
                                             learning_rate=rate, device=device, seed=plan["seed"],
                                             optimizer=optimizer, validation_samples=[], value_mask=plan["mask"],
                                             max_examples=plan["policy_rows"],
                                             max_seconds=max(0.0, work_deadline - time.perf_counter()))
                    require_finite(metrics, "training metrics")
                    require_finite(model.state_dict(), "updated model")
                    require_finite(optimizer.state_dict(), "updated optimizer")
                    arm["training"] = metrics
                    report["training_performed"] |= metrics.get("steps", 0) > 0
                    after_steps = optimizer_steps(optimizer)
                    arm["final_optimizer_steps"] = after_steps
                    if metrics.get("interrupted"):
                        raise AblationDeadline("An arm exceeded its training deadline")
                    if (metrics.get("steps") != args.steps or metrics.get("trained_examples") != plan["policy_rows"] or
                            metrics.get("trained_value_examples") != plan["labels"]["rows"]):
                        raise ValueError("Actual policy/value training exposure differs from frozen arm budget")
                    actual_sources = {key: value for key, value in metrics.get("trained_value_examples_by_source", {}).items() if value}
                    if actual_sources != plan["labels"]["by_source"]:
                        raise ValueError("Actual per-source value exposure differs from the frozen mask")
                    if any(after_steps.get(key) != value + args.steps for key, value in before_steps.items()):
                        raise ValueError("Actual Adam counters differ from the frozen step budget")
                    if any(value != args.steps for key, value in after_steps.items() if key not in before_steps):
                        raise ValueError("New Adam counters differ from the frozen step budget")
                    arm.update(evaluate(model))
                    path = args.output / f"seed-{plan['seed']}-{rate_name(rate)}.pt"
                    check_time(work_deadline, "saving diagnostic checkpoint")
                    save_checkpoint(path, model, purpose="diagnostic-update-size-ablation", resumable=False,
                                    promoted=False, input_checkpoint_sha256=inputs[args.checkpoint],
                                    manifest_sha256=report["manifest_sha256"], learning_rate=rate, seed=plan["seed"])
                    arm.update(complete=True, checkpoint=path.name, checkpoint_sha256=file_sha(path, work_deadline))
                    publish("arm-complete", learning_rate=rate, seed=plan["seed"])
                    del model, optimizer, payload
            report["comparison"] = summarize(report["baseline"], report["arms"], args.learning_rates)
        for path, digest in inputs.items():
            if file_sha(path, deadline) != digest:
                raise ValueError(f"Input changed during ablation: {path}")
        if file_sha(args.output / "manifest.json", deadline) != report["manifest_sha256"]:
            raise ValueError("Frozen manifest changed during ablation")
        report["inputs_unchanged"] = True
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
    result.add_argument("--learning-rates", nargs="+", type=float, default=[.0003, .0001])
    result.add_argument("--steps", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--max-seconds", type=float, default=900)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    for name, default in (("training-games", 16), ("validation-games", 8),
                          ("training-families", 16), ("validation-families", 8),
                          ("training-rows", 512), ("validation-rows", 256)):
        result.add_argument("--min-" + name, type=int, default=default)
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
    if (not 1 <= len(args.learning_rates) <= 4 or
            any(type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate <= 1 for rate in args.learning_rates) or
            len({rate_name(rate) for rate in args.learning_rates}) != len(args.learning_rates)):
        raise ValueError("Provide 1–4 distinct finite learning rates in (0,1]")
    if (not args.model_id or len(set(args.model_id)) != len(args.model_id) or
            any(not isinstance(value, str) or not value.strip() for value in args.model_id)):
        raise ValueError("Provide distinct nonempty exact model IDs")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("max-seconds must be finite and positive")
    if args.min_training_rows > args.steps * args.batch_size:
        raise ValueError("Minimum eligible training rows exceed the policy prefix budget")
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
