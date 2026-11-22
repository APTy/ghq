"""Bounded learning-rate comparison on an immutable curriculum replay.

Every arm reloads the same model AND Adam state. No games are generated, no
winner is selected, and the saved arm checkpoints deliberately lack production
resume references. Validation is measured without gradient updates.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import struct
import time

import numpy as np
import torch

from .curriculum import fingerprint
from .data import SUPPORTED_DATA_VERSIONS, Sample
from .evaluate_replay import breakdown, row_metrics
from .model import load_checkpoint, rules_source_hash, save_checkpoint
from .train import batch, device_name, fit, prepare


class AblationDeadline(TimeoutError):
    pass


def check_time(deadline, stage):
    if deadline is not None and time.perf_counter() >= deadline:
        raise AblationDeadline(f"Time budget exhausted during {stage}")


def file_sha(path, deadline=None):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Input must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            check_time(deadline, f"hashing {path.name}")
            digest.update(chunk)
    return digest.hexdigest()


def reference(parent, metadata, key, pattern, inputs, deadline):
    name, expected = metadata.get(key), metadata.get(key + "_sha256")
    if (not isinstance(name, str) or not re.fullmatch(pattern, name) or
            not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise ValueError(f"Missing exact flat filename and SHA256 for {key}")
    path = parent / name
    if file_sha(path, deadline) != expected:
        raise ValueError(f"Input SHA256 mismatch: {name}")
    inputs[path] = expected
    return path


def read_rows(path, kind, deadline):
    rows = []
    with path.open() as source:
        header = json.loads(next(source))
        if header.get("version") not in SUPPORTED_DATA_VERSIONS or header.get("rules_sha256") != rules_source_hash():
            raise ValueError("Replay schema or rules mismatch")
        if header.get("metadata", {}).get("kind") != kind:
            raise ValueError(f"Expected an explicitly marked {kind} replay")
        for raw in source:
            check_time(deadline, f"loading {kind} replay")
            if raw.strip():
                sample = Sample(**json.loads(raw))
                if sample.outcome is not None and type(sample.outcome) is not int:
                    raise ValueError("Outcome labels must be integer W/D/L or absent")
                sample.validate()
                rows.append(sample)
    if not rows:
        raise ValueError(f"Empty {kind} replay")
    return rows


def verify_partition(training, validation, state, deadline):
    partition = state.get("partition", {})
    registries = [set(partition.get(key, [])) for key in ("train_states", "validation_states")]
    if not all(registries) or registries[0] & registries[1]:
        raise ValueError("Missing or overlapping curriculum partition registries")
    identities = []
    for rows, registry in zip((training, validation), registries):
        families, games, states = set(), set(), set()
        for row in rows:
            check_time(deadline, "checking holdout partitions")
            identity = fingerprint(row)
            if identity not in registry:
                raise ValueError("Replay state is absent from its saved partition registry")
            states.add(identity)
            if row.family_id:
                families.add(row.family_id)
            if row.game_id:
                games.add(row.game_id)
        identities.append((states, families, games))
    if any(left & right for left, right in zip(*identities)):
        raise ValueError("Training and validation states, families, or games overlap")
    return {"canonical_state_overlap": 0, "family_overlap": 0, "game_overlap": 0,
            "training_unique_states": len(identities[0][0]), "validation_unique_states": len(identities[1][0]),
            "note": "Exact saved registries and current rows checked; no claim about unknown prehistory."}


def tree_sha(value):
    """Stable tensor/basic-data identity, independent of checkpoint ZIP metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            value = item.detach().cpu().contiguous()
            digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for value in item:
                visit(value)
        else:
            digest.update((type(item).__name__ + ":" + json.dumps(item, allow_nan=False) + ";").encode())

    visit(value)
    return digest.hexdigest()


def shuffled_order_sha(count, epochs, seed):
    """The index permutations used by train.fit's seeded, in-place shuffles."""
    rng, order, digest = random.Random(seed), list(range(count)), hashlib.sha256()
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            digest.update(struct.pack("<Q", index))
    return digest.hexdigest()


def make_optimizer(model, saved_state, rate):
    if not isinstance(saved_state, dict) or not saved_state.get("state") or not saved_state.get("param_groups"):
        raise ValueError("Ablation requires a populated saved Adam optimizer state")
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-4)
    optimizer.load_state_dict(saved_state)
    restored = tree_sha(optimizer.state_dict())
    original_rates = [group["lr"] for group in optimizer.param_groups]
    # Restore moments/steps FIRST; the ablated learning rate is the only override.
    for group in optimizer.param_groups:
        group["lr"] = rate
    return optimizer, {"restored_optimizer_sha256": restored, "original_learning_rates": original_rates,
                       "effective_learning_rates": [group["lr"] for group in optimizer.param_groups]}


def optimizer_steps(optimizer):
    result = {}
    for parameter, state in optimizer.state_dict()["state"].items():
        value = state.get("step")
        if isinstance(value, torch.Tensor):
            value = value.item()
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError("Saved optimizer has invalid Adam step counters")
        result[str(parameter)] = int(value)
    return result


def gradient_preflight(model, saved_state, samples, *, device, rate, seed, batch_size, deadline):
    """Exercise deterministic backward/Adam once, then discard this model/state."""
    selected = samples[:min(2, batch_size)]
    optimizer, _ = make_optimizer(model, saved_state, rate)
    before = optimizer_steps(optimizer)
    optimizer, metrics = fit(model, selected, epochs=1, batch_size=len(selected), learning_rate=rate,
                             device=device, seed=seed, optimizer=optimizer, validation_samples=[],
                             max_seconds=max(0.0, deadline - time.perf_counter()))
    if metrics.get("interrupted") or metrics.get("steps") != 1:
        raise AblationDeadline("Deterministic gradient preflight did not complete")
    after = optimizer_steps(optimizer)
    if any(after.get(key) != value + 1 for key, value in before.items()):
        raise ValueError("Gradient preflight Adam counters did not advance once")
    if device == "cuda":
        torch.cuda.synchronize()
    return {"complete": True, "device": device, "samples": len(selected), "steps": 1,
            "seconds": metrics.get("seconds"), "discarded_before_baseline_and_arms": True,
            "purpose": "Check deterministic forward/backward/optimizer support; not a measured arm."}


def evaluate_bounded(model, samples, *, device, batch_size, deadline):
    """Use the existing replay metrics with deadline checks for each batch."""
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            check_time(deadline, "validation evaluation")
            selected = samples[start:start + batch_size]
            # Per-record validation/preparation is also bounded.
            records = []
            for sample in selected:
                check_time(deadline, "validation preparation")
                records.extend(prepare([sample]))
            boards, actions, mask, _, _ = batch(records, device)
            policy, wdl = model(boards, actions, mask)
            log_policy, log_wdl = policy.log_softmax(-1).cpu().numpy(), wdl.log_softmax(-1).cpu().numpy()
            for index, sample in enumerate(selected):
                rows.append(row_metrics(sample, log_policy[index, :len(sample.actions)], log_wdl[index]))
            check_time(deadline, "validation evaluation")
    sources, phases = defaultdict(list), defaultdict(list)
    for row in rows:
        sources[row["source"]].append(row)
        phases[row["phase"]].append(row)
    return {"overall": breakdown(rows),
            "by_source": {key: breakdown(value) for key, value in sorted(sources.items())},
            "by_phase": {key: breakdown(value) for key, value in sorted(phases.items())}}


def metric_deltas(reference_metrics, metrics):
    result = {}
    for label, before, after in [("overall", reference_metrics["overall"], metrics["overall"])] + [
            (source, reference_metrics["by_source"][source], value)
            for source, value in metrics["by_source"].items()]:
        result[label] = {}
        for group, key, subgroup in (("policy", "top1_agreement", "non_forced"),
                                     ("policy", "top5_agreement", "non_forced"),
                                     ("policy", "soft_target_nll", "non_forced"),
                                     ("wdl", "logloss", "all"), ("wdl", "brier", "all"),
                                     ("wdl", "accuracy", "all")):
            a, b = before[subgroup][group][key], after[subgroup][group][key]
            result[label][f"{subgroup}_{group}_{key}"] = None if a is None or b is None else b - a
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_ablation(args):
    validate_args(args)
    if args.output.exists():
        raise ValueError("Ablation output must be a new directory")
    started = time.perf_counter()
    deadline = started + args.max_seconds
    work_deadline = deadline - min(10.0, args.max_seconds * .1)
    args.output.mkdir(parents=True)
    report = {"version": 1, "purpose": "fixed-replay learning-rate diagnostic", "resumable": False,
              "promoted": False, "winner": None, "complete": False, "comparison_eligible": False,
              "status": "running", "arms": [], "comparison": None, "inputs_unchanged": None,
              "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "created_at": datetime.now(timezone.utc).isoformat(), "torch": str(torch.__version__),
              "rules_sha256": rules_source_hash(),
              "limitations": ["Imitation/outcome-label diagnostics do not establish playing strength.",
                              "All arm checkpoints are diagnostic and lack production resume references.",
                              "An incomplete comparison cannot select a continuation."]}
    inputs = {}

    def publish(event, **details):
        report["seconds"] = time.perf_counter() - started
        record = {"event": event, "elapsed_seconds": report["seconds"], **details}
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        write_json(args.output / "report.json", report)
        write_json(args.output / "summary.json", {key: report[key] for key in
                   ("status", "complete", "comparison_eligible", "inputs_unchanged", "seconds", "winner", "promoted")})
        print(json.dumps(record, allow_nan=False), flush=True)

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.set_num_threads(args.threads)
        device = device_name(args.device)
        report["device"] = device
        report["deterministic_algorithms"] = True
        inputs[args.checkpoint] = file_sha(args.checkpoint, work_deadline)
        model, payload = load_checkpoint(args.checkpoint, device)
        metadata = payload.get("metadata", {})
        if not payload.get("optimizer_state", {}).get("state"):
            raise ValueError("Checkpoint has no populated optimizer state")
        paths = {key: reference(args.checkpoint.parent, metadata, key, pattern, inputs, work_deadline)
                 for key, pattern in (("replay_buffer", r"replay-buffer-[0-9]{4,}\.jsonl"),
                                      ("validation_buffer", r"validation-buffer-[0-9]{4,}\.jsonl"),
                                      ("curriculum_state", r"curriculum-[0-9]{4,}\.json"))}
        state = json.loads(paths["curriculum_state"].read_text())
        for entry in state.get("history", []):
            reference(args.checkpoint.parent, {"history": entry["file"], "history_sha256": entry["sha256"]},
                      "history", r"history-[0-9a-f]{16}\.pt", inputs, work_deadline)
        training = read_rows(paths["replay_buffer"], "training", work_deadline)
        validation = read_rows(paths["validation_buffer"], "validation", work_deadline)
        report["partition"] = verify_partition(training, validation, state, work_deadline)
        report["inputs"] = {path.name: {"sha256": digest, "bytes": path.stat().st_size}
                            for path, digest in inputs.items()}
        report["checkpoint_iteration"] = metadata.get("iteration")
        expected_steps = args.epochs * math.ceil(len(training) / args.batch_size)
        report.update(training_samples=len(training), validation_samples=len(validation),
                      expected_steps_per_arm=expected_steps,
                      expected_sample_order_sha256=shuffled_order_sha(len(training), args.epochs, args.seed))
        initial_model = tree_sha(model.state_dict())
        initial_optimizer = tree_sha(payload["optimizer_state"])
        report.update(initial_model_sha256=initial_model, initial_optimizer_sha256=initial_optimizer)
        publish("inputs-verified", training_samples=len(training), validation_samples=len(validation),
                expected_steps_per_arm=expected_steps)
        if device == "cuda":
            publish("gradient-preflight-start")
            report["gradient_preflight"] = gradient_preflight(
                model, payload["optimizer_state"], training, device=device, rate=args.learning_rates[0],
                seed=args.seed, batch_size=args.batch_size, deadline=work_deadline)
            # The probe must not alter either the baseline or the measured arms.
            del model, payload
            model, payload = load_checkpoint(args.checkpoint, device)
            if tree_sha(model.state_dict()) != initial_model or tree_sha(payload["optimizer_state"]) != initial_optimizer:
                raise ValueError("Gradient preflight state was not reset exactly")
            publish("gradient-preflight-complete", **report["gradient_preflight"])
        else:
            report["gradient_preflight"] = {"complete": None, "skipped": "CUDA-only runtime compatibility probe"}
        publish("baseline-evaluation-start")
        report["unupdated"] = evaluate_bounded(model, validation, device=device, batch_size=args.batch_size, deadline=work_deadline)
        publish("baseline-evaluated", validation_samples=len(validation))
        del model, payload
        for index, rate in enumerate(args.learning_rates):
            check_time(work_deadline, "arm initialization")
            # Fresh load prevents one arm's parameter or optimizer updates leaking into another.
            model, payload = load_checkpoint(args.checkpoint, device)
            model_identity = tree_sha(model.state_dict())
            if model_identity != initial_model or tree_sha(payload["optimizer_state"]) != initial_optimizer:
                raise ValueError("Arm did not start from the exact same model and optimizer")
            optimizer, restored = make_optimizer(model, payload["optimizer_state"], rate)
            if restored["restored_optimizer_sha256"] != initial_optimizer:
                raise ValueError("Optimizer restoration changed the initial state")
            del payload
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            arm = {"index": index, "learning_rate": rate, "complete": False, "usable_for_comparison": False,
                   "initial_model_sha256": model_identity,
                   "sample_order_sha256": report["expected_sample_order_sha256"],
                   "initial_optimizer_steps": optimizer_steps(optimizer), **restored}
            report["arms"].append(arm)
            publish("arm-start", index=index, learning_rate=rate)
            optimizer, metrics = fit(model, training, epochs=args.epochs, batch_size=args.batch_size,
                                     learning_rate=rate, device=device, seed=args.seed, optimizer=optimizer,
                                     validation_samples=[], max_seconds=max(0.0, work_deadline - time.perf_counter()))
            arm["training"] = metrics
            arm["final_optimizer_steps"] = optimizer_steps(optimizer)
            if metrics.get("interrupted") or metrics.get("steps") != expected_steps:
                report["status"] = "incomplete-training"
                publish("arm-training-incomplete", index=index, steps=metrics.get("steps"))
                raise AblationDeadline("Arm did not complete the required identical optimizer-step budget")
            if any(arm["final_optimizer_steps"].get(key) != value + expected_steps
                   for key, value in arm["initial_optimizer_steps"].items()):
                raise ValueError("Actual Adam step increments differ from the reported training budget")
            publish("arm-trained", index=index, steps=metrics["steps"])
            arm["validation"] = evaluate_bounded(model, validation, device=device, batch_size=args.batch_size, deadline=work_deadline)
            check_time(work_deadline, "saving diagnostic arm")
            checkpoint_path = args.output / f"arm-{index:04d}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, purpose="diagnostic-replay-ablation",
                            resumable=False, promoted=False, input_checkpoint_sha256=inputs[args.checkpoint],
                            learning_rate=rate, seed=args.seed, epochs=args.epochs, training=metrics,
                            validation_replay_sha256=inputs[paths["validation_buffer"]])
            arm.update(complete=True, checkpoint=checkpoint_path.name,
                       checkpoint_sha256=file_sha(checkpoint_path, work_deadline))
            publish("arm-evaluated", index=index, learning_rate=rate, checkpoint=checkpoint_path.name)
            del model, optimizer
        for path, digest in inputs.items():
            if file_sha(path, deadline) != digest:
                raise ValueError(f"Input changed during ablation: {path.name}")
        report["inputs_unchanged"] = True
        steps = [arm["training"]["steps"] for arm in report["arms"]]
        if len(steps) != len(args.learning_rates) or set(steps) != {expected_steps}:
            raise ValueError("Unequal arm budgets cannot be compared")
        report["comparison"] = {"automatic_selection": False,
                                "deltas_vs_unupdated": [metric_deltas(report["unupdated"], arm["validation"])
                                                        for arm in report["arms"]]}
        for arm in report["arms"]:
            arm["usable_for_comparison"] = True
        report.update(complete=True, comparison_eligible=True, status="complete")
        publish("comparison-complete", steps_per_arm=expected_steps, arms=len(steps))
    except AblationDeadline as error:
        report.update(status="incomplete-deadline", error=str(error), comparison=None)
        for arm in report["arms"]:
            arm["usable_for_comparison"] = False
        try:
            report["inputs_unchanged"] = all(file_sha(path, deadline) == digest for path, digest in inputs.items()) if inputs else None
        except AblationDeadline:
            report["inputs_unchanged"] = None
        publish("incomplete", reason=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", comparison=None)
        publish("failed", reason=report["error"])
        raise
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old_cudnn
        if old_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("checkpoint", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    result.add_argument("--learning-rates", type=float, nargs="+", default=[.001, .0003])
    result.add_argument("--epochs", type=int, default=1)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--seed", type=int, default=20261121)
    result.add_argument("--max-seconds", type=float, default=1200)
    result.add_argument("--validate-only", action="store_true")
    return result


def validate_args(args):
    rates = args.learning_rates
    if (not 2 <= len(rates) <= 4 or len(set(rates)) != len(rates) or
            any(not math.isfinite(rate) or not 0 < rate <= 1 for rate in rates)):
        raise ValueError("Provide 2–4 unique finite learning rates in (0,1]")
    for name, maximum in (("epochs", 4), ("batch_size", 512), ("threads", 64)):
        if type(getattr(args, name)) is not int or not 1 <= getattr(args, name) <= maximum:
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    if type(args.seed) is not int or not 0 <= args.seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("max-seconds must be finite and positive")
    return args


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        validate_args(args)
        if args.validate_only:
            print(json.dumps({"valid": True, "mode": "replay-ablation", "training_performed": False}))
            return 0
        run_ablation(args)
    except (ValueError, OSError, RuntimeError) as error:
        command.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
