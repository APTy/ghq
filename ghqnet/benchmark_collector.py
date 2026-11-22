"""Measure spawned collection throughput with a reproducible, bounded grid.

Example: python -m ghqnet.benchmark_collector checkpoint.pt --output grid.json
  --workers 8 16 32 --games 32 --batch-size 16 32 --batch-wait-ms 0 1 5
  --inference-protocol legacy encoded --simulations 128 --max-seconds 1800

No training or promotion. Cases share checkpoint, positions, seeds and search
budgets; each completed case is saved. Turn caps are unfinished games, not draws.
"""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from .collector import CollectionError, CollectionTimeout, collect_games
from .curriculum import opponent_schedule
from .model import TorchEvaluator, load_checkpoint, rules_source_hash
from .rules import e, terminal_value
from .train import device_name


def summarize_reports(reports, collection_seconds, collector=None):
    """Search totals cover returned rollouts; neural rows include aborted work.

    All rates use actual collection wall time, not summed concurrent game time.
    A returned turn-capped rollout is explicitly not a terminal game.
    """
    collector = collector or {}
    total = lambda name: sum(report.get(name, 0) for report in reports)
    fraction = lambda a, b: a / b if a is not None and b else None
    outcomes, reasons = Counter(), Counter()
    for report in reports:
        terminal = report.get("terminal")
        outcome = ("draw" if terminal["winner"] is None else terminal["winner"]) if terminal else "unfinished"
        outcomes[outcome] += 1
        reasons[report.get("reason", "unknown")] += 1
    terminals = len(reports) - outcomes["unfinished"]
    return {"returned_rollouts": len(reports), "terminal_games": terminals,
            "unfinished_rollouts": outcomes["unfinished"], "outcomes": dict(outcomes), "reasons": dict(reasons),
            "actions": total("actions"), "turns": total("turns"),
            "search_simulations": total("search_simulations"), "new_search_nodes": total("search_nodes"),
            "search_totals_scope": "returned rollouts only; interrupted games are not estimated",
            "neural_rows": collector.get("rows"),
            "neural_rows_scope": "all collector inference rows, including work from interrupted games",
            "median_turns": statistics.median([r["turns"] for r in reports]) if reports else None,
            "summed_game_seconds": total("seconds"), "collection_wall_seconds": collection_seconds,
            "returned_rollouts_per_second": fraction(len(reports), collection_seconds),
            "terminal_games_per_second": fraction(terminals, collection_seconds),
            "actions_per_second": fraction(total("actions"), collection_seconds),
            "simulations_per_second": fraction(total("search_simulations"), collection_seconds),
            "new_nodes_per_second": fraction(total("search_nodes"), collection_seconds),
            "neural_rows_per_second": fraction(collector.get("rows"), collection_seconds),
            "root_legal_actions": total("root_legal_actions"),
            "root_visited_actions": total("root_visited_actions"),
            "root_visited_fraction": fraction(total("root_visited_actions"), total("root_legal_actions")),
            "mean_root_policy_entropy_nats": fraction(total("root_policy_entropy_sum"), total("search_calls")),
            "opposite_player_evaluator_fraction": fraction(total("opposite_player_evaluator_calls"), total("evaluator_calls")),
            "searches_with_opposite_player_evaluation_fraction": fraction(total("searches_with_opposite_player_evaluation"), total("search_calls")),
            "chosen_skips": sum(action == "skip" for report in reports for action in report["moves"]),
            "turn_start_skips": total("turn_start_skips"), "hard_examples": total("hard_examples")}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_positions(path, count, seed):
    """Sample distinct exact states from JSONL/replay without loading its labels.

    Only selected states are deserialized and checked against the current rules.
    Selection is a seeded reservoir over unique state strings, then source order.
    Replay headers are checked when present; plain {"state": ...} JSONL is valid.
    """
    rng, selected, seen, rows = random.Random(seed), [], set(), 0
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if line_number == 1 and isinstance(row, dict) and "metadata" in row and "state" not in row:
                if row.get("rules_sha256") != rules_source_hash():
                    raise ValueError("Position replay rules hash does not match this engine")
                continue
            if not isinstance(row, dict) or not isinstance(row.get("state"), str):
                raise ValueError(f"Position row {line_number} must contain a serialized state")
            rows += 1
            state = row["state"]
            digest = hashlib.sha256(state.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            item = (line_number, state, digest)
            if len(selected) < count:
                selected.append(item)
            else:
                index = rng.randrange(len(seen))
                if index < count:
                    selected[index] = item
    if not selected:
        raise ValueError("Position input contains no states")
    selected.sort()
    for line_number, state, _ in selected:
        board = e.BaseBoard.deserialize(state)
        if terminal_value(board) is not None or not list(board.generate_legal_moves()):
            raise ValueError(f"Selected position row {line_number} is terminal or has no legal actions")
    return [item[1] for item in selected], {
        "path": str(path), "sha256": _sha(path), "source_rows": rows,
        "unique_exact_states": len(seen), "selected_states": len(selected),
        "selection": "seeded reservoir over distinct exact state strings, in source order",
        "seed": seed, "selected_line_numbers": [item[0] for item in selected],
        "selected_state_sha256": [item[2] for item in selected],
        "validation_scope": "selected states only; replay labels are unused"}


def build_tasks(args, simulations, repeat, model_id, historical_id=None, positions=()):
    seed = args.seed + repeat * args.games
    schedule = opponent_schedule(args.games, seed=seed, cpu_fraction=args.cpu_fraction,
                                 history_fraction=args.history_fraction)
    tasks = []
    for index, assignment in enumerate(schedule):
        task = dict(assignment, game_id=f"bench-r{repeat:03d}-{index:05d}", seed=seed + index,
                    model_id=model_id, simulations=simulations, max_turns=args.max_turns,
                    opening_turns=args.opening_turns, teacher_ms=args.teacher_ms,
                    teacher_depth=args.teacher_depth, teacher_beam_width=args.teacher_beam_width)
        if assignment["opponent"] == "historical":
            task["opponent_model_id"] = historical_id
        if positions:
            task["start_state"] = positions[index % len(positions)]
            task["opening_turns"] = 0  # Exact supplied states replace random openings.
        tasks.append(task)
    return tasks


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--opening-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=78123)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[32])
    parser.add_argument("--batch-wait-ms", type=float, nargs="+", default=[5])
    parser.add_argument("--inference-protocol", choices=("auto", "legacy", "encoded"), nargs="+", default=["auto"])
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--profile-inference", action="store_true")
    profiles.add_argument("--profile-sample", action="store_true", help="Profile the first setting for each protocol before the unprofiled grid")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--case-max-seconds", type=float)
    parser.add_argument("--request-timeout-seconds", type=float, default=120)
    parser.add_argument("--game-timeout-seconds", type=float, default=300)
    parser.add_argument("--positions", type=Path, help="JSONL state objects or an existing replay buffer; labels are unused")
    parser.add_argument("--position-count", type=int, default=16)
    parser.add_argument("--historical-checkpoint", type=Path)
    parser.add_argument("--history-fraction", type=float, default=0)
    parser.add_argument("--cpu-fraction", type=float, default=0)
    parser.add_argument("--teacher-ms", type=float, default=100)
    parser.add_argument("--teacher-depth", type=int, default=3)
    parser.add_argument("--teacher-beam-width", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def validate_args(args, parser):
    if args.output.exists():
        parser.error("Output exists; select a new report path")
    if (min(args.simulations) < 1 or min(args.workers) < 1 or max(args.workers) > 64 or
            min(args.batch_size) < 1 or max(args.batch_size) > 64 or
            min(args.games, args.max_turns, args.threads, args.repeats, args.position_count,
                args.teacher_depth, args.teacher_beam_width) < 1 or args.opening_turns < 0):
        parser.error("Counts must be positive; workers and batch size cannot exceed 64")
    for name in ("simulations", "workers", "batch_size", "batch_wait_ms", "inference_protocol"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"{name} configurations must be distinct")
    for name in ("max_seconds", "case_max_seconds", "request_timeout_seconds", "game_timeout_seconds", "teacher_ms"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if any(not math.isfinite(value) or value < 0 or value / 1000 >= args.request_timeout_seconds
           for value in args.batch_wait_ms):
        parser.error("batch_wait_ms must be finite, nonnegative and below the request timeout")
    if (any(not math.isfinite(value) or value < 0 for value in (args.history_fraction, args.cpu_fraction)) or
            args.history_fraction + args.cpu_fraction > 1):
        parser.error("Opponent fractions must be finite, nonnegative and sum to at most one")
    if args.history_fraction and not args.historical_checkpoint:
        parser.error("Historical opponents require --historical-checkpoint")


def case_grid(args):
    cases = []
    if args.profile_sample:
        cases.extend((0, args.simulations[0], args.workers[0], args.batch_size[0], args.batch_wait_ms[0], protocol, True, True)
                     for protocol in args.inference_protocol)
    cases.extend((*values, args.profile_inference, False) for values in product(
        range(args.repeats), args.simulations, args.workers, args.batch_size, args.batch_wait_ms, args.inference_protocol))
    return cases


def run_benchmark(args):
    started = time.perf_counter()
    deadline = started + args.max_seconds
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = device_name(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    digest = _sha(args.checkpoint)
    model_id = "sha256:" + digest
    evaluators = {model_id: TorchEvaluator(model, device)}
    historical_id, historical = None, None
    if args.historical_checkpoint:
        historical_digest = _sha(args.historical_checkpoint)
        historical_id = "sha256:" + historical_digest
        if historical_id == model_id:
            raise ValueError("Historical checkpoint must differ from the current checkpoint")
        old_model, _ = load_checkpoint(args.historical_checkpoint, device)
        evaluators[historical_id] = TorchEvaluator(old_model, device)
        historical = {"path": str(args.historical_checkpoint), "sha256": historical_digest}
    positions, position_info = load_positions(args.positions, args.position_count, args.seed) if args.positions else ([], None)
    warm = e.BaseBoard.deserialize(positions[0]) if positions else e.BaseBoard()
    for evaluator in evaluators.values():
        evaluator.predict(warm, list(warm.generate_legal_moves()))
    grid = case_grid(args)
    report = {"benchmark_version": 2, "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
              "historical_checkpoint": historical, "positions": position_info,
              "collector_sha256": _sha(Path(__file__).with_name("collector.py")),
              "selfplay_sha256": _sha(Path(__file__).with_name("selfplay.py")),
              "model_sha256": _sha(Path(__file__).with_name("model.py")),
              "encoding_sha256": _sha(Path(__file__).with_name("encoding.py")),
              "rules_sha256": rules_source_hash(), "device": device, "torch_version": str(torch.__version__),
              "cuda_runtime": torch.version.cuda,
              "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
              "platform": platform.platform(), "parent_threads": args.threads, "worker_threads": 1,
              "simulations": args.simulations, "workers": args.workers, "games_per_case": args.games,
              "max_turns": args.max_turns, "opening_turns": 0 if positions else args.opening_turns, "seed": args.seed,
              "batch_size": args.batch_size, "batch_wait_ms": args.batch_wait_ms,
              "inference_protocol": args.inference_protocol, "profile_inference": args.profile_inference,
              "profile_sample": args.profile_sample,
              "repeats": args.repeats, "cpu_fraction": args.cpu_fraction, "history_fraction": args.history_fraction,
              "teacher_ms": args.teacher_ms, "teacher_depth": args.teacher_depth,
              "teacher_beam_width": args.teacher_beam_width,
              "max_seconds": args.max_seconds, "case_max_seconds": args.case_max_seconds,
              "setup_seconds": time.perf_counter() - started, "planned_cases": len(grid),
              "cases": [], "complete": False, "stop_reason": None,
              "notes": ["No training, checkpoint mutation, promotion, or playing-strength claim.",
                        "Overall budget includes checkpoint loading, position sampling and warmup; an executing operation cannot be preempted.",
                        "Collection wall time includes worker spawn/cleanup and actual rules/search/evaluation.",
                        "Cases marked complete returned every requested rollout; turn/repetition cutoffs are still unfinished games.",
                        "Collector completed_games counts returned worker tasks, including unfinished rollouts; summary terminal_games counts only rules terminals.",
                        "Search totals omit interrupted games; neural rows include their inference work.",
                        "Profiling may synchronize CUDA boundaries and alter throughput; compare unprofiled cases for performance.",
                        "Phase timings may overlap across workers and need not sum to wall time.",
                        "Same seeds preserve workload intent; device/batch floating-point differences can alter later play.",
                        "Requested workers are capped by games; inspect actual_workers before comparing concurrency."]}
    references = {}
    _write(args.output, report)
    for repeat, simulations, workers, batch_size, batch_wait, protocol, profile, diagnostic in grid:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            report["stop_reason"] = {"kind": "overall_deadline"}
            break
        tasks = build_tasks(args, simulations, repeat, model_id, historical_id, positions)
        budget = min(remaining, args.case_max_seconds) if args.case_max_seconds is not None else remaining
        metrics, completed, error = {}, {}, None
        tick = time.perf_counter()
        try:
            collect_games(tasks, evaluators, workers=workers, batch_size=batch_size,
                          batch_wait_ms=batch_wait, request_timeout_s=args.request_timeout_seconds,
                          game_timeout_s=args.game_timeout_seconds, collection_timeout_s=budget,
                          inference_protocol=protocol, profile=profile,
                          metrics=metrics, on_result=lambda index, result: completed.__setitem__(index, result[1]))
        except (CollectionTimeout, CollectionError) as exception:
            error = {"type": type(exception).__name__, "message": str(exception),
                     "stop_reason": getattr(exception, "stop_reason", None)}
            for index, result in zip(exception.completed_indices, exception.results):
                completed.setdefault(index, result[1])
        wall = time.perf_counter() - tick
        reports = [completed[index] for index in sorted(completed)]
        key = (repeat, simulations)
        reference = references.setdefault(key, {"case_index": len(report["cases"]),
                                               "games": {r["game_id"]: r for r in reports}})
        compared = [r for r in reports if r["game_id"] in reference["games"]]
        case = {"case_index": len(report["cases"]), "repeat": repeat, "simulations": simulations,
                "requested_workers": workers, "actual_workers": metrics.get("workers"),
                "batch_size": batch_size, "batch_wait_ms": batch_wait,
                "inference_protocol": protocol, "profile_inference": profile, "diagnostic": diagnostic,
                "task_manifest_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                "scheduled_opponents": dict(Counter(task["opponent"] for task in tasks)),
                "complete": len(reports) == args.games and error is None, "error": error,
                "summary": summarize_reports(reports, wall, metrics), "collector": metrics,
                "timing": {"wall_seconds": wall, "startup_seconds": metrics.get("startup_seconds"),
                           "active_seconds": metrics.get("active_seconds"), "cleanup_seconds": metrics.get("cleanup_seconds"),
                           "phase_seconds": metrics.get("phase_seconds"),
                           "inference_profile": metrics.get("inference_profile")},
                "reference_case_index": reference["case_index"],
                "same_simulation_reference_games": len(compared),
                "moves_match_same_simulation_reference": (all(r["moves"] == reference["games"][r["game_id"]]["moves"]
                                                              for r in compared) if compared else None),
                "outcomes_match_same_simulation_reference": (all((r.get("terminal"), r.get("reason")) ==
                    (reference["games"][r["game_id"]].get("terminal"), reference["games"][r["game_id"]].get("reason"))
                    for r in compared) if compared else None), "games": reports}
        report["cases"].append(case)
        report["seconds"] = time.perf_counter() - started
        report["complete"] = len(report["cases"]) == len(grid) and all(c["complete"] for c in report["cases"])
        _write(args.output, report)
        print(json.dumps({key: case[key] for key in ("case_index", "repeat", "simulations", "requested_workers",
                         "actual_workers", "batch_size", "batch_wait_ms", "inference_protocol", "profile_inference",
                         "diagnostic", "complete", "error", "summary")}), flush=True)
        if error is not None:
            reason = error["stop_reason"] or {}
            kind = reason.get("kind") if isinstance(reason, dict) else reason
            if error["type"] != "CollectionTimeout" or kind != "global_deadline":
                report["stop_reason"] = {"kind": "collection_failure", "error": error}
                _write(args.output, report)
                raise RuntimeError(error["message"])
            if args.case_max_seconds is None or time.perf_counter() >= deadline:
                report["stop_reason"] = {"kind": "overall_deadline"}
                break
    report["seconds"] = time.perf_counter() - started
    report["cases_attempted"] = len(report["cases"])
    report["cases_not_started"] = len(grid) - len(report["cases"])
    _write(args.output, report)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    return run_benchmark(args)


if __name__ == "__main__":
    main()
