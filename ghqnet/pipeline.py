"""Reproducible local/GPU prototype: teacher imitation -> self-play -> training.

Example: python -m ghqnet.pipeline --output .data/neural/pilot --device cpu
Nothing in this module replaces or promotes the live website bot.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .data import read_replay, write_replay
from .model import NetworkConfig, PolicyValueNet, TorchEvaluator, load_checkpoint, save_checkpoint, rules_source_hash
from .selfplay import play_game
from .train import device_name, fit


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--resume", type=Path, help="Resume model/optimizer from a trusted checkpoint")
    result.add_argument("--iterations", type=int, default=None)
    result.add_argument("--games", type=int, default=None, help="Games per update (16 prototype, 200 curriculum)")
    result.add_argument("--teacher-games", type=int, default=None, help="CPU warm-start games; prototype mode only")
    result.add_argument("--teacher-ms", type=float, default=600)
    result.add_argument("--simulations", type=int, default=None)
    result.add_argument("--max-turns", type=int, default=None)
    result.add_argument("--opening-turns", type=int, default=2)
    result.add_argument("--epochs", type=int, default=None)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--replay-size", type=int, default=None)
    result.add_argument("--replay-ratio", type=float, default=None,
                        help="Curriculum only: row exposures per new unique training state (default 1); immutable on resume")
    result.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Initial optimizer rate; resume retains checkpoint optimizer settings")
    result.add_argument("--blocks", type=int, default=6)
    result.add_argument("--channels", type=int, default=96)
    result.add_argument("--seed", type=int, default=20261120)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    result.add_argument("--curriculum", action="store_true", help="Use batched mixed opponents and independent holdouts")
    result.add_argument("--validate-only", action="store_true", help="Validate configuration without generating games or training")
    result.add_argument("--total-games", type=int, default=0, help="Cumulative mixed-game target across resumes; zero uses iteration count")
    result.add_argument("--teacher-positions", type=int, default=50000)
    result.add_argument("--teacher-max-games", type=int, default=2000)
    result.add_argument("--teacher-update-games", type=int, default=200)
    result.add_argument("--teacher-depth", type=int, default=3)
    result.add_argument("--teacher-beam-width", type=int, default=8)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--inference-batch-size", type=int, default=32)
    result.add_argument("--batch-wait-ms", type=float, default=5)
    result.add_argument("--inference-protocol", choices=("auto", "legacy", "encoded"), default="auto")
    result.add_argument("--profile-inference", action="store_true",
                        help="Record inference phases with device synchronization; use only for diagnostics")
    result.add_argument("--game-timeout-seconds", type=float, default=900)
    result.add_argument("--max-seconds", type=float, default=None, help="Soft run deadline including collection and training")
    result.add_argument("--checkpoint-reserve-seconds", type=float, default=120)
    result.add_argument("--cpu-fraction", type=float, default=0.4)
    result.add_argument("--history-fraction", type=float, default=0.2)
    result.add_argument("--history-size", type=int, default=4)
    result.add_argument("--middlegame-fraction", type=float, default=0.25)
    result.add_argument("--start-pool-size", type=int, default=2048)
    result.add_argument("--validation-fraction", type=float, default=0.1)
    result.add_argument("--validation-size", type=int, default=10000)
    result.add_argument("--teacher-replay-fraction", type=float, default=0.2)
    result.add_argument("--hard-replay-fraction", type=float, default=0.1)
    result.add_argument("--input-replay", type=Path, action="append", default=[])
    result.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    return result


def validate_args(args):
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    defaults = {"iterations": (2, 10), "games": (16, 200), "teacher_games": (4, 0),
                "simulations": (64, 256), "max_turns": (100, 150), "epochs": (2, 1),
                "replay_size": (20000, 250000)}
    for name, choices in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, choices[int(args.curriculum)])
    if args.replay_ratio is not None and not args.curriculum:
        raise ValueError("--replay-ratio requires --curriculum")
    if args.curriculum:
        if args.replay_ratio is None:
            args.replay_ratio = 1.0
        if args.replay_ratio < 0:
            raise ValueError("--replay-ratio must be nonnegative")
    for name in ("iterations", "games", "simulations", "max_turns", "epochs", "batch_size", "replay_size", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.teacher_games < 0 or args.opening_turns < 0 or args.teacher_ms <= 0 or args.learning_rate <= 0:
        raise ValueError("Invalid teacher, opening, or learning-rate argument")
    if args.curriculum:
        for name in ("teacher_max_games", "teacher_update_games", "teacher_depth", "teacher_beam_width",
                     "workers", "inference_batch_size", "history_size", "validation_size", "start_pool_size"):
            if getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if min(args.total_games, args.teacher_positions, args.batch_wait_ms) < 0:
            raise ValueError("Curriculum targets and batch wait must be nonnegative")
        if args.game_timeout_seconds <= 0 or args.checkpoint_reserve_seconds <= 0:
            raise ValueError("Timeout and checkpoint reserve must be positive")
        if args.workers > 64 or args.inference_batch_size > 64:
            raise ValueError("Workers and inference batch size must be at most 64")
        if args.max_seconds is not None and args.max_seconds <= args.checkpoint_reserve_seconds:
            raise ValueError("--max-seconds must exceed --checkpoint-reserve-seconds")
        for first, second in ((args.cpu_fraction, args.history_fraction),
                              (args.teacher_replay_fraction, args.hard_replay_fraction)):
            if min(first, second) < 0 or first + second > 1:
                raise ValueError("Curriculum proportions must be nonnegative and sum to at most one")
        if not 0 <= args.validation_fraction < 1 or not 0 <= args.middlegame_fraction <= 1:
            raise ValueError("Invalid validation or middlegame fraction")
        for path in (*args.input_replay, *args.exclude_manifest):
            if not path.is_file():
                raise ValueError(f"Curriculum input does not exist: {path}")
    return args


def run(args):
    validate_args(args)
    if args.validate_only:
        result = {"valid": True, "curriculum": args.curriculum,
                  "games_per_update": args.games, "simulations": args.simulations,
                  "total_games": args.total_games, "replay_size": args.replay_size,
                  "replay_ratio": args.replay_ratio}
        print(json.dumps(result))
        return result
    if args.curriculum:
        from .curriculum_pipeline import run_curriculum
        return run_curriculum(args)
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise ValueError("Output directory is nonempty; choose a new run or explicitly --resume")
    device = device_name(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    optimizer, first_iteration, replay = None, 1, []
    if args.resume:
        checkpoint_path = args.resume.resolve()
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        metadata = checkpoint["metadata"]
        if metadata.get("seed") != args.seed:
            raise ValueError(f"Resume seed must match checkpoint seed {metadata.get('seed')}; pass --seed explicitly")
        first_iteration = metadata.get("iteration", 0) + 1
        buffer_name = metadata.get("replay_buffer")
        buffer_hash = metadata.get("replay_buffer_sha256")
        if not isinstance(buffer_name, str) or Path(buffer_name).name != buffer_name or not buffer_hash:
            raise ValueError("Checkpoint lacks a valid exact replay-buffer reference; raw replay reconstruction is unsupported")
        buffer_path = checkpoint_path.parent / buffer_name
        if not buffer_path.is_file():
            raise ValueError(f"Checkpoint replay buffer is missing: {buffer_path}")
        if hashlib.sha256(buffer_path.read_bytes()).hexdigest() != buffer_hash:
            raise ValueError("Checkpoint replay-buffer hash mismatch")
        replay = read_replay(buffer_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        model = PolicyValueNet(NetworkConfig(args.blocks, args.channels)).to(device)
    # Explicit resume may continue a run or fork into a fresh directory. It
    # must not silently replace a newer completed branch or its configuration.
    for pattern in ("checkpoint-*.pt", "metrics-*.json", "run-*.json"):
        for existing in args.output.glob(pattern):
            suffix = existing.stem.rsplit("-", 1)[-1]
            if suffix.isdigit() and int(suffix) >= first_iteration:
                raise ValueError(f"Resume would overwrite existing run output {existing}; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"device_used": device, "torch": str(torch.__version__), "numpy": np.__version__,
                   "python": platform.python_version(), "network": asdict(model.config),
                   "rules_sha256": rules_source_hash(),
                   "effective_learning_rates": ([group["lr"] for group in optimizer.param_groups]
                                                if optimizer else [args.learning_rate]),
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "started_at": time.time(), "status": "prototype-not-promoted"})
    write_json(args.output / f"run-{first_iteration:04d}.json", config)

    def emit(event):
        print(json.dumps(event), flush=True)
        with (args.output / "events.jsonl").open("a") as output:
            output.write(json.dumps(event) + "\n")

    def collect(iteration, teacher=False):
        evaluator = None if teacher else TorchEvaluator(model, device)
        count = args.teacher_games if teacher else args.games
        games, gathered = [], []
        for index in range(count):
            game_id = f"{iteration:04d}-{'teacher' if teacher else 'selfplay'}-{index:04d}"
            samples, report = play_game(evaluator, teacher=teacher, teacher_ms=args.teacher_ms,
                                       simulations=args.simulations, max_turns=args.max_turns,
                                       opening_turns=args.opening_turns,
                                       seed=args.seed + iteration * 100000 + index, game_id=game_id)
            write_replay(args.output / "replay" / f"{game_id}.jsonl", samples, game=report)
            gathered.extend(samples)
            games.append(report)
            emit({"event": "game", **{key: value for key, value in report.items()
                                      if key not in ("moves", "final_state")}})
        return gathered, games

    def update(iteration, games):
        nonlocal optimizer, replay
        if len(replay) > args.replay_size:
            replay = random.Random(args.seed + iteration).sample(replay, args.replay_size)
        optimizer, metrics = fit(model, replay, epochs=args.epochs, batch_size=args.batch_size,
                                 learning_rate=args.learning_rate, device=device,
                                 seed=args.seed + iteration, optimizer=optimizer)
        report = {"iteration": iteration, "training": metrics, "games": games,
                  "completed_games": sum(not game["truncated"] for game in games),
                  "truncated_games": sum(game["truncated"] for game in games),
                  "promoted": False}
        checkpoint_path = args.output / f"checkpoint-{iteration:04d}.pt"
        # Persist precisely the bounded buffer used for this update. Loading
        # every historical game on resume changes the training distribution
        # and can import stale or unrelated files from an output directory.
        buffer_path = args.output / f"replay-buffer-{iteration:04d}.jsonl"
        write_replay(buffer_path, replay, iteration=iteration, kind="replay-buffer")
        buffer_hash = hashlib.sha256(buffer_path.read_bytes()).hexdigest()
        save_checkpoint(checkpoint_path, model, optimizer, iteration=iteration,
                        seed=args.seed, metrics=metrics, run_config=config,
                        replay_buffer=buffer_path.name, replay_buffer_sha256=buffer_hash)
        write_json(args.output / f"metrics-{iteration:04d}.json", report)
        emit({"event": "trained", "checkpoint": str(checkpoint_path), **metrics})
        return report

    reports = []
    if args.resume is None and args.teacher_games:
        gathered, games = collect(0, teacher=True)
        replay.extend(gathered)
        reports.append(update(0, games))
    for iteration in range(first_iteration, first_iteration + args.iterations):
        gathered, games = collect(iteration)
        replay.extend(gathered)
        reports.append(update(iteration, games))
    summary = {"status": "prototype-not-promoted", "checkpoint": str(args.output /
               f"checkpoint-{first_iteration + args.iterations - 1:04d}.pt"),
               "device": device, "iterations": len(reports),
               "completed_games": sum(report["completed_games"] for report in reports),
               "truncated_games": sum(report["truncated_games"] for report in reports)}
    write_json(args.output / "summary.json", summary)
    emit({"event": "complete", **summary})
    return summary


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
