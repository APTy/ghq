"""Bounded CPU/CUDA throughput measurements; no playing-strength evaluation.

Inputs are preencoded, device-resident copies of the initial position and all
its exact legal actions. Timings exclude rules, encoding, host/device transfer,
and warmup, so they describe network throughput rather than full MCTS speed.
Training uses synthetic targets on disposable model copies; nothing is saved.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .encoding import encode_actions, encode_board
from .model import NetworkConfig, PolicyValueNet, load_checkpoint, rules_source_hash
from .pipeline import write_json
from .rules import e

BENCHMARK_VERSION = 1


def summarize_latencies(latencies_ms, batch_size):
    """Latency quantiles and aggregate boards/second, excluding warmup."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Latencies must be a finite positive sequence")
    if not len(values):
        return {"samples": 0, "median_ms": None, "p95_ms": None,
                "mean_ms": None, "boards_per_second": None, "seconds": 0.0}
    seconds = float(values.sum() / 1000.0)
    return {"samples": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(values.mean()),
            "boards_per_second": float(batch_size * len(values) / seconds), "seconds": seconds}


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation, *, device, batch_size, warmup, repeats, deadline):
    completed_warmup = 0
    for _ in range(warmup):
        if time.perf_counter() >= deadline:
            break
        _synchronize(device)
        operation()
        _synchronize(device)
        completed_warmup += 1
    timings = []
    if completed_warmup == warmup:
        for _ in range(repeats):
            if time.perf_counter() >= deadline:
                break
            _synchronize(device)
            started = time.perf_counter()
            operation()
            _synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)
    return {"batch_size": batch_size, **summarize_latencies(timings, batch_size),
            "requested_repeats": repeats, "warmup_requested": warmup,
            "warmup_completed": completed_warmup,
            "complete": len(timings) == repeats and completed_warmup == warmup}


def _inputs(board, batch_size, device):
    moves = sorted(board.generate_legal_moves(), key=lambda move: move.uci())
    boards = torch.from_numpy(np.repeat(encode_board(board)[None], batch_size, axis=0)).to(device)
    actions = torch.from_numpy(np.repeat(encode_actions(board, moves)[None], batch_size, axis=0)).to(device)
    mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
    return boards, actions, mask


def _devices(requested):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise ValueError("CUDA was requested but PyTorch cannot see a GPU")
    # Measure the requested GPU before the CPU comparison, so a slow CPU does
    # not consume the entire budget before any GPU results are collected.
    return ["cuda", "cpu"] if available and requested != "cpu" else ["cpu"]


def run_benchmark(model, *, device="auto", repeats=10, warmup=2,
                  batch_sizes=(1, 16, 64), train_batch_size=64, train_repeats=3,
                  max_seconds=45):
    """Benchmark disposable copies, preserving the input model and its mode.

    Each device gets an equal share of the total work budget. Incomplete
    measurements are explicit and contain no fabricated timings. An operation
    already executing is allowed to finish; the 45s default leaves headroom
    under the pilot's 60s benchmark allowance for setup and final reporting.
    """
    if repeats < 1 or warmup < 0 or train_repeats < 1 or train_batch_size < 1:
        raise ValueError("Repeat/batch counts must be positive and warmup nonnegative")
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("Inference batch sizes must be positive")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    selected = _devices(device)
    started = time.perf_counter()
    total_deadline = started + max_seconds
    initial = e.BaseBoard()
    legal_count = len(list(initial.generate_legal_moves()))
    report = {"benchmark_version": BENCHMARK_VERSION, "requested_device": device,
              "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(),
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "network_config": asdict(model.config), "rules_sha256": rules_source_hash(),
              "position": "initial", "legal_actions_per_board": legal_count,
              "precision": "float32", "timing_scope": "direct_network_device_resident_inputs",
              "excluded_from_timings": ["rules", "encoding", "host_device_transfers", "warmup"],
              "training_targets": "synthetic uniform policy and draw; disposable weights",
              "max_seconds": max_seconds, "devices": [], "complete": False}
    for name in selected:
        target = torch.device(name)
        deadline = min(total_deadline, time.perf_counter() + max_seconds / len(selected))
        candidate = deepcopy(model).to(target)
        candidate.eval()
        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target)
            properties = torch.cuda.get_device_properties(target)
            device_report = {"device": name, "name": properties.name,
                             "total_memory_bytes": properties.total_memory}
        else:
            device_report = {"device": name, "name": platform.processor() or platform.machine()}
        device_report["inference"] = []
        with torch.inference_mode():
            for size in batch_sizes:
                inputs = _inputs(initial, size, target)
                result = _measure(lambda: candidate(*inputs), device=target, batch_size=size,
                                  warmup=warmup, repeats=repeats, deadline=deadline)
                device_report["inference"].append(result)
                del inputs
        # Include forward, loss, backward, clipping, and AdamW update. Warmup
        # allocates optimizer state before timed updates. No autocast/AMP is
        # used, matching the initial pipeline's full-precision training.
        inputs = _inputs(initial, train_batch_size, target)
        candidate.train()
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3, weight_decay=1e-4)
        policy_target = torch.full(inputs[2].shape, 1.0 / legal_count, device=target)
        outcomes = torch.ones(train_batch_size, dtype=torch.long, device=target)

        def training_step():
            optimizer.zero_grad(set_to_none=True)
            policy, wdl = candidate(*inputs)
            loss = -(policy_target * policy.log_softmax(-1)).sum(-1).mean()
            loss = loss + torch.nn.functional.cross_entropy(wdl, outcomes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()

        device_report["training"] = _measure(
            training_step, device=target, batch_size=train_batch_size,
            warmup=max(1, warmup), repeats=train_repeats, deadline=deadline)
        if target.type == "cuda":
            device_report["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(target)
        device_report["complete"] = (all(item["complete"] for item in device_report["inference"])
                                      and device_report["training"]["complete"])
        report["devices"].append(device_report)
        del optimizer, candidate, inputs, policy_target, outcomes
        if target.type == "cuda":
            torch.cuda.empty_cache()
    report["seconds"] = time.perf_counter() - started
    report["complete"] = all(item["complete"] for item in report["devices"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--train-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=45)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or args.train_repeats < 1 or args.warmup < 0:
        parser.error("Threads/repeats must be positive and warmup nonnegative")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    if args.output.exists():
        parser.error("Benchmark output exists; select a new report path")
    torch.set_num_threads(args.threads)
    torch.manual_seed(4173)
    model = load_checkpoint(args.checkpoint, "cpu")[0] if args.checkpoint else PolicyValueNet(NetworkConfig())
    try:
        report = run_benchmark(model, device=args.device, repeats=args.repeats, warmup=args.warmup,
                               train_repeats=args.train_repeats, max_seconds=args.max_seconds)
    except ValueError as error:
        parser.error(str(error))
    report["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
