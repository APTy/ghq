"""Paired-color checkpoint matches against the production CPU bot.

Both sides receive the same total per-turn wall-clock budget. Maximum-length
and repeated-position cutoffs are unfinished, never counted as draws/wins.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from .mcts import MCTS
from .model import TorchEvaluator, load_checkpoint, rules_source_hash
from .pipeline import write_json
from .rules import e, cpu, PUBLIC
from .selfplay import random_opening
from .train import device_name


def match(evaluator, *, neural_color, seed, time_ms=2400, max_turns=150,
          simulations=100000, opening_turns=2, start_board=None):
    board = start_board.copy() if start_board is not None else random_opening(seed, opening_turns)
    search = MCTS(evaluator, simulations=simulations, seed=seed)
    trajectory, latencies, repeated = [], {"neural": [], "cpu": []}, {}
    reason = "max-turns"
    for _ in range(max_turns):
        if cpu._terminal(board) is not None:
            break
        key = cpu.state_key(board)
        repeated[key] = repeated.get(key, 0) + 1
        if repeated[key] >= 4:
            reason = "repetition-cutoff"
            break
        player = board.turn
        source = "neural" if player == neural_color else "cpu"
        started = time.perf_counter()
        if source == "neural":
            plan = search.plan_turn(board, time_ms=time_ms)
            moves = plan.moves
        else:
            result = json.loads(cpu.analyze_json(board.serialize(), time_ms=time_ms, max_depth=3,
                                                beam_width=8, multi_pv=1))
            moves = result["turnMoves"]
        latencies[source].append((time.perf_counter() - started) * 1000)
        if not moves:
            raise RuntimeError(f"{source} returned an empty nonterminal turn")
        for action in moves:
            if board.turn != player or cpu._terminal(board) is not None:
                raise RuntimeError(f"{source} plan continued beyond its turn")
            legal = {move.uci(): move for move in board.generate_legal_moves()}
            if action not in legal:
                raise RuntimeError(f"{source} supplied illegal action {action}")
            board.push(legal[action])
        if board.turn == player and cpu._terminal(board) is None:
            raise RuntimeError(f"{source} returned an incomplete turn")
        trajectory.append({"player": "RED" if player == e.RED else "BLUE", "moves": moves})
        board.move_stack = []
    terminal = cpu._terminal(board)
    outcome = None
    if terminal is not None:
        reason = terminal["reason"]
        outcome = ("draw" if terminal["winner"] is None else "win" if
                   terminal["winner"] == ("RED" if neural_color == e.RED else "BLUE") else "loss")
    return {"seed": seed, "neural_color": "RED" if neural_color == e.RED else "BLUE",
            "outcome": outcome, "terminal": terminal, "reason": reason,
            "unfinished": terminal is None, "turns": len(trajectory), "trajectory": trajectory,
            "latency_ms": {side: {"mean": statistics.mean(values) if values else 0,
                                   "max": max(values, default=0), "turns": len(values)}
                           for side, values in latencies.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--time-ms", type=float, default=2400)
    parser.add_argument("--max-turns", type=int, default=150)
    parser.add_argument("--simulations", type=int, default=100000)
    parser.add_argument("--opening-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=39173)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.pairs, args.max_turns, args.simulations, args.threads) < 1 or args.time_ms <= 0:
        parser.error("Match counts, budgets and threads must be positive")
    torch.set_num_threads(args.threads)
    device = device_name(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    evaluator = TorchEvaluator(model, device)
    # Warm model kernels before measuring turn latency.
    initial = e.BaseBoard()
    evaluator.predict(initial, list(initial.generate_legal_moves()))
    reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Arena output exists; select a new report path")
    for index in range(args.pairs):
        for color in (e.RED, e.BLUE):
            report = match(evaluator, neural_color=color, seed=args.seed + index,
                           time_ms=args.time_ms, max_turns=args.max_turns,
                           simulations=args.simulations, opening_turns=args.opening_turns)
            reports.append(report)
            totals = {outcome: sum(game["outcome"] == outcome for game in reports)
                      for outcome in ("win", "draw", "loss")}
            totals["unfinished"] = sum(game["unfinished"] for game in reports)
            result = {"checkpoint": str(args.checkpoint),
                      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      "rules_sha256": rules_source_hash(), "cpu_version": cpu.ENGINE_VERSION,
                      "cpu_sha256": hashlib.sha256((PUBLIC / "search.py").read_bytes()).hexdigest(),
                      "time_ms_per_turn": args.time_ms, "device": device, "simulations_cap": args.simulations,
                      "seed": args.seed, "opening_turns": args.opening_turns,
                      "max_turns": args.max_turns, "totals": totals, "games": reports,
                      "promoted": False}
            write_json(args.output, result)
            print(json.dumps({"game": len(reports), "totals": totals,
                              "latency_ms": report["latency_ms"]}), flush=True)


if __name__ == "__main__":
    main()
