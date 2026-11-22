"""Paired-color checkpoint matches against CPU or a historical checkpoint.

Both sides receive the same total per-turn wall-clock budget. Maximum-length
and repeated-position cutoffs are unfinished, never counted as draws/wins.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from .mcts import MCTS
from .model import TorchEvaluator, load_checkpoint, rules_source_hash
from .pipeline import write_json
from .rules import e, cpu, PUBLIC
from .selfplay import random_opening
from .suite import canonical_state_id
from .train import device_name


def match(evaluator, *, neural_color, seed, time_ms=2400, max_turns=150,
          simulations=100000, opening_turns=2, start_board=None, opponent_evaluator=None):
    board = start_board.copy() if start_board is not None else random_opening(seed, opening_turns)
    search = MCTS(evaluator, simulations=simulations, seed=seed)
    opponent_name = "historical" if opponent_evaluator is not None else "cpu"
    opponent_search = MCTS(opponent_evaluator, simulations=simulations, seed=seed + 1) if opponent_evaluator is not None else None
    trajectory, latencies, repeated = [], {"neural": [], opponent_name: []}, {}
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
        source = "neural" if player == neural_color else opponent_name
        started = time.perf_counter()
        if source != "cpu":
            plan = (search if source == "neural" else opponent_search).plan_turn(board, time_ms=time_ms)
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
            "opponent": opponent_name,
            "outcome": outcome, "terminal": terminal, "reason": reason,
            "unfinished": terminal is None, "turns": len(trajectory), "trajectory": trajectory,
            "latency_ms": {side: {"mean": statistics.mean(values) if values else 0,
                                   "max": max(values, default=0), "turns": len(values)}
                           for side, values in latencies.items()}}


def create_openings(*, pairs=10, seed=39173, turns=2):
    """Freeze distinct actual opening states; each is played with both colors."""
    if pairs < 1 or turns < 0 or (turns == 0 and pairs != 1):
        raise ValueError("Use one pair for zero opening turns; otherwise request positive, distinct openings")
    openings, seen = [], set()
    for attempt in range(max(100, pairs * 100)):
        board = random_opening(seed + attempt, turns)
        identity = canonical_state_id(board)
        if identity in seen or cpu._terminal(board) is not None:
            continue
        seen.add(identity)
        openings.append({"id": "opening-" + identity[:20], "canonical_state_id": identity,
                         "seed": seed + attempt, "state": board.serialize()})
        if len(openings) == pairs:
            return {"version": 1, "rules_sha256": rules_source_hash(), "seed": seed,
                    "opening_turns": turns, "openings": openings}
    raise ValueError("Unable to generate the requested number of distinct nonterminal openings")


def validate_openings(manifest):
    if manifest.get("version") != 1 or manifest.get("rules_sha256") != rules_source_hash():
        raise ValueError("Opening manifest schema/rules mismatch")
    openings = manifest.get("openings", [])
    if not openings:
        raise ValueError("Opening manifest is empty")
    seen = set()
    for opening in openings:
        board = e.BaseBoard.deserialize(opening["state"])
        identity = canonical_state_id(board)
        if identity != opening["canonical_state_id"] or identity in seen:
            raise ValueError("Opening identity mismatch or duplicate/mirrored opening")
        if opening["id"] != "opening-" + identity[:20] or not isinstance(opening["seed"], int):
            raise ValueError("Invalid opening identifier or seed")
        if cpu._terminal(board) is not None or not list(board.generate_legal_moves()):
            raise ValueError("Opening must be nonterminal with legal actions")
        seen.add(identity)
    return openings


def summarize_matches(games, *, seed=39173, bootstrap_samples=2000):
    """Resample complete opening pairs, never individual correlated games.

    The interval estimates score conditional on games finishing. Unfinished
    games remain unknown; separate bounds show their possible effect. A single
    opening pair cannot provide a useful confidence interval.
    """
    totals = {outcome: sum(game["outcome"] == outcome for game in games) for outcome in ("win", "draw", "loss")}
    totals["unfinished"] = sum(game["unfinished"] for game in games)
    points = totals["win"] + totals["draw"] * 0.5
    completed = sum(totals[name] for name in ("win", "draw", "loss"))
    groups = {}
    for game in games:
        groups.setdefault(game["opening_id"], []).append(game)
    pairs = []
    for group in groups.values():
        if len(group) == 2 and {game["neural_color"] for game in group} == {"RED", "BLUE"}:
            pairs.append((sum(1 if game["outcome"] == "win" else 0.5 if game["outcome"] == "draw" else 0 for game in group),
                          sum(game["outcome"] is not None for game in group)))
    interval = None
    contributing = sum(number > 0 for _, number in pairs)
    warning = "insufficient_contributing_opening_pairs" if contributing < 2 else None
    if contributing >= 2 and bootstrap_samples > 0:
        rng, samples = random.Random(seed), []
        for _ in range(bootstrap_samples):
            resampled = rng.choices(pairs, k=len(pairs))
            denominator = sum(number for _, number in resampled)
            if denominator:
                samples.append(sum(value for value, _ in resampled) / denominator)
        samples.sort()
        interval = [samples[int((len(samples) - 1) * 0.025)], samples[int((len(samples) - 1) * 0.975)]] if samples else None
        # Resampling observed identical scores cannot estimate tail risk. Do
        # not display [1,1] (or another point interval) as evidence of certainty.
        if interval is not None and interval[0] == interval[1]:
            interval = None
            warning = "degenerate_bootstrap_interval_suppressed"
    finished_pair_scores = [points / 2 for points, count in pairs if count == 2]
    pair_mean = statistics.mean(finished_pair_scores) if finished_pair_scores else None
    # Opening-pair scores are bounded in [0,1]. Hoeffding needs no estimated
    # variance and stays conservative even for uniformly winning tiny samples.
    radius = math.sqrt(math.log(2 / .05) / (2 * len(finished_pair_scores))) if finished_pair_scores else None
    bounded_interval = [max(0., pair_mean - radius), min(1., pair_mean + radius)] if radius is not None else None
    return {"totals": totals, "completed_opening_pairs": len(pairs),
            "independent_opening_clusters": len(groups), "clusters_with_completed_games": contributing,
            "completion_rate": completed / len(games) if games else None,
            "completed_game_score": points / completed if completed else None,
            "completed_only_score_ci95": interval,
            "ci_method": "percentile_bootstrap_by_opening_pair", "bootstrap_samples": bootstrap_samples,
            "ci_warning": warning,
            "ci_note": "Conditional on completion; insufficient or degenerate bootstrap samples give no interval.",
            "fully_finished_pairs": len(finished_pair_scores), "fully_finished_pair_score": pair_mean,
            "fully_finished_pair_score_ci95": bounded_interval,
            "fully_finished_pair_ci_method": "hoeffding_bounded_independent_opening_pair_scores",
            "unfinished_score_bounds": [points / len(games), (points + totals["unfinished"]) / len(games)] if games else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, help="Historical neural opponent; default is the production CPU bot")
    parser.add_argument("--openings", type=Path, help="Reuse a frozen paired-opening JSON manifest")
    parser.add_argument("--write-openings", type=Path, help="Save generated opening manifest (default: OUTPUT.openings.json)")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--time-ms", type=float, default=2400)
    parser.add_argument("--max-turns", type=int, default=150)
    parser.add_argument("--simulations", type=int, default=100000)
    parser.add_argument("--opening-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=39173)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.pairs, args.max_turns, args.simulations, args.threads) < 1 or not math.isfinite(args.time_ms) or args.time_ms <= 0 or args.opening_turns < 0:
        parser.error("Match counts, budgets and threads must be positive")
    if args.output.exists():
        parser.error("Arena output exists; select a new report path")
    if args.openings and args.write_openings:
        parser.error("Use --openings or --write-openings, not both")
    opening_path = args.openings or args.write_openings or args.output.with_suffix(".openings.json")
    if args.openings:
        manifest = json.loads(args.openings.read_text())
    else:
        if opening_path.exists():
            parser.error("Opening output exists; reuse it with --openings or select a new path")
        manifest = create_openings(pairs=args.pairs, seed=args.seed, turns=args.opening_turns)
        write_json(opening_path, manifest)
    openings = validate_openings(manifest)
    torch.set_num_threads(args.threads)
    device = device_name(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    evaluator = TorchEvaluator(model, device)
    opponent_evaluator = None
    if args.opponent_checkpoint:
        opponent_model, _ = load_checkpoint(args.opponent_checkpoint, device)
        opponent_evaluator = TorchEvaluator(opponent_model, device)
    # Warm model kernels before measuring turn latency.
    initial = e.BaseBoard()
    evaluator.predict(initial, list(initial.generate_legal_moves()))
    if opponent_evaluator is not None:
        opponent_evaluator.predict(initial, list(initial.generate_legal_moves()))
    reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for opening in openings:
        for color in (e.RED, e.BLUE):
            report = match(evaluator, neural_color=color, seed=opening["seed"],
                           time_ms=args.time_ms, max_turns=args.max_turns,
                           simulations=args.simulations, start_board=e.BaseBoard.deserialize(opening["state"]),
                           opponent_evaluator=opponent_evaluator)
            report["opening_id"] = opening["id"]
            reports.append(report)
            statistics_report = summarize_matches(reports, seed=args.seed)
            result = {"checkpoint": str(args.checkpoint),
                      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      "opponent": "historical" if args.opponent_checkpoint else "cpu",
                      "opponent_checkpoint": str(args.opponent_checkpoint) if args.opponent_checkpoint else None,
                      "opponent_checkpoint_sha256": hashlib.sha256(args.opponent_checkpoint.read_bytes()).hexdigest() if args.opponent_checkpoint else None,
                      "opening_manifest": str(opening_path),
                      "opening_manifest_sha256": hashlib.sha256(opening_path.read_bytes()).hexdigest(),
                      "rules_sha256": rules_source_hash(), "cpu_version": cpu.ENGINE_VERSION,
                      "cpu_sha256": hashlib.sha256((PUBLIC / "search.py").read_bytes()).hexdigest(),
                      "time_ms_per_turn": args.time_ms, "device": device, "simulations_cap": args.simulations,
                      "seed": args.seed, "opening_turns": manifest.get("opening_turns"),
                      "max_turns": args.max_turns, **statistics_report, "games": reports,
                      "expected_games": len(openings) * 2, "complete": len(reports) == len(openings) * 2,
                      "promoted": False, "promotion": "manual_review_required"}
            write_json(args.output, result)
            print(json.dumps({"game": len(reports), "totals": statistics_report["totals"],
                              "latency_ms": report["latency_ms"]}), flush=True)


if __name__ == "__main__":
    main()
