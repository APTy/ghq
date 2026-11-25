#!/usr/bin/env python3
"""Paired-opening CPU infantry tuning experiment; fixed work, legal replays.

Native Python behavior diagnostic, not an Elo estimate. Each variant plays
itself, or baseline with both colors (--head-to-head). Timeouts invalidate runs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import types

from benchmark_cpu_bot import e, opening, checked_push

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "baseline": (0.035, 0.045, 0.009),
    "less-clustering": (0.035, 0.015, 0.009),
    "moderate": (0.035, 0.035, 0.009),
    "advance": (0.07, 0.015, 0.009),
    "mobility": (0.07, 0.015, 0.025),
}


def engine(name):
    source = (ROOT / "public/search.py").read_text()
    module = types.ModuleType("tuning_" + name.replace("-", "_"))
    sys.modules[module.__name__] = module
    exec(compile(source, "public/search.py", "exec"), module.__dict__)
    advance, support, mobility = CONFIGS[name]
    module.INFANTRY_SUPPORT_WEIGHT = support
    module.INFANTRY_MOBILITY_WEIGHT = mobility
    module.INFANTRY_POSITION = [[advance * module.ADVANCE[c][s] +
                                0.025 * (3.5 - abs(3.5 - s % 8))
                                for s in range(64)] for c in range(2)]
    return module


def play(start, players, args):
    board = start.copy()
    counts = [Counter(), Counter()]
    trajectory = []
    for _ in range(args.turns):
        if players[0]._terminal(board):
            break
        color = board.turn
        bot = players[color]
        stats = counts[color]
        before = board.serialize()
        result = json.loads(bot.analyze_json(before, 60000, args.depth, args.beam, 1))
        if result["timedOut"] or result["depth"] != args.depth:
            if abs(result["score"]) != bot.WIN:
                raise RuntimeError("Incomplete fixed-work search")
        assert before == board.serialize()
        assert result["turnMoves"]
        for uci in result["turnMoves"]:
            assert board.turn == color and bot._terminal(board) is None
            move = e.Move.from_uci(uci)
            kind = board.piece_type_at(move.from_square) if move.from_square is not None else move.unit_type
            is_infantry = kind in (e.INFANTRY, e.ARMORED_INFANTRY, e.AIRBORNE_INFANTRY)
            stats["actions"] += 1
            if is_infantry and move.name == "Reinforce":
                stats["deployments"] += 1
            if is_infantry and move.name == "Move":
                stats["infantry_moves"] += 1
                if bot.ADVANCE[color][move.to_square] > bot.ADVANCE[color][move.from_square]:
                    stats["forward_moves"] += 1
            if (is_infantry and move.capture_preference is not None) or move.auto_capture_type == "free":
                stats["infantry_captures"] += 1
            checked_push(board, move, "tuning")
        assert board.turn != color or bot._terminal(board)
        stats["turns"] += 1
        infantry = (board.infantry | board.armored_infantry | board.airborne_infantry) & board.occupied_co[color]
        stats["infantry_samples"] += infantry.bit_count()
        for square in e.scan_forward(infantry):
            exits = bot._safe_destinations(board, square, color).bit_count()
            stats["blocked_samples"] += exits == 0
            stats["cramped_samples"] += exits <= 2
        trajectory.append({"state": before, "moves": result["turnMoves"]})
        board.move_stack = []
    return {"stats": [dict(c) for c in counts], "terminal": players[0]._terminal(board),
            "finalState": board.serialize(), "trajectory": trajectory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--turns", type=int, default=80)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--head-to-head", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or args.turns < 1 or not 1 <= args.depth <= 8 or not 2 <= args.beam <= 24:
        parser.error("Invalid search or game limits")
    if args.output.exists():
        parser.error("Output already exists")
    bots = {name: engine(name) for name in CONFIGS}
    report = {"config": {**vars(args), "output": str(args.output)}, "weights": CONFIGS,
              "sourceSha256": hashlib.sha256((ROOT / "public/search.py").read_bytes()).hexdigest(), "games": []}
    for variant in args.variants:
        for pair in range(args.pairs):
            start, prefix = opening(args.seed + 7919 * pair)
            for color in (e.RED, e.BLUE):
                players = [bots[variant], bots[variant]]
                if args.head_to_head:
                    players[not color] = bots["baseline"]
                initial = start if args.head_to_head or color == e.RED else start.mirror()
                game = play(initial, players, args)
                game.update(variant=variant, pair=pair, candidateColor=color, opening=prefix)
                report["games"].append(game)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(variant, pair, color, game["terminal"], game["stats"], flush=True)
    summary = {}
    for variant in args.variants:
        games = [g for g in report["games"] if g["variant"] == variant]
        total = Counter()
        for game in games:
            for color, stats in enumerate(game["stats"]):
                if not args.head_to_head or color == game["candidateColor"]:
                    total.update(stats)
        summary[variant] = {
            "games": len(games), "completed": sum(g["terminal"] is not None for g in games),
            "counts": dict(total),
            "per100Turns": {key: round(100 * total[key] / total["turns"], 2)
                            for key in ("deployments", "infantry_moves", "forward_moves", "infantry_captures")},
            "blockedPercent": round(100 * total["blocked_samples"] / max(1, total["infantry_samples"]), 2),
            "crampedPercent": round(100 * total["cramped_samples"] / max(1, total["infantry_samples"]), 2),
        }
        if args.head_to_head:
            summary[variant]["wins"] = sum(g["terminal"] is not None and
                g["terminal"]["winner"] == ("RED" if g["candidateColor"] == e.RED else "BLUE") for g in games)
    report["summary"] = summary
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
