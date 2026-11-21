#!/usr/bin/env python3
"""Reproducible native-CPU smoke arena against the unchanged legacy ValuePlayer.

Run from the repository root, for example:
  python3 scripts/benchmark_cpu_bot.py --games 4 --time-ms 100 --max-turns 80

Every opening is played twice, with the new engine assigned each color. Opening
actions are chosen from sorted legal moves using fixed seeds and are included in
the JSON report. Wall-clock search budgets make exact game results machine
dependent. The legacy bot completes its usual greedy actions without a timer;
both engines' actual turn latency is reported. This is a smoke benchmark, not an
Elo estimate. Move/time limits are unfinished games, never adjudicated draws.
"""

import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "public"))
SEARCH_SOURCE_SHA256 = hashlib.sha256((ROOT / "public" / "search.py").read_bytes()).hexdigest()
import engine as e
import search


def mirror_move(move):
    result = move.copy()
    for field in ("from_square", "to_square", "capture_preference"):
        square = getattr(result, field)
        if square is not None:
            setattr(result, field, 63 - square)
    if result.orientation is not None:
        result.orientation = (result.orientation + 4) % 8
    return result


def legacy_move(board):
    """ValuePlayer maximizes Blue's value, so rotate Red into that perspective."""
    perspective = board.mirror() if board.turn == e.RED else board.copy()
    move = e.ValuePlayer(perspective).get_next_move()
    if move is None:
        raise RuntimeError("Legacy bot returned no move in a nonterminal position")
    return mirror_move(move) if board.turn == e.RED else move


def checked_push(board, move, player):
    if not board.is_legal(move):
        raise RuntimeError(f"{player} selected illegal action {move.uci()} in {board.serialize()}")
    board.push(move)


def play_turn(board, player, args):
    """Return a new board; never mutate the caller's state."""
    before = board.serialize()
    working = board.copy()
    working.move_stack = []
    color = working.turn
    started = time.perf_counter()
    moves, depths = [], []
    search_calls, nodes = 0, 0
    while working.turn == color and working.outcome() is None:
        if len(moves) >= 67:  # Automatic captures remove pieces; ordinary turns have three actions.
            raise RuntimeError(f"{player} failed to finish a turn")
        if player == "legacy":
            plan = [legacy_move(working).uci()]
        else:
            remaining_ms = max(0.0, args.time_ms - (time.perf_counter() - started) * 1000)
            result = json.loads(search.analyze_json(
                working.serialize(), remaining_ms, args.max_depth, args.beam_width, 1
            ))
            search_calls += 1
            depths.append(result["depth"])
            nodes += result["nodes"]
            plan = result["turnMoves"] or ([result["bestMove"]] if result["bestMove"] else [])
            if not plan:
                raise RuntimeError("CPU search returned no move in a nonterminal position")
        for uci in plan:
            if working.turn != color or working.outcome() is not None:
                break
            checked_push(working, e.Move.from_uci(uci), player)
            moves.append(uci)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if board.serialize() != before:
        raise RuntimeError(f"{player} mutated its input position")
    return working, {
        "player": player, "color": "RED" if color == e.RED else "BLUE",
        "elapsedMs": round(elapsed_ms, 3), "moves": moves,
        "depths": depths, "searchCalls": search_calls, "nodes": nodes,
    }


def opening(seed, turns=2):
    board = e.BaseBoard()
    rng = random.Random(seed)
    prefix = []
    for _ in range(turns):
        color = board.turn
        while board.turn == color and board.outcome() is None:
            moves = sorted(board.generate_legal_moves(), key=lambda move: move.uci())
            moves = [move for move in moves if move.name != "Skip"] or moves
            move = rng.choice(moves)
            checked_push(board, move, "opening")
            prefix.append(move.uci())
    if board.outcome() is not None:
        raise RuntimeError(f"Opening seed {seed} unexpectedly produced a terminal game")
    board.move_stack = []
    return board, prefix


def play_game(start, new_color, args):
    board = start.copy()
    turns = []
    started = time.perf_counter()
    limit_reason = None
    while board.outcome() is None:
        if len(turns) >= args.max_turns:
            limit_reason = "move-limit"
            break
        if args.max_seconds and time.perf_counter() - started >= args.max_seconds:
            limit_reason = "time-limit"
            break
        player = "cpu" if board.turn == new_color else "legacy"
        board, turn = play_turn(board, player, args)
        turns.append(turn)
    outcome = board.outcome()
    result = "uncompleted" if limit_reason else (
        "draw" if outcome.winner is None else "win" if outcome.winner == new_color else "loss"
    )
    return {
        "cpuColor": "RED" if new_color == e.RED else "BLUE",
        "result": result,
        "reason": limit_reason or outcome.termination,
        "turnCount": len(turns),
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "finalHeuristicScoreRed": round(search.evaluate(board), 4),
        "finalState": board.serialize(),
        "turns": turns,
    }


def tactical_positions():
    """Small, independent captures; successful lines may differ between runs."""
    for name, infantry in (
        ("immediate-hq-capture", (e.D5, e.F4)),
        ("two-action-hq-coordination", (e.C4, e.E3)),
    ):
        board = e.BaseBoard()
        board.clear_board()
        board.set_piece_at(e.A1, e.Piece(e.HQ, e.RED))
        board.set_piece_at(e.E5, e.Piece(e.HQ, e.BLUE))
        for square in infantry:
            board.set_piece_at(square, e.Piece(e.INFANTRY, e.RED))
        list(board._generate_free_captures(board.turn))
        yield name, board


def run_tactics(args):
    results = []
    for name, board in tactical_positions():
        for mirrored in (False, True):
            start = board.mirror() if mirrored else board
            for player in ("cpu", "legacy"):
                final, details = play_turn(start, player, args)
                outcome = final.outcome()
                results.append({
                    "name": name, "mirrored": mirrored, "player": player,
                    "solved": outcome is not None and outcome.winner == start.turn,
                    **details,
                })
    return results


def distribution(values):
    if not values:
        return {"count": 0, "medianMs": None, "p95Ms": None, "maxMs": None}
    ordered = sorted(values)
    return {
        "count": len(values), "medianMs": round(statistics.median(values), 3),
        "p95Ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
        "maxMs": round(ordered[-1], 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=4, help="Even number of games; paired colors (default: 4)")
    parser.add_argument("--time-ms", type=float, default=100, help="Requested CPU search budget per turn (default: 100)")
    parser.add_argument("--max-turns", type=int, default=80, help="Turns per game before marking unfinished (default: 80)")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=3, help="Search depth in player turns")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--max-seconds", type=float, default=0, help="Optional per-game wall-clock cap; 0 disables")
    parser.add_argument("--tactics-only", action="store_true", help="Run the small tactical comparison without arena games")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even number of at least 2")
    if args.time_ms < 0 or args.max_turns < 1 or args.max_seconds < 0:
        parser.error("time budgets must be nonnegative and --max-turns must be positive")
    if not 2 <= args.beam_width <= 24 or not 1 <= args.max_depth <= 5:
        parser.error("--beam-width must be 2..24 and --max-depth must be 1..5")

    tactics = run_tactics(args)
    games = []
    if not args.tactics_only:
        for pair in range(args.games // 2):
            opening_seed = args.seed + pair * 7919
            start, prefix = opening(opening_seed, turns=2 + 2 * (pair % 2))
            for color in (e.RED, e.BLUE):
                game = play_game(start, color, args)
                game.update({"pair": pair + 1, "openingSeed": opening_seed,
                             "openingMoves": prefix, "initialState": start.serialize()})
                games.append(game)
                print(f"Game {len(games)}/{args.games}: CPU {game['cpuColor']} {game['result']} "
                      f"({game['reason']}, {game['turnCount']} turns)", file=sys.stderr, flush=True)

    turns = [turn for game in games for turn in game["turns"]]
    counts = Counter(game["result"] for game in games)
    baseline_source = "\n".join(inspect.getsource(function) for function in (
        e.ValuePlayer, e.evaluate_board_for_blue, e._get_color_score
    )) + repr(e.POSITION_GRADIENT) + repr(e.PIECE_VALUES)
    report = {
        "engineVersion": search.ENGINE_VERSION,
        "searchSourceSha256": SEARCH_SOURCE_SHA256,
        "pythonVersion": sys.version,
        "legacyBaselineSha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "config": vars(args),
        "summary": {
            "wins": counts["win"], "losses": counts["loss"], "draws": counts["draw"],
            "uncompleted": counts["uncompleted"],
            "uncompletedReasons": dict(Counter(game["reason"] for game in games if game["result"] == "uncompleted")),
            "turnLatency": {player: distribution([turn["elapsedMs"] for turn in turns if turn["player"] == player])
                            for player in ("cpu", "legacy")},
            "completedSearchDepths": dict(sorted(Counter(depth for turn in turns for depth in turn["depths"]).items())),
            "tacticsSolved": {player: sum(item["solved"] for item in tactics if item["player"] == player)
                              for player in ("cpu", "legacy")},
            "tacticsPerPlayer": len(tactics) // 2,
            "interpretation": "Native smoke test only; unfinished games are not draws and no Elo is inferred.",
        },
        "tactics": tactics,
        "games": games,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
