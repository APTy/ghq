#!/usr/bin/env python3
"""Profile complete CPU self-play with identical unprofiled/profiled work.

Fixed depth/beam, no opening randomness. A search timeout invalidates the run;
turn-capped games are explicitly unfinished. Native Python, not Pyodide or
neural MCTS. Output includes the trajectory and standard cProfile artifacts.
"""
import argparse
from collections import Counter
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "public"))
import engine as e
import search


def play(args):
    board = e.BaseBoard()
    turns = []
    started = time.perf_counter()
    while search._terminal(board) is None and len(turns) < args.max_turns:
        before = board.serialize()
        color = board.turn
        tick = time.perf_counter()
        result = json.loads(search.analyze_json(
            before, args.search_ceiling_ms, args.depth, args.beam_width, 1))
        if result["timedOut"]:
            raise RuntimeError("Search timed out; raise --search-ceiling-ms for fixed-work profiling")
        moves = result["turnMoves"]
        if not moves:
            raise RuntimeError("Nonterminal search returned no plan")
        for uci in moves:
            move = e.Move.from_uci(uci)
            if board.turn != color or search._terminal(board) is not None or not board.is_legal(move):
                raise RuntimeError(f"Invalid turn plan: {moves}")
            board.push(move)
        if board.turn == color and search._terminal(board) is None:
            raise RuntimeError("Incomplete turn plan")
        board.move_stack = []
        turns.append(dict(state=before, moves=moves, nodes=result["nodes"],
                          depth=result["depth"], seconds=time.perf_counter() - tick))
        if len(turns) % 20 == 0:
            print(f"{len(turns)} turns, {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
    return dict(seconds=time.perf_counter() - started, turns=turns,
                terminal=search._terminal(board), final_state=board.serialize())


def profile_rows(profiler):
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        try:
            filename = str(Path(filename).relative_to(ROOT))
        except ValueError:
            pass
        rows.append(dict(file=filename, line=line, function=function, calls=calls,
                         primitive_calls=primitive, self_seconds=own,
                         cumulative_seconds=cumulative))
    return sorted(rows, key=lambda row: row["self_seconds"], reverse=True)


def replay(game, repeats):
    """Rules only, same complete trajectory; no search/evaluation or JSON I/O."""
    moves = [e.Move.from_uci(uci) for turn in game["turns"] for uci in turn["moves"]]
    for _ in range(repeats):
        board = e.BaseBoard()
        for move in moves:
            legal = list(board.generate_legal_moves())
            if move not in legal:
                raise RuntimeError("Replay action is illegal")
            board.push(move)
            board.move_stack = []
            board.outcome()
        if board.serialize() != game["final_state"]:
            raise RuntimeError("Replay final state differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--search-ceiling-ms", type=float, default=60000)
    parser.add_argument("--replay-repeats", type=int, default=30)
    args = parser.parse_args()
    if not (1 <= args.depth <= 8 and 2 <= args.beam_width <= 24 and
            args.max_turns > 0 and args.replay_repeats > 0 and
            0 < args.search_ceiling_ms < float("inf")):
        parser.error("Invalid search or repeat configuration")
    if args.output.exists():
        parser.error("Output directory exists; choose a new directory")
    args.output.mkdir(parents=True)
    baseline = play(args)
    profiler = cProfile.Profile()
    profiled = profiler.runcall(play, args)
    signature = lambda game: [(turn["state"], turn["moves"], turn["nodes"], turn["depth"]) for turn in game["turns"]]
    if signature(baseline) != signature(profiled) or baseline["final_state"] != profiled["final_state"]:
        raise RuntimeError("Profile changed the game/search work")
    profiler.dump_stats(str(args.output / "selfplay.prof"))
    tick = time.perf_counter()
    replay(baseline, args.replay_repeats)
    replay_seconds = time.perf_counter() - tick
    rules_profiler = cProfile.Profile()
    rules_profiler.runcall(replay, baseline, args.replay_repeats)
    rules_profiler.dump_stats(str(args.output / "rules-replay.prof"))
    turns = baseline["turns"]
    report = dict(config={**vars(args), "output": str(args.output)},
                  python=sys.version, platform=platform.platform(),
                  source_sha256={name: hashlib.sha256((ROOT / "public" / name).read_bytes()).hexdigest()
                                 for name in ("engine.py", "search.py")},
                  scope="CPU self-play, fixed depth/beam; imports excluded; native Python",
                  complete=baseline["terminal"] is not None, identical_work=True,
                  summary=dict(turns=len(turns), actions=sum(len(t["moves"]) for t in turns),
                               nodes=sum(t["nodes"] for t in turns),
                               terminal=baseline["terminal"],
                               depth_counts=dict(Counter(t["depth"] for t in turns)),
                               baseline_seconds=baseline["seconds"],
                               profiled_seconds=profiled["seconds"],
                               median_turn_seconds=statistics.median(t["seconds"] for t in turns) if turns else None,
                               replay_seconds_per_game=replay_seconds / args.replay_repeats),
                  functions=profile_rows(profiler), rules_functions=profile_rows(rules_profiler),
                  game=baseline)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
