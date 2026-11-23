#!/usr/bin/env python3
"""Low-overhead CPU sampling of native self-play (POSIX SIGPROF required).

Alternates uninstrumented and sampled complete games with identical fixed work.
Samples describe process CPU time, not wall-clock waiting. Python signals are
handled at interpreter checkpoints, so long native calls can bias attribution.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import signal
import sys
import time

from profile_selfplay import ROOT, play


def category(stack):
    names = {name for _, _, name in stack}
    if "evaluate" in names:
        for name, label in (
            ("_safe_destinations", "evaluation: mobility"),
            ("_mobility_masks", "evaluation: mobility"),
            ("_artillery_capture_threats", "evaluation: artillery threats"),
            ("_credible_fire", "evaluation: fire reliability"),
            ("_find_free_captures_for_cluster", "evaluation: capture allowance"),
            ("_find_adjacency_clusters", "evaluation: infantry topology"),
        ):
            if name in names:
                return label
        return "evaluation: remaining scoring and helpers"
    for name, label in (
        ("push", "rules: applying candidate/game actions"),
        ("generate_legal_moves", "rules: legal move generation"),
        ("_copy", "search: board copies"),
        ("state_key", "search: state keys"),
    ):
        if name in names:
            return label
    return "search control, sorting, bookkeeping and other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--interval-ms", type=float, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--search-ceiling-ms", type=float, default=60000)
    args = parser.parse_args()
    if not hasattr(signal, "ITIMER_PROF"):
        parser.error("POSIX ITIMER_PROF is required")
    if not (args.runs > 0 and 0 < args.interval_ms < float("inf") and
            1 <= args.depth <= 8 and 2 <= args.beam_width <= 24 and
            args.max_turns > 0 and 0 < args.search_ceiling_ms < float("inf")):
        parser.error("Invalid benchmark configuration")
    if args.output.exists():
        parser.error("Choose a new output directory")
    if signal.getitimer(signal.ITIMER_PROF) != (0.0, 0.0):
        parser.error("Another process CPU timer is already active")
    args.output.mkdir(parents=True)
    stacks = Counter()

    def sample(signum, frame):
        stack = []
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename.startswith(str(ROOT)):
                stack.append((filename[len(str(ROOT)) + 1:], frame.f_lineno, frame.f_code.co_name))
            frame = frame.f_back
        stacks[tuple(stack)] += 1

    previous = signal.getsignal(signal.SIGPROF)
    signature = None
    rows = []
    try:
        signal.signal(signal.SIGPROF, sample)
        for run in range(args.runs):
            for sampled in (False, True):
                wall, cpu = time.perf_counter(), time.process_time()
                if sampled:
                    interval = args.interval_ms / 1000
                    signal.setitimer(signal.ITIMER_PROF, interval, interval)
                try:
                    game = play(args)
                finally:
                    signal.setitimer(signal.ITIMER_PROF, 0)
                row = dict(run=run + 1, sampled=sampled,
                           wall_seconds=time.perf_counter() - wall,
                           cpu_seconds=time.process_time() - cpu)
                current = ([(t["state"], t["moves"], t["nodes"], t["depth"]) for t in game["turns"]],
                           game["final_state"])
                if signature is None:
                    signature = current
                    (args.output / "game.json").write_text(json.dumps(game, indent=2) + "\n")
                if current != signature:
                    raise RuntimeError("Instrumentation changed game/search work")
                row.update(turns=len(game["turns"]), actions=sum(len(t["moves"]) for t in game["turns"]),
                           nodes=sum(t["nodes"] for t in game["turns"]), terminal=game["terminal"])
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)
    groups, leaves, inclusive = Counter(), Counter(), Counter()
    for stack, count in stacks.items():
        groups[category(stack)] += count
        if stack:
            leaves[stack[0]] += count
        for filename, name in {(filename, name) for filename, _, name in stack}:
            inclusive[(filename, name)] += count
    total = sum(stacks.values())
    report = dict(config={**vars(args), "output": str(args.output)}, python=sys.version,
                  platform=platform.platform(), runs=rows, samples=total, identical_work=True,
                  source_sha256={name: hashlib.sha256((ROOT / "public" / name).read_bytes()).hexdigest()
                                 for name in ("engine.py", "search.py")},
                  exclusive_categories=[dict(category=k, samples=v, percent=100*v/total)
                                        for k, v in groups.most_common()],
                  leaf_lines=[dict(file=k[0], line=k[1], function=k[2], samples=v, percent=100*v/total)
                              for k, v in leaves.most_common()],
                  inclusive_functions=[dict(file=k[0], function=k[1], samples=v, percent=100*v/total)
                                       for k, v in inclusive.most_common()])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["exclusive_categories"], indent=2))


if __name__ == "__main__":
    main()
