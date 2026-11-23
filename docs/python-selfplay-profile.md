# Python self-play profile

Measured November 23, 2026 on macOS 15.7.4 arm64, CPython 3.13.1.
This profiles `public/search.py` playing both colors using `public/engine.py`.
It does not measure neural MCTS, GPU collection, or browser/Pyodide throughput.

## Optimization retest

Implemented all four targeted changes in `public/engine.py` and `public/search.py`:
empty-attacker/isolated-cluster capture exits, a search-owned topology cache capped
at 4,096 entries, precomputed positional/coverage/HQ-distance weights, and one
derived-mask rebuild per piece move. The topology cache stores immutable tuples;
current artillery orientations still determine capture allowances. Arithmetic
grouping and iteration order are preserved to avoid changing floating-point ties.

Alternating full-game runs used the saved original modules and optimized modules
in one process, with identical fixed search work:

| Variant | Individual unprofiled runs | Mean |
| --- | --- | ---: |
| Original | 11.153 s, 14.235 s | 12.694 s |
| Optimized | 9.171 s, 8.961 s | 9.066 s |

That is **28.6% less wall time**, or 1.40× throughput, in this small sample. Host
timings varied, so this is not a precise universal speedup. Every run retained the
same 104 turns, 315 actions, 165,748 search score calls, and Blue HQ-capture win.
An additional optimized profile run took 8.609 s unprofiled and 26.711 s profiled.

| Profile hotspot | Original cumulative seconds | Optimized cumulative seconds |
| --- | ---: | ---: |
| Static evaluation | 23.240 | 18.582 |
| Capture allowance | 5.830 | 1.165 |
| Adjacency clustering | 3.481 | 2.320 |
| Apply candidate action | 4.372 | 3.610 |
| Artillery coverage | 1.923 | 1.480 |

These nested times overlap. Capture allowance fell from 18.4% to 4.4% of the
profile. Adjacency-generator calls/resumptions fell from 2,695,982 to 1,545,843;
artillery-coverage calls fell from 155,756 to 79,932. Evaluation remains the main
bottleneck at 69.6%, with candidate application at 13.5%.

Rules-only replay initially produced inconsistent timings (including a slower
optimized result). A follow-up alternated 24 batches of ten full-game replays,
120 games per variant: original averaged 31.64 ms wall / 30.26 ms process CPU;
optimized averaged 27.05 ms wall / 26.94 ms CPU. This supports an improvement,
but the millisecond timings remain sensitive to host load.

Exact differential checks compared 29,130 legal successors from every recorded
turn-start position and its color mirror. Legal action sets, resulting serialized
states, and evaluation scores match the original exactly. Regression tests also
check recomputed artillery/infantry masks across legal successors, cached versus
uncached scores, cache eviction, and isolated-cluster allowance masks.
Final full Python suite: **342 passed, 236 subtests passed** in 37.61 s, with
the existing `TestCase` collection warning. `git diff --check` is clean.

Artifacts: `.data/profiles/selfplay-optimized/`,
`optimization-comparison.json`, `optimization-equivalence.json`, and
`rules-interleaved-comparison.json` under `.data/profiles/`. The original module
snapshot is `.data/profiles/before-optimization/`; the comparison driver is
`.data/profiles/compare_optimization.py`.

The sections below retain the original measurements and prioritization as the
baseline for this retest.

## Existing coverage and reproduction

`scripts/benchmark_cpu_bot.py` already measures timed CPU-versus-legacy strength
and latency. `ghqnet.benchmark` measures isolated network throughput, explicitly
excluding rules and encoding. `ghqnet.benchmark_collector` measures collection
throughput and optional inference phases. These do not replace a function-level
profile of a full CPU self-play game.

```sh
.venv-neural/bin/python -B scripts/profile_selfplay.py \
  --output .data/profiles/selfplay-depth1
```

Choose a new output directory when rerunning. The script saves a full trajectory,
source hashes, function timings/call counts, and two standard cProfile files.
View them with `python3 -m pstats PATH/selfplay.prof` (then `sort cumulative`,
`stats 30`). The local measured artifacts are in `.data/profiles/selfplay-depth1/`;
that directory is intentionally gitignored.

The benchmark starts at the standard initial position, fixes depth to one player
turn and beam width to four, and checks that profiling preserves every position,
move, depth, node count, and final state. The 60-second per-search ceiling is a
safety limit: hitting it raises an error instead of comparing different work.
Turn limits remain explicitly unfinished. Imports/startup are excluded.

## Full-game result

| Measurement | Result |
| --- | ---: |
| Outcome | Blue wins by HQ capture |
| Player turns / actions | 104 / 315 |
| Search score calls (reported nodes) | 165,748 |
| Static evaluation calls | 124,765 |
| Unprofiled wall time | 12.00 s |
| Profiled wall time | 31.68 s |
| Median unprofiled turn | 94.0 ms |
| Rules-only replay per game, averaged over 30 repetitions | 28.49 ms |

The first 40 turns consume 7.44 seconds (62% of baseline time); turns 41–80
consume 3.45 seconds; the remaining 24 consume 1.11 seconds. Workload and board
complexity change substantially over the game, so opening-only microbenchmarks
would be misleading. Four later unprofiled runs, including the shortcut experiment
below, ranged from 10.62 to 12.11 seconds.

## Where time goes

Percentages below use cProfile's total self time as denominator and each named
function's cumulative time, including its callees. Rows overlap and **must not be
added**. Profiling inflates runtime by 2.64× and can distort the relative cost of
small Python calls; use unprofiled timings for throughput claims.

| Function / activity | Profile share | Calls |
| --- | ---: | ---: |
| Static evaluation: `search.evaluate` | 73.4% | 124,765 |
| Capture allowance: `_find_free_captures_for_cluster` | 18.4% | 4,028,236 |
| Apply candidate action: `BaseBoard.push` | 13.8% | 126,866 |
| Adjacency clustering: `_find_adjacency_clusters` | 11.0% | 2,695,982 |
| Evaluation mobility: `_safe_destinations` | 7.2% | 2,438,043 |
| Recompute artillery coverage: `get_bombarded_squares` | 6.1% | 155,756 |
| Board copying: `BaseBoard.copy` | 1.8% | 165,748 |
| Search state keys | 1.6% | 165,852 |
| Legal move generation | 0.8% | 136,373 |

Generator call counts include resumptions, not just distinct invocations.
Capture/cluster costs occur inside evaluation, move execution, and generation.
Search accounts for 99.8% of the full-game profile. The rules-only replay performs
full legal enumeration, verifies each chosen action, pushes it, and checks the
outcome; it does not include candidate exploration. Its low runtime does not mean
rules operations are free when search repeats them hundreds of thousands of times.

## Improvements to prioritize

1. **Cheap exits in capture evaluation.** Evaluation asks for capture allowance
   for both colors of every infantry cluster. Empty attacking sides and isolated
   singleton clusters can often return immediately. A temporary singleton/no-enemy
   shortcut preserved the complete trajectory and search work. Alternating baseline,
   shortcut, shortcut, baseline runs averaged 11.536 s versus 11.147 s (3.4% less
   time), but individual timings vary enough that this is preliminary, not a proven
   speedup. The experiment is saved locally as `shortcut-experiment.json`, with its
   driver at `.data/profiles/check_shortcut.py`. No production change was made.
2. **Reuse infantry topology across artillery orientations.** Search evaluates many
   rotations with unchanged infantry positions. A bounded, search-local cache of
   pure adjacency-cluster results is a promising next experiment. Key it on both
   occupancy masks and the infantry mask. Capture-allowance caching needs additional
   artillery/orientation inputs; caching mutable pending-capture state is unsafe.
   Measure hit rate and memory before keeping it. This is a hypothesis, not a measured
   gain.
3. **Precompute evaluation constants.** Square/color coverage weights, advancement,
   center bonuses, and HQ-distance terms currently repeat arithmetic and bit scans.
   Precompute tables and consider bounded coverage-mask score caching. Preserve
   arithmetic order where exact floating-point score equality matters. The evaluator
   itself has 19.9% exclusive time, separate from its helper calls.
4. **Avoid rebuilding coverage twice per artillery move.** `_move_piece` removes and
   reinserts a piece; both paths recompute that color's entire artillery coverage.
   Deferring recomputation until the move is complete could reduce this 6.1% hotspot.
   This needs more care than a lookup table because captures and overlapping fire
   must retain exact behavior. Infantry adjacency updates have a similar pattern.

Board copying and serialization are lower-priority targets in this workload.
Do not start with an undo-stack rewrite or a native-language port based on these
numbers. Search already caches full-state evaluations; topology caching would
target reuse across *different* rule states.

This is one complete, deterministic, shallow game, not a representative strength
or production-throughput estimate. Repeat with deeper/wider searches and neural
self-play before generalizing to the training collector. All percentages describe
this measured workload, not guaranteed optimization savings.

Validation: baseline and profiled trajectories/search work match; 30 unprofiled
and 30 profiled rules replays reach the identical final state with every action
legal. Engine integrity tests: 14 passed, 12 subtests passed. Production rules and
search code were not modified.

Source SHA-256:

- `engine.py`: `c51d2cc122937b02d9ab9711740f6257df142c8eb4f1dd356a70eb0713fececf`
- `search.py`: `fae295d671af07a38407467d41df6746a22811b44ecf4f2573e13aca5fa10818`
