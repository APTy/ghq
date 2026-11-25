# Infantry tuning — November 25, 2026

Keep a single behavior change: infantry support weight **0.045 → 0.035**
(22% lower). Advance and mobility weights remain 0.035 and 0.009.
The named constants in `public/search.py` make future comparisons easier.

The old support reward counted each neighboring pair twice, for a total of
0.09 infantry units. Advancing one rank earns only 0.035. This creates an
incentive to reinforce clusters and stay together. Reducing the support reward
weakens that incentive while keeping the existing capture, isolation, HQ safety,
and airborne-option evaluation.

## Results

Native Python, fixed depth 2, beam 4, three fresh opening seeds (20261201 plus
multiples of 7919), each played in its original and mirrored orientation.
Each configuration plays itself. These are six games per configuration but
only three distinct opening families. Search timeouts invalidate a run; all
actions are checked against the rules engine.

Compare the **first 30 player turns of every game**, so an earlier finish cannot
artificially lower board population. Each column contains 180 player turns:

| Measurement | Original 0.045 | Selected 0.035 |
| --- | ---: | ---: |
| Infantry deployments | 92 | 83 |
| Infantry moves | 114 | 151 |
| Forward infantry moves | 82 | 113 |
| Infantry captures | 3 | 9 |
| Mean infantry on board for the moving side | 9.77 | 9.16 |
| Infantry with zero movement destinations | 22.46% | 15.11% |
| Infantry with at most two movement destinations | 61.68% | 58.80% |

Infantry includes regular, armored, and airborne units. Captures include their
explicit capture actions and automatic free captures, excluding bombardments.
Movement destinations use the evaluator's geometry after the player's turn;
this measures crowding/engagement/fire restrictions, not a guaranteed safe move
after an enemy reply. Population and crowding are sampled after each player
turn, for that player's pieces.

Across the full games (80-turn cap), original self-play finished 0/6; selected
self-play finished 3/6. Full-game infantry captures per 100 player turns rose
from 6.04 to 13.86, and blocked-infantry samples fell from 19.61% to 10.57%.
Different game lengths make these full-game figures less directly comparable.

Six separate depth-1, beam-4 games against the original bot, with paired colors
and three opening seeds starting at 20261125, ended **1 win, 1 loss, 4 unfinished**.
Unfinished games are not draws. This is evidence of changed behavior in a small
sample, not a demonstrated playing-strength increase or browser benchmark.

## Alternatives and checks

The initial depth-1 screen compared original settings, support 0.015 alone,
support 0.015 with advance 0.07, and those two changes with mobility 0.025.
All reduced crowding and increased captures in the screened opening family.
However, support 0.015 failed two existing paradrop regressions. Values 0.020
and 0.025 also failed both; 0.030 failed one. Values 0.035 and 0.040 passed both.
The selected 0.035 passed the full suite: **347 tests and 236 subtests**.
No regression expectations were weakened.

## Reproduction and artifacts

From the repository root, use a fresh output path for each command:

```sh
.venv-neural/bin/python -B scripts/benchmark_infantry.py --output /tmp/infantry-selfplay.json --variants baseline moderate --pairs 3 --seed 20261201 --turns 80 --depth 2
.venv-neural/bin/python -B scripts/benchmark_infantry.py --output /tmp/infantry-matches.json --variants moderate --pairs 3 --turns 80 --head-to-head
.venv-neural/bin/python -B -m pytest tests -q
```

Local artifacts in `.data/infantry-tuning/`: `screen.json` (initial variants),
`validation.json` (deeper baseline/aggressive candidate), `moderate.json`
(selected deeper candidate), `moderate-matches.json` (against original),
`matched-30-turns.json` (equal-horizon aggregate), and `moderate-tests.log`.
Game reports retain every pre-turn state and selected action, configuration,
and search source fingerprint. To reproduce the equal-horizon aggregate, replay
the first 30 trajectory entries of each game and sample the moving side after
each turn using the definitions above.

Reload the browser after changing Python sources so workers load the new bot.
