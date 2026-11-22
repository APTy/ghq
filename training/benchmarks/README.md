# Frozen evaluation collection

`cases.jsonl` contains **250 unique states from 61 families**, with 500 color runs:

| Objective | Unique states |
| --- | ---: |
| Capture infantry / armored infantry / airborne infantry | 31 / 12 / 30 |
| Capture regular / armored / heavy artillery | 72 / 43 / 34 |
| Capture HQ | 4 |
| Actively defend an immediately threatened HQ | 24 |

The 226 capture cases come from 58 different games in the November 21 T4 pilot. Of those, 18 have two chosen actions in their witness and 208 have one. The 24 constructed defense positions form three attacker-type template families; translations of a template are kept together. Each case has an exact, legal, complete-turn witness, checked in both colors. A passive skip fails every objective. Mirrored runs share one case ID and do not increase the unique-state or family count.

These are narrow positive objectives. A capture witness proves that the stated material gain or HQ capture is attainable in the remaining turn. It does **not** establish an optimal move, a favorable trade after the reply, or a forced game win unless the HQ is actually captured. Defense gates enumerate every forced-capture branch and every first chosen enemy action; they do **not** prove survival of the opponent's whole three-action turn. The corpus is weighted toward immediate captures and does not measure general positional strength. Full matches remain necessary.

`candidates.jsonl` contains 200 additional positions for later review. Their policies and observed game outcomes are provenance, never ground truth. `unverified` and `teacher_approximate` records have no scored objective and are excluded from tactical pass rates. The generator never derives a correctness label merely from a losing game.

The source pilot games were used to train historical checkpoints 0003–0006. **This collection is diagnostic, not unseen evaluation for those checkpoints.** A fresh training run can reserve these game families. `reserved.manifest.json` excludes all 64 source replay hashes, game IDs, and families, plus the 250 selected tactical state fingerprints. It intentionally does not exclude every ordinary opening state from those games. Fresh curriculum IDs use a separate namespace. Both color versions are covered by the fingerprint `sha256(min(board.serialize(), board.mirror().serialize()).encode("ascii"))`.

The frozen train/dev/test partitions have 197/17/36 states in 48/5/8 families. Every source-game family and constructed template stays within one split. All three partitions are currently reserved from training through the full exclusion manifest; a `train` label does not authorize importing it automatically. The split manifests pin the SHA of the common case file. Candidate records share the same family partitioning. No command here trains or promotes a model.

Run the original ten regression color runs plus the frozen test partition:

```sh
.venv-neural/bin/python -m ghqnet.tactics CHECKPOINT.pt \
  --suite training/benchmarks/cases.jsonl --split test \
  --output .data/neural/tactics-test.json --device cpu --threads 2 \
  --time-ms 2400 --simulations 100000 --seed 4173
```

Use `--split all` for the full 250-case diagnostic, or `--skip-regressions` for only the selected frozen cases. Results include per-color pass counts, cases passing both colors, categories, and family counts. The original ten regressions remain the default when `--suite` is omitted.

The legacy regression anchors are not listed in `reserved.manifest.json`. They are not explicitly injected into training, but can arise naturally during teacher or self-play games. Their reports do not establish absence of training overlap. The manifest exclusions described above apply to the separate frozen collection.

Reproduce this collection into a **new** directory (the generator refuses overwrites):

```sh
.venv-neural/bin/python -m ghqnet.suite \
  --replays .data/neural/gpu-pilot-20261121/remote/ghq-train-20261121-133423-0ad864/run/replay \
  --output .data/neural/evaluation-rebuild --count 250 --candidate-count 200 \
  --defenses 24 --seed 4173 --source-usage known_training
```

Game files must include their actual action trajectory and final board. The miner validates the retained trajectory and every state against the authoritative engine. Constructed positions are explicitly marked as such; legal engine validation is not presented as a proof of reachability from the initial board.

For matches, freeze one opening manifest and reuse it against CPU and historical checkpoints:

```sh
.venv-neural/bin/python -m ghqnet.arena CANDIDATE.pt \
  --output .data/neural/arena-cpu.json --write-openings .data/neural/openings.json \
  --pairs 20 --device cpu --threads 2 --time-ms 2400

.venv-neural/bin/python -m ghqnet.arena CANDIDATE.pt \
  --opponent-checkpoint HISTORICAL.pt --openings .data/neural/openings.json \
  --output .data/neural/arena-historical.json --device cpu --threads 2 --time-ms 2400
```

Each opening is played with candidate Red and candidate Blue, at the same turn budget. Bootstrap intervals resample opening pairs, not individual correlated games. The score interval is explicitly conditional on completed games; unfinished games are separately counted and produce lower/upper score bounds. Fewer than two contributing opening pairs or a degenerate bootstrap produces no bootstrap interval. A separate conservative Hoeffding interval uses only fully finished pair scores and does not imply certainty after a handful of uniform wins. These intervals describe the sampled opening distribution. Run timed matches sequentially on an otherwise idle CPU. Promotion always requires manual review.
