# GHQ neural prototype

This is an experimental policy/value bot with a working training and evaluation
pipeline. The website continues to use `ghq-cpu-4`. A small training run proves
that the pipeline works; it does **not** establish that the checkpoint is stronger.

## Design

| Component | Implementation |
| --- | --- |
| Rules | Imports the same `public/engine.py` as the website |
| State | 41 planes: pieces, directions, reserves, bombardment, infantry adjacency, actions used, used pieces, pending captures, and draw state |
| Canonicalization | Blue inputs rotate 180 degrees and swap colors; search executes original legal moves |
| Network | Six residual blocks, 96 channels, approximately 1.13 million parameters; configurable smaller models for tests |
| Policy | One shared scorer over **each exact legal action**, gathering source, destination, and capture-victim features plus orientation/action/deployment embeddings |
| Value | Win/draw/loss for the player whose action it is; search uses W minus L |
| Search | PUCT over individual actions; retains every legal action; value changes sign only when the player changes |
| Turn planning | Reuses searched positions across actions, with one shared time limit for the complete turn, including automatic captures |
| Checkpoints | Atomic saves with network/schema versions, rules hash, optimizer, seed, and exact replay snapshot identity |

This avoids the old adapter's unconditional value sign changes after every
action and its incomplete state keys. Different capture victims are distinct
policy entries, including the heavy-versus-armored-artillery decision. Learning
can account for support, retreat space, and exposed guns through the board
features and outcomes, without defining another fixed offside penalty.

Search currently uses ordinary PUCT. Curriculum mode runs independent games in
CPU worker processes and combines their leaf evaluations through a shared GPU
batcher. Each worker has at most one pending evaluation; actual batches depend
on how many games request the same frozen model concurrently. CPU move generation,
encoding, interprocess traffic, and CPU-opponent turns can still limit throughput.
Gumbel search and controlled search-budget comparisons remain experiments to do;
neither a larger simulation count nor a larger GPU is proof of stronger play.

## Install and verify

From the repository root:

```sh
python3 -m venv .venv-neural
.venv-neural/bin/python -m pip install -r requirements-neural.txt pytest
.venv-neural/bin/python -m pytest tests training/gcp -q
```

Local implementation was exercised on Python 3.13.1, PyTorch 2.15.0, and NumPy
2.5.5 on Apple Silicon CPU. Requirements deliberately accept compatible PyTorch
2.x versions so the GPU image can retain its CUDA installation. The container
pins PyTorch 2.7.1. The real CUDA path completed a 64-game T4 pilot on
2026-11-21; results and limitations are in [GCP.md](GCP.md#first-gpu-result-2026-11-21).

## Run the original small prototype

```sh
.venv-neural/bin/python -m ghqnet.pipeline \
  --output .data/neural/smoke \
  --iterations 1 --games 2 --teacher-games 1 --teacher-ms 100 \
  --simulations 4 --max-turns 8 --epochs 3 \
  --blocks 1 --channels 16 --batch-size 32 --device cpu
```

This intentionally tiny command checks collection, learning, serialization, and
loading. Its low search budgets and short games are unsuitable for strength
claims. Use a new output directory for each experiment. A more useful CPU pilot:

```sh
.venv-neural/bin/python -m ghqnet.pipeline \
  --output .data/neural/pilot \
  --iterations 2 --games 4 --teacher-games 2 --teacher-ms 600 \
  --simulations 32 --max-turns 80 --epochs 4 --device cpu
```

The pipeline first imitates the production CPU bot's selected actions, then
trains on neural MCTS visit distributions and terminal game outcomes. Teacher
targets are one-hot actions, not a fabricated full search distribution. Early
self-play samples visit distributions with training-only root noise. Later
actions select the most visited move. Only the engine can declare a draw:
length/repetition cutoffs retain policy examples but have **no value label**.
There is no heuristic-score-to-win-probability conversion.

Each run writes per-game JSONL replays with complete serialized states and exact
legal action lists, event logs, iteration metrics, bounded replay snapshots, and
`checkpoint-NNNN.pt`. Metrics distinguish completed games from unfinished games.
In this original prototype mode, loss measurements are training-set diagnostics,
not held-out generalization or playing strength. Curriculum mode below adds an
independent validation partition. Neither kind of loss establishes match strength
or calibrated win probabilities.

Seeds and artifacts make runs auditable. The CPU teacher uses a wall-clock search
budget, so rerunning collection on a different load or machine can select
different moves. Resume preserves replay data; it does not make future timed
teacher searches bit-for-bit deterministic.

Resume a completed checkpoint with its adjacent replay snapshot:

```sh
.venv-neural/bin/python -m ghqnet.pipeline \
  --resume .data/neural/pilot/checkpoint-0002.pt \
  --output .data/neural/pilot \
  --iterations 2 --games 16 --simulations 128 --max-turns 150 --device cpu
```

`--iterations` counts additional iterations. Resume restores optimizer state and
the exact bounded replay buffer, verifies its hash, and rejects overwriting newer
outputs. Keep the original seed. To branch an experiment from an older checkpoint,
choose a fresh output directory. Checkpoints from before replay snapshots were
introduced cannot be used for training resume. They can still be analyzed.

## Batched training curriculum

`--curriculum` enables the larger, resumable training loop. Start with a bounded
local integration check; these short games and tiny search budgets test the
implementation, not playing strength:

```sh
.venv-neural/bin/python -m ghqnet.pipeline \
  --curriculum --output .data/neural/curriculum-smoke --device cpu --threads 1 \
  --teacher-positions 8 --teacher-max-games 8 --teacher-update-games 4 \
  --teacher-ms 25 --teacher-depth 3 --teacher-beam-width 8 \
  --games 8 --total-games 8 --iterations 1 --simulations 8 --max-turns 6 \
  --workers 2 --inference-batch-size 2 --batch-wait-ms 3 \
  --blocks 1 --channels 16 --epochs 1 --batch-size 16 --replay-size 500 \
  --max-seconds 120 --checkpoint-reserve-seconds 15 \
  --exclude-manifest training/benchmarks/reserved.manifest.json
```

A larger experimental configuration is below. `--validate-only` resolves and
checks the arguments without collecting games or training; remove that flag to
run it on a prepared CUDA machine. The targets are an initial research budget,
not a prediction that 2,000 games will beat CPU4 or finish within the time limit.
Use the bounded cloud launcher described in [GCP.md](GCP.md) for rented hardware.

```sh
.venv-neural/bin/python -m ghqnet.pipeline \
  --curriculum --validate-only --output .data/neural/curriculum-2000 \
  --seed 20261121 --device cuda --threads 2 \
  --teacher-positions 50000 --teacher-max-games 2000 --teacher-update-games 200 \
  --teacher-ms 2400 --teacher-depth 4 --teacher-beam-width 12 \
  --total-games 2000 --games 32 --iterations 128 \
  --simulations 256 --max-turns 150 --opening-turns 2 \
  --workers 16 --inference-batch-size 16 --batch-wait-ms 5 --game-timeout-seconds 3600 \
  --cpu-fraction 0.4 --history-fraction 0.2 --history-size 4 \
  --middlegame-fraction 0.25 --start-pool-size 2048 \
  --replay-size 250000 --teacher-replay-fraction 0.2 --hard-replay-fraction 0.1 \
  --validation-fraction 0.1 --validation-size 10000 \
  --epochs 1 --batch-size 64 --max-seconds 12600 --checkpoint-reserve-seconds 120 \
  --exclude-manifest training/benchmarks/reserved.manifest.json
```

Collection begins with CPU-versus-CPU games until it obtains the requested number
of accepted training positions or reaches `--teacher-max-games` scheduled
attempts (including watchdog failures). It updates and checkpoints at most every
`--teacher-update-games` attempts. Position counts include
repeated observations; unique canonical states are reported separately. A bounded
bootstrap that reaches its game cap below the position target reports that
shortfall rather than continuing indefinitely.

Mixed collection schedules 40% current-model self-play, 40% games against the
fixed CPU opponent, and 20% games against frozen historical checkpoints. A saved
schedule cursor keeps proportions and alternating colors balanced across update
and resume boundaries; small chunks do not restart rounding. The fractions are
configurable. Each collection uses one historical snapshot, rotating across
updates, so a few historical games do not each require a separate inference
batch. Historical
slots use the CPU opponent until an older snapshot is available; reports retain
both requested and actual opponent types. Mixed CPU/historical games train on
the current model's decisions only. Ordinary self-play retains both sides.
Selected middlegames come only from the training partition and retain their
source family identity. No state with an outstanding draw offer is added to that
start pool.

Teacher policies remain one-hot chosen actions; no full search distribution or
calibrated value is inferred from a CPU heuristic score. Neural policy targets
remain MCTS visit distributions. Genuine draws keep draw labels; length and
repetition cutoffs keep policy targets but have no WDL target. `--total-games`
counts collected mixed games across resumes, including cutoffs, so inspect HQ
wins, draws, unfinished games, lengths, and skip rates alongside the game count.
The iteration count limits updates in the current invocation; set it high enough
for the remaining target when reducing games per update. The wall-time limit
still bounds the run. A batch cannot exceed the worker count because each game
has at most one pending inference request.

Replay retention reserves at least the requested teacher and hard-example shares
when those examples are available, then fills the remaining capacity from all
remaining records. These are retention quotas, not extra loss weights or invented
labels. Current automatic `hard` tags identify legal captures of an HQ, artillery,
or airborne unit, or a position where our HQ is under bombardment. Such positions
are tactically relevant; the tag does not prove the move is good or the position
is a verified puzzle. Their policy and outcome targets are unchanged. Trusted
additional replays can be supplied with repeated `--input-replay PATH` on a fresh
curriculum, subject to the same exclusions and partitions.

Curriculum training has a separate **row-exposure budget**. `--replay-ratio 1`
(the initial conservative default) earns one training-row exposure per newly
admitted, distinct canonical training state. This is not necessarily training
that new row: the budget samples a deterministic shuffled prefix of the retained
replay. Duplicate states, validation rows, exclusions, and checkpoint saves earn
no extra budget. A smaller `--games` batch therefore no longer buys another full
replay epoch. The ratio is a setting to evaluate, not a proven optimum for either
teacher bootstrap or self-play.
Equal exposure budgets still allow different minibatch boundaries and optimizer
trajectories when collection cadence changes.

The persisted `replay_exposure` ledger records admitted unique states, actual
trained examples, and remaining fractional credit. Only completed optimizer
batches spend credit; a partial final batch respects the exact row limit.
Deadlines and the `--epochs` upper bound leave unused credit for a later update
with newly admitted training states. Saving alone, failed collection, and
collection containing no novel training states do not train. A zero-step save
also leaves an existing frozen-opponent history unchanged.

Resume requires the same replay ratio as the saved ledger. Older checkpoints
without this accounting retain all replay and model/optimizer state but receive
zero retroactive credit, with that migration recorded explicitly. Historical
exposure and pending untrained rows cannot be reconstructed reliably from a
buffer or Adam step count. Future admissions earn credit normally. Prototype
mode retains its existing epoch behavior; `--replay-ratio` requires `--curriculum`.

Families are assigned to training or validation by a seeded hash. A game and its
continuations remain together. Canonical state fingerprints treat a 180-degree
color swap as the same position; a record already seen in the other partition is
omitted. Those exclusions survive replay eviction and checkpoint resume. The
bounded validation buffer is measured separately and is never passed to optimizer
updates. This prevents exact-state leakage; it does not make positions from
unrelated games strategically independent or prove general win-probability
calibration. Start with fresh weights for a clean evaluation: importing an old
checkpoint cannot undo its prior exposure to a reserved case.

The benchmark manifest reserves all listed source-game identities and replay
hashes, plus the selected tactical state fingerprints. All frozen suite splits,
including its partition named `train`, are currently reserved. The training
process receives exclusions, not the tactical solutions. See
[the benchmark notes](benchmarks/README.md) for provenance and objective limits.

Every completed game is written immediately. At the internal deadline, completed
data and the current model/optimizer are checkpointed; training checks the budget
while preparing records, between optimizer steps, and between validation batches.
The cloud VM's deletion deadline is a separate outer bound. Interrupted scheduled
games are discarded and their IDs recorded, while future IDs remain unique.
An individual game's absolute watchdog retires only that worker/game and starts
a replacement for remaining tasks. It contributes no replay rows or fabricated
draw; healthy games continue. Global budget exhaustion checkpoints and stops.
Worker startup, evaluator request, and inference failures are distinct fatal
conditions, saved with their cause and completed progress. An entirely
unproductive collection stops with `curriculum-no-progress`, rather than looping
forever. Timeout events include the game ID, elapsed time and inference activity;
that activity is not a count of completed turns.
Resumption is auditable, not a claim of bit-for-bit agreement with an uninterrupted
wall-clock run.

Keep these adjacent files when moving a curriculum checkpoint:

- `checkpoint-NNNN.pt` and its exact `replay-buffer-NNNN.jsonl`.
- `validation-buffer-NNNN.jsonl` and `curriculum-NNNN.json`, both hash-identified
  by the checkpoint.
- The bounded set of `history-<hash>.pt` files referenced by the curriculum state.

Resume using the same seed, validation fraction, opponent proportions, and
desired targets. For example,
after copying the complete checkpoint bundle to `RUN`:

```sh
.venv-neural/bin/python -m ghqnet.pipeline \
  --curriculum --resume RUN/checkpoint-0005.pt --output RUN \
  --seed 20261121 --device cuda --threads 2 \
  --teacher-positions 50000 --teacher-max-games 2000 --teacher-update-games 200 \
  --teacher-ms 2400 --teacher-depth 4 --teacher-beam-width 12 \
  --total-games 2000 --games 32 --iterations 128 --simulations 256 --max-turns 150 \
  --workers 16 --inference-batch-size 16 --validation-fraction 0.1 \
  --game-timeout-seconds 3600 \
  --max-seconds 12600 --checkpoint-reserve-seconds 120
```

The actual checkpoint number is reported in `summary.json`. Bootstrap counters,
the exact replay and validation buffers, split registries, historical model
identities, optimizer state, and cumulative mixed-game count resume from that
checkpoint. Pending teacher collection continues if its target/cap was not
reached. `--iterations` limits additional mixed updates; `--total-games` remains
a cumulative cap. Use a new output directory to fork an older checkpoint.

Before scaling further, compare complete collection throughput and learning at
128/256/512 simulations using the same starting model, opponent mix, and compute
budget. Collection wall time is the correct throughput denominator: per-game
times overlap across workers. Record actual batch sizes, CPU time, root action
coverage, opponent-reply coverage, and held-out results. A Gumbel/sequential-halving
ablation is still pending. Stronger teacher alternatives and broader reviewed
positional objectives also remain possible improvements; current tactical tags
and capture witnesses do not replace those experiments.

## Measure collector throughput

`ghqnet.benchmark_collector` measures real spawned games and shared inference
without training, changing checkpoints, or promoting a model. The collector's
default `auto` protocol sends numeric board/action inputs from each worker when
the evaluator supports `predict_encoded_batch`, including `TorchEvaluator`.
Custom evaluators with only `predict_batch` retain the `legacy` protocol, which
sends serialized rules states and actions for the parent to decode and encode.
Force `legacy` or `encoded` to compare the two paths; forcing `encoded` on an
unsupported evaluator fails explicitly. Legal action order and model identity
remain part of the request in both paths.

Start with a small local check using an existing checkpoint and a new report path:

```sh
.venv-neural/bin/python -m ghqnet.benchmark_collector CHECKPOINT.pt \
  --output .data/neural/collector-local.json --device cpu --threads 1 \
  --games 2 --max-turns 2 --opening-turns 0 --simulations 8 \
  --workers 1 2 --batch-size 2 --batch-wait-ms 0 \
  --inference-protocol legacy encoded --profile-sample \
  --seed 78123 --repeats 1 --case-max-seconds 20 --max-seconds 120
```

`--workers`, `--batch-size`, `--batch-wait-ms`, `--simulations`, and
`--inference-protocol` accept lists and form a Cartesian grid. A single batch
size or wait value remains valid. `--repeats` changes the seeded workload between
repeats while keeping it identical across configurations within each repeat.
Keep the checkpoint, positions, opponent mix, seeds, and search budget fixed
when comparing throughput. Use at least as many games as requested workers:
`actual_workers` exposes the cap when, for example, 32 workers are requested for
only 16 games. An inference batch also cannot exceed concurrent worker requests
for the same model.

The default grid is unprofiled. `--profile-sample` first runs one diagnostic case
per requested protocol, using the first worker/batch/wait/simulation settings,
then runs the unprofiled grid. Alternatively, `--profile-inference` profiles every
case; these flags are mutually exclusive. Profiling measures worker preparation,
IPC, decoding, queue wait, and model packing/transfers/forward/postprocessing.
Supporting CUDA evaluators synchronize phase boundaries, which can change
throughput. Use diagnostic cases to locate costs and compare **unprofiled** cases
to choose settings. Worker phase totals can overlap and need not sum to wall time.
The first case for each repeat/simulation budget supplies the move-parity
reference; with `--profile-sample`, that reference is a diagnostic case, not a
throughput baseline.

For representative positions, add `--positions RUN/replay-buffer-NNNN.jsonl
--position-count 16`. Plain JSONL objects containing a serialized `state` are also
accepted. Selection uses a seeded reservoir over distinct exact state strings;
the report records the source hash, selected line numbers, and state hashes.
Only selected states are validated, and replay labels are unused. Supplied
states replace random openings and are cycled across games. A one-turn cap from
a mid-turn position may measure only the remaining part of that turn, so these
short cases do not establish sustained full-game throughput or opponent balance.

Add `--historical-checkpoint RUN/history-HASH.pt --history-fraction 0.2` to use
one frozen historical opponent, and `--cpu-fraction 0.4` for CPU opponents; the
remaining fraction is current-model self-play. Fractions must sum to at most
one. `--teacher-ms`, `--teacher-depth`, and `--teacher-beam-width` fix the CPU
opponent's search budget. Timed CPU search and floating-point differences can
still change later moves, so inspect matched task hashes and parity results
before interpreting a speed comparison.

Each case is saved atomically to the report as it finishes. `--case-max-seconds`
limits one case; `--max-seconds` bounds the whole grid, including loading,
position sampling, and warmup. A running inference call cannot be preempted.
Deadline-limited cases retain completed results and are marked incomplete;
startup, request, or evaluator failures stop the benchmark with their cause.
Reports retain startup, active collection, cleanup, and optional phase timings.
Simulation/new-node rates cover returned rollouts only; neural-row totals also
include work from interrupted games. All headline rates use actual collection
wall time rather than summed concurrent game times. `returned_rollouts` includes
turn/repetition cutoffs, whereas `terminal_games` counts only rules terminals;
neither a returned short rollout nor raw collector `completed_games` means a
finished game. Rank matched, complete, unprofiled cases rather than partially
completed workloads.

The full bounded GPU grid is in
[experiments/collector-tuning.json](experiments/collector-tuning.json). It fixes
checkpoint 0004's replay/history companions and compares 8/16/32 workers,
16/32-row requested batches, 0/1/5 ms waits, and both protocols. See
[isolated collector performance experiments](GCP.md#isolated-collector-performance-experiments)
for `--collector-benchmark-config`, verified resume/source bundles, output files,
and the separate VM deletion bound. Update companion paths when changing the
checkpoint.

[experiments/collector-confirmation.json](experiments/collector-confirmation.json)
follows that screen with four-turn rollouts, two seeded repeats, 16/32 workers,
32-row maximum batches, and a 1 ms wait. It reverses protocol order and exercises
both sides of CPU/historical matchups. Select settings using complete matched
cases and collection wall time, including spawn/cleanup, then check neural rows
and CPU search work for trajectory differences. Four-turn results still do not
establish sustained full-game throughput.

### Measured T4 settings (2026-11-21)

On one T4 with 16 vCPUs, the 38-case screen completed all cases. Each of the 18
unprofiled legacy/encoded pairs returned identical moves and neural-row counts;
encoded transport improved wall-clock throughput by 29–46%. Diagnostic parent
request decoding fell from 5.65 seconds to 0.03 seconds. Feature preparation now
happens in workers; the GPU forward computation itself did not get faster.

The eight-case confirmation used 32 four-turn rollouts per case, 256 search
simulations, 16 replay positions, two seeded repeats, and the 40% CPU / 20%
historical / 40% current-model opponent mix. Both protocols used batch 32 and a
1 ms wait. Collection time includes worker startup and cleanup:

| Protocol | Workers | Mean collection time | Pooled neural rows/second |
| --- | ---: | ---: | ---: |
| Legacy | 16 | 97.05 s | 735 |
| Encoded | 16 | 66.08 s | 1,080 |
| Legacy | 32 | 85.84 s | 831 |
| Encoded | 32 | 54.23 s | 1,316 |

Encoded transport improved matched throughput by 47% at 16 workers and 58% at
32 workers. Moving encoded collection from 16 to 32 workers added 21.8% in both
repeats. The combined comparison is 1.79×, but its legacy baseline uses batch
32/wait 1, not the old production batch 16/wait 5. Both protocols also use the
optimized encoder, so these comparisons isolate transport and concurrency.

Use [curriculum-resume-optimized-2000.json](experiments/curriculum-resume-optimized-2000.json)
for the next mixed-collection resume: encoded inference, 32 workers, batch 32,
wait 1 ms. It preserves the learner's 256 simulations, CPU opponent's 3.2-second
search budget, curriculum, and training limits. Across worker counts, 61/64
move sequences and 62/64 final states matched; every divergence began with a
timed CPU-opponent decision. The 32-worker configuration searched about 8% fewer
CPU nodes. This is a measured throughput tradeoff, not a measured strength loss
or gain. Teacher bootstrap labels remain unchanged; these settings were tested
for mixed collection, not a new CPU-only teacher bootstrap.

The [performance record](benchmarks/collector-throughput-20261121.json) contains
the run IDs, source/checkpoint hashes, per-case measurements, phase profiles,
and parity details. All 46 GPU cases completed, both temporary VMs were deleted,
and the existing training run was left unchanged. Four-turn results do not
establish completed games per hour or playing strength.

## Evaluate before promoting

Check imitation and outcome prediction on the saved validation replay:

```sh
.venv-neural/bin/python -m ghqnet.evaluate_replay \
  --checkpoint RUN/checkpoint-0000.pt \
  --heldout-replay RUN/validation-buffer-0000.jsonl \
  --output RUN/heldout-0000.json --device cpu --threads 1
```

This read-only report separates sources, the first/second/third chosen actions,
automatic captures, and positions with only one legal action. Use its non-forced
teacher agreement and negative log likelihood to judge imitation; fresh-turn Skip
rates help diagnose passivity. WDL log loss, Brier score, and accuracy use only
actual outcome labels. Canonical unique-state counts expose repeated positions.
The default evaluates the full replay; optional `--max-samples N --seed SEED`
selects a reproducible family-balanced subset. A replay explicitly marked as
training, or matching the checkpoint's training-buffer hash, is rejected.
Validation-buffer identity is reported when available, but this diagnostic does
not independently audit all historical training exposure. None of these metrics
substitute for playing-strength matches.

Analyze any serialized position or board FEN without touching the running app:

```sh
.venv-neural/bin/python -m ghqnet.analyze \
  .data/neural/pilot/checkpoint-0002.pt --time-ms 2400
```

Run paired matches with both colors, identical seeded openings, and the same
**total per-turn** time limit as the CPU bot:

```sh
.venv-neural/bin/python -m ghqnet.arena \
  .data/neural/pilot/checkpoint-0002.pt \
  --output .data/neural/pilot/arena.json --pairs 50 \
  --time-ms 2400 --max-turns 150 --device cpu
```

The report includes wins, draws, losses, unfinished games, full move trajectories,
latency, checkpoint/CPU/rules hashes, and uncertainty across opening pairs.
Unfinished games stay outside the W/D/L counts and have separate worst/best score
bounds; do not hide them when interpreting results. A simulation ceiling can
cause neural turns to use less than their allotted time. An inference call in
flight cannot be interrupted, so the deadline is soft by one call's latency.

Promotion should require the reported paradrop/large-artillery tactical cases to
pass in both colors, a clear advantage over `ghq-cpu-4` across enough paired games
to assess uncertainty, few unfinished games, and acceptable browser latency.
The legacy regression fixtures are evaluation cases, not automatically injected
into training. Their anchors are not listed in `benchmarks/reserved.manifest.json`
and may occur naturally during teacher or self-play games; these checks do not
establish unseen-position generalization. The separate 250-case frozen suite has
explicit exclusions for fresh runs that use its manifest. Evaluate against
frozen previous checkpoints as well as the CPU bot
before judging self-play progress. The current pipeline does not automatically
promote a checkpoint or claim a calibrated evaluation bar.

After a strong model passes those gates, export and validate browser inference,
then connect its search plans and calibrated outcome estimates to the existing
analysis UI. ONNX export, browser integration, outcome calibration, Gumbel search,
and a multi-machine opponent league are subsequent work. The implemented
curriculum uses one parent inference broker and local worker processes.

Run the legacy tactical regression checks (four reported positions and an immediate HQ
capture, each in both colors):

```sh
.venv-neural/bin/python -m ghqnet.tactics \
  .data/neural/pilot/checkpoint-0002.pt \
  --output .data/neural/pilot/tactics.json --time-ms 2400 --device cpu
```

Some safety gates can pass by playing quietly, and immediate HQ captures are
explicitly protected in search. Passing these checks does not replace match
evaluation.

The frozen collection adds 250 distinct states in 61 families, including positive
capture and immediate HQ-defense objectives that cannot pass by skipping. It is
mostly one-action captures, with 18 two-action witnesses and 24 defense states
from three templates. A witness proves a narrow attainable objective, not that a
trade is favorable after the reply or that the move is strategically best. Cases
mined from the old pilot are not unseen tests of checkpoints trained on that pilot.
Use the [benchmark notes](benchmarks/README.md) for exact provenance and splits.

```sh
.venv-neural/bin/python -m ghqnet.tactics CHECKPOINT.pt \
  --suite training/benchmarks/cases.jsonl --split test \
  --output .data/neural/tactics-test.json --time-ms 2400 --device cpu --threads 2

.venv-neural/bin/python -m ghqnet.arena CHECKPOINT.pt \
  --write-openings .data/neural/openings-100.json --pairs 100 \
  --output .data/neural/arena-cpu.json --time-ms 2400 --device cpu --threads 2

.venv-neural/bin/python -m ghqnet.arena CHECKPOINT.pt \
  --opponent-checkpoint PREVIOUS.pt --openings .data/neural/openings-100.json \
  --output .data/neural/arena-previous.json --time-ms 2400 --device cpu --threads 2
```

One hundred opening pairs means 200 games per opponent; this is a proposed
evaluation budget, not an automatic promotion threshold. Keep timed matches
sequential on an otherwise idle machine, reuse the same frozen openings, and
examine uncertainty and unfinished rates. Tactical scores explain particular
failures; they do not substitute for the match results.

## Fixed-replay learning-rate diagnostic

When a new checkpoint regresses, compare update behavior before collecting more
games. `ghqnet.ablate_replay` loads the same curriculum checkpoint, replay and
held-out buffer for each arm. It restores Adam's moments and step counters before
overriding only the learning rate, uses identical sample order and update counts,
and reports validation metrics separately by data source.

```sh
.venv-neural/bin/python -m ghqnet.ablate_replay CHECKPOINT.pt \
  --output .data/neural/replay-ablation-local --device cpu \
  --learning-rates 0.001 0.0003 --epochs 1 --batch-size 128 \
  --threads 2 --seed 20261121 --max-seconds 1200
```

The checkpoint's exact replay, validation, curriculum and history companions must
be beside it. Inputs are verified and never rewritten. Both arms must finish the
same complete epoch; partial results are ineligible for comparison. No games are
generated and no winner is selected automatically. Saved arm checkpoints are
diagnostic, not directly resumable curriculum checkpoints. A validation improvement
is not a playing-strength claim; use development tactics and new paired matches
before considering a production change. See [GCP.md](GCP.md) for the bounded GPU
workload and [the example configuration](experiments/replay-ablation-20261121.json).

## Value-target experiment preparation

Replay v3 records both players' immutable model identities, their search and
exploration settings, and the actual terminal result. Versions 1 and 2 remain
readable, but their missing provenance is unknown. A row's `mcts` source alone
does not establish self-play. Raw outcomes are preserved; an optional training
mask controls which outcomes contribute to value loss. Policy targets stay in
every batch. Metrics distinguish raw labels, eligible labels and actual
value-training exposures.

The [frozen next experiment](experiments/value-targets-20261121.json) starts from
checkpoint4, with 128 training and 32 development continuations from separate
existing families. The same checkpoint plays both sides throughout collection.
Roots inherit their original partition; old state registries, mirrors, reserved
cases and the current arena's opening roots remain excluded from cross-partition
reuse. Development families have been used before: this is a development screen,
not untouched promotion evidence.

Prepare roots and hash the inputs without collecting games, running inference,
or training (run from the repository root):

```sh
.venv-neural/bin/python -m ghqnet.value_experiment_data prepare \
  --config training/experiments/value-targets-20261121.json \
  --output .data/neural/value-targets-20261121
```

After the existing timed arena finishes, collection and assembly are separate,
explicit commands. Collection uses batched inference and never updates weights.
The frozen cohort has a two-hour total collection budget and 900-second game
watchdogs. Cutoffs retain unknown outcomes; missing games block assembly rather
than becoming draws or an omitted unsuccessful subset. Only known unstarted
jobs can continue automatically on a later explicit collection invocation;
interrupted or failed games require a recorded recovery decision.

```sh
.venv-neural/bin/python -m ghqnet.value_experiment_data collect \
  .data/neural/value-targets-20261121 --device cpu
.venv-neural/bin/python -m ghqnet.value_experiment_data assemble \
  .data/neural/value-targets-20261121
```

Assembly selects 1,920 distinct legacy and 2,176 fresh training states, plus
1,024 legacy and 1,024 fresh development states, with family-balanced sampling.
Fresh value rows require verified terminal outcomes; unfinished games remain in
the raw collection. Insufficient data fails the readiness check. No root,
state, family or outcome is resampled to improve the comparison.

Then freeze the training comparison, still without running it:

```sh
.venv-neural/bin/python -m ghqnet.ablate_value_targets \
  --checkpoint .data/neural/curriculum-resume-20261121/resume-input/checkpoint.pt \
  --training .data/neural/value-targets-20261121/dataset/training.jsonl \
  --validation .data/neural/value-targets-20261121/dataset/validation.jsonl \
  --partition .data/neural/value-targets-20261121/dataset/partition.json \
  --model-id fd6a4fcd68305438a65ce709c064b22a546dac93700b1f53084e0b07868e48bb \
  --output .data/neural/value-targets-20261121/comparison-prepared
```

To execute later, use a different new output directory and add `--execute`.
Both arms restore identical weights and Adam state, then see the same policy
rows in the same order for 32 batches of 128, across three fixed shuffle seeds.
Arm B uses only current self-play value labels. Arm A randomly selects mixed
terminal labels, matching B's labeled count in each exact batch. This prevents
fewer value exposures from masquerading as a better label source. Masks, order,
input hashes, label counts, source mix and family counts are recorded before
inference. Actual completed policy and value exposures must match the plan.

Fresh held-out self-play value loss is primary. Legacy policy retention remains
a check, while teacher/mixed value loss is diagnostic. The configuration records
the advancement criteria; the runner reports every arm and does not select or
promote a winner. A promising result still needs development tactics and new
paired strength matches. This first comparison also changes which positions
receive value supervision, so it cannot prove mixed-opponent outcomes are wrong.
Comparing old and fresh outcomes at identical roots would be a stronger later
control. No GPU provisioning or new jobs occur during preparation.

## Smaller updates and a bounded strength screen

`training/experiments/update-size-20261122.json` freezes the follow-up on the
saved value-target dataset. `ghqnet.ablate_update_size` tests learning rates
0.0003 and 0.0001 with the same three shuffle seeds, 32 batches of 128 policy
rows, and 2,176 verified self-play value labels. Every run restores checkpoint4
and its original Adam state. The previous 0.001 result is descriptive reference;
the saved development data is reused for calibration, not independent proof.

`ghqnet.update_size_decision` chooses the fixed representative seed using the
predeclared qualification and near-tie rules. If neither rate qualifies on the
prediction metrics, it still selects one explicitly exploratory candidate for
the bounded match test. It never chooses a favorable seed. The separate
`ghqnet.update_evaluation` runner checks baseline and candidate on the same 44
tactical color runs, then plays 48 games from eight fresh paired openings:
candidate–baseline, candidate–CPU, and baseline–CPU. Legal, complete, finite
plans and both immediate HQ captures are required; other tactical pass counts
and teacher agreement remain diagnostic. Games run sequentially at 2.4 seconds
per whole turn, with explicit game and phase caps. Unfinished games are unknown
outcomes. No automatic extension or promotion is allowed.

Local progress and immutable inputs live under
`.data/neural/update-size-20261122/`: `execution.json` identifies the active
phase; `comparison/report.json` retains all six training runs; `decision.json`
records selection; `tactics/decision.json` records the hard gate and advisory
regressions; `arena/report.json` saves each game, trajectory, and paired summary.
The source hashes are frozen before execution, so do not edit engine, model,
search, or experiment code during the timed evaluation. Preserve the arena
openings and trajectories as evaluation data for subsequent learning cycles.

## GCP

Project `play-ghq` has dedicated private training infrastructure and a completed
single-T4 pilot. Its VM and boot disk were deleted; image/checkpoint storage is
retained. See [GCP.md](GCP.md) for measurements, the bounded L4/T4 launcher,
infrastructure prerequisites, immutable container build, checkpoint uploads,
and deletion deadline. Quota is not a capacity reservation.

## First local result (2026-11-20)

The full 6-block/96-channel CPU pilot at
`.data/neural/prototype-20261120/checkpoint-0002.pt` trained on 1,209 positions
from two teacher games and eight self-play games. Nine games completed; one hit
the 80-turn limit. The resulting buffer has 968 true outcome labels. The final
iteration's combined training loss fell from 3.704 to 2.820.

At the normal 2.4-second turn budget it passed **4 of 10** tactical regression
cases: the empty paradrop and immediate HQ capture in both colors. It failed
the trapped-paradrop, heavy-versus-armored selection, and large-gun safety cases
in both colors. This is a working prototype, **not a stronger replacement yet**.
The report is `.data/neural/prototype-20261120/tactics.json`.

In one paired opening (both colors) against `ghq-cpu-4` at 2.4 seconds per turn,
the prototype lost both games, with no draws or unfinished games. Its mean turn
latency was approximately 2.3 seconds; the largest observed overrun was 7.4 ms.
This two-game baseline is too small for an Elo estimate. Full trajectories and
budgets are in `.data/neural/prototype-20261120/arena.json`.

Collection took approximately 85 seconds and training 30 seconds on the local
CPU. These measurements justify using CPU for this pipeline pilot; they do not
predict throughput for a large batched training run.

After the GPU pilot changes, the implementation suite passes 184 tests and
75 parametrized subtests (with one existing test-collection warning).
That validates the pipeline and rules handling, not the learned playing strength.
The subsequent GPU pilot added 64 self-play games and four updates. It still
lost both paired CPU matches. Its final tactical score rose to 6/10 only because
it skipped in both trapped-paradrop cases; heavy-target selection and large-gun
safety still failed. The last two collection batches were entirely double-skip
draws. This supports adding fixed CPU/historical opponents and broader teacher
examples before a larger run, while retaining the reported positions as
regression checks. Batch inference across concurrent games is the measured
throughput opportunity; increasing GPU size alone does not fix serial search.
