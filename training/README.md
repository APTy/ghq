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

This first version uses ordinary PUCT, not Gumbel search. The evaluator supports
batched positions, but self-play currently searches one game at a time. Parallel
self-play with a shared inference batcher is the next throughput improvement
before committing to a large GPU run. CPU move generation remains a potential
bottleneck even with CUDA inference.

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
pins PyTorch 2.7.1; its real GPU path needs a separate smoke test.

## Run a small complete experiment

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
Loss measurements are training-set diagnostics, not held-out generalization or
playing strength. The model must see substantially more diverse completed games
before its WDL estimates can be trusted.

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

## Evaluate before promoting

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
latency, and checkpoint/CPU/rules hashes. Unfinished games stay outside the W/D/L
counts; do not hide them when interpreting results. A simulation ceiling can
cause neural turns to use less than their allotted time. An inference call in
flight cannot be interrupted, so the deadline is soft by one call's latency.

Promotion should require the reported paradrop/large-artillery tactical cases to
pass in both colors, a clear advantage over `ghq-cpu-4` across enough paired games
to assess uncertainty, few unfinished games, and acceptable browser latency.
The regression fixtures are evaluation cases, not automatically injected into
training. Evaluate against frozen previous checkpoints as well as the CPU bot
before judging self-play progress. The current pipeline does not automatically
promote a checkpoint or claim a calibrated evaluation bar.

After a strong model passes those gates, export and validate browser inference,
then connect its search plans and calibrated outcome estimates to the existing
analysis UI. ONNX export, browser integration, outcome calibration, Gumbel search,
and a distributed opponent league are subsequent work, not implemented features
of this prototype.

Run the held-out tactical gates (four reported positions and an immediate HQ
capture, each in both colors):

```sh
.venv-neural/bin/python -m ghqnet.tactics \
  .data/neural/pilot/checkpoint-0002.pt \
  --output .data/neural/pilot/tactics.json --time-ms 2400 --device cpu
```

Some safety gates can pass by playing quietly, and immediate HQ captures are
explicitly protected in search. Passing these checks does not replace match
evaluation.

## GCP

Project `play-ghq` was verified active with unused L4/T4 quota in `us-central1`.
No GPU resources were provisioned for the local prototype. See [GCP.md](GCP.md)
for the bounded single-L4 launcher, infrastructure prerequisites, immutable
container build, checkpoint uploads, and deletion deadline. Quota is not a
capacity reservation. A GPU pilot should first measure completed examples per
second and training throughput before scaling.

## First local result (2026-11-20)

The full 6-block/96-channel CPU pilot at
`.data/neural/prototype-20261120/checkpoint-0002.pt` trained on 1,209 positions
from two teacher games and eight self-play games. Nine games completed; one hit
the 80-turn limit. The resulting buffer has 968 true outcome labels. The final
iteration's combined training loss fell from 3.704 to 2.820.

At the normal 2.4-second turn budget it passed **4 of 10** held-out tactical
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

The implementation suite passes 141 tests (including 56 parametrized subtests).
That validates the pipeline and rules handling, not the learned playing strength.
The next training experiment needs broader teacher positions and completed
games, with these tactical cases retained as evaluation gates.
