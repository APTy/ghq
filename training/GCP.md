# Bounded GHQ GPU training

Start with local correctness and throughput measurements in [README.md](README.md).
This job is a pilot, not a trained production bot or a claim of playing strength.
It runs one GPU VM, using project `play-ghq` explicitly. The default is
`g2-standard-4` (one NVIDIA L4); `--gpu t4 --cpus 16` selects
`n1-standard-16` with one NVIDIA T4 and sixteen CPUs for game workers.
T4 supports `--cpus 4`, `8`, or `16`; its default remains four.
There is no automatic fallback or multi-instance retry.
The default is **read-only preflight**. `run.py --launch` creates the job;
building/uploading the container is a separate step.

## Current readiness

The following dedicated prerequisites have been provisioned in `play-ghq` as of
2026-11-21:

| Resource | Configuration |
| --- | --- |
| Output/image bucket | `gs://play-ghq-neural-611639590301`, `us-central1` |
| Worker service account | `ghq-training@play-ghq.iam.gserviceaccount.com` |
| Worker permissions | Bucket-only `roles/storage.objectUser`; no project roles |
| Custom VPC | `ghq-training` |
| Subnet | `ghq-training-us-central1`, `10.77.0.0/24`, Private Google Access enabled |

Regional L4 and preemptible L4 quotas each reported limit 16, usage 0. The project
quota response omitted `GPUS_ALL_REGIONS`; this is recorded as unknown, not zero
or unlimited. Compute Engine still enforces the applicable quota at creation.
The first live pilot used a T4 in `us-central1-f` after L4 capacity failures and
an L4 startup failure revealed the missing Docker runtime described below.
No bucket lifecycle policy is configured; uploaded image archives, bundles, and
results remain until explicitly removed or a lifecycle policy is added.

## Existing resources required

- A dedicated Cloud Storage bucket in the selected region, with uniform bucket
  access and public access prevention. Plan how retained artifacts will be removed.
  Store only this experiment's image archives and outputs there.
- A dedicated service account with **no project roles** and
  `roles/storage.objectUser` on that bucket. It can read the image archive and
  create/replace output objects, but has no Compute, project-editor, or deployment
  permissions. Use the attached identity; never download service-account keys.
- A subnet with Private Google Access and working Google API routing and DNS.
  This pilot uses the dedicated subnet above and does not need Cloud NAT.
  The VM has **no external IP**, creates no firewall
  rules, and exposes no training service. The training container has no network.
- Compute, Storage, and IAM APIs enabled; billing enabled; at least one L4 of
  regional GPU quota and sufficient disk quota. G2 does not require a separate
  CPU quota; the T4 option checks the requested N1 CPU count as well as T4 GPU quota. The
  launcher enforces global GPU quota when that metric is reported.
  GPU VMs with predefined runtimes can consume preemptible GPU quota even with
  `STANDARD` provisioning, so reported preemptible L4 quota is checked too.
  Quota does not guarantee
  current GPU capacity. The preflight checks APIs, billing, identity, bucket,
  private connectivity settings, image, and GPU quotas without changing them.

These resources are prerequisites; the launcher does not grant IAM, enable APIs,
create buckets, or edit networks. The launching identity needs permission to create/delete the VM and attach
the training identity (`iam.serviceAccounts.actAs`), and to inspect the prerequisites
and read the output bucket. Organizational policies and inherited/group IAM still
need to permit this plan; preflight inspects direct project and bucket bindings.

## Build and publish an immutable image

Run from the repository root with Docker available. Build only after the local
smoke tests pass. The Dockerfile uses the official PyTorch 2.7.1 / CUDA 12.8 runtime
pinned by digest. Local testing may use a newer PyTorch release within
`requirements-neural.txt`; cloud compatibility must be verified separately.
`training/Dockerfile.dockerignore` limits the build context to the Python training
sources, rules/search, requirements, and Dockerfile; unrelated data and credentials
are excluded.

```sh
docker buildx build --platform linux/amd64 -f training/Dockerfile \
  -t ghq-train:pilot --load .
docker run --rm ghq-train:pilot --help
docker image inspect ghq-train:pilot --format '{{.Id}}'
docker save ghq-train:pilot | gzip > /tmp/ghq-train-pilot.tar.gz
shasum -a 256 /tmp/ghq-train-pilot.tar.gz
```

Record the printed **Docker image ID** and **archive SHA256**. Upload under a unique
path in the dedicated bucket (this upload is a write):

```sh
gcloud storage cp /tmp/ghq-train-pilot.tar.gz \
  gs://play-ghq-neural-611639590301/images/UNIQUE_BUILD/ghq-train-pilot.tar.gz \
  --project=play-ghq
```

The archive is large (several GB), so upload once and reuse it. Copying the archive
from GCS lets the private VM avoid registry credentials and public internet
access. Its container runtime can be installed from the pinned offline bundle
below. The guest verifies the image archive SHA256
before loading it. Docker's containerd image store and classic image store can
expose different immutable IDs for the same archive. The worker first looks up
the requested ID locally. If unavailable, it requires exactly one `Loaded image:`
tag from the verified archive and resolves that tag to its local immutable ID.
It records the requested ID, resolved ID, and loaded tag in `hardware.json`.
Every container run uses `--pull=never`; the fallback never fetches an image.

## Offline container runtime bundle

The selected `common-cu129-ubuntu-2204-nvidia-580-v20261111` image was observed to
include NVIDIA drivers but **no Docker executable**. Supply an offline runtime
bundle for this image. On an internet-connected workstation, build one with the
reusable utility:

```sh
python3 training/gcp/build_runtime.py --output /tmp/ghq-runtime-UNIQUE_BUILD
cat /tmp/ghq-runtime-UNIQUE_BUILD/artifact.json
gcloud storage cp /tmp/ghq-runtime-UNIQUE_BUILD/runtime.zip \
  gs://play-ghq-neural-611639590301/runtime/UNIQUE_BUILD/runtime.zip \
  --project=play-ghq
```

The builder downloads official Linux x86_64 Docker **29.10.0** static binaries and
the four NVIDIA Container Toolkit **1.20.2-1** amd64 packages. Their sizes and
SHA256 values are pinned from the vendors' published checksums. It writes
`downloads/manifest.json`, a deterministic `runtime.zip`, and `artifact.json`
containing the ZIP and manifest hashes. The output directory must be new or
empty; the tool never replaces an existing bundle, installs packages, or uploads
files. Updating versions also requires updating the worker's runtime allowlist.

Pass both `--runtime-bundle-uri` and `--runtime-bundle-sha256` to the launcher,
using the ZIP hash from `artifact.json`. The fresh VM reads the bundle from the
dedicated bucket, verifies its ZIP hash and each manifest entry, and installs
the local packages without contacting public package repositories. The Docker
daemon uses a Unix socket with bridge networking disabled. Existing Docker is
preserved; CUDA is still checked inside the container before training starts.
Installation details are saved in `runtime-install.log` and
`runtime-manifest.json` when installation is needed. This runtime bundle is
separate from the large training image archive and the optional resume bundle.

## Reuse the runtime image with reviewed training sources

Changes to Python training code do not require uploading another multi-GB
PyTorch image. Build a source-only overlay after local tests and a smoke run
against the pinned container pass:

```sh
python3 training/gcp/build_source.py --output /tmp/ghq-source-UNIQUE_BUILD.zip
gcloud storage cp /tmp/ghq-source-UNIQUE_BUILD.zip \
  gs://play-ghq-neural-611639590301/source/UNIQUE_BUILD/source.zip \
  --project=play-ghq
```

The deterministic archive includes `ghqnet/*.py`, `public/engine.py`,
`public/search.py`, and `training/benchmarks/*.manifest.json`. It excludes
training outputs and checkpoint files. Supply `--source-bundle-uri` and
`--source-bundle-sha256` using the printed archive SHA256. The worker verifies
the archive, its path allowlist, and every file hash before mounting it read-only
over `/app`. Dependencies continue to come from the pinned image. A dependency
change requires building and validating a new image.

For the curriculum, pass `--pipeline-config /path/to/experiment.json`. This is a
JSON object mapping pipeline option names (without leading `--`) to scalar
values or boolean flags. It **replaces** the launcher's legacy training flags.
Output, device, and resume paths remain controlled by the worker. Validate the
same options with `python -m ghqnet.pipeline --validate-only --output /tmp/check`
before launch. Set `max-seconds` below the VM deadline to leave time for startup,
checkpointing, and final upload. The checked-in experiment configuration records
the intended game/position targets; a runtime limit can stop before those targets.

## Isolated collector performance experiments

Use `--collector-benchmark-config training/experiments/collector-tuning.json`
instead of `--pipeline-config` to run only `ghqnet.benchmark_collector`. This mode
requires a verified resume bundle and does not call the training pipeline or
modify the checkpoint. It uses the same pinned image, source overlay, private
network, telemetry, output synchronization, and automatic VM deletion as a
training job. The checked-in tuning configuration refers to checkpoint 0004's
replay and historical companion; change these paths when using another bundle.

The JSON accepts the benchmark's reviewed scalar/list options, including
worker counts, inference batch sizes, wait times, protocols, and repeated cases.
`profile-sample` runs one diagnostic case per protocol before the unprofiled
grid. Diagnostic CUDA phase timings explicitly synchronize the device; compare
unprofiled cases when selecting settings. `positions` may reference a replay or
validation companion under `/resume/`; labels are not used or trained on.

Set an explicit `max-seconds`, leaving at least 600 seconds before the VM's
`--max-minutes` deadline for setup and upload (1200 seconds on a 30-minute VM).
An additional worker watchdog stops the benchmark container if its own deadline
does not return. `case-max-seconds` limits one grid configuration; interrupted
cases remain incomplete and cannot be ranked as completed work. The overall
budget includes checkpoint loading, input sampling, and warmup.

Outputs are `collector-benchmark.json`, `collector-benchmark.log`, and the usual
`hardware.json`, `gpu-metrics.csv`, and completion record. A completed short
rollout is not necessarily a terminal game. Compare neural rows, simulations,
active collection time, startup cost, and phase timings; this experiment does
not establish playing strength or a sustained full-game rate.

The completed T4 measurements and selected mixed-collection settings are in
[the training guide](README.md#measured-t4-settings-2026-11-21) and
[collector-throughput-20261121.json](benchmarks/collector-throughput-20261121.json).
For a subsequent resume, use
[curriculum-resume-optimized-2000.json](experiments/curriculum-resume-optimized-2000.json)
with a source bundle containing encoded inference support. The configuration
changes only the four measured performance controls; it does not update a VM
already running from a frozen source bundle.

## Optional resume bundle

Build the exact bundle from a trusted checkpoint with the neural environment:

```sh
.venv-neural/bin/python training/gcp/build_resume.py \
  --checkpoint RUN/checkpoint-0001.pt --output /tmp/ghq-resume-UNIQUE_RUN.zip
```

The builder reads tensor/basic-data checkpoints with `weights_only=True`, checks
every referenced companion hash, and emits a separate ZIP manifest containing
the archive hash. It refuses existing outputs. The manifest stays beside the
ZIP and is not an extra entry inside the worker's resume archive.

To continue a run, create a trusted ZIP containing flat, regular files:
`checkpoint.pt` and the replay buffer named in that checkpoint's metadata, for
example `replay-buffer-0002.jsonl`. Curriculum checkpoints also require their
referenced `validation-buffer-NNNN.jsonl`, `curriculum-NNNN.json`, and every
`history-<16hex>.pt` referenced by that curriculum state. Rename the checkpoint inside
the ZIP to `checkpoint.pt`; keep the replay buffer's original name and contents.
The checkpoint must reference its companions' exact SHA256 values. The worker
rejects unexpected files, directories, symlinks, path traversal, more than 32
entries, and more than 4 GiB of total uncompressed data. The compressed ZIP also
has a 4 GiB limit.

```sh
shasum -a 256 /tmp/ghq-resume.zip
gcloud storage cp /tmp/ghq-resume.zip \
  gs://play-ghq-neural-611639590301/resume/UNIQUE_RUN/ghq-resume.zip \
  --project=play-ghq
```

Add both `--resume-bundle-uri` and `--resume-bundle-sha256` to the launcher command.
The worker verifies the ZIP hash, extracts it into a separate directory, and
mounts it read-only at `/resume`. Training receives
`--resume /resume/checkpoint.pt`; the pipeline independently verifies the replay
buffer hash. Pass the checkpoint's original `--seed`; legacy prototype resumes
use `--teacher-games 0`. Curriculum resumes keep `--curriculum` and the same
holdout configuration; an unfinished teacher bootstrap continues automatically.
Resume restores model, optimizer, iteration, exact replay/validation buffers,
curriculum counters, and historical opponents; new
artifacts go under the new run prefix rather than overwriting the inputs.

## Review the plan, then launch

Substitute real values; the command below performs **reads only**:

```sh
python3 training/gcp/run.py \
  --project play-ghq --zone us-central1-a \
  --bucket play-ghq-neural-611639590301 --subnet ghq-training-us-central1 \
  --service-account ghq-training@play-ghq.iam.gserviceaccount.com \
  --image-uri gs://play-ghq-neural-611639590301/images/UNIQUE_BUILD/ghq-train-pilot.tar.gz \
  --image-sha256 ARCHIVE_SHA256 \
  --image-id sha256:DOCKER_IMAGE_ID_HEX \
  --runtime-bundle-uri gs://play-ghq-neural-611639590301/runtime/UNIQUE_BUILD/runtime.zip \
  --runtime-bundle-sha256 RUNTIME_ZIP_SHA256 \
  --max-minutes 60 --iterations 2 --games 16 --simulations 64 \
  --max-turns 100 --epochs 2 --teacher-games 4 --teacher-ms 600 \
  --seed 20261120 --threads 2 --batch-size 64
```

For the T4 configuration, add `--gpu t4` and select a zone with T4 support
(the first pilot used `us-central1-f`). This is an explicit hardware choice;
available quota does not guarantee capacity for either GPU type.

For a resumed run, replace `--teacher-games 4` with `--teacher-games 0`, use the
checkpoint's seed, and append:

```sh
  --resume-bundle-uri gs://play-ghq-neural-611639590301/resume/UNIQUE_RUN/ghq-resume.zip \
  --resume-bundle-sha256 RESUME_ZIP_SHA256
```

To execute the printed plan, repeat the exact command with `--launch`. Keep the
launcher running. It creates the VM, waits for the uploaded
completion record, and deletes the unique VM and its boot disk on completion or
interruption. Training runs this command in the container:

```sh
python -m ghqnet.pipeline --output /output/run --device cuda \
  --iterations 2 --games 16 --simulations 64 --max-turns 100 --epochs 2 \
  --teacher-games 4 --teacher-ms 600 --seed 20261120 --threads 2 --batch-size 64
```

Before training, the worker runs `python -m ghqnet.benchmark --device cuda
--repeats 10 --threads 2 --output /output/benchmark.json`. A resumed run benchmarks
the resumed checkpoint. Benchmark output and worker logs are outside `/output/run`,
so the pipeline still receives a fresh output directory. `hardware.json` records
GPU/driver, host, PyTorch/CUDA, and image details. During training,
`gpu-metrics.csv` records GPU utilization, memory, power, and temperature every
five seconds. These measurements are needed to assess whether GPU acceleration
helps the actual workload.

The default image family is Google's current Ubuntu 22.04 CUDA 12.9 / NVIDIA 580
Deep Learning VM family. Preflight resolves it to an exact image name and launch
uses that name. On 2026-11-20 this resolved to
`common-cu129-ubuntu-2204-nvidia-580-v20261111`. That image needs the offline
runtime bundle described above. Startup checks NVIDIA drivers, Docker, and CUDA
access inside the container, and fails rather than falling back to CPU or
fetching unreviewed packages. Future image releases may have a different
container runtime configuration.

## Runtime bound, outputs, and cleanup

The VM defaults to a platform-enforced `--max-run-duration=60m` with
`--instance-termination-action=DELETE`, a 100 GB auto-deleted boot disk, and no
automatic restart. The pilot script permits 15–240 minutes, with no automatic
retries or additional instances. If the launcher disconnects, the GCE runtime
deadline remains the backstop. Google documents that termination can begin up to
30 seconds after the requested duration. On successful completion the launcher deletes the
VM immediately, rather than paying for the rest of the time limit.

Completed files are uploaded every 60 seconds to the printed
`gs://BUCKET/runs/INSTANCE/` prefix. Checkpoints must be written atomically by the
pipeline; `.tmp` files are excluded. A final upload runs before the completion
record is published. The shutdown hook also attempts a final upload. A hard
termination or storage outage can lose progress since the last successful sync;
unfinished games are not magically checkpointed. GCS artifacts remain after VM
deletion. Inspect `completion.json`, `benchmark.json`, `hardware.json`,
`gpu-metrics.csv`, `training.log`, and pipeline metrics before
using any model. `upload-failed.json` means the final upload did not succeed.

The duration is a **compute runtime bound, not a dollar budget**. On 2026-11-21,
the published `us-central1` G2 rate was $0.706832276/hour and 100 GiB of balanced
disk was approximately $0.0136986/hour: approximately **$0.72 for one hour**, or
about **$0.73 allowing the documented termination delay**. This excludes GCS
storage/operations, transfer, and taxes. Check prices before subsequent runs.
The T4 configuration's published rate was approximately $0.55/hour including
four N1 CPUs and the same disk, or $0.51 for its 55-minute runtime bound.
For concurrent collection, sixteen N1 CPUs cost $0.759996/hour and the T4 costs
$0.35/hour. Including the same disk gives approximately $1.124/hour, or $4.50
for a four-hour bound, before storage, transfer, and taxes. The machine is deleted
as soon as its job completes, even when the configured bound is longer.
Retained image archives/checkpoints continue to incur storage charges. This bucket
has no lifecycle policy: explicitly remove obsolete prefixes or configure a
policy when the retention requirements are known. Pipeline artifacts are under `runs/INSTANCE/run/`; worker logs and
job metadata are directly under `runs/INSTANCE/`. The pipeline receives a fresh
subdirectory so worker files cannot trigger its nonempty-output protection.
Worker uploads disable gcloud's parallel composite upload feature, which avoids
bucket-metadata permission checks and keeps the identity limited to Object User.

If the create command times out or is interrupted, cleanup first describes the
named VM and verifies this run's metadata and training label before deleting it.
If ownership cannot be confirmed, it leaves the resource untouched and reports
the uncertain result; the requested platform DELETE deadline remains the backstop.
During monitoring, transient cloud errors retry within the existing deadline;
only confirmed resource absence ends the wait early. Cleanup makes at most three
bounded deletion attempts and checks that the VM is absent. Unconfirmed cleanup
returns failure and prints the exact command to inspect or retry.
A prematurely stopped VM clears its runtime deadline and can retain a disk if the launcher is gone;
verify the labeled resources after interrupted runs. The cleanup command for the
specific printed instance is:

```sh
gcloud compute instances delete EXACT_PRINTED_INSTANCE \
  --project=play-ghq --zone=us-central1-a
gcloud compute instances list --project=play-ghq \
  --filter='labels.managed-by=ghq-training'
```

## First GPU result (2026-11-21)

Run `ghq-train-20261121-133423-0ad864` used one T4 on `n1-standard-4` in
`us-central1-f`, with a 55-minute DELETE deadline. It resumed the local
`prototype-20261120/checkpoint-0002.pt`, preserving its optimizer and exact replay
buffer. Configuration: four additional iterations, 16 games each, 64 simulations
per action, a 150-turn cap, four epochs per update, batch size 64, two CPU threads,
seed 20261120, and no additional teacher games. The unchanged network has six
96-channel residual blocks and 1,132,012 parameters.

The worker completed successfully in 20m13s including setup. Collection took
941.26s and four training updates took 38.45s. It added 3,815 positions, leaving
5,024 replay positions with 4,633 genuine terminal outcome labels. The 64 games
produced eight HQ wins, 55 double-skip draws, and one repetition cutoff. That
cutoff has no value labels. Artifacts were fully uploaded and downloaded; the
launcher deleted the VM, and independent listings confirmed zero remaining
training VMs and zero `ghq-train-` boot disks. The bucket, service account, and
private network remain for reuse. Retained storage still incurs charges.

| Collection batch | Generating checkpoint | HQ wins | Double-skip draws | Cutoffs | Median turns |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | 2 | 7 | 9 | 0 | 27 |
| 4 | 3 | 1 | 14 | 1 | 15.5 |
| 5 | 4 | 0 | 16 | 0 | 13.5 |
| 6 | 5 | 0 | 16 | 0 | 14 |

The next checkpoint is trained after each collection batch: the last row is
checkpoint 5 self-play, not a measurement of checkpoint 6 against itself.
Shorter games increased games/second without demonstrating better play. Replay
analysis also found more start-of-turn Skip choices, though this was not
monotonically increasing and state variety remained high.

The direct-network benchmark used preencoded, device-resident initial positions
with all 64 legal actions. It excludes rules, encoding, transfers, and warmup;
these are throughput opportunities, not end-to-end search speedups.

| Measurement | VM CPU | T4 |
| --- | ---: | ---: |
| Batch 1 inference, positions/s | 265 | 417 |
| Batch 16 inference, positions/s | 637 | 6,036 |
| Batch 64 inference, positions/s | 818 | 14,252 |
| Synthetic batch 64 training step, mean ms | 250.0 | 19.8 |

Batch 64 gives about 34.2 times the T4's batch 1 throughput. Actual serial
self-play achieved 4.05 actions/s and 259.4 simulations/s; evaluator calls used
84.6% of search time. GPU utilization averaged 35.8% over 199 five-second
samples, with a 38% median. The matched first game took **5.75s on the local Mac
CPU versus 14.45s on the T4 VM**, with identical action sequence and final state.
That is a platform comparison: CPUs, operating systems, and PyTorch versions
also differ. It does not isolate the effect of GPU hardware.

All four new checkpoints were tested locally on CPU at the baseline's 2.4s
per-turn budget, seed 4173, and two threads. Their tactical scores were
**4/10, 4/10, 4/10, and 6/10**, compared with the original 4/10. The final two
extra passes came from simply playing `skip` in both trapped-paradrop positions.
Heavy-versus-armored selection and large-gun safety still failed in both colors.
All evaluated plans were legal and complete.

Final checkpoint 6 then lost both paired matches against `ghq-cpu-4`, with no
draws or unfinished games, at 2.4s per turn (seed 39173, two opening turns, an
80-turn cap, local CPU). The original checkpoint also lost both games under
these settings; rules and CPU source hashes match. Two games do not establish an
Elo or statistical equivalence. This pilot provides no evidence for promotion;
the website still uses `ghq-cpu-4`.

The next experiment should combine concurrent games through a shared inference
batcher and mix fixed CPU/historical-checkpoint opponents with self-play, backed
by broader teacher examples. Preserve real draw labels; silently punishing
legal draws would change the learning objective. Judge the changes using the
same held-out tactical gates plus a larger paired arena, not training loss or
game-completion rate.

Reproducibility records under `.data/neural/gpu-pilot-20261121/`:

- `results/summary.json`: hardware, benchmarks, training, GPU samples, and all evaluations.
- `results/pilot-measurements.json`: per-iteration replay and outcome analysis.
- `remote/ghq-train-20261121-133423-0ad864/`: complete worker outputs and local evaluation reports.
- `artifacts.json`, `attempts.json`, and `cleanup.json`: immutable input identities, launch attempts, and cleanup verification.
- `local-cpu-game.json`: matched local CPU collection measurement.

Cloud artifacts remain at
`gs://play-ghq-neural-611639590301/runs/ghq-train-20261121-133423-0ad864/`.
The final checkpoint SHA256 is
`8cd26368c9e03db8b7f415a82277aed303653494af0040f3c08b35e84209d3b3`.

## Larger curriculum experiment (2026-11-21)

The run was `ghq-train-20261121-144021-275981` in `us-central1-f`:
one T4 on `n1-standard-16`, with a four-hour DELETE deadline. Its configuration
is [experiments/curriculum-2000.json](experiments/curriculum-2000.json).
It starts a fresh network, collects up to 50,000 teacher training positions
(maximum 2,000 teacher games), then targets 2,000 additional mixed games with
256 simulations per action. Teacher search uses a 3.2s budget, depth cap four,
and beam width twelve. Actual completed depth can be lower than the cap.

The curriculum uses sixteen game workers, shared inference batches, a 250,000
position replay cap, retained teacher/tactical quotas, independent validation,
and the frozen evaluation exclusions. The network remains six blocks with
96 channels. The internal 3.5-hour deadline leaves time for startup and final
checkpoint upload; the configured game counts are targets, not reported results.
The exact source bundle SHA256 is
`6b08ca26aaf46979e14a86d46e1c64b22b584584a700083beab52218996ce722`.

Before launch, the pinned Linux/PyTorch image passed a real teacher/collection/
training smoke, configuration validation, and sixteen spawned-worker collection.
A matched local CPU collector benchmark took 16.18s with one worker and 12.42s
with four, with identical moves (1.30x end-to-end speedup). This does not establish
GPU collection throughput or playing strength. Experiment metadata and local
checks are in `.data/neural/curriculum-20261121/`; cloud outputs use
`gs://play-ghq-neural-611639590301/runs/ghq-train-20261121-144021-275981/`.
No candidate has been promoted to the website.

The old pilot's final checkpoint also completed a larger baseline screen:
**0 wins, 0 draws, 20 losses**, all HQ captures, across ten paired openings at
2.4s per turn on the local CPU. No games hit the 100-turn limit or repetition
cutoff. The report and frozen openings are under
`.data/neural/curriculum-20261121/evaluation/`. Reuse those openings for a
candidate screen; use a separate, larger match set for a promotion decision.
This is a baseline result, not a result for the new training run.

The first fresh checkpoint, `checkpoint-0000.pt`, completed one optimization
pass after 100 teacher games: 15,867 training rows and 2,712 validation rows.
Its data hashes and partition registries were verified. Training contains
14,728 unique canonical states across 85 families; validation contains 2,465
states across 14 families, with no cross-partition or reserved-suite overlap.
These are early bootstrap measurements, before the 50,000-position target:

| Same validation positions | Old pilot checkpoint 6 | Fresh checkpoint 0 |
| --- | ---: | ---: |
| Teacher top-1 agreement, non-forced | 12.3% | 23.3% |
| Teacher top-5 agreement, non-forced | 30.3% | 52.1% |
| WDL log loss | 4.731 | 0.596 |
| Fresh-turn policy argmax is Skip | 13.9% | 0.0% |

The policy comparison uses 2,592 non-forced positions; the Skip comparison uses
875 fresh-turn choices. WDL labels in this validation batch contain wins and
losses, with no draws. These correlated positions measure imitation and outcome
prediction under teacher play, not playing strength or general calibration.
The fresh checkpoint SHA256 is
`946f6e574de8032554d4789d90fcd6a54ceeaad38d75ff11c905de2b8d01a123`.

### Completed run and check-in

The worker finished at 17:10 UTC with exit code zero and a successful final
upload. The VM and attached disk were independently verified deleted. This was
an orderly shutdown of an incomplete curriculum: 364 teacher games supplied
61,768 retained training rows, but only 27 of the intended 2,000 mixed games
completed. Final checkpoint 0004 trained for 502 optimizer steps over 64,182
rows, including 2,414 mixed-game rows. All eight files needed to resume are
downloaded and hash-verified under the experiment's `completed/` directory.

Mixed collection stopped after 1,803.7 seconds; the entire pipeline used 8,738
of its 12,600-second budget. This strongly implicates the 1,800-second per-game
watchdog. The collector cancels all outstanding tasks on any collection timeout,
and the pipeline discards the exception text and reports `curriculum-deadline`.
The exact trigger/game was therefore not recorded. Fix timeout scope and
diagnostics before resuming: one slow game must not cancel an otherwise healthy
batch. Of 200 scheduled games, 16 were in flight and 157 had not started when
the batch stopped. Actual collection throughput was 672 evaluated rows/second;
historical models fragmented inference into batches averaging 1.21–1.57 rows.
These measurements are not the resident-input network-only GPU benchmark.

On the same final validation buffer (7,635 rows, 7,275 non-forced), policy top-1
agreement improved from 23.6% at checkpoint 0000 to 34.2% at checkpoint 0004;
top-5 improved from 52.1% to 66.3%. WDL log loss changed from 0.584 to 0.555.
These are imitation/outcome-prediction metrics, not match strength. The final
checkpoint still passes 6/10 original tactical regression color runs, failing
the heavy-over-armored choice and large-gun safety cases. Final training and
validation registries have no canonical-state or family overlap; all reserved
tactical states and original regression anchors are empirically absent.

The early checkpoint 0000 completed its paired screen at 0 wins, 0 draws, and
20 losses against CPU4. Checkpoint 0004 has not yet had that match screen, and
no neural candidate has been promoted. The final checkpoint SHA256 is
`fd6a4fcd68305438a65ce709c064b22a546dac93700b1f53084e0b07868e48bb`.
Full check-in evidence is in
`.data/neural/curriculum-20261121/completed/check-in.json`.

### Corrected resume

The resume configuration is
[experiments/curriculum-resume-2000.json](experiments/curriculum-resume-2000.json).
Run `ghq-train-20261121-175146-6b82c4` was launched in `us-central1-f` on
one T4 with sixteen vCPUs. The platform DELETE bound and auto-delete disk were
verified after creation.
It uses checkpoint 0004 and its exact eight-file resume bundle, preserving
61,768 teacher positions and 27 completed mixed games. It skips teacher bootstrap,
collects 32 games per update with sixteen workers, uses one historical model per
collection, and preserves the opponent/color schedule across smaller chunks and
resumes. The cumulative target remains 2,000 mixed games. The 128-update cap
allows progress toward that target; the 3.5-hour pipeline and four-hour VM DELETE
bounds still take precedence.

Individual game watchdogs now retire only the overdue game and replace its
worker; healthy games continue. The watchdog is one hour, and its event records
the game ID, elapsed time and inference activity. Unfinished watchdog games add
no replay rows or synthetic draws. Global deadline, startup timeout, request
timeout and evaluator failure have distinct saved reasons. Status snapshots
preserve pipeline status even when the worker exits zero after a clean upload.

Before launch, 181 neural/cloud tests and 104 subtests passed. A pinned Linux
smoke with sixteen real spawned workers resumed the exact checkpoint, routed
current/historical inference, preserved every existing replay row and partition,
and saved/reloaded without changing the model or optimizer (training was
explicitly disabled for this smoke). A separate real-worker integration test
exercised repeated individual timeouts, continued updates and checkpoint resume.
Smoke outputs are test artifacts and must never be used as production resumes.

The reviewed source SHA256 is
`91dd9c281aa757081eb3984138dc0f25ba8ddb1546bfce7e90f78e99d916c9ca`;
the resume ZIP SHA256 is
`514156bb9dbfcb20537040a6cb9b7274136e3443b41887edfa7d5a786fa23782`.
The reused runtime and container are the same pinned versions as the prior run.
Launch details and verification artifacts are recorded in
`.data/neural/curriculum-resume-20261121/experiment.json`.
The first uploaded collection-start record confirmed CUDA and inherited counters
of 364 teacher games, 61,768 teacher positions and 27 mixed games. Bootstrap was
skipped. The first 32-game batch schedules thirteen self-play, thirteen CPU and
six historical games, using exactly two evaluator model identities.

Read a progress snapshot without changing the job:

```sh
python3 training/gcp/status.py --project play-ghq \
  --run-uri gs://play-ghq-neural-611639590301/runs/ghq-train-20261121-175146-6b82c4 \
  --output .data/neural/curriculum-resume-20261121/status
```

Snapshots distinguish raw collected samples from the training-position counter
saved at a checkpoint. Missing final summaries are normal while a job is active;
an upload timestamp alone does not prove VM liveness or successful completion.

## Bounded replay ablation

`--replay-ablation-config training/experiments/replay-ablation-20261121.json`
selects an offline diagnostic instead of the training pipeline or collector
benchmark. Use the same verified resume, source, runtime and image arguments as
training, with `--max-minutes 30`. The reviewed configuration compares learning
rates `0.001` and `0.0003` for one epoch on identical checkpoint/replay inputs,
with a 1,200-second workload budget. It does not collect new games.

The source bundle must contain `ghqnet/ablate_replay.py`. The worker mounts inputs
read-only, invokes that module explicitly, and writes `replay-ablation.log` plus
`replay-ablation/{events.jsonl,summary.json,report.json,arm-*.pt}`. The workload
watchdog stops its own container; the existing upload and VM deletion paths still
apply. Launcher preflight remains read-only unless `--launch` is supplied.

Inspect `report.json` for `complete`, `comparison_eligible`, equal optimizer-step
counts and unchanged input hashes. A zero process exit code alone does not make an
interrupted diagnostic usable. Arm checkpoints intentionally lack curriculum
resume references; neither validation gains nor this small experiment authorize
production promotion.

## Verification and sources

```sh
python3 -m unittest discover -s training/gcp -p 'test_*.py' -v
python3 -m py_compile training/gcp/build_runtime.py
python3 training/gcp/build_runtime.py --help
bash -n training/gcp/startup.sh training/gcp/shutdown.sh
```

The offline tests exercise the read-only default, explicit launch boundary,
private-network/runtime flags, quota handling, resume ZIP validation, Docker ID
portability, read-only mounts, and completion/interruption cleanup. They do not
replace a real GPU smoke run.

After the pilot and launcher reliability fixes, the complete suite passed
184 tests and 75 subtests; the cloud-only suite passed 41 tests. The existing
`tests/alpha_zero/test_cases.py` collection warning remains.

- [Google: current Deep Learning VM images](https://docs.cloud.google.com/deep-learning-vm/docs/images)
- [Google: limit VM runtime](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime)
- [Google: create-instance flags](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/create)
- [Google: Private Google Access](https://docs.cloud.google.com/vpc/docs/private-google-access)
- [Google: Cloud Storage IAM roles](https://docs.cloud.google.com/storage/docs/access-control/iam-roles)
- [Google: GPU and disk pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized)
- [Google: N1 VM pricing](https://cloud.google.com/compute/vm-instance-pricing)
- [Google: attached GPU pricing](https://cloud.google.com/compute/gpus-pricing)
- [Google: disk pricing](https://cloud.google.com/compute/disks-image-pricing)
- [Google: GPU quota behavior](https://docs.cloud.google.com/compute/resource-usage)
- [Google: parallel composite upload permissions](https://docs.cloud.google.com/storage/docs/parallel-composite-uploads)
- [PyTorch: pinned container](https://hub.docker.com/layers/pytorch/pytorch/2.7.1-cuda12.8-cudnn9-runtime/images/sha256-c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2)
- [Docker: pinned static binary checksum](https://download.docker.com/linux/static/stable/x86_64/docker-29.10.0.tgz.sha256)
- [NVIDIA: official package checksums](https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages)
