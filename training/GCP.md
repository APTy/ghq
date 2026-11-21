# Bounded GHQ GPU pilot

Start with local correctness and throughput measurements in [README.md](README.md).
This job is a pilot, not a trained production bot or a claim of playing strength.
It runs one `g2-standard-4` VM (one NVIDIA L4), using project `play-ghq` explicitly.
The default is **read-only preflight**. `run.py --launch` creates the job;
building/uploading the container is a separate step.

## Current readiness

Read-only checks on 2026-11-20 found an active project and unused `us-central1`
quota for 16 L4 GPUs and 8 T4 GPUs, but no dedicated training bucket or training identity.
The existing US Central subnets have Private Google Access disabled. Do not use
the Terraform state bucket or deployment identities for training. No cloud job
has been launched and the container/GPU path has not been tested on a VM.

## Existing resources required

- A dedicated Cloud Storage bucket in the selected region, with uniform bucket
  access, public access prevention, and an appropriate retention/lifecycle policy.
  Store only this experiment's image archives and outputs there.
- A dedicated service account with **no project roles** and
  `roles/storage.objectUser` on that bucket. It can read the image archive and
  create/replace output objects, but has no Compute, project-editor, or deployment
  permissions. Use the attached identity; never download service-account keys.
- An existing subnet with Private Google Access and working Google API routing
  and DNS. Enabling Private Google Access on the selected existing subnet avoids
  creating NAT or a new network. The VM has **no external IP**, creates no firewall
  rules, and exposes no training service. The training container has no network.
- Compute, Storage, and IAM APIs enabled; billing enabled; at least one L4 of
  regional/global GPU quota plus machine/CPU/disk quota. Quota does not guarantee
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

Record the printed **Docker image ID** and **archive SHA256**. Choose an existing
dedicated bucket, then upload under a unique path (this upload is a write):

```sh
gcloud storage cp /tmp/ghq-train-pilot.tar.gz \
  gs://YOUR_TRAINING_BUCKET/images/UNIQUE_BUILD/ghq-train-pilot.tar.gz \
  --project=play-ghq
```

The archive is large (several GB), so upload once and reuse it. Copying the archive
from GCS lets the private VM avoid registry credentials, public internet access,
and package installation during startup. The guest verifies the archive SHA256
and runs the loaded image by immutable Docker ID.

## Review the plan, then launch

Substitute real values; the command below performs **reads only**:

```sh
python3 training/gcp/run.py \
  --project play-ghq --zone us-central1-a \
  --bucket YOUR_TRAINING_BUCKET --subnet YOUR_EXISTING_SUBNET \
  --service-account ghq-training@play-ghq.iam.gserviceaccount.com \
  --image-uri gs://YOUR_TRAINING_BUCKET/images/UNIQUE_BUILD/ghq-train-pilot.tar.gz \
  --image-sha256 ARCHIVE_SHA256 \
  --image-id sha256:DOCKER_IMAGE_ID_HEX \
  --max-minutes 120 --iterations 2 --games 16 --simulations 64 \
  --max-turns 100 --epochs 2
```

To execute the printed plan, repeat the exact command with `--launch`. Keep the
launcher running. It creates the VM, waits for the uploaded
completion record, and deletes the unique VM and its boot disk on completion or
interruption. Training runs this command in the container:

```sh
python -m ghqnet.pipeline --output /output/run --device cuda \
  --iterations 2 --games 16 --simulations 64 --max-turns 100 --epochs 2
```

The default image family is Google's current Ubuntu 22.04 CUDA 12.9 / NVIDIA 580
Deep Learning VM family. Preflight resolves it to an exact image name and launch
uses that name. On 2026-11-20 this resolved to
`common-cu129-ubuntu-2204-nvidia-580-v20261111`. Startup checks NVIDIA drivers,
Docker, and CUDA access inside the container, and fails rather than falling back
to CPU or installing unreviewed packages. This is not a guarantee that every
future image family release will retain the same container runtime configuration.

## Runtime bound, outputs, and cleanup

The VM has a platform-enforced `--max-run-duration=120m` with
`--instance-termination-action=DELETE`, a 100 GB auto-deleted boot disk, and no
automatic restart. The pilot script permits 15–240 minutes, with no automatic
retries or additional instances. If the launcher disconnects, the GCE runtime
deadline remains the backstop. On successful completion the launcher deletes the
VM immediately, rather than paying for the rest of the time limit.

Completed files are uploaded every 60 seconds to the printed
`gs://BUCKET/runs/INSTANCE/` prefix. Checkpoints must be written atomically by the
pipeline; `.tmp` files are excluded. A final upload runs before the completion
record is published. The shutdown hook also attempts a final upload. A hard
termination or storage outage can lose progress since the last successful sync;
unfinished games are not magically checkpointed. GCS artifacts remain after VM
deletion. Inspect `completion.json`, `training.log`, and pipeline metrics before
using any model. `upload-failed.json` means the final upload did not succeed.

The duration is a **compute runtime bound, not a dollar budget**. Check the current
regional G2 price plus 100 GB disk usage, GCS storage/operations, and any transfer
charges before launch. Retained image archives/checkpoints continue to incur
storage charges; use the bucket lifecycle policy or explicitly remove obsolete
run prefixes. Pipeline artifacts are under `runs/INSTANCE/run/`; worker logs and
job metadata are directly under `runs/INSTANCE/`. The pipeline receives a fresh
subdirectory so worker files cannot trigger its nonempty-output protection.

If the create command times out or is interrupted, cleanup first describes the
named VM and verifies this run's metadata and training label before deleting it.
If ownership cannot be confirmed, it leaves the resource untouched and reports
the uncertain result; the requested platform DELETE deadline remains the backstop.
A prematurely stopped VM can retain a disk if the launcher is gone;
verify the labeled resources after interrupted runs. The cleanup command for the
specific printed instance is:

```sh
gcloud compute instances delete EXACT_PRINTED_INSTANCE \
  --project=play-ghq --zone=us-central1-a
gcloud compute instances list --project=play-ghq \
  --filter='labels.managed-by=ghq-training'
```

## Verification and sources

```sh
python3 -m unittest discover -s training/gcp -p 'test_*.py' -v
bash -n training/gcp/startup.sh training/gcp/shutdown.sh
```

The offline tests exercise the read-only default, explicit launch boundary,
private-network/runtime flags, and completion/interruption cleanup. They do not
replace a real GPU smoke run.

- [Google: current Deep Learning VM images](https://docs.cloud.google.com/deep-learning-vm/docs/images)
- [Google: limit VM runtime](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime)
- [Google: create-instance flags](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/create)
- [Google: Private Google Access](https://docs.cloud.google.com/vpc/docs/private-google-access)
- [Google: Cloud Storage IAM roles](https://docs.cloud.google.com/storage/docs/access-control/iam-roles)
- [Google: GPU pricing](https://cloud.google.com/compute/gpus-pricing)
- [PyTorch: pinned container](https://hub.docker.com/layers/pytorch/pytorch/2.7.1-cuda12.8-cudnn9-runtime/images/sha256-c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2)
