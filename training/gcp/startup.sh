#!/usr/bin/env bash
set -euo pipefail
# All executable code is supplied as reviewed VM metadata; never eval metadata.
mkdir -p /opt/ghq-job
curl --fail --silent --show-error --retry 5 \
  -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/ghq-runner \
  -o /opt/ghq-job/worker.py
exec python3 /opt/ghq-job/worker.py
