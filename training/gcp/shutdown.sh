#!/usr/bin/env bash
set -euo pipefail
if [ -f /opt/ghq-job/worker.py ]; then
  exec python3 /opt/ghq-job/worker.py --shutdown
fi
