#!/usr/bin/env bash
set -euo pipefail

# Train all experiment configs sequentially.
# Usage: (cd models && CUDA_VISIBLE_DEVICES=0,1 bash train_all.sh)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for exp in "${DIR}"/experiments/*.json; do
  echo "=== Training ${exp} ==="
  PYTHONPATH="${DIR}/.." python -m models.cli --experiment "${exp}" --mode train
done
