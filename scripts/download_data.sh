#!/usr/bin/env bash
# The raw recordings (165 episodes, 14 GB) from Hugging Face into data/:
#   data/mug_pnp_dataset/<episode>/ and data/mug_pnp_failed/<episode>/: color.mp4, depth.mkv, frames.csv, meta.json
#   bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n humanoidschallenge \
  hf download mcarroll101/HChallenge-mug-pnp --repo-type dataset --local-dir data --exclude README.md
