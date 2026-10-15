#!/usr/bin/env bash
# The dataset from Hugging Face into data/ (165 episodes, 19 GB):
#   data/mug_pnp_dataset/<episode>/ (100 kept) and data/mug_pnp_failed/<episode>/ (65 failed), each with
#   raw recording   color.mp4, depth.mkv, frames.csv, meta.json
#   processed       sim.npz, flow_sim.npz, video_sim.mp4, scene.json (made by scripts/preprocess_data.sh)
#   bash scripts/download_data.sh            raw + processed: enough to train and evaluate
#   bash scripts/download_data.sh --raw      raw only, then run scripts/preprocess_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ARGS=(--exclude README.md --exclude .gitattributes)
if [ "${1:-}" = "--raw" ]; then
  ARGS=(--include "*/color.mp4" --include "*/depth.mkv" --include "*/frames.csv" --include "*/meta.json")
fi
conda run --no-capture-output -n humanoidschallenge \
  hf download mcarroll101/HChallenge-mug-pnp --repo-type dataset --local-dir data "${ARGS[@]}"
