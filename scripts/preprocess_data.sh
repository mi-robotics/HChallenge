#!/usr/bin/env bash
# Raw recordings -> training data (data/README.md), for every episode in data/mug_pnp_dataset and data/mug_pnp_failed:
#   1. real2sim: MediaPipe hand -> table frame -> cleaned path -> retarget -> sim scene -> replay
#   2. sim.npz: the replay again, saving the robot's actions, state and camera images (kept + replayed failed)
#   3. action flow: SAM 2 mug masks -> CoTracker points -> 3D flow -> flow_sim.npz + video_sim.mp4 on the robot's clock
#   bash scripts/preprocess_data.sh
# Finished stages are skipped, so it can be re-run after an interruption. The splits are data/splits.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n humanoidschallenge python"

$PY replay/run_dataset.py data/mug_pnp_dataset
$PY replay/run_dataset.py data/mug_pnp_failed
$PY replay/record_sim.py
$PY replay/record_sim.py $($PY -c "from dataset import episode_dirs; print(*episode_dirs('failed'))")
$PY flow/run_flows.py
