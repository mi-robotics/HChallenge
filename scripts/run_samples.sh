#!/usr/bin/env bash
# End-to-end samples with our weights: the WAM (v1) plans each test episode from its first real frame, the policy
# follows the plan in sim (with pauses), and each run is rendered as on the website's WAM page:
#   my recording + ground truth flow | generated video + planned flow | robot wrist camera | robot third person
# Needs scripts/setup.sh, scripts/download_data.sh and scripts/download_weights.sh (or your own trained weights).
#   bash scripts/run_samples.sh [seed]     -> outputs/wam/v1/samples_s<seed>/videos_pauses/<episode>.mp4 (~10 min)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n humanoidschallenge python"
SEED=${1:-0}
OUT=outputs/wam/v1/samples_s$SEED

$PY wam/plan.py outputs/wam/v1/lora_20000.pt --out $OUT --splits test --seed $SEED
$PY policy/eval_s1.py outputs/s1/ablation/full_wrist_state_no_img_pos/ckpt_20000.pt --plan $OUT --pauses \
  --splits test --seeds $SEED --workers 4 --label "WAM v1 + pauses"
echo "videos: $OUT/videos_pauses/"
