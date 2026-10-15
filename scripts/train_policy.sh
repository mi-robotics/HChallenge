#!/usr/bin/env bash
# The flow-conditioned policy (wrist camera + robot state + hand and mug flow), then its evaluation in sim with the
# ground truth flow: every split, 5 seeds x 5 attempts per episode.
#   bash scripts/train_policy.sh        (training ~30 min, evaluation ~35 min on an RTX 5090)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n humanoidschallenge python"
OUT=outputs/s1/ablation/full_wrist_state_no_img_pos

$PY policy/train_s1.py --cameras eye_in_hand --flow full --img-aug --obs-pos-embed --no-img-pos-embed --out $OUT
$PY policy/eval_s1.py $OUT/ckpt_20000.pt --attempts 5 --workers 8
