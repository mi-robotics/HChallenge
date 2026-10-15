#!/usr/bin/env bash
# End-to-end evaluation: the WAM (v1) plans each episode from its first real frame, then the policy follows the plan
# in sim with the replay's pauses inserted. Run scripts/train_policy.sh and scripts/train_wam.sh first.
#   val + test: 5 plans (seeds 0-4) x 5 policy seeds; train + failed: the seed 0 plans x 5 policy seeds
#   bash scripts/eval_wam.sh            (planning ~2.5 h, evaluation ~20 min on an RTX 5090)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="conda run --no-capture-output -n humanoidschallenge python"
WAM=outputs/wam/v1/lora_20000.pt
POLICY=outputs/s1/ablation/full_wrist_state_no_img_pos/ckpt_20000.pt

$PY wam/plan.py $WAM --out outputs/wam/v1/e2e --splits train val test failed
for k in 1 2 3 4; do
  $PY wam/plan.py $WAM --out outputs/wam/v1/e2e_s$k --splits val test --seed $k
done
for d in e2e e2e_s1 e2e_s2 e2e_s3 e2e_s4; do
  $PY policy/eval_s1.py $POLICY --plan outputs/wam/v1/$d --pauses --splits val test --workers 8
done
$PY policy/eval_s1.py $POLICY --plan outputs/wam/v1/e2e --pauses --splits train failed --workers 8
$PY wam/aggregate_e2e.py
