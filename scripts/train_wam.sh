#!/usr/bin/env bash
# The world action model (v1): Cosmos-Predict2-2B fine-tuned with LoRA to predict 1.6 s of video + flow.
#   bash scripts/train_wam.sh           (~18 h on an RTX 5090; needs the gated Cosmos weights, see setup.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n humanoidschallenge python wam/train.py --out outputs/wam/v1
