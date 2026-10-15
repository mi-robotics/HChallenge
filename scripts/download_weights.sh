#!/usr/bin/env bash
# Our trained weights from Hugging Face into outputs/ (750 MB):
#   outputs/s1/ablation/full_wrist_state_no_img_pos/ckpt_20000.pt   flow-conditioned policy (wrist camera)
#   outputs/wam/v1/lora_20000.pt, config.json                        world action model v1 (LoRA)
#   outputs/wam/prompt_t5.pt                                         the prompt's T5 embedding
#   bash scripts/download_weights.sh
# The WAM also needs nvidia/Cosmos-Predict2-2B-Video2World (gated, see scripts/setup.sh); it is downloaded on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
conda run --no-capture-output -n humanoidschallenge \
  hf download mcarroll101/HChallenge-models --local-dir outputs --exclude README.md --exclude .gitattributes
