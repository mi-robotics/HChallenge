#!/usr/bin/env bash
# Conda env, pip packages, third-party code at the commits we used, and model checkpoints.
#   bash scripts/setup.sh
# Needs conda, git, wget, ffmpeg and an NVIDIA GPU with CUDA 12.8 drivers (we used an RTX 5090).
# The WAM base model (nvidia/Cosmos-Predict2-2B-Video2World) is gated: accept its licence on Hugging Face and run
# `hf auth login` before scripts/train_wam.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-humanoidschallenge}
RUN="conda run --no-capture-output -n $ENV"

conda create -y -n $ENV python=3.10
$RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$RUN pip install -r requirements.txt
$RUN pip install --no-deps mediapipe==0.10.35  # its opencv-contrib dependency clashes with robosuite's opencv

clone() {  # folder, url, commit
  [ -d third_party/$1 ] || git clone -q $2 third_party/$1
  git -C third_party/$1 checkout -q $3
}
mkdir -p third_party
clone LIBERO https://github.com/Lifelong-Robot-Learning/LIBERO.git 8f1084e
clone co-tracker https://github.com/facebookresearch/co-tracker.git 82e02e8
clone sam2 https://github.com/facebookresearch/sam2.git 2b90b9f
# LIBERO's pinned requirements are old; requirements.txt has what it needs.
$RUN pip install --no-deps -e third_party/LIBERO --config-settings editable_mode=compat
$RUN pip install --no-deps -e third_party/co-tracker --config-settings editable_mode=compat
SAM2_BUILD_CUDA=0 $RUN pip install --no-deps --no-build-isolation -e third_party/sam2 --config-settings editable_mode=compat

wget -nc -q -P third_party/mediapipe \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
wget -nc -q -P third_party/cotracker_ckpt https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget -nc -q -P third_party/sam2_ckpt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# LIBERO reads its paths from a config file; keep it in the repo, not in ~/.libero.
LIBERO_ROOT=$PWD/third_party/LIBERO/libero/libero
mkdir -p third_party/libero_config
cat > third_party/libero_config/config.yaml <<EOF
assets: $LIBERO_ROOT/assets
bddl_files: $LIBERO_ROOT/bddl_files
benchmark_root: $LIBERO_ROOT
datasets: $LIBERO_ROOT/../datasets
init_states: $LIBERO_ROOT/init_files
EOF
conda env config vars set -n $ENV LIBERO_CONFIG_PATH=$PWD/third_party/libero_config MUJOCO_GL=egl
echo "done: conda activate $ENV"
