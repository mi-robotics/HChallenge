# Humanoid Challenge

Videos of my own hand moving a mug onto a plate drive a Panda arm in a LIBERO simulation: from one real image of the
scene and a text instruction, a video world model (Cosmos) plans the hand and mug motion as action flow, and a
flow-conditioned policy carries it out in sim.

**Project details, results and videos: https://mi-robotics.github.io/HChallenge/**

## Repository structure

```
├── calibration/    D405 camera wrapper, ArUco table marker and printable targets (the table frame)
├── replay/         record demos, track the hand, retarget it to the robot and replay it in sim (real2sim)
├── flow/           action flow: hand and mug points tracked with CoTracker3 and SAM 2, lifted to 3D, put on the robot's clock
├── sim/            the LIBERO task (cup on plate) and scene helpers: make_env, move_to, set_gripper
├── data/           dataset README, train / val / test / failed split (the episodes themselves are not in git)
├── policy/         flow-conditioned action policy (flow matching, DINOv2): train and closed-loop eval in sim
├── wam/            world action model: Cosmos-Predict2 fine-tuned (LoRA) to predict video + flow; planning and end-to-end eval
├── site/           the project website (GitHub Pages, deployed by .github/workflows/pages.yml)
├── scripts/        setup, data download and preprocessing, training and evaluation (below)
├── dataset.py      the episodes as the policy and the WAM read them
└── requirements.txt
```

Not in git: `data/` episodes (18.5 GB, see `data/README.md`), `outputs/` (checkpoints, evals) and `third_party/`
(LIBERO, CoTracker3, SAM 2 and their checkpoints, installed by `scripts/setup.sh`).

## Reproduce

Linux, conda, ffmpeg and an NVIDIA GPU (we used one RTX 5090). Times are for that GPU.

With our data and weights:

```
bash scripts/setup.sh              # conda env humanoidschallenge, third-party code and checkpoints
bash scripts/download_data.sh      # dataset from Hugging Face into data/ (raw + processed, 19 GB)
bash scripts/download_weights.sh   # our policy and WAM weights into outputs/ (750 MB)
bash scripts/run_samples.sh        # WAM plans -> policy in sim on the test episodes, rendered videos (~10 min)
```

Everything from scratch:

```
bash scripts/download_data.sh --raw   # raw recordings only (14 GB)
bash scripts/preprocess_data.sh       # real2sim replay, robot data and action flow for every episode
bash scripts/train_policy.sh          # flow-conditioned policy (wrist camera), then its sim evaluation (~1 h)
bash scripts/train_wam.sh             # world action model v1 (~18 h)
bash scripts/eval_wam.sh              # end-to-end: WAM plans -> policy in sim, with pauses (~3 h)
```

The WAM's base model, nvidia/Cosmos-Predict2-2B-Video2World, is gated: accept its licence on Hugging Face and run
`hf auth login` first.

Data: https://huggingface.co/datasets/mcarroll101/HChallenge-mug-pnp ·
weights: https://huggingface.co/mcarroll101/HChallenge-models
