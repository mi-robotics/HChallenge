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
├── PROJECT_PLAN.md phases, steps, key decisions and measured results
├── CLAUDE.md       working notes: environment, conventions
└── requirements.txt
```

Not in git: `data/` episodes (18.5 GB, see `data/README.md`), `outputs/` (checkpoints, evals) and `third_party/`
(LIBERO, CoTracker3, SAM 2 and their checkpoints; install steps in `requirements.txt` and `CLAUDE.md`).
