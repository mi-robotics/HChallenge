# HumanoidsChallenge

Submission for the Humanoid "Internship, Robot Learning Research" challenge:
use data I personally recorded (phone video of my hand) to drive a Panda arm in a
simple sim (LIBERO / robosuite). Submitted as a public GitHub repo with a README.

- Reviewers judge: creativity (my own data must play a role), sim performance and/or
  world-model quality, implementation simplicity, clear results with **no AI slop**.
- Full plan: `PROJECT_PLAN.md` (phases 1–6 + write-up). It is the source of truth for
  scope, key decisions and checkboxes.

## How we work

- One phase at a time, one step at a time. Don't start the next step until the user agrees.
- **No speculative code or structure.** Only create files/folders the current step needs.
  The repo layout in `PROJECT_PLAN.md` is a target, not something to scaffold up front.
- After every step:
  1. Tick the checkbox(es) in `PROJECT_PLAN.md`; update the Key decisions table if a decision changed.
  2. Write a step report `docs/steps/<phase>.<step>-<slug>.html` (e.g. `1.1-sim-scene.html`:
     what we did, why, commands, results, numbers, what didn't work) and publish it as an artifact.
  3. In `docs/tracker.html`, mark the step done (state 1), set the next step to 2, add the
     report URL to that step, and republish to the same artifact URL. Steps in the tracker
     mirror the numbered steps in `PROJECT_PLAN.md` one-to-one.
  4. Commit (and push when the user asks).
- Keep writing plain and specific. Measured numbers over adjectives. Record failures too.

## Environment

- Conda env: `humanoidschallenge` (Python 3.10, chosen for LIBERO/robosuite compatibility).
  Run things with `conda run -n humanoidschallenge ...` or activate it first.
- Install packages only when a step needs them, and record them in `requirements.txt`
  at that point.
- LIBERO is cloned (commit 8f1084e) into `third_party/` (gitignored) and installed with
  `pip install --no-deps -e third_party/LIBERO --config-settings editable_mode=compat`.
  Its pinned requirements are old, so we skip them; `requirements.txt` has what it actually needs.
- The env sets `LIBERO_CONFIG_PATH=third_party/libero_config` (absolute path) and `MUJOCO_GL=egl`
  via `conda env config vars`. Don't touch `~/.libero/`, which belongs to another env.

## Sim

- Task: `sim/cup_on_plate.bddl` ("put the cup on the plate"), red mug + plate on the LIBERO
  living-room table. Robot base is at -x and reaches towards +x.
- `sim/scene.py`: `make_env`, `reset`, `move_to(target_xyz)`, `set_gripper`. Control is
  OSC_POSE delta at 20 Hz; action[:3] = 1 means 5 cm per step; gripper -1 open, +1 close.
- The mug (9–10 cm wide, 13.5 cm tall) is too wide for the Panda's 8 cm grip: grasp the rim with
  one finger inside, at cup centre + 4 cm in y (fingers close along world y).
- Training runs on Google Colab GPU; sim + teleop run locally.

## Repo

- Remote: `git@github.com:mi-robotics/HumanoidsChallenge.git` (branch `main`).
- Don't commit large data or videos (`data/`, raw recordings). Small GIFs for the README go in `media/`.

## Artifacts

- Progress tracker: `docs/tracker.html` → https://claude.ai/artifact/CpWJFnat7EFLwbwQXKYquD
- Step reports: `docs/steps/*.html`, one artifact each, linked from the tracker.
