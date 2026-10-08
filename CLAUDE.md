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
  2. Write a step report `docs/steps/NN-<slug>.html` (what we did, why, commands, results,
     numbers, what didn't work) and publish it as an artifact.
  3. Update `docs/tracker.html` (phase status + link to the new step report) and republish it
     to the same artifact URL.
  4. Commit (and push when the user asks).
- Keep writing plain and specific. Measured numbers over adjectives. Record failures too.

## Environment

- Conda env: `humanoidschallenge` (Python 3.10, chosen for LIBERO/robosuite compatibility).
  Run things with `conda run -n humanoidschallenge ...` or activate it first.
- Install packages only when a step needs them, and record them in `requirements.txt`
  at that point.
- Training runs on Google Colab GPU; sim + teleop run locally.

## Repo

- Remote: `git@github.com:mi-robotics/HumanoidsChallenge.git` (branch `main`).
- Don't commit large data or videos (`data/`, raw recordings). Small GIFs for the README go in `media/`.

## Artifacts

- Progress tracker: `docs/tracker.html` → https://claude.ai/artifact/CpWJFnat7EFLwbwQXKYquD
- Step reports: `docs/steps/*.html`, one artifact each, linked from the tracker.
