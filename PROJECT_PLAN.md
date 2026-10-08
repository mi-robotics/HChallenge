# Project Plan: Human Video → Action Flow → Robot in Sim

Humanoid Robot Learning Internship Challenge

---

## The idea in one paragraph

I record myself doing a simple task with my hand (pushing a cup to a plate) while my hand **teleoperates a robot in simulation** at the same time. That gives me paired data: real video, my hand's motion as **action flow** (trails of tracked points), and the robot's actions in sim. I then train two models:

1. **Flow generator**: looks at a real photo + instruction and predicts the hand's action flow ("how should a hand move to do this?").
2. **Flow-to-action policy**: takes that flow and turns it into robot actions in sim ("how does the robot make that motion happen?").

At the end: **real photo + instruction → flow → robot does the task in sim.**

---

## Key decisions (fill in / confirm as we go)

| Decision | Current choice | Notes |
|---|---|---|
| Task | Push a cup onto a plate | Pushing avoids grasping, the hardest part. Grasping can be a stretch goal. |
| Simulator | LIBERO (robosuite underneath), Panda arm | May need a simple custom scene: table, cup, plate. |
| Camera | Phone on a tripod first, head/chest mount later | Tripod is easier to debug. |
| Calibration | ArUco marker taped to the table | Gives a shared "table frame" for real and sim. |
| Hand tracking | MediaPipe Hands (fast, works live) | HaMeR only if needed offline for better quality. |
| Point tracking | CoTracker (offline) | For clean action flow in the dataset. |
| Flow format | Points × timesteps × (x, y, z, visible), in table coordinates | Same format everywhere. Defined in Phase 3. |
| Compute | Laptop for sim + teleop, Google Colab GPU for training | |
| Deadline | Fri 9 Oct 2026, 23:59 BST | From the job posting. |

---

## Repo structure

```
project/
├── PROJECT_PLAN.md        ← this file
├── README.md              ← final write-up for reviewers
├── calibration/           ← ArUco marker, camera calibration scripts
├── sim/                   ← sim scene setup, robot control helpers
├── replay/                ← Phase 1: offline video → sim replay
├── teleop/                ← Phase 2: live hand → sim teleop
├── flow/                  ← Phase 3: video → action flow
├── data/                  ← Phase 4: recorded episodes (not committed if large)
├── policy/                ← Phase 5: flow → action policy
├── generator/             ← Phase 6: image + instruction → flow
├── eval/                  ← evaluation scripts and result tables
└── media/                 ← GIFs and videos for the README
```

---

## Phase 1: Video → Sim Replication

**Goal:** Take one recorded video of my hand and make the sim robot copy the motion afterwards (offline, not live).

**Why first:** This builds every piece the later phases need (calibration, hand tracking, mapping to sim, robot control) without the pressure of doing it live.

### Steps

1. **Set up the sim scene.** Panda arm, table, cup, plate. Make sure I can move the gripper to a target position and open/close it.
2. **Print and place the ArUco marker** on the table. Calibrate my phone camera once (checkerboard photos).
3. **Record one short demo** (5–10 seconds) of pushing the cup to the plate, with the marker visible.
4. **Track my hand** in every frame with MediaPipe. Get wrist and fingertip positions.
5. **Convert to table coordinates** using the marker, so positions are in centimetres on the table, not pixels.
6. **Clean it up.** Smooth the trajectory and resample it to the sim's control rate (20 Hz).
7. **Map hand → robot.** Wrist position → gripper position (with an offset and scale). Thumb–index distance → gripper open/close.
8. **Match the starting scene.** Detect where the cup is in the first frame and put the sim cup in the same place.
9. **Replay in sim** and record a video.

### Done when

- [ ] Sim scene works and the gripper can be moved to target positions
- [ ] Hand positions are in table coordinates (sanity check: they land on the table, not floating or below it)
- [ ] The sim robot follows the motion from my video
- [ ] I have a side-by-side video: real demo | sim replay

### Watch out for

- Depth (distance from camera) from a single phone camera is noisy. The marker helps; keep the camera angle steady.
- If the replayed push misses the cup, check the offset between my wrist and where my fingers actually touched the cup.

---

## Phase 2: Video → Sim Teleop (live)

**Goal:** Move my hand in front of the camera and have the sim robot follow in real time.

### Steps

1. **Stream the phone camera** to my laptop (an IP webcam app or similar).
2. **Run the Phase 1 pipeline live:** marker → hand tracking → table coordinates → robot target, every frame.
3. **Smooth the motion** so the robot doesn't jitter (a simple filter; keep lag low).
4. **Run the sim loop at 20 Hz**, always using the latest hand position.
5. **Show the sim on screen** next to the camera feed so I can watch the robot while moving my hand.
6. **Measure latency** (time from hand movement to robot movement). Note it in the README.
7. **Add a recording button / hotkey** that saves everything for one episode (used in Phase 4).
8. **Add a reset**: puts the sim cup back where the real cup is.

### Done when

- [ ] Robot follows my hand live with acceptable lag (aim for under ~200 ms)
- [ ] I can push the sim cup onto the sim plate by moving my hand
- [ ] Recording saves an episode to disk
- [ ] Short screen recording of live teleop for the README

### Watch out for

- If the camera moves (head mount), the marker must stay visible so the table frame can be recomputed every frame.
- Too much smoothing = laggy robot; too little = shaky robot. Tune it once and write down the setting.

---

## Phase 3: Video → Action Flows

**Goal:** Turn videos into action flow: trails of tracked points, in one fixed format used by every model.

### Steps

1. **Define the flow format** (write it down in `flow/FORMAT.md`):
   - Which points: wrist, thumb tip, index tip, plus a few points on the back of the hand
   - Horizon: the next T steps (start with T = 16 at 10 Hz ≈ 1.6 seconds)
   - Each point per step: (x, y, z) in table coordinates + visible yes/no
   - Shape: `[num_points, T, 4]`
2. **Run CoTracker** on recorded videos to get clean point trails (better than live MediaPipe).
3. **Convert trails to table coordinates** using the marker (reuse Phase 1 code).
4. **Also track points on the cup** (object flow). Needed for an ablation later.
5. **Write a visualiser** that draws the trails on top of the video. Use it constantly to check things look right.
6. **Write a function that slices** a full episode into training samples: (current time step → next T steps of flow).

### Done when

- [ ] `flow/FORMAT.md` exists and every script uses that format
- [ ] Hand flow and cup flow can be extracted from any recorded video
- [ ] Visualiser shows trails that match what my hand actually did
- [ ] Slicing function produces training samples

### Watch out for

- Fingers hide each other during contact. Mark those points as not visible instead of guessing.
- Keep the number of points and timesteps the same everywhere; mismatched shapes are the most common bug.

---

## Phase 4: Data Collection

**Goal:** Record a paired dataset using the Phase 2 teleop setup.

### What one episode contains

| Item | Source |
|---|---|
| Instruction | Typed or spoken ("push the cup to the plate") |
| Real video | Phone camera |
| Hand action flow | Phase 3 pipeline on the real video |
| Cup flow | Phase 3 pipeline on the real video |
| Sim robot actions | Teleop, recorded every step |
| Sim images + robot state | Sim, recorded every step |
| Success flags | Did the real cup reach the plate? Did the sim cup? |

### Protocol

1. Place the cup and plate in a new position each episode (mark a rough grid of positions on the table).
2. Press reset so the sim matches the real scene.
3. Press record, push the **real** cup with my hand while watching the **sim** robot follow.
4. Stop recording. Note whether both real and sim succeeded.
5. Repeat. Target **50–100 episodes**. Optional: a few instruction variations ("push the cup left", "push the cup to the plate").

### After recording

- [ ] Run the Phase 3 flow extraction on all episodes
- [ ] Drop episodes where real and sim drifted apart (e.g. sim cup missed)
- [ ] Split by episode: train / validation / test
- [ ] Keep some cup starting positions **only in the test set** (to test generalisation)
- [ ] Write `data/README.md`: how many episodes, how many kept, what was dropped and why

### Watch out for

- Vary lighting and positions a bit; too-similar episodes make models look better than they are.
- Back up the raw videos; everything else can be regenerated from them.

---

## Phase 5: Flow → Action Policy (training and evaluation)

**Goal:** Train a policy that takes action flow and outputs robot actions in sim.

### Model

- **Inputs:** sim camera image, robot state, target hand action flow (next T steps)
- **Outputs:** next few robot actions (gripper movement + open/close)
- **Start simple:** a small network (flow + state → actions, no image) to check the pipeline works
- **Then:** a small ACT or diffusion policy with the image included (LeRobot has implementations)

### Steps

1. Build the training dataset from Phase 4 (flow, sim obs, robot actions).
2. Train the simple baseline. Check the loss goes down and predictions look sensible.
3. Train the main policy on Colab.
4. Evaluate in sim by feeding **recorded** flows from held-out test episodes.

### Evaluation

| Metric | What it tells me |
|---|---|
| Success rate (cup on plate) | Does it complete the task? |
| Trail-following error | How closely does the gripper follow the flow? |
| Results on unseen cup positions | Does it generalise? |

### Comparisons (these make the results meaningful)

- [ ] **Baseline:** raw hand-coded retargeting (Phase 1 rule) on the same flows
- [ ] **Learned policy** vs that baseline
- [ ] **Ablation:** policy conditioned on hand flow vs cup flow vs both
- [ ] **Robustness:** add noise to the input flow; how fast does success drop?

### Done when

- [ ] Results table with all comparisons above
- [ ] Videos of successes **and** failures

---

## Phase 6: Flow Generator (training and evaluation)

**Goal:** Train a model that looks at a real photo + instruction and predicts the hand action flow.

### Model

- **Inputs:** real image (first frame of an episode or current frame), instruction text
- **Output:** hand action flow in the Phase 3 format
- **Start simple:** frozen pretrained image encoder (e.g. DINOv2) + simple text embedding + small network that outputs the trails
- **Upgrade if needed:** a small diffusion model over trails (handles cases with more than one valid way to do the task)

### Steps

1. Build the dataset: (real image, instruction) → hand flow, from Phase 4.
2. Train the simple version. Visualise predicted trails on test images.
3. Upgrade only if predictions are blurry/averaged.
4. **End-to-end test:** real photo → generator → flow → Phase 5 policy → sim robot.

### Evaluation

| Metric | What it tells me |
|---|---|
| Trail error vs real recorded flow (test set) | Is the generator accurate? |
| End-to-end success rate in sim | Does the whole system work? |
| End-to-end vs using the real recorded flow | How much is lost by generating the flow? |

### Done when

- [ ] Predicted trails drawn on real test photos look sensible
- [ ] End-to-end success rate reported
- [ ] Hero demo video: real photo with predicted trail | sim robot executing it

---

## Final: Write-up (README)

Keep it short, specific, and honest. Reviewers explicitly said no AI slop.

- [ ] One-paragraph summary + hero GIF at the top
- [ ] Diagram of the pipeline (real photo → generator → flow → policy → sim)
- [ ] How to run it (install, commands, Colab links)
- [ ] Results tables from Phases 5 and 6
- [ ] Design choices and **why**
- [ ] What worked, what didn't, and what I'd do next
- [ ] Failure videos, not just successes

---

## How to work on this with Claude Code

- Work **one phase at a time**. Start each session with: *"Read PROJECT_PLAN.md. We're working on Phase X."*
- Tick the checkboxes as things get done so progress is always visible.
- After each phase, write a few notes under it: what worked, what changed, numbers measured. These become the README later.
- If a decision changes (e.g. task, flow format), update the **Key decisions** table first.
- Always keep something working end-to-end before making it fancier.
