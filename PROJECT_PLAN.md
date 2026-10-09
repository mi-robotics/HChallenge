# Project Plan: Human Video → Action Flow → Robot in Sim

Humanoid Robot Learning Internship Challenge

---

## The idea in one paragraph

I record myself doing a simple task with my hand (picking up a cup and putting it on a plate), then **replay each recording on a robot in simulation** and keep the episodes where the sim robot succeeds. That gives me paired data: real video, my hand's motion as **action flow** (trails of tracked points), and the robot's actions in sim. I then train two models:

1. **Flow generator**: looks at a real photo + instruction and predicts the hand's action flow ("how should a hand move to do this?").
2. **Flow-to-action policy**: takes that flow and turns it into robot actions in sim ("how does the robot make that motion happen?").

At the end: **real photo + instruction → flow → robot does the task in sim.**

---

## Key decisions (fill in / confirm as we go)

| Decision | Current choice | Notes |
|---|---|---|
| Task | Pick up a cup and put it on a plate | Changed from pushing in step 1.1: in sim the mug tips on the plate rim or shoves the plate (0/21 push runs succeeded). A rim grasp (one finger inside the mug) succeeded 8/8. |
| Simulator | LIBERO (robosuite underneath), Panda arm | May need a simple custom scene: table, cup, plate. |
| Camera | Intel RealSense D405 (RGB + depth, 1280×720 @ 30 fps) on a fixed mount | Switched from phone in step 1.2: factory intrinsics (no checkerboard calibration), and depth gives the hand's 3D position directly. |
| Calibration | ArUco marker taped to the table | Gives a shared "table frame" for real and sim. Printer scales to 89%: marker is 89 mm, board squares 26.7 mm (set in `calibration/make_targets.py`). |
| Hand tracking | MediaPipe Hands (fast, works live) | HaMeR only if needed offline for better quality. |
| Point tracking | CoTracker (offline) | For clean action flow in the dataset. |
| Flow format | hand [6, 16, 4] + cup [8, 16, 4]: points × timesteps (10 Hz) × (x, y, z, visible), metres, table frame | Defined in step 3.1: `flow/FORMAT.md`, constants in `flow/format.py`. |
| Sim actions for the dataset | Offline replay of recorded videos; failed replays dropped | Changed after step 1.9: replay with rim-grasp constraints succeeded 5/5, so live teleop (Phase 2) was dropped. |
| Compute | Laptop for recording + sim replay, Google Colab GPU for training | |
| Deadline | ___ | |

---

## Repo structure

```
project/
├── PROJECT_PLAN.md        ← this file
├── README.md              ← final write-up for reviewers
├── calibration/           ← printable targets, D405 camera access, table-marker pose
├── sim/                   ← sim scene setup, robot control helpers
├── replay/                ← Phase 1: offline video → sim replay
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
2. **Print and place the ArUco marker** on the table. Check the D405 factory calibration against the ChArUco board and the depth sensor.
3. **Record demos** (5 × 10–11 s, `replay/record.py`) of picking up the cup and putting it on the plate, with the marker visible.
4. **Track my hand** in every frame with MediaPipe. Get wrist and fingertip positions.
5. **Convert to table coordinates** using the marker, so positions are in centimetres on the table, not pixels.
6. **Clean it up.** Smooth the trajectory and resample it to the sim's control rate (20 Hz).
7. **Map hand → robot.** Pinch point (thumb/index midpoint) → gripper position, relative to the grasp point. Thumb–index distance → gripper open/close.
8. **Match the starting scene.** Locate the cup and plate (from the grasp and release points; white objects give no D405 depth) and place the sim cup and plate the same distance apart.
9. **Replay in sim** and record a video.

### Done when

- [x] Sim scene works and the gripper can be moved to target positions
- [x] Hand positions are in table coordinates (sanity check: they land on the table, not floating or below it)
- [x] The sim robot follows the motion from my video
- [x] I have a side-by-side video: real demo | sim replay

### Notes

- **1.1** Pushing a mug onto a plate failed in sim (0/21): it tips on the rim or shoves the plate. Switched to pick-and-place; scripted rim grasp 8/8.
- **1.2** Switched phone → D405. Printer scales A4 to 89%; with measured sizes, depth and image-based distance agree within 1.7 mm at 35 cm. Table marker: 30/30 frames, pose jitter 1–2 mm.
- **1.3** 5 demos, 307–345 frames each, 0 dropped, marker visible 99–100% of frames. I grasp the mug around its body from the side, while the sim robot uses a top-down rim grasp: Phase 1 maps the wrist path and grasp timing, not the grasp style. My left hand is on the keyboard at the start/end (pressing s/e): track the right hand only. The cup and plate are further from the camera than the marker (beyond the D405's best range of 50 cm).
- **1.4** MediaPipe hand landmarker, right hand only (labelled "Right" on this camera; the docs' mirrored-image caveat does not apply). Right hand found in 70 / 87 / 100 / 100 / 98% of frames. Demo 1 loses it for 2.7 s (frames 34–113) because the cup was at the top edge of the image and the hand left the frame; demos 2 and 5 have only 1–5-frame gaps. When the hand is found, D405 depth is valid at the thumb and index tips 100% of the time and at the wrist 84–100%; wrist depth 40–72 cm.
- **Setup change (after 1.4):** camera re-aimed and marker moved for a bigger workspace. Marker rotated so x points away from the camera (camera now at x = −41.9, y = 0.3, z = 43.9 cm in the table frame, i.e. where the sim robot base is). Marker 61 cm away, 30/30 frames, jitter 1–3 mm. The first 5 demos were moved to `data/demos_old/` and re-recorded with this setup.
- **1.3 / 1.4 re-run (new setup):** 5 demos, 300–339 frames, 0 dropped, marker visible 100%. Right hand found in 100 / 100 / 100 / 93 / 93% of frames (gaps: demo 4 frames 99–127 during the reach, demo 5 frames 184–204); depth valid at wrist and tips 99–100%, wrist 37–63 cm from the camera. The left hand is out of view.
- **1.5** Landmarks → table frame: undistort pixel, scale by D405 depth (z along the optical axis), then marker pose. The camera is fixed during a demo, so one pose per demo (per-axis median; per-frame spread < 1 mm). Checks: bare table near the marker comes out at z = −2.3…+2.4 mm (median); a plane fitted to the whole table tilts 1.1–3.4° from the marker's z, i.e. ≤ 2 cm height error where the hand works (left as is for now). Bone lengths are realistic and stable (wrist–knuckle 7.9–8.5 cm, index finger 7.4–7.9 cm). The wrist stays 15–28 cm above the table (I grasp from above/side); fingertips go down to 5–10 cm when grasping. 0–8 frames per demo have a fingertip below −1 cm: depth from the table behind a thin finger; to be filtered in 1.6. The hand works 25–80 cm to the side of the marker (camera looks off to the side), so 1.7/1.8 should map to sim using the cup/plate positions, not the marker origin.
- **1.6** `replay/clean.py` → `traj.npz` (t, wrist, thumb, index, aperture, filled) at 20 Hz. Spike rule (> 3 cm from a 7-frame rolling median, or z < −1 cm) dropped only 6 samples, all on the index tip. Gaps filled by interpolation: up to 26 frames in demos 4–5 (longest 0.7 s). Zero-phase Gaussian smoothing (σ = 67 ms): wrist acceleration RMS 5–14 → 0.4–0.8 m/s², cleaned line stays on the raw points. Thumb–index distance works as a grasp signal: ~4.5 cm open on approach, ~1.5 cm while carrying (pinch on the mug), opens at release.
- **1.7** Changed the mapping from wrist → gripper to **pinch point (thumb/index midpoint) → gripper fingertip centre**: it is where the fingers touch the mug, while the wrist sits 10+ cm above it at a pose-dependent offset. Target = pinch − pinch at grasp + sim rim-grasp point, so the real (~9 cm) and sim (13.5 cm) mug heights cancel; no rotation or scale. Gripper: one grasp per demo; close at the first thumb–index < 2.5 cm, release at the first > 3.0 cm after a 3 cm lift. A plain hysteresis gave extra events (relaxed hand re-closing after release, a 0.35 s flicker in demo 4, demo 4 releasing by moving away with fingers only at ~3.3 cm); debouncing broke demos 2–3. Check: all 5 release points land within std 0.2 / 0.4 / 0.6 cm (x/y/z) of each other, i.e. on the saucer, which did not move; grasp heights 6.9–8.5 cm (just under a ~9 cm mug's rim); lift 7–15 cm.
- **1.8** Depth-based cup/saucer detection failed: both are white and the D405 returns almost no depth on blank white surfaces (the "saucer" it found was a cable). Located them from the hand instead: grasp point = on the cup, release point = over the saucer. Projected into the first frame, the grasp point lands on the mug rim and the release xy (at saucer height) on the saucer centre in all 5 demos. Sim: cup at the BDDL region centre (−0.11, 0), plate at cup + (release − grasp)xy (15–38 cm away). All replay targets within 0.43–0.78 m of the robot base (Panda reach 0.855 m) and ≥ 9 cm above the table. Added a sim camera at the D405's pose relative to the cup (same orientation, fovy 58.6°) for side-by-side videos; needed `sim.forward()` after moving it.
- **1.9** `replay/replay.py`: matched scene, one control step per 20 Hz sample, LIBERO On(cup, plate) after 1 s of settling. **Raw hand path: 2/5.** Failures were all the embodiment gap (I pinch from the side; the robot grasps with one finger inside the mug): the gripper knocked the mug over on approach (demo 3), dragged it off the plate when moving away sideways after release (demo 2), and closed while still descending (demo 5). **With three local rim-grasp constraints: 5/5**, cups end 0.1–1.8 cm from the plate centre, sim cup lift 6.7–13.8 cm vs hand 7.2–15.3 cm. Constraints: (1) approach from ≥ 8 cm above the grasp point until the *gripper* (not the target, the arm lags fast motion by up to 10 cm) is within 3 cm horizontally; (2) at the grasp, 0.5 s arrive with fingers open, then 0.5 s close; (3) after release, hold 0.25 s and rise 10 cm before following the hand. A first version that applied the clearance to the target and closed during the settle got 3/5. Tracking error (gripper vs target) median 8–34 mm, p95 30–91 mm: the OSC controller lags fast hand motion (up to 0.43 m/s).
- **1.9 ablation** (`replay/ablation.py`, all 8 subsets × 5 demos): none 2/5; approach / settle / lift alone 3/5 each; approach+settle 4/5, approach+lift 4/5, settle+lift 3/5; all three 5/5. Each constraint is needed: without approach demo 3 is knocked over (gripper comes in 2–3 cm above grasp height, pushes the mug 4 cm before the fingers close); without settle demo 5 closes on air (fingers close 3.2 cm above the grasp point, cup lifted 0.5 cm); without lift-out demo 2 is dragged 3.8 cm off the plate centre. Approach alone breaks demo 5's grasp (the gripper arrives later than the hand's close command), which settle fixes. Parameters were tuned on these same 5 demos.

### Watch out for

- D405 depth is most accurate within 50 cm and has holes on blank white surfaces (no texture to match). The marker sits about 52 cm away.
- If the replayed grasp misses the rim, check the offset between my wrist and where my fingers actually closed on the cup.

---

## Phase 2: Video → Sim Teleop (live) — dropped

Dropped after step 1.9. Offline replay of a recorded video already gets the sim cup onto the plate (5/5 with the rim-grasp constraints), so the dataset's sim actions come from replaying recorded videos (Phase 4), and failed replays are dropped. Live teleop would have let me correct the robot while recording; offline replay can't, so some episodes will be lost to failed replays instead.

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
6. ~~Write a function that slices a full episode into training samples~~ Moved to Phase 5 step 1 (see notes).

### Done when

- [x] `flow/FORMAT.md` exists and every script uses that format
- [x] Hand flow and cup flow can be extracted from any recorded video
- [x] Visualiser shows trails that match what my hand actually did
- [x] ~~Slicing function produces training samples~~ moved to Phase 5 step 1

### Notes

- **3.1** `flow/FORMAT.md` + `flow/format.py`. Hand points: wrist, thumb tip, index tip (pinch point and grasp signal), index/middle/pinky knuckles (back of the hand, gives orientation). Cup: 8 points. T = 16 at 10 Hz = now + 1.5 s ahead (~4 cm per step at the p95 hand speed 0.39 m/s). Absolute table-frame metres; hidden points interpolated only to avoid NaNs, flagged visible = 0 for masking; past the episode end the last frame repeats. Checked by building the hand flow for all 5 demos from the MediaPipe landmarks: 100–111 steps each, visible 93–100%. Open issue for 3.4: no D405 depth on the white mug. Stand-in mug flow for the render: a 4 cm-radius cylinder centred 4 cm from the grasp point along thumb → index (the index tip is inside the mug), 4 rim + 4 base points, moved with the pinch point from grasp to release. A rigid fit to the back of the hand (which would also rotate the mug) put the mug base 1–24 cm above the table at release; translation only gives 0.8–2.3 cm (saucer ~1.5 cm), and the release positions of all 5 demos agree within 1 cm in x and 5 cm in y.

- **3.2** `flow/track_points.py`: CoTracker3 offline on the local RTX 5090, 34 s for all 5 demos. Queries = the 6 hand points from MediaPipe. With one query frame per demo (at the grasp) the pinky knuckle drifted (p95 287 px from MediaPipe in demo 4): it is hidden behind the hand in 70–80% of frames in demos 2 and 4, as the camera sees the thumb side. Querying every 0.5 s (20–23 query frames) and taking each frame from the nearest query: median 2–5 px, p95 4–30 px, max 74 px from MediaPipe, measured ≥ 4 frames from a query (1 px ≈ 0.8 mm at 50 cm). Tracks cover every frame, including MediaPipe's gaps. CoTracker's visibility (96–99%) says whether the surface at that pixel is visible, not the landmark: a query on a hidden pinky knuckle starts on whatever is in front of it. Added an occlusion check: MediaPipe's metric 3D hand (world landmarks, now saved by `track_hand.py`; landmark pixels unchanged, 0 px) is fitted to the D405 points with a RANSAC similarity fit (11–13 of 21 inliers within 1.5 cm, scale 0.83–0.93), giving each point's expected depth; D405 depth at the tracked pixel then agrees within a median ±2 cm. Occluded = CoTracker says so, or something > 3 cm in front, or > 5 cm behind (pixel off the hand), or no depth. Using MediaPipe's z directly gave per-landmark biases of −6 to +5 cm (wrist "occluded" 92–100%). Result: pinky knuckle occluded 7–32%, wrist 1–17%, the rest ≤ 9%. Mug: the 8 stand-in points (now on the camera-facing half of the mug) queried in the first and last frame; CoTracker follows them within a median 1–6 px of the hand-derived stand-in, an independent check of the stand-in. Mug occluded 7–26%: hand in front, track off the mug onto the table (depth > 5 cm behind; seen in demo 4), or > 40 px from the stand-in.

- **3.3** `flow/to_flow.py` → `flow.npz` per demo (100–113 steps). Hand: tracked pixel + D405 depth + marker pose. Mug: tracked pixel + the stand-in's depth. Each 10 Hz step = mean of the visible camera frames within ±50 ms; nearest-frame sampling left about twice the jitter of the smoothed step 1.6 path (pinch 2nd difference 5.5–10 mm vs 4–8 mm after averaging; step 1.6: 2.6–6 mm). Pinch vs step 1.6: median 1–2 mm, p95 5–7 mm, outside MediaPipe's gaps. Inside them, step 1.6 only interpolates: in demo 5 (0.7 s gap) the tracks show the hand swinging ~10 cm towards the camera and partly out of the image, which the interpolation missed (max 101 mm apart). Added to 3.2: points within 10 px of the image border are hidden. Visible: hand 95–97%, mug 86–98%. No visible point below −1 cm. Mug ends 0.4–2.2 cm above its start (on the saucer).

- **3.4** Mug flow = CoTracker pixel + depth from the stand-in mug, now moved with the CoTracker pinch point (the MediaPipe one interpolated straight through demo 5's gap). Independent check `flow/mug_check.py`: PnP of the 8 tracked mug points (≥ 5 visible, 83–97% of frames), no hand involved. Mug standing still: PnP vs flow median 3–10 mm, p95 8–23 mm. Carried: median 8–56 mm, max 43–120 mm, the gap lying along the camera's line of sight, and PnP jumps back by up to 10 cm at the release, which a real mug cannot do. With ~100 px between the points a small spread in the tracks reads as a large change in distance, so PnP is not used for depth. Real mug depth needs texture on the mug (tape) in the Phase 4 recordings. Hidden mug points are now filled by moving them with the visible ones; linear interpolation drew them up to 32 cm off the mug over hidden runs of up to 3 s.
- **3.4 mug points, geometric vs mask** (`flow/mug_sam.py`, `flow/mug_compare.py`): SAM 2 (large) segments the mug in every frame from one click in the first frame (98 s for 5 demos); masks checked by eye at 5 times per demo: whole mug incl. handle, saucer and hand excluded. Mask method: 8 points by farthest-point sampling on the first-frame mask (eroded 8 px). Both tracked by CoTracker from the first frame only, scored against the masks. Geometric: start on the mug 88–100% (demo 4: 1 of 8 starts on the mug's lower edge and stays on the table for the whole video), on the mug when visible 83–100%, > 20 px off 0–7.3%. Mask: start on the mug 100%, on the mug 98.6–100%, > 20 px off 0%; visible 88–91% vs 95–100% (edge and handle points hide as the mug turns). The mask method needs one click instead of the grasp and an assumed mug size. Switched the flow to the mask points (`flow/track_points.py`): mug visible 88–91%, PnP vs flow still median 2–12 mm standing, carried median 10–34 mm; mug ends 1.4–1.9 cm above its start.
- **3.5** `flow/visualise.py`: every frame shows the flow sample starting there (next 16 steps, 1.5 s) for the 6 hand and 8 mug points, hidden steps in red. Used it to find the mug-fill problem above.
- **3.6 moved to Phase 5 step 1.** Cutting a sample out of `flow.npz` is one line (already in `flow/visualise.py`); the real work is aligning each sample to the sim replay (20 Hz, plus time added by the rim-grasp constraints), which needs the Phase 4 replays.

### Watch out for

- Fingers hide each other during contact. Mark those points as not visible instead of guessing.
- Keep the number of points and timesteps the same everywhere; mismatched shapes are the most common bug.

---

## Phase 4: Data Collection

**Goal:** Record a paired dataset: real videos, replayed offline in sim with the Phase 1 pipeline. Keep the episodes where the sim cup ends on the plate.

### What one episode contains

| Item | Source |
|---|---|
| Instruction | "put the cup on the plate" (optional variations) |
| Real video | D405 colour + depth (`replay/record.py`) |
| Hand action flow | Phase 3 pipeline on the real video |
| Cup flow | Phase 3 pipeline on the real video |
| Sim robot actions | Offline replay (`replay/replay.py`, rim-grasp constraints), every step |
| Sim images + robot state | Sim, recorded every step of the replay |
| Success flag | LIBERO On(cup, plate) after the replay |

### Steps

1. **Record episodes** with `replay/record.py`. Place the cup and plate in a new position each time (mark a rough grid on the table). Target **50–100 episodes**.
2. **Run the Phase 1 pipeline** on every episode: track hand → table coordinates → clean → retarget → match scene.
3. **Replay each episode in sim** with the rim-grasp constraints, saving actions, robot state and images per step.
4. **Keep the successful replays.** Write down how many were dropped and why (missed grasp, hand lost, etc.).
5. **Extract hand and cup flow** (Phase 3) for the kept episodes.
6. **Split by episode** into train / validation / test, with some cup positions **only in the test set**. Write `data/README.md`: how many episodes, how many kept, what was dropped and why.

### Done when

- [ ] 50–100 recorded episodes, each replayed in sim
- [ ] Kept episodes have sim actions, state, images and flow saved in one format
- [ ] `data/README.md` with counts and the split

### Notes

- **4.1** `python replay/record.py data/mug_pnp_dataset` (recorder now takes the output folder; `x` discards a take). 100 episodes, 9.8 min, 4.4–9.9 s each (median 5.7 s; the Phase 1 demos were 10–11 s, so about twice as fast). 1 dropped frame in total, marker found in every first frame, 7.5 GB. Cup and saucer moved every episode. The 5 Phase 1 demos are not in the dataset (saucer always in one place).
- **4.2** `replay/run_dataset.py`: Phase 1 pipeline + sim replay on every episode, each in its own process (MediaPipe and MuJoCo's EGL renderer in one long process segfaulted in scene_match on episode 5; the same episode ran fine alone). Unchanged Phase 1 replay: **34/100**. 9 no grasp found in the hand track (2 never pinched below 2.5 cm, 2 closed without rising 3 cm, 5 never reopened above 3 cm after placing, 4 of them with 30–47% of the hand track gap-filled); 2 hand lost > 1.5 s (NaN trajectory); 55 replayed and failed, 32 of them with the gripper resting ~5 cm up on the mug's rim when the fingers closed. The approach let the descent start within 3 cm (horizontal) of the grasp point; in Phase 1 the hand always came from one direction, here it does not. Half-speed replay of the 55 rescued only 12, so speed is not the main cause. **Descending only within 1 cm: 60/100** (0 on the rim; Phase 1 demos still 5/5). Remaining 29 replay failures: 22 lifted but put down 3–29 cm off the plate (8 let go ≥ 8 cm above the grasp height), 4 knocked off the table, 3 closed but not lifted. Kept episodes: cup ends median 0.8 cm, max 3.0 cm from the plate centre. Shorter episodes fail more (before the fix: 1/11 under 5 s). The 40 failures moved to `data/mug_pnp_failed/` (not deleted); batch-1 summary in `data/mug_pnp_dataset/replay_summary_batch1.json`.
- **4.1/4.2 batch 2** (40 more, median 5.65 s): **19/40** kept, total **79**. 4 hand lost (NaN trajectory), 3 no grasp found, 12 lifted but put down off the plate, 1 not lifted, 1 knocked off the table. Cause of most: I had suggested putting the index finger inside the mug; then the thumb-index distance during the carry is 3–5 cm, above the 3 cm "open" threshold, so the release fires 0.35–0.85 s after the grasp and the robot lets go mid-carry; the finger inside the mug is also hidden (32–61% of the hand track gap-filled around the grasp). Tried detecting the release from the hand setting the cup down (pinch back within 3 cm of grasp height after rising 5 cm) instead of the finger gap, on scratch copies of 128 episodes: batch 1 60 → 57, batch 2 19 → 17. Not adopted. For the next batch: pinch the rim from the outside, as in batch 1. Batch-2 failures moved to `data/mug_pnp_failed/` (61 there now).
- **Batches 3–4** (outside pinch again): 21 recorded, 18 kept, 1 failed, 2 hung in MediaPipe hand tracking (2% CPU for 8+ min; `run_dataset.py` now times an episode out after 300 s); then 4 more, 4/4 kept. **Dataset: 101 successful episodes** in `data/mug_pnp_dataset/`, 64 dropped in `data/mug_pnp_failed/` (165 recorded, 61% kept).

### Watch out for

- Vary lighting and positions a bit; too-similar episodes make models look better than they are.
- Back up the raw videos; everything else can be regenerated from them.
- `scene_match.py` puts the sim cup at one fixed spot and only the cup → plate offset follows the real scene. If varied absolute positions matter for Phase 5, place the sim cup at the real cup's table position instead.
- Keeping only successful replays filters on the hand-designed constraints too; report the kept fraction.
- **Limitation (accepted for now): the robot always grasps the same spot on the rim.** The replay anchors the hand path so the moment I close lands on the sim rim-grasp point (cup centre + 4 cm in y, fixed because the fingers close along world y), so where I grasp around the rim is not copied. It does not cause failures (none of the 40 in step 4.2). A fix would take the grasp angle around the mug (mug centre from the SAM mask) and rotate the gripper yaw to match; the replay is position-only today.
- Same anchoring, matters for Phase 5: the sim cup always starts at one spot, but the flow is in real table coordinates, so a flow points to where the real cup was. Before Phase 5: place the sim cup at the real cup's position, or give the policy the flow relative to the cup.

---

## Phase 5: Flow → Action Policy (training and evaluation)

**Goal:** Train a policy that takes action flow and outputs robot actions in sim.

### Model

- **Inputs:** sim camera image, robot state, target hand action flow (next T steps)
- **Outputs:** next few robot actions (gripper movement + open/close)
- **Start simple:** a small network (flow + state → actions, no image) to check the pipeline works
- **Then:** a small ACT or diffusion policy with the image included (LeRobot has implementations)

### Steps

1. Build the training dataset from Phase 4 (flow, sim obs, robot actions). Includes the slicing moved from step 3.6: sample s = flow steps s … s+15 (last step repeated past the end), paired with the sim observation and actions at the matching replay step (replay is 20 Hz, flow 10 Hz, plus any time the replay constraints add).
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

- [ ] **Baseline:** hand-coded retargeting (Phase 1 rule + rim-grasp constraints) on the same flows
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
