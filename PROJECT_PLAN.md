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
| Compute | Local RTX 5090 for recording, sim replay and training | Changed in Phase 5: the training split (two 256 × 256 cameras, 5.8 GB uint8) fits on the GPU, so no data loader; Colab not needed. |
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
2. **Run the Phase 1 pipeline** on every episode: track hand → table coordinates → clean → retarget → match scene → replay. Keep the successful replays; write down how many were dropped and why (missed grasp, hand lost, etc.). (Was steps 2 and 4.)
3. **Replay each kept episode in sim**, saving actions, robot state and images per step.
4. **Extract hand and cup flow** (Phase 3) for every episode, kept and failed (failed ones may be useful later).
5. **Split by episode** into train / validation / test, with some cup positions **only in the test set**. Write `data/README.md`: how many episodes, how many kept, what was dropped and why.

### Done when

- [x] 50–100 recorded episodes, each replayed in sim
- [x] Kept episodes have sim actions, state, images and flow saved in one format (per episode folder: `sim.npz` + `flow.npz`)
- [x] `data/README.md` with counts and the split

### Notes

- **4.1** `python replay/record.py data/mug_pnp_dataset` (recorder now takes the output folder; `x` discards a take). 100 episodes, 9.8 min, 4.4–9.9 s each (median 5.7 s; the Phase 1 demos were 10–11 s, so about twice as fast). 1 dropped frame in total, marker found in every first frame, 7.5 GB. Cup and saucer moved every episode. The 5 Phase 1 demos are not in the dataset (saucer always in one place).
- **4.2** `replay/run_dataset.py`: Phase 1 pipeline + sim replay on every episode, each in its own process (MediaPipe and MuJoCo's EGL renderer in one long process segfaulted in scene_match on episode 5; the same episode ran fine alone). Unchanged Phase 1 replay: **34/100**. 9 no grasp found in the hand track (2 never pinched below 2.5 cm, 2 closed without rising 3 cm, 5 never reopened above 3 cm after placing, 4 of them with 30–47% of the hand track gap-filled); 2 hand lost > 1.5 s (NaN trajectory); 55 replayed and failed, 32 of them with the gripper resting ~5 cm up on the mug's rim when the fingers closed. The approach let the descent start within 3 cm (horizontal) of the grasp point; in Phase 1 the hand always came from one direction, here it does not. Half-speed replay of the 55 rescued only 12, so speed is not the main cause. **Descending only within 1 cm: 60/100** (0 on the rim; Phase 1 demos still 5/5). Remaining 29 replay failures: 22 lifted but put down 3–29 cm off the plate (8 let go ≥ 8 cm above the grasp height), 4 knocked off the table, 3 closed but not lifted. Kept episodes: cup ends median 0.8 cm, max 3.0 cm from the plate centre. Shorter episodes fail more (before the fix: 1/11 under 5 s). The 40 failures moved to `data/mug_pnp_failed/` (not deleted); batch-1 summary in `data/mug_pnp_dataset/replay_summary_batch1.json`.
- **4.1/4.2 batch 2** (40 more, median 5.65 s): **19/40** kept, total **79**. 4 hand lost (NaN trajectory), 3 no grasp found, 12 lifted but put down off the plate, 1 not lifted, 1 knocked off the table. Cause of most: I had suggested putting the index finger inside the mug; then the thumb-index distance during the carry is 3–5 cm, above the 3 cm "open" threshold, so the release fires 0.35–0.85 s after the grasp and the robot lets go mid-carry; the finger inside the mug is also hidden (32–61% of the hand track gap-filled around the grasp). Tried detecting the release from the hand setting the cup down (pinch back within 3 cm of grasp height after rising 5 cm) instead of the finger gap, on scratch copies of 128 episodes: batch 1 60 → 57, batch 2 19 → 17. Not adopted. For the next batch: pinch the rim from the outside, as in batch 1. Batch-2 failures moved to `data/mug_pnp_failed/` (61 there now).
- **Batches 3–4** (outside pinch again): 21 recorded, 18 kept, 1 failed, 2 hung in MediaPipe hand tracking (2% CPU for 8+ min; `run_dataset.py` now times an episode out after 300 s); then 4 more, 4/4 kept. **Dataset: 101 successful episodes** in `data/mug_pnp_dataset/`, 64 dropped in `data/mug_pnp_failed/` (165 recorded, 61% kept).
- **4.2 scene from the hand vs SAM (pilot, not adopted).** The broken-scene case (sim plate placed on top of the cup, cup pushed away before the replay starts) happened in 15 of the 50 replayed failures and 0 of the 101 kept: the plate goes at cup + (release − grasp) of the pinch, and an early release puts it 0.6–11 cm from the cup. Pilot on 62 failed + 20 kept episodes: SAM 2 mug mask (click from the hand at the first pinch), mug base per frame = ray through the bottom of the mask meets the table plane (+ mug radius away from the camera). On a checked example the SAM cup and plate positions were right and the hand's were not. But: 20/82 first-frame masks were wrong (table, saucer, hand, cable; caught by area 4–25k px, h/w 0.8–1.7, fill > 0.6). (a) Positions and grasp/release times from the mug's motion: failures 7/62 rescued, kept 9/20 still succeed. (b) Positions from SAM, grasp from the hand, release = first hand opening once the mug is nearly at rest: failures 6/62, kept 13/20. Why it hurts the kept ones: the replay only needs the plate where the robot lets go, and the robot carries the cup by the hand's pinch path; the SAM plate offset differs from the pinch's by a median 2 cm (max 34 cm), so a correct scene with the hand's carry puts the cup partly off the plate. Using real positions would need the robot path made relative to the mug too. Kept the hand-based scene. Pilot scripts were scratch only; `mug_mask.npz` stays in the 82 episode folders.
- **4.2 hand vs SAM, debugged.** They agree: in 19 kept episodes the pinch at the grasp is 2.8–6.9 cm from SAM's mug centre (rim ≈ 4.5–5 cm) at 8.0–9.6 cm height (mug ≈ 9 cm), and at the release 2.1–6.2 cm from SAM's final mug centre. The pilot's 2 cm median gap was comparing the mug centre's displacement with the pinch's (the mug turns in the hand); the sim cup cannot turn in the gripper. **D405 depth reads short at range**, measured against the marker's table plane on 11 episodes: 0–1% at 50 cm, 2.2–3.6% at 90 cm, 4.7% at 120 cm (per-pixel IQR ±3–5%); the step 1.4 check only looked within 25 cm of the marker. The mugs sit 50–94 cm from the camera. Pinch height at the grasp rises 1.4 cm per metre of range (corr 0.27). A per-episode correction fitted on the table moved the median grasp and release heights by only 0.3 cm, and the re-run kept 87 instead of 101: 5 dropped by a new grasp check (correct), 12 flipped to failures, 7 of them borderline set-downs (the sim plate's rim is 3.6 cm above the table, the real saucer ~1.5 cm, so the sim cup lands on the plate rim and a few mm decide it). Correction not adopted: the robot follows the same measured pinch path the flow is built from, so flow and robot match regardless (gripper path = pinch path + one per-episode offset, sim grasp point − real grasp point; Phase 5 must shift the flows by it). Known and left for now: in 5 kept episodes the grasp is detected 13–17 cm above the table (above the mug), which puts that offset 5–8 cm wrong although the replay succeeds; a check on the pinch height at the grasp (5–13 cm) would drop them. For future recordings: camera closer (work area < ~60 cm from it).
- **4.3** `replay/record_sim.py` → `sim.npz` per kept episode: per control step (20 Hz) the action sent, joint position/velocity, gripper fingers, end-effector pose, cup/plate state, third-person camera (256 × 256) and gripper camera (128 × 128), plus the real-video time each step follows and the table → sim offset (sim grasp point − real grasp point) that puts the hand flow on the robot's path. Observations before each action. Third-person camera: LIBERO's agentview moved back and up (50° above the table, 1.7 m from the workspace centre, `sim/scene.py` THIRD_PERSON) because the plate left the default view in some episodes; it must be set after every reset (reset rebuilds the model). 100/101 succeed when recorded: 20261009_112930 failed (cup 1.5 cm from the plate centre; borderline set-down that passed in 4.2) and was moved to `data/mug_pnp_failed/`. **Dataset: 100 episodes**, 18,622 steps (142–265 per episode, median 177), 2.0 GB.
- **4.4** `flow/run_flows.py`: per episode SAM 2 mug mask and timing (`flow/mug_sam.py`) → CoTracker hand + mug points → `flow.npz` (FORMAT.md) → `flow_overlay.mp4`; writes `<dataset>/flow_summary.json`. **Flow for 163/163 episodes with a hand track**: 100 kept + 63 failed (the other 2 failed episodes hung in MediaPipe and have no hand track). The step 3.4 click on the mug is replaced by SAM 2's automatic mask generator with a mug filter (area 2.5–30k px, h/w 0.85–1.9, fill > 0.6, median grey > 100; the mug in shadow is ~127, so the pilot's 140 missed it), picking the candidate nearest the hand's first pinch; in 3 failed episodes the mug was not found in frame 0 and the last frame was used. Masks propagated forwards and backwards. Grasp/release times now come from the mug's motion (mask base on the table plane moves > 2 cm from its start / is within 2 cm of its end), not from the hand, because the hand events are wrong in many failed episodes (e.g. 112033: hand grasp at 5.15 s, mug moves at 3.47 s); 12 failed episodes have no grasp from the hand track (no grasp found or hand lost) and still get a flow; in the others the hand events differ from the mug motion by up to 3.8 s. Kept episodes: the mug starts moving a median 0.54 s after the hand pinch closes (p10–p90 0.23–0.90 s) and stops 0.73 s before the hand opens (0.35–1.20 s); failed: −0.45…1.62 s and −1.07…1.50 s. 10,391 flow steps at 10 Hz (44–105 per episode). Hand visible median 98% (min 91%), mug visible median 88% kept / 85% failed (min 40% / 32%; the hand covers the mug while carrying). Run time 56 min on the RTX 5090 (6 test episodes done earlier). Checked by eye on the overlays of 6 test episodes and the 4 in the report, not on all 163. **Flow vs sim robot** (100 kept, `replay/record_sim.py --video --flow` projects the flow + sim_minus_real into the third-person camera): while carrying, the flow pinch point is a median 2 mm (max 2.3 cm) from the replay target, so the flow is the robot's commanded path; the gripper lags that target by ~0.2 s (4 steps): 5.4 cm median at the same step, 1.0 cm (p90 3.0 cm) with the lag removed. The mug flow + offset is a median 6.9 cm from the sim cup in xy at the start (fixed sim grasp spot vs where I grasped the real rim), so for the mug flow the offset should be from the cup, not the grasp point. **Flow on the robot's clock:** `flow/sim_clock.py` → `flow_sim.npz` per kept episode, one row per `sim.npz` step (20 Hz), flow.npz interpolated at the real time each sim step follows; the 55 inserted steps per episode (grasp settle/close, release hold/rise, final 1 s) hold the flow still. Same length as the robot data (median 177 steps vs 61 flow steps at 10 Hz on the real clock). Sample at sim step i = rows i:i+32:2 (FORMAT.md).

- **4.5** `data/split.py` → `data/splits.json`, by episode: the 100 kept episodes shuffled (seed 0) into **train 80 / val 10 / test 10** (14,759 / 1,988 / 1,875 sim steps). A held-out table region (test_novel, cup y < −0.61 m) was tried and dropped: it would be all out of distribution. **Paired training data on the robot's clock**, one row / frame per sim step (20 Hz): `video_sim.mp4` (real video with the 55 inserted steps as held frames; `flow/sim_clock.py`), `flow_sim.npz` (now also `frame`, the colour frame per step) and `sim.npz`. Also recorded for the failed episodes whose replay runs (`replay(..., record=True)`): 46 of 65, 7,876 steps; 20261009_112930 succeeded this time (passed 4.2, failed 4.3: borderline set-downs are not repeatable). 17 failed have real-clock flow only (no grasp in the hand track / hand lost), 2 nothing past the raw recording. Paired video 291 MB in total (H.264, 1280 × 720). `data/README.md` documents where the data lives, every file in an episode folder with what made it and whether training needs it, frames and clocks, failure reasons. Not needed by any code: the overlay videos (`hand_overlay`, `flow_overlay`, `replay.mp4`, 320 MB) and `replay_trace.npz` (superseded by sim.npz); kept for now. Training images, measured on 26,498 steps (18,622 kept): raw uint8 RGB is 2.8 GB at 224 × 224 for the kept episodes (4.0 GB with failed), 2.1 GB at 256 × 144, 12.9 GB at 640 × 360, 51.5 GB at full 1280 × 720; storing each camera frame once and indexing by `frame` saves 30% (the pauses repeat frames).

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

1. **Image-only S1 baseline (no flow).** The S1 action policy from HVLA (TheWisp/lerobot, flow matching, S2 removed): two sim cameras (third person + gripper, 256 → 224 for DINOv2) + robot state → 32-step action chunk; train on the 80 train episodes, evaluate closed loop in sim on the test scenes. What a normal visuomotor policy gets from this data, for the flow-conditioned policy to beat.
2. Build the flow-conditioned training dataset from Phase 4 (flow, sim obs, robot actions), all on the robot's clock: sample at sim step i = `flow_sim.npz` rows i … i+30 step 2 (16 steps at 10 Hz), with `sim.npz` and `video_sim.mp4` at step i.
3. Train the simple flow baseline. Check the loss goes down and predictions look sensible.
4. Train the main policy (flow + images + state).
5. Evaluate in sim by feeding **recorded** flows from held-out test episodes.

### Notes

- **5.1** Image-only S1 baseline (`policy/s1_model.py`, `train_s1.py`, `eval_s1.py`): HVLA's flow-matching S1 (TheWisp/lerobot 13f05f3, Apache-2.0) without S2 or training-time RTC; DINOv2 ViT-S/14 fine-tuned on the third-person and gripper cameras (256 → 224, ImageNet-normalised; HVLA feeds [0, 1]) + state token (eef pos, quat, fingers) → 32-step chunk of (xyz, gripper); 93.5 M params. Gripper camera first re-recorded at 256 for all 146 replayed episodes (kept 100/100 again; 112930 failed again, so it has now gone pass / fail / pass / fail). Trained on the 80 train episodes held on the GPU (6.6 GB), batch 64, LR 2.5e-5 cosine, 20k steps, 56 min. Train loss 1.9 → 0.02; **val loss lowest at 2k (0.57), then rises to 1.42**: not a bug (ckpt 5k on the same code: train 0.13, val 0.74), the model memorises 80 episodes. Closed loop (episode's scene and start pose, recorded length + 3 s): replaying the recorded actions through the evaluator reproduces success 6/6. First protocol (15 Euler steps, execute 8, 2 runs): val 5k 12/20, 10k 10/20, 15k 16/20, 20k 14/20; two runs of one checkpoint disagree on 2/10 episodes, so selection on 10 val episodes is noise. **Eval protocol now fixed:** 20k checkpoint only, 10 Euler steps, execute 10 of 32, splits train/val/test/failed, 5 seeds (torch seed 1000*seed + episode index), mean ± std of success % over seeds; 730 runs, 10 min on 8 workers. **baseline_v1:** train 85.5 ± 2.4, val 82.0 ± 4.0, test 60.0 ± 8.9, failed 42.6 ± 3.3. **baseline_v1_img_aug** (LeRobot's default ImageTransforms, HVLA has none: 3 of 6 per image, brightness/contrast/saturation/hue/sharpness/affine, batched GPU ops matching torchvision to 2e-7, no slowdown): train 88.5 ± 3.9, val 88.0 ± 11.7, **test 72.0 ± 7.5**, failed 42.6 ± 1.1. Val loss min 0.44 at 3k (vs 0.57 at 2k), 0.90 at 20k (vs 1.42): augmentation slows the memorising, does not stop it. Failures in the videos: cup set down beside the plate or tipped over.

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

### Model (world action model, `wam/`)

- **Base:** Cosmos-Predict2-2B Video2World (`nvidia/Cosmos-Predict2-2B-Video2World`, diffusers format; gated, accept
  the licence on Hugging Face), fine-tuned with LoRA (rank 64 on attention + MLP, 91.8M of 2,048M parameters).
- **Inputs:** the current real frame (256 × 448) and the prompt "pick up the mug and put it on the plate" (T5).
- **Outputs, jointly denoised:** the next 16 frames at 10 Hz (1.6 s) and the flow for the same 16 steps
  (6 hand + 8 mug points, x y z in the table frame + visible = 896 numbers).
- **Flow in the latent sequence** (the Cosmos Policy trick, arXiv 2601.16163): the 896 numbers are repeated
  32× to fill one latent frame (16 × 32 × 56) and averaged back out. Latents: [frame now (given)] [4 latents =
  16 future frames] [flow]. The flow frame is last so the causal VAE decodes the video without it.
- **Data:** robot clock with the pauses (`video_sim.mp4` + `flow_sim.npz`), so the predicted flow is in the
  policy's format and goes to the S1 policy unchanged.
- Dropped: generate video then track it (scenario 1): the generated video has no depth, and our flow is 3D.

### Steps

1. Build the dataset and the fine-tuning code (`wam/data.py`, `model.py`, `train.py`); smoke tests without the
   Cosmos weights.
2. Train on the train split; visualise predicted video + trails on val/test frames.
3. **End-to-end test:** real photo → WAM → flow → Phase 5 policy (`full_wrist_state`) → sim robot, chunk by chunk
   (each chunk starts from the last predicted frame).

### Notes

- **6.1** `wam/` (diffusers 0.41 `CosmosTransformer3DModel` + Wan2.1 VAE, peft LoRA; not NVIDIA's cosmos-policy
  stack, which needs Docker + Megatron; cloned for reference to `third_party/cosmos-policy`, commit 18a2acc). Noise and parametrisation as diffusers' Cosmos2 Video2World pipeline
  (x_t = (1 − t) x0 + t ε, t = σ/(1+σ), the network predicts ε − x0; given frame clean with t = 1e-4); training σ
  lognormal(1.386, 1.2) as Cosmos Policy. Sample at step i: video frames i, i+2, …, i+32 and flow rows i … i+30
  (same rows as the policy). Smoke tests (`wam/smoke.py`, no Cosmos weights): samples match the video/flow rows;
  tile → untile error 5e-7; VAE round trip 41.1 dB PSNR; the recorded flow projected onto the 256 × 448 frames
  sits on the hand and mug. Overfit on one batch of 4 with a 2-layer random transformer: flow loss 2.28 → 0.24,
  sampled-flow error 85 → 44 mm in 1,500 steps. Full-size 2B (random weights, LoRA r64, gradient checkpointing,
  bf16): batch 8 = 4.15 s/step, 13.3 GB (VAE encode 0.94 s of it; transformer 3.25 s); batch 16 = 6.6 s + VAE,
  21.4 GB; without checkpointing batch 8 runs out of memory. Sampling one 1.6 s chunk: 2.2 s (20 steps), 1.1 s
  (10 steps). 20k steps at batch 8 ≈ 23 h (≈ 18 h with the latents cached). Not run yet: real Cosmos weights (gated)
  and the T5 prompt embedding (from the same repo).
- **6.2** Three LoRA runs, same config (20k steps, batch 8, LR 1e-4; 3.2 / 4.3 / 2.7 s per step = 18 / 24 / 15 h on the 5090). A first run drew independent
  noise for each of the 32 flow copies and learned the mean flow (205 mm vs the mean-flow predictor's 200 mm);
  one noise draw tiled like the flow fixed it. **v1** video + flow: 1.6 s val flow error 60 mm from the image alone
  (mean over steps 10k–20k; hold still 70 mm), 9.6 s chained rollouts 102 mm (4 val episodes, one sample each).
  **v2** + the flow's current row as a condition (given clean half the time): 61 mm from the image alone, 41 mm with
  the recorded current row given, rollouts 110 mm when chained on its own last row. **v3** video only: video loss
  0.018 vs 0.025 (v1) at 18k–20k. Sampler: Cosmos Policy's 5 steps / σ_min 4 gives the same flow error as 35 steps /
  σ_min 0.002, but the moving hand is visibly cleaner at 15–35 steps.
- **6.3** `wam/plan.py`: from each episode's first real frame, chained 1.6 s chunks (35 steps, σ_min 0.002, v2 hands each
  chunk's last flow row to the next) until the policy's horizon (recorded length + 3 s) is covered; 146 episodes in
  86 min. `policy/eval_s1.py --plan` feeds that flow to `full_wrist_state_no_img_pos` (wrist camera + state + hand and
  mug flow) instead of the recorded flow; the real→sim offset (sim_minus_real) is still the episode's recorded one.
  Success, 1 seed, 1 attempt (train / val / test / failed): recorded flow (seed 0 of the step 5 eval) 98 / 60 / 80 / 7 %;
  **v1 46 / 20 / 30 / 4 %; v2 34 / 10 / 30 / 4 %.** Plan vs recorded flow at the same step: v1 77 / 119 / 114 / 120 mm,
  v2 76 / 114 / 91 / 118 mm (counts timing errors too). The continuation does not help end to end. Train is far above
  val/test in plan error, so the WAM partly memorised its 80 episodes. Failures in the videos: the cup set down short of
  the plate. The sim run is cheap (146 runs in 1.9 min); sharing the GPU with Cosmos sampling starves the policy.
  **Pauses added** (`--pauses`): the plans rarely hold still (pinch moves < 2 mm per 0.1 s in 7 % of plan steps vs 32 %
  of recorded ones), so the replay's pauses are inserted after planning: grasp / release from the plan's thumb and
  index tips with the step 1.7 rule (found in 141 / 146 v1, 136 / 146 v2 plans), flow held 1.0 s at the grasp and
  0.75 s at the release. v1 58 / 20 / 20 / 4 %, v2 39 / 0 / 40 / 2 %: train up (v1 +18 / −9 episodes), val and test
  within one episode. **v2 + recorded first flow row** (`wam/plan.py --given-state`, the hand and mug as recorded in
  the first frame given to the first chunk): first 1.6 s chunk 36 mm from the recorded flow vs 59 mm from the image
  alone (val + test), whole episode 73 / 105 / 93 / 111 mm; success 38 / 30 / 20 / 4 %, with pauses 52 / 40 / 20 / 4 %.
  With 10 val / test episodes and 1 seed none of these differences is outside one or two episodes.

### Evaluation

| Metric | What it tells me |
|---|---|
| Trail error vs real recorded flow (test set) | Is the generator accurate? |
| End-to-end success rate in sim | Does the whole system work? |
| End-to-end vs using the real recorded flow | How much is lost by generating the flow? |

### Done when

- [ ] Predicted trails drawn on real test photos look sensible
- [x] End-to-end success rate reported
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
- [ ] Project site on GitHub Pages (`site/`: Overview, Data, Policy, WAM, Code; deployed by `.github/workflows/pages.yml`). Skeleton with empty pages done; content to come

---

## How to work on this with Claude Code

- Work **one phase at a time**. Start each session with: *"Read PROJECT_PLAN.md. We're working on Phase X."*
- Tick the checkboxes as things get done so progress is always visible.
- After each phase, write a few notes under it: what worked, what changed, numbers measured. These become the README later.
- If a decision changes (e.g. task, flow format), update the **Key decisions** table first.
- Always keep something working end-to-end before making it fancier.
