# Dataset: put the cup on the plate

Real recordings of my hand moving a mug onto a saucer (Intel RealSense D405), each replayed offline by a Panda in
LIBERO, with hand and mug action flow extracted from the video. Recorded 2026-10-09.

The episode data is not in git (16 GB). This file, `split.py` and `splits.json` are.

## Where it lives

```
data/
├── mug_pnp_dataset/      100 kept episodes: the sim replay put the cup on the plate (11 GB)
├── mug_pnp_failed/       65 dropped episodes: replay failed or the pipeline could not process them (5.2 GB)
├── demos/                5 Phase 1 demos (saucer always in one place; not in the dataset)
├── demos_old/            5 demos from the first camera setup (not used)
├── calibration/          ChArUco board captures used to check the D405 factory intrinsics (step 1.2)
├── split.py              writes splits.json
└── splits.json           train / val / test / test_novel episode lists
```

One folder per episode, named by the recording time: `<dataset>/20261009_HHMMSS/`.

## Counts

165 recorded, 100 kept (61%). Every kept episode replayed successfully twice: in step 4.2 and again with
recording on in step 4.3.

| why dropped | episodes |
|---|---|
| no grasp found in the hand track (thumb–index never closed below 2.5 cm, or never reopened above 3 cm) | 12 |
| hand lost for > 1.5 s (NaN trajectory) | 6 |
| MediaPipe hung (300 s timeout); no hand track, no flow | 2 |
| replay ran, cup not lifted (< 2 cm) | 8 |
| replay ran, cup lifted but ended 3.1–37 cm from the plate centre (one knocked off the table) | 36 |
| passed in step 4.2, failed when re-recorded in 4.3 (cup 1.5 cm from the plate centre) | 1 |
| **total** | **65** |

Batch 2 (40 episodes, index finger inside the mug) kept only 19: the thumb–index gap stayed above the 3 cm
release threshold, so the robot let go mid-carry. Details per batch: `PROJECT_PLAN.md`, step 4.2 notes.

## Splits (`splits.json`, step 4.5)

Split by episode. Cup position = mean xy of the 8 mug flow points at the first step (table frame).

| split | episodes | sim steps | cup y (cm) |
|---|---|---|---|
| train | 70 | 12,845 | −61 … −19 |
| val | 10 | 1,938 | −57 … −23 |
| test | 10 | 1,854 | −59 … −20 |
| test_novel | 10 | 1,985 | −84 … −61 |
| failed (flow only) | 45 | – | −56 … −13 |
| failed_novel (flow only) | 18 | – | −88 … −64 |

- `test_novel`: the 10 kept episodes with the cup furthest to the right of the marker (y < −0.613 m). No
  train/val/test episode starts there, but the boundary is tight: the nearest train episode is 2 mm inside it. The rest are shuffled with seed 0.
- `failed` episodes have flow but no sim actions; they can add flow-only training data (e.g. Phase 6).
  `failed_novel` are in the held-out region and must not be trained on.
- The far-right region fails more often: 18 of the 63 failed episodes with flow are there, vs 10 of the 100 kept
  (the cup is 0.8–1 m from the camera there, where D405 depth reads 2–4% short).
- In sim the cup always starts at the same spot (−0.11, 0) and only the plate moves (cup + the real cup→plate
  offset), so for the sim robot `test_novel` means new absolute flow positions and plate offsets, not a new cup spot.

## Frames and clocks

- **Table frame**: ArUco marker (id 0) at the origin, x away from the camera across the table, y left, z up,
  metres. Table axes = sim world axes.
- **Table → sim world**: `p_sim = p_table + sim_minus_real` (`sim.npz`), one translation per episode that puts
  the pinch at the moment of grasp on the sim rim-grasp point. It does not put the mug flow on the sim cup (median
  6.9 cm off, because I grasp the real rim at different spots).
- **Clocks**: camera 30 fps; `traj`/`retarget` 20 Hz on the real clock; `flow.npz` 10 Hz on the real clock;
  `sim.npz` and `flow_sim.npz` 20 Hz on the robot's clock. A sim step follows the real video at `t_real`; 55 steps
  per episode are inserted by the replay (grasp settle and close, release hold and rise, final 1 s wait), and
  `t_real` holds during them. All `t` are seconds since the first camera frame.

## Files in an episode folder

Raw recording (`replay/record.py`). Everything else can be regenerated from these four.

| file | contents |
|---|---|
| `color.mp4` | 1280 × 720 colour, 30 fps |
| `depth.mkv` | depth aligned to colour, lossless uint16, metres = value × `depth_scale` (1e-4) |
| `frames.csv` | per frame: index, camera frame number, timestamp_ms |
| `meta.json` | instruction ("put the cup on the plate"), fps, frames, dropped_frames, duration_s, image_size, camera_matrix, dist_coeffs, depth_scale |

Hand → robot pipeline (Phase 1 code, run by `replay/run_dataset.py`, step 4.2).

| file | contents |
|---|---|
| `hand.npz` | MediaPipe right hand per camera frame: detected [F], score [F], uv [F,21,2] px, depth [F,21] m (0 = none), world [F,21,3], t_ms [F] |
| `hand_overlay.mp4` | colour video with the hand landmarks drawn |
| `hand_table.npz` | landmarks in the table frame: p [F,21,3] m, valid [F,21], cam_in_table [3] |
| `traj.npz` | cleaned, 20 Hz: t [N], wrist / thumb / index [N,3], aperture [N] (thumb–index, m), filled [N] (gap-filled) |
| `retarget.npz` | t [N], rel [N,3] (pinch − pinch at grasp), gripper [N] (−1 open / +1 close), i_grasp, i_release, real_grasp [3], real_release [3] |
| `scene.json` | sim_cup_xy, sim_plate_xy, sim_grasp_point, plate_minus_cup_cm, reach checks, sim_camera |
| `replay.json` | step 4.2 replay result: success, tracking_error_mm, cup / hand max lift, cup_final_xy_to_plate_cm |
| `replay_trace.npz` | step 4.2 replay per step: target, eef, cup [S,3], gripper [S], table_z, plate_xy, grasp |
| `replay.mp4` | real video next to the step 4.2 sim replay |

Sim recording (`replay/record_sim.py`, step 4.3; kept episodes only). `sim.npz`, per control step (20 Hz), observed
before the action:

| key | shape | contents |
|---|---|---|
| `action` | [S,7] | OSC_POSE delta: xyz in [−1, 1] (±5 cm), rotation 0, gripper −1 open / +1 close |
| `joint_pos`, `joint_vel` | [S,7] | arm joints (rad, rad/s) |
| `gripper_qpos` | [S,2] | finger positions |
| `eef_pos`, `eef_quat` | [S,3], [S,4] | end-effector pose, sim world |
| `cup_pos`, `cup_quat`, `plate_pos` | [S,3], [S,4], [S,3] | object state, sim world (cup_pos is the mug base) |
| `agentview` | [S,256,256,3] uint8 | third-person RGB (`sim/scene.py` THIRD_PERSON) |
| `eye_in_hand` | [S,128,128,3] uint8 | gripper camera RGB |
| `t_real` | [S] | real-video time this step follows (s) |
| `sim_minus_real` | [3] | table → sim world offset |
| `success`, `hz`, `sim_plate_xy` | | LIBERO On(cup, plate), 20, plate position |

Flow (`flow/run_flows.py`, step 4.4; all episodes with a hand track). Format: `flow/FORMAT.md`.

| file | contents |
|---|---|
| `mug_mask.npz` | SAM 2 mug mask per camera frame (packed [F,720,160] uint8; `flow/mug_sam.py` load_masks), frame (where the mug was found), t_grasp, t_release (from the mug's motion) |
| `cotracker.npz` | CoTracker per camera frame: uv [F,6,2], visible [F,6] (hand points), cup_uv [F,8,2], cup_visible [F,8] |
| `flow.npz` | real clock, 10 Hz: t [N], hand [6,N,4], cup [8,N,4] (x, y, z table frame, visible) |
| `flow_sim.npz` | robot clock (kept only): one row per `sim.npz` step: t [S], t_real [S], hand [6,S,4], cup [8,S,4] |
| `flow_overlay.mp4` | colour video with the flow drawn |

A flow sample (the next 1.5 s, 16 steps at 10 Hz): from `flow.npz` at step i, `hand[:, i:i+16]`; from
`flow_sim.npz` at sim step i, `hand[:, i:i+32:2]` (last row repeated past the end).

Hand points: wrist, thumb tip, index tip, index / middle / pinky knuckle. Pinch point = mean of thumb and index tips.

## Summary files

| file | contents |
|---|---|
| `mug_pnp_dataset/replay_summary.json`, `mug_pnp_failed/replay_summary.json` | step 4.2 per episode: stage reached, error, replay.json fields |
| `mug_pnp_dataset/replay_summary_batch{1,2}.json` | the same for recording batches 1 and 2 when they were run |
| `*/flow_summary.json` | step 4.4 per episode: stage, error, mug_frame, t_grasp, t_release, steps, hand / mug visible fraction |

## Regenerate

```
python replay/run_dataset.py data/mug_pnp_dataset     # hand track → replay (step 4.2)
python replay/record_sim.py                           # sim.npz for kept episodes (step 4.3)
python flow/run_flows.py                              # masks → flow → flow_sim.npz (step 4.4)
python data/split.py                                  # splits.json (step 4.5)
```
