# Dataset: put the cup on the plate

Real recordings of my hand moving a mug onto a saucer (Intel RealSense D405), each replayed offline by a Panda in
LIBERO, with hand and mug action flow extracted from the video. Recorded 2026-10-09.

The episode data is not in git (18.5 GB). This file, `split.py` and `splits.json` are.

## Where it lives

```
data/
├── mug_pnp_dataset/      100 kept episodes: the sim replay put the cup on the plate (12 GB)
├── mug_pnp_failed/       65 failed episodes: replay failed, or the hand track gave no grasp (6.5 GB)
├── demos/                5 Phase 1 demos (saucer always in one place; not in the dataset)
├── demos_old/            5 demos from the first camera setup (not used)
├── calibration/          ChArUco board captures used to check the D405 factory intrinsics (step 1.2)
├── split.py              writes splits.json
└── splits.json           train / val / test (kept) and failed / failed_no_robot episode lists
```

One folder per episode, named by the recording time: `<dataset>/20261009_HHMMSS/`.

## Training data: three files per episode, matched step for step

Everything for training is on the robot's clock: one row (or video frame) per sim control step, 20 Hz.

| file | contents |
|---|---|
| `video_sim.mp4` | the real colour video with the pauses: frame i = the camera frame sim step i follows (1280 × 720, 20 fps) |
| `flow_sim.npz` | hand and mug flow with the pauses: row i = the flow at that same moment |
| `sim.npz` | robot data: action, proprioception, end-effector pose, object state, third-person and gripper camera images |

Same length in all three (episode 20261009_111136: 254 frames, 254 rows, 254 steps). The replay inserts 55 steps
per episode that have no real-video moment of their own (settle and close at the grasp, hold and rise after the
release, a final 1 s wait); during them the real video and the flow hold still while the robot moves.

Who has it: the 100 kept episodes and 46 of the 65 failed ones (those whose replay ran; `sim.npz` `success` says
how it ended). The other 19 failed episodes have no robot data: 17 have real-clock flow only (`flow.npz`; no grasp
found in the hand track, or the hand was lost), 2 have nothing past the raw recording (MediaPipe hung).

## Splits (`splits.json`, step 4.5)

| list | episodes | sim steps | what |
|---|---|---|---|
| train | 80 | 14,759 | kept, shuffled with seed 0 |
| val | 10 | 1,988 | kept |
| test | 10 | 1,875 | kept |
| failed | 46 | 7,876 | failed, with robot data and paired video + flow |
| failed_no_robot | 17 | – | failed, real-clock flow only |

20261009_112930 is in `failed`: it passed in step 4.2, failed in 4.3 (cup 1.5 cm from the plate centre), passed
once more when the failed episodes were first recorded and failed again in the step 5.1 re-recording; the replay
is not fully repeatable on borderline set-downs. In that re-recording all 100 kept episodes succeeded again.

## Why episodes failed

165 recorded, 100 kept (61%).

| why | episodes |
|---|---|
| no grasp found in the hand track (thumb–index never closed below 2.5 cm, or never reopened above 3 cm) | 12 |
| hand lost for > 1.5 s (NaN trajectory) | 6 |
| MediaPipe hung (300 s timeout); no hand track, no flow | 2 |
| replay ran, cup not lifted (< 2 cm) | 8 |
| replay ran, cup lifted but ended 3.1–37 cm from the plate centre (one knocked off the table) | 36 |
| passed in step 4.2, failed in the 4.3 recording (cup 1.5 cm from the plate centre) | 1 |
| **total** | **65** |

Batch 2 (40 episodes, index finger inside the mug) kept only 19: the thumb–index gap stayed above the 3 cm
release threshold, so the robot let go mid-carry. Details per batch: `PROJECT_PLAN.md`, step 4.2 notes.

## Frames and clocks

- **Table frame**: ArUco marker (id 0) at the origin, x away from the camera across the table, y left, z up,
  metres. Table axes = sim world axes.
- **Table → sim world**: `p_sim = p_table + sim_minus_real` (`sim.npz`), one translation per episode that puts
  the pinch at the moment of grasp on the sim rim-grasp point. It does not put the mug flow on the sim cup (median
  6.9 cm off, because I grasp the real rim at different spots). The sim cup always starts at (−0.11, 0); only the
  plate moves (cup + the real cup→plate offset).
- **Clocks**: camera 30 fps; `traj` / `retarget` 20 Hz and `flow.npz` 10 Hz on the real clock; `video_sim.mp4`,
  `flow_sim.npz` and `sim.npz` 20 Hz on the robot's clock. `t_real` (in `sim.npz` and `flow_sim.npz`) is the
  real-video time each sim step follows; it holds during the inserted steps. All times are seconds since the first
  camera frame.

## Every file in an episode folder

Training data (robot clock; kept episodes and the 46 replayed failed ones):

| file | what it is | made by | needed? |
|---|---|---|---|
| `video_sim.mp4` | real colour video on the robot's clock (with the pauses), 20 fps | `flow/sim_clock.py` | training |
| `flow_sim.npz` | t [S], t_real [S], frame [S] (`color.mp4` frame shown at each step), hand [6,S,4], cup [8,S,4] (x, y, z table frame, visible) | `flow/sim_clock.py` | training |
| `sim.npz` | robot data per step, observed before the action (keys below) | `replay/record_sim.py` | training |

`sim.npz`:

| key | shape | contents |
|---|---|---|
| `action` | [S,7] | OSC_POSE delta: xyz in [−1, 1] (±5 cm), rotation 0, gripper −1 open / +1 close |
| `joint_pos`, `joint_vel` | [S,7] | arm joints (rad, rad/s) |
| `gripper_qpos` | [S,2] | finger positions |
| `eef_pos`, `eef_quat` | [S,3], [S,4] | end-effector pose, sim world |
| `cup_pos`, `cup_quat`, `plate_pos` | [S,3], [S,4], [S,3] | object state, sim world (cup_pos is the mug base) |
| `agentview` | [S,256,256,3] uint8 | third-person RGB (`sim/scene.py` THIRD_PERSON) |
| `eye_in_hand` | [S,256,256,3] uint8 | gripper camera RGB (128 × 128 until step 5.1, then all 146 replayed episodes were re-recorded at 256) |
| `t_real` | [S] | real-video time this step follows (s) |
| `sim_minus_real` | [3] | table → sim world offset |
| `success`, `hz`, `sim_plate_xy` | | LIBERO On(cup, plate) at the end, 20, plate position |

Raw recording (`replay/record.py`). The source of everything else; keep.

| file | what it is |
|---|---|
| `color.mp4` | 1280 × 720 colour, 30 fps |
| `depth.mkv` | depth aligned to colour, lossless uint16, metres = value × `depth_scale` (1e-4); 11 GB of the 18.5 GB |
| `frames.csv` | per camera frame: index, camera frame number, timestamp_ms |
| `meta.json` | instruction ("put the cup on the plate"), fps, frames, dropped_frames, duration_s, image_size, camera_matrix, dist_coeffs, depth_scale |

Pipeline steps in between (small; kept so the slow models never have to run again: SAM 2 + CoTracker alone took
56 min over the dataset):

| file | what it is | made by | used by |
|---|---|---|---|
| `hand.npz` | MediaPipe right hand per camera frame: detected, score, uv [F,21,2] px, depth [F,21] m, world [F,21,3], t_ms | `replay/track_hand.py` | to_table; flow (timestamps, occlusion check) |
| `hand_table.npz` | the 21 landmarks in the table frame: p [F,21,3], valid [F,21], cam_in_table | `replay/to_table.py` | clean |
| `traj.npz` | cleaned 20 Hz hand: t, wrist / thumb / index [N,3], aperture, filled | `replay/clean.py` | retarget |
| `retarget.npz` | robot commands from the hand: t, rel [N,3] (pinch − pinch at grasp), gripper (−1/+1), i_grasp, i_release, real_grasp, real_release | `replay/retarget.py` | replay (the robot's path) |
| `scene.json` | sim scene: sim_cup_xy, sim_plate_xy, sim_grasp_point, plate_minus_cup_cm, reach checks, sim_camera | `replay/scene_match.py` | replay |
| `mug_mask.npz` | SAM 2 mug mask per camera frame (packed bits; `flow/mug_sam.py` load_masks), frame where the mug was found, t_grasp / t_release from the mug's motion | `flow/mug_sam.py` | track_points, to_flow |
| `cotracker.npz` | CoTracker per camera frame: uv [F,6,2], visible [F,6] (hand points), cup_uv [F,8,2], cup_visible [F,8] | `flow/track_points.py` | to_flow |
| `flow.npz` | flow on the real clock, 10 Hz: t [N], hand [6,N,4], cup [8,N,4] (`flow/FORMAT.md`) | `flow/to_flow.py` | sim_clock; the only flow for failed_no_robot |

Step 4.2 results, superseded by `sim.npz` for training:

| file | what it is | needed? |
|---|---|---|
| `replay.json` | replay outcome: success, tracking_error_mm, cup / hand max lift, cup_final_xy_to_plate_cm | rows of `replay_summary.json`; small |
| `replay_trace.npz` | per step: target, eef, cup, gripper, table_z, plate_xy, grasp | no (sim.npz has the same and more); 2.5 MB in total |

Videos for looking at, not used by any code (320 MB in total, can be regenerated):

| file | what it is |
|---|---|
| `hand_overlay.mp4` | colour video with the MediaPipe hand drawn |
| `flow_overlay.mp4` | colour video with the flow drawn |
| `replay.mp4` | real video next to the step 4.2 sim replay |

A flow sample (the next 1.5 s, 16 steps at 10 Hz): from `flow_sim.npz` at sim step i, `hand[:, i:i+32:2]`; from
`flow.npz` at step i, `hand[:, i:i+16]` (last row repeated past the end). Hand points: wrist, thumb tip, index tip,
index / middle / pinky knuckle; pinch point = mean of thumb and index tips.

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
python replay/record_sim.py data/mug_pnp_failed/<ep>  # sim.npz for a failed episode whose replay runs
python flow/run_flows.py                              # masks → flow → flow_sim.npz + video_sim.mp4 (step 4.4)
python data/split.py                                  # splits.json (step 4.5)
```
