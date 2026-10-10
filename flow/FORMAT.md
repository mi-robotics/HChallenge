# Action flow format

Action flow is trails of tracked points: where each point is now and where it goes over the next T steps.
Every script and model in Phases 3–6 uses this format. The constants live in `flow/format.py`; import them,
don't retype them.

## One sample

```
hand  float32 [6, 16, 4]     points × timesteps × (x, y, z, visible)
cup   float32 [8, 16, 4]
```

- **Timesteps:** T = 16 at 10 Hz. Step 0 is the current time t0, step k is t0 + k / 10 s, so a sample covers
  1.5 s ahead. In the 5 demos the hand reaches the cup in 1.2–5.0 s and carries it for 3.3–7.3 s; at the
  fastest hand speed (p95 0.39 m/s) one step is about 4 cm.
- **x, y, z:** metres in the table frame (ArUco marker at the origin, x away from the camera, y left, z up;
  see `calibration/`). Absolute positions, not offsets: any re-centring is the model's job.
- **visible:** 1 if the point was tracked in that frame *and* the D405 had depth there, else 0. Hidden points
  are never guessed: their x, y, z are filled only so the array has no NaNs (hand: linear interpolation
  from the nearest visible frames, held flat at the ends; mug: moved with the mug's visible points), and losses
  must mask them with `visible`.
- **Past the end of an episode** the last frame is repeated (same x, y, z, visible).

## Points

Hand, in this order (MediaPipe landmark index in brackets):

| # | Point | Why |
|---|---|---|
| 0 | wrist [0] | hand position |
| 1 | thumb tip [4] | thumb–index distance is the grasp signal (close < 2.5 cm, open > 3.0 cm, step 1.7) |
| 2 | index tip [8] | midpoint of 1 and 2 = pinch point, which drives the gripper |
| 3 | index knuckle [5] | with 0, 4, 5: four points on the back of the hand, nearly rigid, so together they give the hand orientation |
| 4 | middle knuckle [9] | |
| 5 | pinky knuckle [17] | |

Cup: 8 points spread over the SAM 2 mug mask in the first frame (`flow/mug_sam.py`, farthest-point sampling; steps 3.2, 3.4). Image position: CoTracker; hidden when outside that frame's mug mask. Depth: the D405 has none on the white mug, so it comes from where the point's camera ray meets a 4 cm-radius cylinder that moves with the tracked pinch point, capped at the rim (`flow/to_flow.py`; checked against PnP in `flow/mug_check.py`).

## One episode

`flow.npz` in the demo folder (`data/demos/<timestamp>/`):

```
t     float32 [N]           seconds since the first camera frame, 10 Hz
hand  float32 [6, N, 4]
cup   float32 [8, N, 4]
```

Sample at step i = `hand[:, i:i+16]` (padded as above). Slicing is step 3.6.

## One episode on the robot's clock (kept and replayed failed episodes)

`flow_sim.npz` (`flow/sim_clock.py`, step 4.4): the same points with one row per `sim.npz` control step (20 Hz),
so flow and robot data have the same length. Also written for the failed episodes whose replay ran (step 4.5). Each row is `flow.npz` at the real-video time that sim step follows
(`t_real`); during the steps the replay inserts (settle and close at the grasp, hold and rise after the release,
the final 1 s wait; 55 steps per episode) the points stand still.

```
t       float32 [N_sim]          sim time, step / 20
t_real  float32 [N_sim]          real-video time followed (from sim.npz)
frame   int     [N_sim]          color.mp4 frame shown at each step (nearest camera frame to t_real)
hand    float32 [6, N_sim, 4]    table frame
cup     float32 [8, N_sim, 4]
```

Sample at sim step i = `hand[:, i:i+32:2]` (16 steps at 10 Hz of robot time, padded as above).
`video_sim.mp4` next to it is the real video on the same clock: video frame i = `color.mp4` frame `frame[i]`.

## Table frame → sim

Table and sim axes are aligned, so it is a translation, fixed per episode by the grasp:
`p_sim = p_table + sim_grasp_point - real_grasp`, with `sim_grasp_point` from `scene.json` and
`real_grasp` from `retarget.npz` (step 1.8).
