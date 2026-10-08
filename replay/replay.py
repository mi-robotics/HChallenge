"""Replay each demo in sim and record a side-by-side video (real | sim from the D405's pose).

  python replay/replay.py [--raw | --only approach,settle,lift] [data/demos/<demo> ...]     (default: all demos)

Per demo: matched scene (step 1.8), gripper moved to the first target (not part of the replay), then one
control step per 20 Hz sample: end-effector target = pinch trajectory anchored at the sim rim-grasp point,
gripper open/close from the hand (step 1.7). After the last sample the sim runs 1 s more so the cup settles,
then LIBERO's On(cup, plate) check decides success.

Rim-grasp constraints (default: all three; --raw replays the hand path as is; --only picks a subset for ablations). The robot grasps with one finger inside
the mug, while I pinch it from the side, so following the hand exactly knocks the mug over on approach and
drags it off the plate after release. Three local changes, the hand path is kept everywhere else:
  approach  before the grasp, the target stays >= 8 cm above the grasp point until the gripper itself (not the
            target: the arm lags fast hand motion) is within 3 cm of it horizontally, then descends
  settle    at the grasp point: 0.5 s with fingers open (arrive), then 0.5 s closing (grip), then follow the hand
  lift out  after release, hold 0.25 s, then rise 10 cm straight up before following the hand again

Writes replay<tag>.mp4 + replay<tag>.json + replay_trace<tag>.npz (per-step target, gripper, cup) to each demo folder
(tag: "" all constraints, "_raw" none, "_<subset>" otherwise), and media/phase1/task9/replay.mp4 / replay_raw.mp4 (all demos tiled).
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "replay"))
from scene import CLOSE, CUP, OPEN, make_env, move_to, place, render, reset, set_camera, step_towards  # noqa: E402

MEDIA = ROOT / "media" / "phase1" / "task9"
HZ = 20
W, H = 640, 360
CONSTRAINTS = ("approach", "settle", "lift")
CLEAR_M, NEAR_M, ARRIVE_S, GRIP_S, LIFT_M, OPEN_HOLD_S = 0.08, 0.03, 0.5, 0.5, 0.10, 0.25


def approach_clearance(eef, grasp):
    """Minimum target height before the grasp: CLEAR_M above the grasp point while the gripper is >= 2 * NEAR_M
    away horizontally, falling linearly to 0 at NEAR_M."""
    d_xy = np.linalg.norm(eef[:2] - grasp[:2])
    return grasp[2] + CLEAR_M * np.clip((d_xy - NEAR_M) / NEAR_M, 0, 1)


def rim_grasp_constraints(targets, gripper, i_grasp, i_release, use=CONSTRAINTS):
    """Returns new (targets, gripper, source index per step, approach mask) with the settle / lift-out changes.
    The approach clearance depends on where the gripper actually is, so it is applied in the control loop."""
    g = targets[i_grasp]
    tg = targets.copy()
    # Lift out: after release, stay above the release point until LIFT_M higher, then rejoin the hand path.
    r = tg[i_release].copy()
    lift_steps = int(LIFT_M / 0.01) if "lift" in use else 0  # 1 cm per step
    lift = np.array([r + [0, 0, LIFT_M * (k + 1) / lift_steps] for k in range(lift_steps)]).reshape(-1, 3)
    after = tg[i_release:].copy()
    if "lift" in use:
        after[:, 2] = np.maximum(after[:, 2], r[2] + LIFT_M)
    n_arrive, n_grip = (int(ARRIVE_S * HZ), int(GRIP_S * HZ)) if "settle" in use else (0, 0)
    n_hold = int(OPEN_HOLD_S * HZ) if "lift" in use else 0
    n_settle = n_arrive + n_grip
    src = np.r_[np.arange(i_grasp), [i_grasp] * n_settle, np.arange(i_grasp, i_release), [i_release] * (n_hold + lift_steps),
                np.arange(i_release, len(tg))].astype(int)
    new_t = np.concatenate([tg[:i_grasp], np.reshape([g] * n_settle, (-1, 3)), tg[i_grasp:i_release],
                            np.reshape([r] * n_hold, (-1, 3)), lift, after])
    new_g = np.concatenate([gripper[:i_grasp], [OPEN] * n_arrive, [CLOSE] * n_grip, gripper[i_grasp:i_release],
                            [gripper[i_release]] * (n_hold + lift_steps), gripper[i_release:]])
    approach = (np.arange(len(new_t)) < i_grasp) & ("approach" in use)
    return new_t, new_g, src, approach


def tag(use):
    return "" if set(use) == set(CONSTRAINTS) else "_raw" if not use else "_" + "+".join(use)


def replay(demo, env, use=CONSTRAINTS):
    demo = Path(demo)
    rt = np.load(demo / "retarget.npz")
    sc = json.loads((demo / "scene.json").read_text())

    reset(env)
    obs = place(env, sc["sim_cup_xy"], sc["sim_plate_xy"])
    targets = rt["rel"] + np.array(sc["sim_grasp_point"])
    gripper = rt["gripper"]
    i_grasp, i_release = int(rt["i_grasp"]), int(rt["i_release"])
    grasp_point = targets[i_grasp].copy()
    src = np.arange(len(targets))  # which hand sample (and real frame) each sim step corresponds to
    approach = np.zeros(len(targets), bool)
    if use:
        targets, gripper, src, approach = rim_grasp_constraints(targets, gripper, i_grasp, i_release, use)
    cam = sc["sim_camera"]
    set_camera(env, cam["name"], np.array(cam["pos"]), np.array(cam["R_world_from_cv"]), cam["fovy_deg"])
    table_z = obs[f"{CUP}_pos"][2]

    start = targets[0].copy()
    if approach[0]:
        start[2] = max(start[2], grasp_point[2] + CLEAR_M)
    obs, pre_steps, _ = move_to(env, obs, start, gripper=OPEN, max_steps=150)
    sim_frames, err, cup_z, tgts, eefs, cups = [], [], [], [], [], []
    for k in range(len(targets)):
        tgt = targets[k].copy()
        if approach[k]:
            tgt[2] = max(tgt[2], approach_clearance(obs["robot0_eef_pos"], grasp_point))
        obs, *_ = env.step(step_towards(obs, tgt, gripper[k]))
        err.append(np.linalg.norm(obs["robot0_eef_pos"] - tgt))
        tgts.append(tgt)
        eefs.append(obs["robot0_eef_pos"].copy())
        cups.append(obs[f"{CUP}_pos"].copy())
        cup_z.append(obs[f"{CUP}_pos"][2] - table_z)
        sim_frames.append(render(env, cam["name"], W, H))
    for _ in range(HZ):  # let the cup settle
        obs, *_ = env.step(np.r_[np.zeros(6), gripper[-1]])
        sim_frames.append(render(env, cam["name"], W, H))
    success = bool(env.check_success())

    err, cup_z = np.array(err), np.array(cup_z)
    held = slice(i_grasp, i_release)
    result = {
        "constraints": list(use),
        "success": success,
        "pre_roll_steps": int(pre_steps),
        "tracking_error_mm": {"median": float(np.median(err) * 1000), "p95": float(np.percentile(err, 95) * 1000),
                              "max": float(err.max() * 1000)},
        "cup_max_lift_cm": float(cup_z.max() * 100),
        "hand_max_lift_cm": float((rt["rel"][held, 2].max()) * 100),
        "cup_final_xy_to_plate_cm": float(np.linalg.norm(obs[f"{CUP}_pos"][:2] - np.array(sc["sim_plate_xy"])) * 100),
    }
    (demo / f"replay{tag(use)}.json").write_text(json.dumps(result, indent=2))
    np.savez(demo / f"replay_trace{tag(use)}.npz", target=np.array(tgts), eef=np.array(eefs), cup=np.array(cups),
             gripper=gripper, table_z=table_z, plate_xy=np.array(sc["sim_plate_xy"]), grasp=grasp_point)
    print(f"{demo.name}: {'SUCCESS' if success else 'FAIL'} | tracking error median {result['tracking_error_mm']['median']:.0f} "
          f"p95 {result['tracking_error_mm']['p95']:.0f} max {result['tracking_error_mm']['max']:.0f} mm | cup lifted "
          f"{result['cup_max_lift_cm']:.1f} cm (hand {result['hand_max_lift_cm']:.1f}) | cup ends "
          f"{result['cup_final_xy_to_plate_cm']:.1f} cm from plate centre")
    write_side_by_side(demo, rt["t"][src], sim_frames, success, f"replay{tag(use)}.mp4")
    return result


def write_side_by_side(demo, t, sim_frames, success, name):
    """Real frame for each sim step's hand sample | sim frame, at 20 fps. (During the inserted settle / lift-out
    steps the real frame holds.)"""
    cap = cv2.VideoCapture(str(demo / "color.mp4"))
    real = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        real.append(cv2.resize(f, (W, H)))
    hand = np.load(demo / "hand.npz")
    t_cam = (hand["t_ms"] - hand["t_ms"][0]) / 1000
    n_extra = len(sim_frames) - len(t)
    t_all = np.r_[t, np.full(n_extra, t[-1])]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{2 * W}x{H}", "-r", str(HZ), "-i", "-", "-c:v", "libx264", "-crf", "23",
                           "-pix_fmt", "yuv420p", str(demo / name)], stdin=subprocess.PIPE)
    for k, (tk, s) in enumerate(zip(t_all, sim_frames)):
        r = real[min(int(np.searchsorted(t_cam, tk)), len(real) - 1)].copy()
        s = np.ascontiguousarray(s[:, :, ::-1])
        cv2.putText(r, "real", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(s, "sim replay", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if k >= len(t):
            label, colour = ("cup on plate", (60, 200, 60)) if success else ("cup not on plate", (40, 40, 230))
            cv2.putText(s, label, (12, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        ff.stdin.write(np.hstack([r, s]).tobytes())
    ff.stdin.close()
    ff.wait()


def tile(demos, name, out_path):
    """Side-by-side videos stacked as a 2-column grid of (real | sim) pairs."""
    n = len(demos)
    tw, th = 960, 270
    f = "".join(f"[{i}:v]scale={tw}:{th},drawtext=text='demo {i + 1}':x=10:y={th - 30}:fontsize=20:fontcolor=white:"
                f"box=1:boxcolor=black@0.5[v{i}];" for i in range(n))
    cells = [f"{(i % 2) * tw}_{(i // 2) * th}" for i in range(n + 1)]
    duration = max(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                         str(Path(d) / name)], capture_output=True, text=True).stdout) for d in demos)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for d in demos:
        cmd += ["-i", str(Path(d) / name)]
    cmd += ["-f", "lavfi", "-i", f"color=c=black:s={tw}x{th}:r={HZ}", "-filter_complex",
            f"{f}[{n}:v]trim=duration=15[v{n}];{''.join(f'[v{i}]' for i in range(n + 1))}"
            f"xstack=inputs={n + 1}:layout={'|'.join(cells)}:shortest=0,trim=duration={duration}[out]",
            "-map", "[out]", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", str(out_path)]
    subprocess.run(cmd, check=True)
    print("wrote", out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    use = CONSTRAINTS
    if "--raw" in args:
        use = ()
        args.remove("--raw")
    if "--only" in args:
        i = args.index("--only")
        use = tuple(c for c in CONSTRAINTS if c in args[i + 1].split(","))
        del args[i:i + 2]
    demos = args or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    env = make_env(img_size=64)
    results = [replay(d, env, use) for d in demos]
    env.close()
    print(f"constraints {'+'.join(use) or 'none'}: success {sum(r['success'] for r in results)}/{len(results)}")
    if tag(use) in ("", "_raw"):  # ablation videos stay in the demo folders
        MEDIA.mkdir(parents=True, exist_ok=True)
        tile(demos, f"replay{tag(use)}.mp4", MEDIA / f"replay{tag(use)}.mp4")
