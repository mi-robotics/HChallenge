"""Replay kept episodes in sim and save what the robot did and saw, per control step (20 Hz), to sim.npz.

  python replay/record_sim.py [--video out.mp4] [data/mug_pnp_dataset/<episode> ...]   (default: all kept episodes)

Same replay as step 4.2 (replay/replay.py, all rim-grasp constraints), with record=True. sim.npz per episode:
  action [N,7]         the OSC_POSE delta action sent (xyz in [-1, 1] = +-5 cm, rotation 0, gripper -1 open / +1 close)
  joint_pos, joint_vel [N,7], gripper_qpos [N,2]          proprioception, observed before the action
  eef_pos [N,3], eef_quat [N,4]                           end-effector pose (world frame)
  cup_pos, cup_quat, plate_pos                            object state
  agentview [N,128,128,3], eye_in_hand [N,128,128,3]      third-person (sim/scene.py THIRD_PERSON: moved back and up
                                                          so the whole workspace is in view) and gripper camera, uint8 RGB
  t_real [N]           time in the real video of the hand sample each step follows (s); the inserted settle /
                       lift-out steps and the final 1 s hold the time, so steps are not evenly spaced in t_real
  sim_minus_real [3]   offset from the table frame to the sim world (sim grasp point - real grasp point): adding it
                       to the hand flow puts the flow on the robot's path
  success, hz, sim_plate_xy
--video renders, for the episodes given, the real video next to the two sim cameras, one row per episode.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "replay"))
from replay import HZ, make_env, replay  # noqa: E402

IMG = 128


def render(episodes, out_path, cell=256):
    rows = []
    for k, ep in enumerate(episodes):
        ep = Path(ep)
        sim = np.load(ep / "sim.npz")
        t_cam = np.loadtxt(ep / "frames.csv", delimiter=",", skiprows=1)[:, 2]
        t_cam = (t_cam - t_cam[0]) / 1000
        cap, real = cv2.VideoCapture(str(ep / "color.mp4")), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            real.append(cv2.resize(f, (cell * 16 // 9 // 2 * 2, cell)))  # even width for H.264
        frames = []
        for i in range(len(sim["action"])):
            r = real[min(int(np.searchsorted(t_cam, sim["t_real"][i])), len(real) - 1)].copy()
            views = [cv2.resize(np.ascontiguousarray(sim[v][i][:, :, ::-1]), (cell, cell), interpolation=cv2.INTER_NEAREST)
                     for v in ("agentview", "eye_in_hand")]
            for img, label in zip([r] + views, [f"episode {k + 1}: real", "sim: third person", "sim: gripper camera"]):
                cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(np.hstack([r] + views))
        rows.append(frames)
    n = max(len(f) for f in rows)
    h, w = rows[0][0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{w}x{h * len(rows)}", "-r", str(HZ), "-i", "-", "-c:v", "libx264", "-crf", "24",
                           "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(np.vstack([f[min(i, len(f) - 1)] for f in rows]).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    video = None
    if "--video" in args:
        i = args.index("--video")
        video = Path(args[i + 1])
        del args[i:i + 2]
    episodes = args or sorted(str(p) for p in (ROOT / "data" / "mug_pnp_dataset").iterdir() if p.is_dir())
    env = make_env(img_size=IMG)
    results = [replay(ep, env, record=True) for ep in episodes]
    env.close()
    print(f"success {sum(r['success'] for r in results)}/{len(results)}")
    if video:
        video.parent.mkdir(parents=True, exist_ok=True)
        render(episodes, video)
