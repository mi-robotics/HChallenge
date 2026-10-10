"""Put an episode's flow and real video on the robot's clock: one row / frame per sim.npz control step (20 Hz), so
video, flow and robot data have the same length and pause together.

  python flow/sim_clock.py [<episode> ...]     (default: every episode in mug_pnp_dataset and mug_pnp_failed with sim.npz)

Each sim step follows the real video at time t_real (sim.npz). The flow points at that step are flow.npz interpolated
at t_real: during the steps the replay inserts (settle and close at the grasp, hold and rise after the release, the
final 1 s wait) t_real holds, so the points stand still while the robot moves. Writes flow_sim.npz:
  t     [N]          sim time, step / 20 Hz (N = len(sim.npz action))
  t_real [N]         copied from sim.npz
  frame [N]          color.mp4 frame shown at each step (nearest camera frame to t_real)
  hand  [6, N, 4], cup [8, N, 4]   table frame, as flow.npz (add sim.npz sim_minus_real for the sim world)
Sample at sim step i (FORMAT.md, 16 steps at 10 Hz of robot time) = hand[:, i:i+32:2], last row repeated past the end.
Also writes video_sim.mp4: color.mp4 frames[frame], 1280 x 720 at 20 fps (one video frame per sim step).
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM_HZ = 20


def at_times(t, pts, tq):
    """pts [P, N, 4] sampled at t -> [P, len(tq), 4] at tq: xyz linear, visible from the nearest flow step."""
    xyz = np.stack([[np.interp(tq, t, p[:, k]) for k in range(3)] for p in pts]).transpose(0, 2, 1)
    near = np.clip(np.round(np.interp(tq, t, np.arange(len(t)))).astype(int), 0, len(t) - 1)
    return np.concatenate([xyz, pts[:, near, 3:]], axis=2).astype(np.float32)


def to_sim_clock(ep):
    ep = Path(ep)
    f, sim = np.load(ep / "flow.npz"), np.load(ep / "sim.npz")
    tq = sim["t_real"]
    t_cam = np.loadtxt(ep / "frames.csv", delimiter=",", skiprows=1)[:, 2]
    t_cam = (t_cam - t_cam[0]) / 1000
    frame = np.clip(np.round(np.interp(tq, t_cam, np.arange(len(t_cam)))).astype(int), 0, len(t_cam) - 1)
    out = dict(t=(np.arange(len(tq)) / SIM_HZ).astype(np.float32), t_real=tq.astype(np.float32), frame=frame,
               hand=at_times(f["t"], f["hand"], tq), cup=at_times(f["t"], f["cup"], tq))
    np.savez(ep / "flow_sim.npz", **out)
    write_video(ep, frame)
    return out


def write_video(ep, frame):
    cap, frames = cv2.VideoCapture(str(ep / "color.mp4")), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    frame = np.minimum(frame, len(frames) - 1)  # color.mp4 can hold one frame fewer than frames.csv
    h, w = frames[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(SIM_HZ), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                           str(ep / "video_sim.mp4")], stdin=subprocess.PIPE)
    for i in frame:
        ff.stdin.write(frames[i].tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    eps = sys.argv[1:] or sorted(p for d in ("mug_pnp_dataset", "mug_pnp_failed") for p in (ROOT / "data" / d).iterdir()
                                 if (p / "sim.npz").exists() and (p / "flow.npz").exists())
    for ep in eps:
        o = to_sim_clock(ep)
        print(f"{Path(ep).name}: {len(o['t'])} steps ({o['t'][-1] + 1 / SIM_HZ:.2f} s), "
              f"{int((np.diff(o['t_real']) == 0).sum())} held")
