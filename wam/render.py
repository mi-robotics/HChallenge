"""Draw a flow sample [16, 14, 4] (table frame) on 256 x 448 frames with the episode's marker pose; the end-to-end
videos (step 6.3)."""
import json
import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT / "replay"))
from to_table import demo_marker_pose  # noqa: E402
from track_points import HAND_COLOUR, YELLOW  # noqa: E402
from visualise import draw  # noqa: E402


def camera(ep):
    meta = json.loads((ep / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, t, _ = demo_marker_pose(ep, K, dist)
    return K, dist, cv2.Rodrigues(R)[0], t, meta["image_size"]


def overlay(frame, flow, cam):
    """frame [H, W, 3] RGB uint8, flow [16, 14, 4] -> a copy with the 14 trails drawn (hand on top)."""
    K, dist, rvec, t, (w0, h0) = cam
    h, w = frame.shape[:2]
    pts = flow.transpose(1, 0, 2)  # [14, 16, 4]
    uv = cv2.projectPoints(pts[..., :3].reshape(-1, 3).astype(np.float64), rvec, t, K, dist)[0].reshape(14, -1, 2)
    img = np.ascontiguousarray(frame[..., ::-1])
    draw(img, uv * [w / w0, h / h0], pts[..., 3] > 0, HAND_COLOUR + [YELLOW] * 8, 6, scale=0.35)
    return np.ascontiguousarray(img[..., ::-1])


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
                          stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
    ff.stdin.close()
    ff.wait()


def write_panels(ep, plan, sim_frames, path):
    """One video per end-to-end episode at the robot's 20 Hz: real video + recorded flow | generated video + planned
    flow (wam/plan.py) | robot wrist camera | robot third-person camera. sim_frames: (third-person, wrist) per step.
    Each video shows the 1.6 s flow chunk its frame belongs to (the real video holds its last frame at the end)."""
    sys.path.insert(0, str(ROOT / "wam"))
    from data import H, read_flow, read_frames
    cam = camera(ep)
    real, real_flow = read_frames(ep / "video_sim.mp4"), read_flow(ep)
    gen, gen_flow = read_frames(plan / "plans" / f"{ep.name}.mp4"), np.load(plan / "plans" / f"{ep.name}.npz")["flow"]
    out = []
    for i, (third, wrist) in enumerate(sim_frames):
        j = i // 2  # 10 Hz frame of the plan; chunk c covers frames 16 c + 1 ... 16 c + 16 (frame 0 = the real image)
        c = min(max(j - 1, 0) // 16, len(gen_flow) // 16 - 1)
        g = overlay(gen[min(j, len(gen) - 1)], gen_flow[16 * c:16 * c + 16], cam)
        r = min(i, len(real) - 1)
        rows = np.minimum(32 * (r // 32) + np.arange(0, 32, 2), len(real_flow) - 1)
        a = overlay(real[r], real_flow[rows], cam)
        panels = [a, g, cv2.resize(wrist, (H, H)), cv2.resize(third, (H, H))]
        for img, text in zip(panels, ("real + recorded flow", "generated + planned flow", "robot wrist", "robot")):
            cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panels[0], f"t = {i / 20:.1f} s", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        out.append(np.concatenate(panels, 1))
    write_mp4(path, out, 20)
