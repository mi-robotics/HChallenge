"""Draw a flow sample [16, 14, 4] (table frame) on 256 x 448 frames with the episode's marker pose."""
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
