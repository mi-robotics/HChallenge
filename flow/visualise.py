"""Draw the action flow on the demo video: at every frame, the flow sample that starts there (the next 16 steps at
10 Hz, 1.5 s) for the 6 hand points and the 8 mug points, projected into the image with the marker pose.

  python flow/visualise.py [data/demos/<demo> ...]     (default: all demos)

Each trail is one point's future path; dots are the 16 steps, the white ring is step 0 (now). Steps the flow marks
hidden (visible = 0, position interpolated) are drawn red.

Writes flow_overlay.mp4 to each demo folder and media/phase3/task5/flow.mp4 (all demos tiled, with a legend).
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT / "replay"))
from format import HZ, T  # noqa: E402
from to_table import demo_marker_pose  # noqa: E402
from track_points import BLUE, GREEN, HAND_COLOUR, ORANGE, RED, YELLOW, read_video, tile  # noqa: E402

MEDIA = ROOT / "media" / "phase3" / "task5"
LEGEND = [(BLUE, "wrist + knuckles", "dot"), (ORANGE, "thumb tip", "dot"), (GREEN, "index tip", "dot"),
          (YELLOW, "mug", "dot"), (RED, "hidden (interpolated)", "dot"), ((90, 90, 90), f"now; trail = next {T} steps", "ring")]


def visualise(demo):
    demo = Path(demo)
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, tv, _ = demo_marker_pose(demo, K, dist)
    rvec = cv2.Rodrigues(R)[0]
    flow, hand_mp = np.load(demo / "flow.npz"), np.load(demo / "hand.npz")
    pts = np.concatenate([flow["hand"], flow["cup"]])  # [14, N, 4]
    colours = HAND_COLOUR + [YELLOW] * len(flow["cup"])
    N = pts.shape[1]
    uv = cv2.projectPoints(pts[..., :3].reshape(-1, 3).astype(np.float64), rvec, tv, K, dist)[0].reshape(len(pts), N, 2)
    frames = read_video(demo / "color.mp4")
    t_cam = (hand_mp["t_ms"][:len(frames)] - hand_mp["t_ms"][0]) / 1000

    h, w = frames.shape[1:3]
    out = demo / "flow_overlay.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(meta["fps"]), "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                           str(out)], stdin=subprocess.PIPE)
    for i, img in enumerate(frames):
        img = img.copy()
        s = min(int(round(t_cam[i] * HZ)), N - 1)
        steps = np.minimum(np.arange(s, s + T), N - 1)  # past the end: last frame repeated (FORMAT.md)
        for p in list(range(len(flow["hand"]), len(pts))) + list(range(len(flow["hand"]))):  # hand on top
            q = uv[p, steps].astype(int)
            vis = pts[p, steps, 3] > 0
            for k in range(T - 1):
                cv2.line(img, tuple(q[k]), tuple(q[k + 1]), colours[p] if vis[k + 1] else RED, 3, cv2.LINE_AA)
            for k in range(T):
                cv2.circle(img, tuple(q[k]), 5, colours[p] if vis[k] else RED, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(q[0]), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"t = {t_cam[i]:.1f} s", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    return out, len(frames)


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    results = [visualise(d) for d in demos]
    MEDIA.mkdir(parents=True, exist_ok=True)
    tile([r[0] for r in results], MEDIA / "flow.mp4", max(r[1] for r in results) / 30, LEGEND)
