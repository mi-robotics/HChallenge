"""Track the right hand in recorded demos with MediaPipe, and read D405 depth at each landmark.

  python replay/track_hand.py [data/demos/<demo> ...]     (default: all demos)

Writes to each demo folder:
  hand.npz          per frame t: detected [T], score [T], uv [T,21,2] pixels, depth [T,21] metres
                    (0 = no valid depth), world [T,21,3] MediaPipe's metric 3D hand shape (metres, around the
                    hand's centre, camera-aligned axes; used for the occlusion check in step 3.2), t_ms [T]
  hand_overlay.mp4  colour video with the tracked hand drawn on it
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "third_party" / "mediapipe" / "hand_landmarker.task"
WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def depth_frames(path, size):
    w, h = size
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE)
    try:
        while True:
            buf = p.stdout.read(w * h * 2)
            if len(buf) < w * h * 2:
                return
            yield np.frombuffer(buf, np.uint16).reshape(h, w)
    finally:  # also runs when the caller stops early
        p.kill()
        p.wait()


def depth_at(depth_m, uv, r=3):
    """Median of valid depth in a (2r+1)^2 patch; 0 if none."""
    h, w = depth_m.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < w and 0 <= v < h):
        return 0.0
    patch = depth_m[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
    patch = patch[patch > 0]
    return float(np.median(patch)) if patch.size else 0.0


def track(demo):
    demo = Path(demo)
    meta = json.loads((demo / "meta.json").read_text())
    size = tuple(meta["image_size"])
    t_ms = np.array([float(r["timestamp_ms"]) for r in csv.DictReader(open(demo / "frames.csv"))])
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))

    T = len(t_ms)
    detected, score = np.zeros(T, bool), np.zeros(T)
    uv, depth, world = np.zeros((T, 21, 2)), np.zeros((T, 21)), np.zeros((T, 21, 3))
    cap = cv2.VideoCapture(str(demo / "color.mp4"))
    w, h = size
    out = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{w}x{h}", "-r", str(meta["fps"]), "-i", "-", "-c:v", "libx264", "-crf", "23",
                            "-pix_fmt", "yuv420p", str(demo / "hand_overlay.mp4")], stdin=subprocess.PIPE)
    for t, d_raw in enumerate(depth_frames(demo / "depth.mkv", size)):
        ok, bgr = cap.read()
        if not ok or t >= T:
            break
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            int(t_ms[t] - t_ms[0]))
        # Checked on the demos: the moving (right) hand is labelled "Right" on this camera; the left
        # hand resting on the keyboard is "Left".
        rights = [(hd[0].score, lm, wl) for hd, lm, wl in zip(res.handedness, res.hand_landmarks, res.hand_world_landmarks)
                  if hd[0].category_name == "Right"]
        if rights:
            s, lm, wl = max(rights, key=lambda x: x[0])
            detected[t], score[t] = True, s
            uv[t] = [(p.x * w, p.y * h) for p in lm]
            world[t] = [(p.x, p.y, p.z) for p in wl]
            d_m = d_raw.astype(np.float32) * meta["depth_scale"]
            depth[t] = [depth_at(d_m, p) for p in uv[t]]
            for a, b in CONNECTIONS:
                cv2.line(bgr, tuple(uv[t][a].astype(int)), tuple(uv[t][b].astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
            for i in (WRIST, THUMB_TIP, INDEX_TIP):
                cv2.circle(bgr, tuple(uv[t][i].astype(int)), 6, (0, 0, 255), -1)
            cv2.putText(bgr, f"wrist depth {depth[t, WRIST] * 100:.1f} cm" if depth[t, WRIST] else "wrist depth: none",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr, "right hand not detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        out.stdin.write(bgr.tobytes())
    out.stdin.close()
    out.wait()
    landmarker.close()

    np.savez(demo / "hand.npz", detected=detected, score=score, uv=uv, depth=depth, world=world, t_ms=t_ms)
    key = depth[detected][:, [WRIST, THUMB_TIP, INDEX_TIP]]
    print(f"{demo.name}: right hand in {detected.mean() * 100:.0f}% of {T} frames | valid depth when detected: "
          f"wrist {(key[:, 0] > 0).mean() * 100:.0f}%  thumb tip {(key[:, 1] > 0).mean() * 100:.0f}%  "
          f"index tip {(key[:, 2] > 0).mean() * 100:.0f}% | wrist depth range "
          f"{key[key[:, 0] > 0, 0].min() * 100:.0f}-{key[key[:, 0] > 0, 0].max() * 100:.0f} cm")


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    for d in demos:
        track(d)
