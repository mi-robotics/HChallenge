"""Track the flow's hand points through each demo video with CoTracker3 (offline model).

  python flow/track_points.py [data/demos/<demo> ...]     (default: all demos)

Query points: the 6 hand points of flow/FORMAT.md, taken from the MediaPipe landmarks (step 1.4) every QUERY_EVERY
frames where MediaPipe found the hand, each tracked forwards and backwards through the whole video. In each frame,
a point's position and visibility come from its query nearest in time. (With one query frame per demo, the pinky
knuckle, hidden behind the hand in 70-80% of frames in demos 2 and 4, drifted up to 287 px.)
Check: pixel distance to the MediaPipe landmark in frames >= 4 from a query frame where MediaPipe found the hand.

Writes cotracker.npz to each demo folder (uv [T, 6, 2] pixels, visible [T, 6], source query frame [T]) and
media/phase3/task2/tracks.mp4 (all demos tiled: CoTracker dots, MediaPipe rings).
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
from format import HAND_MEDIAPIPE, HAND_POINTS  # noqa: E402
from cotracker.predictor import CoTrackerPredictor  # noqa: E402

CHECKPOINT = ROOT / "third_party" / "cotracker_ckpt" / "scaled_offline.pth"
MEDIA = ROOT / "media" / "phase3" / "task2"
QUERY_EVERY = 15  # frames (0.5 s)


def read_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return np.stack(frames)
        frames.append(f)


def track(demo, model):
    demo = Path(demo)
    hand = np.load(demo / "hand.npz")
    frames = read_video(demo / "color.mp4")
    detected = np.flatnonzero(hand["detected"])
    qf = np.unique(detected[np.abs(detected[None] - np.arange(0, len(frames), QUERY_EVERY)[:, None]).argmin(1)])
    P = len(HAND_MEDIAPIPE)
    uv_q = hand["uv"][qf][:, list(HAND_MEDIAPIPE)].reshape(-1, 2)  # [Q * P, 2], query-major
    queries = torch.tensor(np.c_[np.repeat(qf, P), uv_q], dtype=torch.float32, device="cuda")[None]
    video = torch.from_numpy(frames[..., ::-1].copy()).permute(0, 3, 1, 2)[None].float().cuda()  # B T 3 H W, RGB
    with torch.no_grad():
        tracks, vis = model(video, queries=queries, backward_tracking=True)
    tracks = tracks[0].cpu().numpy().reshape(len(frames), len(qf), P, 2)
    vis = vis[0].cpu().numpy().reshape(len(frames), len(qf), P) > 0.5
    src = np.abs(np.arange(len(frames))[:, None] - qf[None]).argmin(1)  # nearest query per frame
    uv, visible = tracks[np.arange(len(frames)), src], vis[np.arange(len(frames)), src]
    np.savez(demo / "cotracker.npz", uv=uv, visible=visible, query_frame=qf[src])

    # Check against MediaPipe away from the query frames (at a query frame the two agree by construction).
    mp = hand["uv"][:, list(HAND_MEDIAPIPE)]
    away = hand["detected"] & (np.abs(np.arange(len(frames)) - qf[src]) >= 4)
    err = np.linalg.norm(uv - mp, axis=2)[away]  # [frames, 6]
    print(f"{demo.name}: {len(qf)} query frames | visible {visible.mean() * 100:.0f}% | px from MediaPipe, frames >= 4 from "
          f"a query, median / p95 / max: " + ", ".join(f"{n} {np.median(e):.0f}/{np.percentile(e, 95):.0f}/{e.max():.0f}"
                                                       for n, e in zip(HAND_POINTS, err.T)))
    return frames, uv, visible, mp, hand["detected"], qf


def overlay(demo, frames, uv, visible, mp, detected, q):
    """Write cotracker_overlay.mp4: CoTracker dots (hollow when not visible), MediaPipe rings."""
    meta = json.loads((Path(demo) / "meta.json").read_text())
    h, w = frames.shape[1:3]
    out = Path(demo) / "cotracker_overlay.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(meta["fps"]), "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                           str(out)], stdin=subprocess.PIPE)
    # BGR of dataviz slots 1-3: back of hand (wrist + knuckles), thumb tip, index tip
    colour = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (214, 120, 42), (214, 120, 42), (214, 120, 42)]
    for i, img in enumerate(frames):
        img = img.copy()
        for j, c in enumerate(colour):
            if detected[i]:
                cv2.circle(img, tuple(mp[i, j].astype(int)), 13, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(uv[i, j].astype(int)), 7, c, -1 if visible[i, j] else 2, cv2.LINE_AA)
        label = f"frame {i}" + (" (query)" if i in q else "")
        cv2.putText(img, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    return out


def tile(videos, out_path, seconds):
    tw, th = 640, 360
    n = len(videos)
    f = "".join(f"[{i}:v]scale={tw}:{th},drawtext=text='demo {i + 1}':x=10:y={th - 30}:fontsize=20:fontcolor=white:"
                f"box=1:boxcolor=black@0.5[v{i}];" for i in range(n))
    cells = [f"{(i % 3) * tw}_{(i // 3) * th}" for i in range(n + 1)]
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for v in videos:
        cmd += ["-i", str(v)]
    cmd += ["-f", "lavfi", "-i", f"color=c=white:s={tw}x{th}:r=30:d={seconds}", "-filter_complex",
            f"{f}[{n}:v]null[v{n}];{''.join(f'[v{i}]' for i in range(n + 1))}"
            f"xstack=inputs={n + 1}:layout={'|'.join(cells)}:shortest=0[out]",
            "-map", "[out]", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", str(out_path)]
    subprocess.run(cmd, check=True)
    print("wrote", out_path)


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    model = CoTrackerPredictor(checkpoint=str(CHECKPOINT), offline=True, window_len=60).cuda()
    results = [track(d, model) for d in demos]
    videos = [overlay(d, *r) for d, r in zip(demos, results)]
    MEDIA.mkdir(parents=True, exist_ok=True)
    tile(videos, MEDIA / "tracks.mp4", max(len(r[0]) for r in results) / 30)
