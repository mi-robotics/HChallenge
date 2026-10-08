"""Track the flow's hand and mug points through each demo video with CoTracker3 (offline model), and flag occlusions.

  python flow/track_points.py [data/demos/<demo> ...]     (default: all demos)

Hand: the 6 hand points of flow/FORMAT.md, queried from the MediaPipe landmarks (step 1.4) every QUERY_EVERY frames
where MediaPipe found the hand, each tracked forwards and backwards through the whole video. In each frame, a point
comes from its query nearest in time. (With one query frame per demo, the pinky knuckle, hidden behind the hand in
70-80% of frames in demos 2 and 4, drifted up to 287 px.)
Check: pixel distance to the MediaPipe landmark in frames >= 4 from a query frame where MediaPipe found the hand.

Mug: 8 points spread over the SAM 2 mug mask in the first frame (flow/mug_sam.py, run it first), tracked from there.
(Before: 8 points on a cylinder placed from the hand, re-queried in the last frame; in demo 4 one started on the
mug's lower edge and stayed on the table. Comparison: flow/mug_compare.py.)

Occlusion. CoTracker's visibility says whether the surface at the tracked pixel is visible, not whether the point is:
a query on a hidden pinky knuckle starts on whatever is in front of it. So a point is also occluded if the D405 depth
at its pixel disagrees with where the point should be:
  hand  expected position: MediaPipe's metric 3D hand shape (world landmarks) fitted to the D405 points of its 21
        landmarks with a RANSAC similarity fit (rotation, translation, scale; inliers within 1.5 cm). Occluded if the
        D405 sees something > IN_FRONT_M in front of it, > BEHIND_M behind it (the pixel is off the hand), or has
        no depth there, or it is within EDGE_PX of the image border (demo 5: the hand leaves the image and the tracks
        sit on the border). Frames without MediaPipe or with < 8 landmarks with depth: CoTracker's visibility and
        the border check only.
        (Using MediaPipe's z directly, without the fit, gave per-landmark biases of -6 to +5 cm: the wrist was
        "occluded" in 92-100% of frames.)
  mug   hidden unless CoTracker calls it visible and it lies inside that frame's SAM 2 mug mask (dilated by
        MASK_TOL_PX): the mask leaves out the hand in front of the mug and anything off the mug.

Writes cotracker.npz to each demo folder (uv [T, 6, 2], visible [T, 6], cup_uv [T, 8, 2], cup_visible [T, 8];
visible = not occluded) and media/phase3/task2/tracks.mp4 (all demos tiled, with a legend).
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
sys.path.insert(0, str(ROOT / "replay"))
from format import HAND_MEDIAPIPE, HAND_POINTS  # noqa: E402
from mug_sam import load_masks, mask_points  # noqa: E402
from track_hand import depth_at, depth_frames  # noqa: E402
from cotracker.predictor import CoTrackerPredictor  # noqa: E402

CHECKPOINT = ROOT / "third_party" / "cotracker_ckpt" / "scaled_offline.pth"
MEDIA = ROOT / "media" / "phase3" / "task2"
QUERY_EVERY = 15  # frames (0.5 s)
IN_FRONT_M, BEHIND_M = 0.03, 0.05
FIT_INLIER_M = 0.015
MASK_TOL_PX = 4
EDGE_PX = 10
# BGR of dataviz slots 1-4: back of hand (wrist + knuckles), thumb tip, index tip, mug; red = occluded
BLUE, ORANGE, GREEN, YELLOW, RED = (214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (40, 40, 220)
HAND_COLOUR = [BLUE, ORANGE, GREEN, BLUE, BLUE, BLUE]


def read_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return np.stack(frames)
        frames.append(f)


def run_cotracker(model, frames, qf, uv_q):
    """qf [Q] query frames, uv_q [Q, P, 2] -> tracks [T, Q, P, 2], visible [T, Q, P]."""
    Q, P = uv_q.shape[:2]
    queries = torch.tensor(np.c_[np.repeat(qf, P), uv_q.reshape(-1, 2)], dtype=torch.float32, device="cuda")[None]
    video = torch.from_numpy(frames[..., ::-1].copy()).permute(0, 3, 1, 2)[None].float().cuda()  # B T 3 H W, RGB
    with torch.no_grad():
        tracks, vis = model(video, queries=queries, backward_tracking=True)
    return tracks[0].cpu().numpy().reshape(-1, Q, P, 2), vis[0].cpu().numpy().reshape(-1, Q, P) > 0.5


def similarity(a, b):
    """s, R, t minimising |s R a + t - b| (Umeyama)."""
    ca, cb = a.mean(0), b.mean(0)
    u, sv, vt = np.linalg.svd((b - cb).T @ (a - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    R = u @ D @ vt
    s = (sv * np.diag(D)).sum() / ((a - ca) ** 2).sum()
    return s, R, cb - s * R @ ca


def expected_hand(world, measured, rng, iters=60):
    """Fit MediaPipe's hand shape (world [21, 3]) to D405 points (measured [21, 3], NaN where no depth) with RANSAC;
    returns the fitted [21, 3] in the camera frame."""
    ok = np.isfinite(measured[:, 0])
    w, m = world[ok], measured[ok]
    best = None
    for _ in range(iters):
        k = rng.choice(len(w), 4, replace=False)
        s, R, t = similarity(w[k], m[k])
        inl = np.linalg.norm(s * w @ R.T + t - m, axis=1) < FIT_INLIER_M
        if best is None or inl.sum() > best.sum():
            best = inl
    s, R, t = similarity(w[best], m[best])
    return s * world @ R.T + t


def track(demo, model):
    demo = Path(demo)
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    hand = np.load(demo / "hand.npz")
    frames = read_video(demo / "color.mp4")
    n = len(frames)
    idx = np.arange(n)

    # Hand.
    detected = np.flatnonzero(hand["detected"])
    qf = np.unique(detected[np.abs(detected[None] - np.arange(0, n, QUERY_EVERY)[:, None]).argmin(1)])
    tracks, vis = run_cotracker(model, frames, qf, hand["uv"][qf][:, list(HAND_MEDIAPIPE)])
    src = np.abs(idx[:, None] - qf[None]).argmin(1)  # nearest query per frame
    uv, co_vis = tracks[idx, src], vis[idx, src]

    # Mug: points from the first-frame SAM 2 mask.
    masks = load_masks(demo)[:n]
    ctracks, cvis = run_cotracker(model, frames, np.array([0]), mask_points(masks[0])[None])
    cup_uv, cup_co_vis = ctracks[:, 0], cvis[:, 0]

    # Occlusion from D405 depth at the tracked pixels.
    scale = meta["depth_scale"]
    d_hand = np.zeros((n, len(HAND_POINTS)))
    for i, d_raw in enumerate(depth_frames(demo / "depth.mkv", tuple(meta["image_size"]))):
        if i >= n:
            break
        d_m = d_raw.astype(np.float32) * scale
        d_hand[i] = [depth_at(d_m, p) for p in uv[i]]
    rng = np.random.default_rng(0)
    expected = np.full((n, len(HAND_POINTS)), np.nan)
    for i in range(n):
        ok = hand["depth"][i] > 0
        if hand["detected"][i] and ok.sum() >= 8:
            ray = cv2.undistortPoints(hand["uv"][i].reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            measured = np.where(ok[:, None], np.c_[ray, np.ones(21)] * hand["depth"][i][:, None], np.nan)
            expected[i] = expected_hand(hand["world"][i], measured, rng)[list(HAND_MEDIAPIPE), 2]
    checked = np.isfinite(expected)
    in_front = checked & (d_hand > 0) & (d_hand < expected - IN_FRONT_M)
    off_hand = checked & (d_hand > expected + BEHIND_M)
    no_depth = checked & (d_hand == 0)
    h, w = frames.shape[1:3]
    at_edge = (uv[..., 0] < EDGE_PX) | (uv[..., 0] > w - 1 - EDGE_PX) | (uv[..., 1] < EDGE_PX) | (uv[..., 1] > h - 1 - EDGE_PX)
    visible = co_vis & ~in_front & ~off_hand & ~no_depth & ~at_edge
    tol = np.stack([cv2.dilate(m.astype(np.uint8), np.ones((2 * MASK_TOL_PX + 1,) * 2, np.uint8)) for m in masks]) > 0
    cu = np.clip(cup_uv[..., 0].round().astype(int), 0, w - 1)
    cv = np.clip(cup_uv[..., 1].round().astype(int), 0, h - 1)
    on_mug = tol[idx[:, None], cv, cu]
    cup_visible = cup_co_vis & on_mug
    np.savez(demo / "cotracker.npz", uv=uv, visible=visible, cup_uv=cup_uv, cup_visible=cup_visible)

    # Check against MediaPipe away from the query frames (at a query frame the two agree by construction).
    mp = hand["uv"][:, list(HAND_MEDIAPIPE)]
    away = hand["detected"] & (np.abs(idx - qf[src]) >= 4)
    err = np.linalg.norm(uv - mp, axis=2)[away]
    print(f"{demo.name}: {len(qf)} hand query frames | px from MediaPipe, frames >= 4 from a query, median / p95 / max: "
          + ", ".join(f"{p} {np.median(e):.0f}/{np.percentile(e, 95):.0f}/{e.max():.0f}" for p, e in zip(HAND_POINTS, err.T)))
    print(f"   depth-checked {checked[:, 0].mean() * 100:.0f}% of frames | occluded %: " + ", ".join(
        f"{p} {(~v).mean() * 100:.0f} (CoTracker {(~c).mean() * 100:.0f}, in front {f.mean() * 100:.0f}, off hand "
        f"{o.mean() * 100:.0f}, no depth {z.mean() * 100:.0f}, edge {e.mean() * 100:.0f})"
        for p, v, c, f, o, z, e in zip(HAND_POINTS, visible.T, co_vis.T, in_front.T, off_hand.T, no_depth.T, at_edge.T)))
    print(f"   mug: occluded {(~cup_visible).mean() * 100:.0f}% (CoTracker {(~cup_co_vis).mean() * 100:.0f}, outside "
          f"the mask {(~on_mug).mean() * 100:.0f})")
    return frames, uv, visible, cup_uv, cup_visible, mp, hand["detected"], qf


def overlay(demo, frames, uv, visible, cup_uv, cup_visible, mp, detected, q):
    """Write cotracker_overlay.mp4: CoTracker dots (red when occluded), MediaPipe rings."""
    meta = json.loads((Path(demo) / "meta.json").read_text())
    h, w = frames.shape[1:3]
    out = Path(demo) / "cotracker_overlay.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(meta["fps"]), "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                           str(out)], stdin=subprocess.PIPE)
    for i, img in enumerate(frames):
        img = img.copy()
        for j, c in enumerate(cup_uv[i]):
            cv2.circle(img, tuple(c.astype(int)), 8, YELLOW if cup_visible[i, j] else RED, -1, cv2.LINE_AA)
        for j, c in enumerate(HAND_COLOUR):
            if detected[i]:
                cv2.circle(img, tuple(mp[i, j].astype(int)), 14, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(uv[i, j].astype(int)), 8, c if visible[i, j] else RED, -1, cv2.LINE_AA)
        label = f"frame {i}" + (" (hand query)" if i in q else "")
        cv2.putText(img, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    return out


def legend(path, w=640, h=360, rows=None):
    img = np.full((h, w, 3), 255, np.uint8)
    rows = rows or [(BLUE, "wrist + knuckles (CoTracker)", "dot"), (ORANGE, "thumb tip", "dot"), (GREEN, "index tip", "dot"),
            (YELLOW, "mug (CoTracker)", "dot"), (RED, "occluded", "dot"), ((90, 90, 90), "MediaPipe landmark", "ring")]
    for k, (c, text, kind) in enumerate(rows):
        y = 50 + k * 50
        if kind == "ring":
            cv2.circle(img, (50, y), 14, c, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, (50, y), 10, c, -1, cv2.LINE_AA)
        cv2.putText(img, text, (85, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 30, 26), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def tile(videos, out_path, seconds, legend_rows=None):
    tw, th = 640, 360
    n = len(videos)
    legend_png = out_path.parent / "legend.png"
    legend(legend_png, tw, th, legend_rows)
    f = "".join(f"[{i}:v]scale={tw}:{th},drawtext=text='demo {i + 1}':x=10:y={th - 30}:fontsize=20:fontcolor=white:"
                f"box=1:boxcolor=black@0.5[v{i}];" for i in range(n))
    cells = [f"{(i % 3) * tw}_{(i // 3) * th}" for i in range(n + 1)]
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for v in videos:
        cmd += ["-i", str(v)]
    cmd += ["-loop", "1", "-framerate", "30", "-t", str(seconds), "-i", str(legend_png), "-filter_complex",
            f"{f}[{n}:v]format=yuv420p[v{n}];{''.join(f'[v{i}]' for i in range(n + 1))}"
            f"xstack=inputs={n + 1}:layout={'|'.join(cells)}:shortest=0[out]",
            "-map", "[out]", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", str(out_path)]
    subprocess.run(cmd, check=True)
    legend_png.unlink()
    print("wrote", out_path)


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    model = CoTrackerPredictor(checkpoint=str(CHECKPOINT), offline=True, window_len=60).cuda()
    results = [track(d, model) for d in demos]
    videos = [overlay(d, *r) for d, r in zip(demos, results)]
    MEDIA.mkdir(parents=True, exist_ok=True)
    tile(videos, MEDIA / "tracks.mp4", max(len(r[0]) for r in results) / 30)
