"""Compare two ways of choosing the 8 mug points to track, scored against the SAM 2 mug masks (flow/mug_sam.py).

  python flow/mug_compare.py [data/demos/<demo> ...]     (default: all demos)

  geometric  (current) points on a cylinder placed from the hand: centre 4 cm from the grasp point along
             thumb -> index, two rings of 4 on the camera-facing half (flow/format.py). Needs the grasp and an
             assumed mug size.
  mask       (new) 8 points spread over the SAM 2 mug mask in the first frame (eroded by ERODE_PX, farthest-point
             sampling from the mask's centre). Needs one click on the mug.

Both are tracked by CoTracker from the first frame only (same settings), so the comparison is about where the points
start, not the re-query at the last frame that flow/track_points.py also uses.
Per method: share of point-frames that CoTracker calls visible, and of those, the share inside the mug mask (dilated
by TOL_PX), and the share more than 20 px outside it (the track left the mug).

Writes media/phase3/task4/mug_points.mp4 (geometric | mask, demo by demo) and media/phase3/task4/mug_points.json.
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
from cotracker.predictor import CoTrackerPredictor  # noqa: E402
from format import cup_from_hand  # noqa: E402
from mug_sam import load_masks  # noqa: E402
from to_flow import tracked_hand  # noqa: E402
from track_points import CHECKPOINT, RED, YELLOW, read_video, run_cotracker  # noqa: E402

MEDIA = ROOT / "media" / "phase3" / "task4"
ERODE_PX, TOL_PX, FAR_PX, N = 8, 4, 20, 8


def geometric_points(demo):
    t, hand, _, _, (K, dist, R, tv, _) = tracked_hand(demo)
    p = cup_from_hand(demo, t, hand)[:, 0, :3]
    return cv2.projectPoints(p.astype(np.float64), cv2.Rodrigues(R)[0], np.ravel(tv), K, dist)[0][:, 0]


def mask_points(mask):
    """N points spread over the eroded mask: farthest-point sampling, starting nearest the mask's centre."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * ERODE_PX + 1,) * 2, np.uint8))
    vu = np.argwhere(m)
    pts = vu[:, ::-1].astype(float)  # (u, v)
    chosen = [np.linalg.norm(pts - pts.mean(0), axis=1).argmin()]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(N - 1):
        chosen.append(d.argmax())
        d = np.minimum(d, np.linalg.norm(pts - pts[chosen[-1]], axis=1))
    return pts[chosen]


def score(uv, vis, masks):
    """uv [T, P, 2], vis [T, P], masks [T, H, W] -> (visible share, inside share of visible, far-off share of visible)."""
    T, H, W = masks.shape
    tol = np.stack([cv2.dilate(m.astype(np.uint8), np.ones((2 * TOL_PX + 1,) * 2, np.uint8)) for m in masks]).astype(bool)
    dist_out = np.stack([cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in masks])
    u = np.clip(uv[..., 0].round().astype(int), 0, W - 1)
    v = np.clip(uv[..., 1].round().astype(int), 0, H - 1)
    t = np.arange(T)[:, None]
    inside, far = tol[t, v, u], dist_out[t, v, u] > FAR_PX
    return vis.mean(), inside[vis].mean(), far[vis].mean(), inside


def compare(demo, model):
    demo = Path(demo)
    frames, masks = read_video(demo / "color.mp4"), load_masks(demo)
    n = min(len(frames), len(masks))
    frames, masks = frames[:n], masks[:n]
    out = {}
    for name, q in (("geometric", geometric_points(demo)), ("mask", mask_points(masks[0]))):
        tracks, vis = run_cotracker(model, frames, np.array([0]), q[None])
        uv, vis = tracks[:, 0], vis[:, 0]
        v_share, inside, far, inside_pf = score(uv, vis, masks)
        out[name] = {"query_inside_first_mask": float(score(q[None], np.ones((1, N), bool), masks[:1])[1]),
                     "visible": float(v_share), "inside_mask_when_visible": float(inside),
                     "over_20px_outside_when_visible": float(far), "uv": uv, "vis": vis, "inside": inside_pf}
        print(f"{demo.name} {name:9s}: start points inside the first mask {out[name]['query_inside_first_mask'] * 100:.0f}% "
              f"| CoTracker visible {v_share * 100:.0f}% | of those inside the mug mask {inside * 100:.1f}%, > {FAR_PX} px "
              f"outside {far * 100:.1f}%")
    return frames, masks, out


def video(results, out_path):
    """Per demo: geometric | mask side by side, mask outline in white, points yellow on the mug, red off it."""
    w, h = 640, 360
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * w}x{h}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", str(out_path)],
                          stdin=subprocess.PIPE)
    sx = w / results[0][0].shape[2]
    for k, (frames, masks, out) in enumerate(results):
        for i in range(len(frames)):
            tiles = []
            for name in ("geometric", "mask"):
                img = cv2.resize(frames[i], (w, h))
                cs, _ = cv2.findContours(cv2.resize(masks[i].astype(np.uint8), (w, h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
                r = out[name]
                for p in range(N):
                    if r["vis"][i, p]:
                        c = YELLOW if r["inside"][i, p] else RED
                        cv2.circle(img, tuple((r["uv"][i, p] * sx).astype(int)), 5, c, -1, cv2.LINE_AA)
                label = f"demo {k + 1}: {'geometric (from the hand)' if name == 'geometric' else 'mask (SAM 2)'}"
                cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                tiles.append(img)
            ff.stdin.write(np.hstack(tiles).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out_path)


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    model = CoTrackerPredictor(checkpoint=str(CHECKPOINT), offline=True, window_len=60).cuda()
    results = [compare(d, model) for d in demos]
    MEDIA.mkdir(parents=True, exist_ok=True)
    summary = {Path(d).name: {m: {k: v for k, v in r[2][m].items() if k not in ("uv", "vis", "inside")} for m in r[2]}
               for d, r in zip(demos, results)}
    (MEDIA / "mug_points.json").write_text(json.dumps(summary, indent=2))
    video(results, MEDIA / "mug_points.mp4")
