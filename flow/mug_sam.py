"""Segment the mug in every frame with SAM 2, and time when it is picked up and put down.

  python flow/mug_sam.py [<episode> ...]     (default: data/demos)

The mug is the SAM 2 automatic-mode proposal that looks like a mug (size, aspect, fill, colour) nearest to the
hand's first pinch; the video predictor tracks that mask through the video. Picked up / put down: when the mask's
base point on the table first leaves / last returns within MOVE_M of its resting place.
mask_points() picks the 8 mug points to track. Writes mug_mask.npz: masks [T, H, W], frame, t_grasp, t_release.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT / "replay"))
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from format import N_CUP_POINTS  # noqa: E402
from to_table import demo_marker_pose  # noqa: E402

CHECKPOINT = ROOT / "third_party" / "sam2_ckpt" / "sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
ERODE_PX = 8
MUG_AREA_PX, MUG_H_OVER_W, MUG_FILL, MUG_GREY = (2500, 30000), (0.85, 1.9), 0.6, 100  # saucers have h/w <= 0.7
MOVE_M, SAUCER_M, MUG_R_M = 0.02, 0.015, 0.045


def hand_hint(demo, K, dist, R, tv):
    """Pixel of the pinch point the first time the hand closes (thumb-index < 2.5 cm), or the image centre."""
    if not (demo / "traj.npz").exists():
        return np.array([K[0, 2], K[1, 2]])
    tr = np.load(demo / "traj.npz")
    ok = ~np.isnan(tr["aperture"]) & ~np.isnan(tr["thumb"]).any(1) & ~np.isnan(tr["index"]).any(1)
    if not ok.any():
        return np.array([K[0, 2], K[1, 2]])
    closed = np.flatnonzero(ok & (tr["aperture"] < 0.025))
    k = int(closed[0]) if len(closed) else int(np.flatnonzero(ok)[np.nanargmin(tr["aperture"][ok])])
    p = (tr["thumb"][k] + tr["index"][k]) / 2
    return cv2.projectPoints(p[None], cv2.Rodrigues(R)[0], np.ravel(tv), K, dist)[0].ravel()


def find_mug(frames, hint, generator):
    """(frame index, mask) of the mug-like proposal nearest the hint, in the first frame or else the last."""
    for fi in (0, len(frames) - 1):
        img = frames[fi]
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            proposals = generator.generate(np.ascontiguousarray(img[:, :, ::-1]))
        best = None
        for p in proposals:
            m = p["segmentation"]
            v, u = np.nonzero(m)
            h, w = v.ptp() + 1, u.ptp() + 1
            if (MUG_AREA_PX[0] <= len(v) <= MUG_AREA_PX[1] and MUG_H_OVER_W[0] <= h / w <= MUG_H_OVER_W[1]
                    and len(v) / (h * w) > MUG_FILL and np.median(grey[m]) > MUG_GREY):
                d = np.hypot(u.mean() - hint[0], v.mean() - hint[1])
                if best is None or d < best[0]:
                    best = (d, m)
        if best is not None:
            return fi, best[1]
    raise ValueError("no mug-like object in the first or last frame")


def mug_motion(demo, masks, t_cam):
    """(t_grasp, t_release) in camera time from the mug base's position on the table (see module docstring)."""
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, tv, _ = demo_marker_pose(demo, K, dist)
    o = -R.T @ np.ravel(tv)
    n = min(len(masks), len(t_cam))
    uv = np.full((n, 2), np.nan)
    for i in range(n):
        v, u = np.nonzero(masks[i])
        if len(v) >= 200:
            uv[i] = [np.median(u[v >= v.max() - 3]), v.max()]
    ok = ~np.isnan(uv[:, 0])
    d = np.c_[cv2.undistortPoints(uv[ok][:, None], K, dist)[:, 0], np.ones(ok.sum())] @ R  # rays, table frame
    base = {}
    for h in (0.0, SAUCER_M):
        q = o[:2] + ((h - o[2]) / d[:, 2])[:, None] * d[:, :2]
        away = (q - o[:2]) / np.linalg.norm(q - o[:2], axis=1, keepdims=True)
        base[h] = np.full((n, 2), np.nan)
        base[h][ok] = q + MUG_R_M * away
    seen = np.flatnonzero(ok)  # the mug can be hidden by the hand at the start
    start = np.median(base[0.0][seen[:10]], 0)
    end = np.median(base[SAUCER_M][seen[-10:]], 0)
    moved = np.nan_to_num(np.linalg.norm(base[0.0] - start, axis=1), nan=0) > MOVE_M
    run = np.flatnonzero(np.convolve(moved, np.ones(3), "valid") == 3)
    away_end = np.flatnonzero(np.nan_to_num(np.linalg.norm(base[SAUCER_M] - end, axis=1), nan=0) > MOVE_M)
    if not len(run) or not len(away_end):
        raise ValueError("the mug never moves more than 2 cm")
    return float(t_cam[run[0]]), float(t_cam[min(away_end[-1] + 1, n - 1)])


def segment(demo, predictor, generator):
    from track_points import read_video
    demo = Path(demo)
    frames = read_video(demo / "color.mp4")
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, tv, _ = demo_marker_pose(demo, K, dist)
    fi, first = find_mug(frames, hand_hint(demo, K, dist, R, tv), generator)
    tmp = demo / "_frames_jpg"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    masks = np.zeros(frames.shape[:3], bool)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(tmp))
            predictor.add_new_mask(state, frame_idx=fi, obj_id=1, mask=first)
            for reverse in ((False, True) if fi > 0 else (False,)):
                for i, _, logits in predictor.propagate_in_video(state, start_frame_idx=fi, reverse=reverse):
                    masks[i] = (logits[0, 0] > 0).cpu().numpy()
    finally:
        shutil.rmtree(tmp)
    hand = np.load(demo / "hand.npz")
    t_cam = (hand["t_ms"] - hand["t_ms"][0]) / 1000
    t_grasp, t_release = mug_motion(demo, masks, t_cam)
    np.savez_compressed(demo / "mug_mask.npz", packed=np.packbits(masks, axis=-1), shape=masks.shape, frame=fi,
                        t_grasp=t_grasp, t_release=t_release)
    area = masks.sum((1, 2))
    print(f"{demo.name}: mug found in frame {fi} | mask area median {np.median(area):.0f} px, empty in "
          f"{(area == 0).sum()} frames | mug moves {t_grasp:.2f}-{t_release:.2f} s")
    return masks


def load_models():
    model = build_sam2(CONFIG, str(CHECKPOINT), device="cuda")
    generator = SAM2AutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.9)
    return build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device="cuda"), generator


def mask_points(mask, n=N_CUP_POINTS):
    """n points spread over the eroded mask: farthest-point sampling, starting nearest the mask's centre."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * ERODE_PX + 1,) * 2, np.uint8))
    pts = np.argwhere(m)[:, ::-1].astype(float)  # (u, v)
    chosen = [np.linalg.norm(pts - pts.mean(0), axis=1).argmin()]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(n - 1):
        chosen.append(d.argmax())
        d = np.minimum(d, np.linalg.norm(pts - pts[chosen[-1]], axis=1))
    return pts[chosen]


def load_masks(demo):
    m = np.load(Path(demo) / "mug_mask.npz")
    return np.unpackbits(m["packed"], axis=-1)[..., :m["shape"][2]].astype(bool)


def mug_frame(demo):
    """Frame the mug was found in (0 for older masks)."""
    m = np.load(Path(demo) / "mug_mask.npz")
    return int(m["frame"]) if "frame" in m else 0


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    predictor, generator = load_models()
    for d in demos:
        segment(d, predictor, generator)
