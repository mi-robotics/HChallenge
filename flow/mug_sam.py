"""Segment the mug in every frame with SAM 2 (video predictor), from one click in the first frame.

  python flow/mug_sam.py [data/demos/<demo> ...]     (default: all demos)

The click is the hand-derived mug centre (flow/format.py stand-in, from the MediaPipe hand) projected into the first
frame: a stand-in for a human click; the mask itself comes from the image only. mask_points() picks the 8 mug points
to track (flow/track_points.py) from the first-frame mask. Run this before flow/track_points.py.

Writes mug_mask.npz to each demo folder (packed boolean masks [T, H, W]).
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
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402
from format import N_CUP_POINTS, cup_from_hand, from_mediapipe  # noqa: E402
from to_table import demo_marker_pose  # noqa: E402

CHECKPOINT = ROOT / "third_party" / "sam2_ckpt" / "sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
ERODE_PX = 8


def click(demo):
    """Projected centre of the stand-in mug in the first frame."""
    meta = json.loads((Path(demo) / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, tv, _ = demo_marker_pose(Path(demo), K, dist)
    t, hand, _ = from_mediapipe(Path(demo), hz=None)
    c = cup_from_hand(Path(demo), t, hand)[:, 0, :3].mean(0)
    return cv2.projectPoints(c[None].astype(np.float64), cv2.Rodrigues(R)[0], np.ravel(tv), K, dist)[0].ravel()


def segment(demo, predictor):
    from track_points import read_video
    demo = Path(demo)
    frames = read_video(demo / "color.mp4")
    tmp = demo / "_frames_jpg"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    xy = click(demo)
    masks = np.zeros(frames.shape[:3], bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(tmp))
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=np.array([xy], np.float32),
                                        labels=np.array([1], np.int32))
        for i, _, logits in predictor.propagate_in_video(state):
            masks[i] = (logits[0, 0] > 0).cpu().numpy()
    shutil.rmtree(tmp)
    np.savez_compressed(demo / "mug_mask.npz", packed=np.packbits(masks, axis=-1), shape=masks.shape, click=xy)
    area = masks.sum((1, 2))
    print(f"{demo.name}: click {xy.round()} | mask area first frame {area[0]} px, median {np.median(area):.0f}, "
          f"min {area.min()} px, empty in {(area == 0).sum()} frames")
    return masks


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


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device="cuda")
    for d in demos:
        segment(d, predictor)
