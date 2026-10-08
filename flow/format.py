"""Action flow format constants (see flow/FORMAT.md) and a shape/value check.

  python flow/format.py [data/demos/<demo>]     (default: first demo)

Running it builds the hand part of an episode in this format from the step 1.5 MediaPipe landmarks (stand-in
until CoTracker, step 3.2), checks it, and draws two samples on the camera image:
media/phase3/task1/flow_samples.jpg.
"""
import json
import sys
from pathlib import Path

import numpy as np

HZ = 10
T = 16
HAND_POINTS = ("wrist", "thumb_tip", "index_tip", "index_knuckle", "middle_knuckle", "pinky_knuckle")
HAND_MEDIAPIPE = (0, 4, 8, 5, 9, 17)
N_CUP_POINTS = 8
DTYPE = np.float32


def check(flow, n_points, n_steps=None):
    """Assert flow is [n_points, n_steps, 4] float32, finite, with visible in {0, 1}."""
    assert flow.dtype == DTYPE, flow.dtype
    assert flow.ndim == 3 and flow.shape[0] == n_points and flow.shape[2] == 4, flow.shape
    assert n_steps is None or flow.shape[1] == n_steps, flow.shape
    assert np.isfinite(flow).all(), "NaN/inf in flow"
    assert np.isin(flow[..., 3], (0, 1)).all(), "visible must be 0 or 1"


def fill_hidden(p, visible):
    """Linearly interpolate x, y, z over hidden frames ([N, 3], [N]); hold flat at the ends."""
    out = p.copy()
    k = np.flatnonzero(visible)
    for j in range(3):
        out[:, j] = np.interp(np.arange(len(p)), k, p[k, j])
    return out


def from_mediapipe(demo):
    """Hand flow for a whole episode from hand_table.npz (30 fps) -> (t [N], hand [6, N, 4]) at HZ."""
    ht, hand = np.load(demo / "hand_table.npz"), np.load(demo / "hand.npz")
    t_cam = (hand["t_ms"] - hand["t_ms"][0]) / 1000
    t = np.arange(0, t_cam[-1] + 1e-9, 1 / HZ)
    frame = np.abs(t_cam[None] - t[:, None]).argmin(1)  # nearest camera frame per flow step
    flow = np.zeros((len(HAND_POINTS), len(t), 4), DTYPE)
    for i, lm in enumerate(HAND_MEDIAPIPE):
        vis = ht["valid"][frame, lm]
        flow[i, :, :3] = fill_hidden(ht["p"][frame, lm], vis)
        flow[i, :, 3] = vis
    return t.astype(DTYPE), flow, frame


def draw_samples(demo, t, flow, frame, starts, out):
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "replay"))
    from to_table import demo_marker_pose

    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, tv, _ = demo_marker_pose(demo, K, dist)
    rvec = cv2.Rodrigues(R)[0]
    cap = cv2.VideoCapture(str(demo / "color.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    # BGR of dataviz slots 1-3: back of hand (wrist + knuckles), thumb tip, index tip
    colour = {0: (214, 120, 42), 3: (214, 120, 42), 4: (214, 120, 42), 5: (214, 120, 42), 1: (52, 104, 235), 2: (122, 175, 27)}
    tiles = []
    for s in starts:
        img = frames[frame[s]].copy()
        for i in range(len(HAND_POINTS)):
            win = flow[i, s:s + T]
            uv = cv2.projectPoints(win[:, :3].astype(np.float64), rvec, tv, K, dist)[0].reshape(-1, 2).astype(int)
            for k in range(len(uv) - 1):
                cv2.line(img, tuple(uv[k]), tuple(uv[k + 1]), colour[i], 3, cv2.LINE_AA)
            for k, (u, v) in enumerate(uv):
                if win[k, 3]:
                    cv2.circle(img, (u, v), 5, colour[i], -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (u, v), 5, colour[i], 1, cv2.LINE_AA)  # hidden: hollow
            cv2.circle(img, tuple(uv[0]), 9, (255, 255, 255), 2, cv2.LINE_AA)  # step 0 = now
        label = f"t0 = {t[s]:.1f} s, next {T} steps at {HZ} Hz"
        cv2.rectangle(img, (0, 0), (cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] + 24, 48), (0, 0, 0), -1)
        cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (960, 540)))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([tiles[0], np.full((540, 8, 3), 255, np.uint8), tiles[1]]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    demo = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((root / "data" / "demos").iterdir())[0]
    t, hand, frame = from_mediapipe(demo)
    check(hand, len(HAND_POINTS))
    sample = hand[:, 0:T]
    check(sample, len(HAND_POINTS), T)
    print(f"{demo.name}: episode hand {hand.shape} {hand.dtype}, {len(t) / HZ:.1f} s; sample {sample.shape}; "
          f"visible {hand[..., 3].mean() * 100:.0f}% (per point: "
          + ", ".join(f"{n} {hand[i, :, 3].mean() * 100:.0f}%" for i, n in enumerate(HAND_POINTS)) + ")")
    rt = np.load(demo / "retarget.npz")
    i_grasp = int(round(rt["t"][int(rt["i_grasp"])] * HZ))
    draw_samples(demo, t, hand, frame, [max(0, i_grasp - 12), i_grasp + 10], root / "media" / "phase3" / "task1" / "flow_samples.jpg")
