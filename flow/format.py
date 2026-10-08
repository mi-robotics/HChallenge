"""Action flow format constants (see flow/FORMAT.md) and a shape/value check.

  python flow/format.py [data/demos/<demo>]     (default: first demo)

Running it builds an episode in this format from stand-in data, checks it, and draws three samples on the camera
image (media/phase3/task1/flow_samples.jpg):
  hand  step 1.5 MediaPipe landmarks (until CoTracker, step 3.2)
  cup   derived from the hand (until step 3.4; the D405 has no depth on the white mug): two rings of 4 points on the
        camera-facing half of a cylinder (radius CUP_R_M, at 80% and 35% of the grasp height) whose centre is
        CUP_R_M from the grasp point along thumb -> index (the index tip is inside the mug). The cup is still until the grasp, moves with the pinch point (translation only) until the
        release, then stays where it was let go. All marked visible. (A rigid fit to the back of the hand, which
        would also rotate the cup, put the cup base 1-24 cm above the table at release: rotation noise times the
        hand-to-cup distance. Translation only gives 0.8-2.3 cm, i.e. on the ~1.5 cm saucer.)
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
CUP_R_M = 0.04


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


def from_mediapipe(demo, hz=HZ):
    """Hand flow for a whole episode from hand_table.npz (30 fps) -> (t [N], hand [6, N, 4], camera frame [N]) at
    hz, or at every camera frame if hz is None."""
    ht, hand = np.load(demo / "hand_table.npz"), np.load(demo / "hand.npz")
    t_cam = (hand["t_ms"] - hand["t_ms"][0]) / 1000
    t = t_cam if hz is None else np.arange(0, t_cam[-1] + 1e-9, 1 / hz)
    frame = np.abs(t_cam[None] - t[:, None]).argmin(1)  # nearest camera frame per flow step
    flow = np.zeros((len(HAND_POINTS), len(t), 4), DTYPE)
    for i, lm in enumerate(HAND_MEDIAPIPE):
        vis = ht["valid"][frame, lm]
        flow[i, :, :3] = fill_hidden(ht["p"][frame, lm], vis)
        flow[i, :, 3] = vis
    return t.astype(DTYPE), flow, frame


def cup_from_hand(demo, t, hand):
    """Stand-in cup flow [8, N, 4] (see module docstring)."""
    rt = np.load(demo / "retarget.npz")
    i_grasp = int(np.abs(t - rt["t"][int(rt["i_grasp"])]).argmin())
    i_release = int(np.abs(t - rt["t"][int(rt["i_release"])]).argmin())
    thumb, index = hand[1, i_grasp, :3], hand[2, i_grasp, :3]
    g = (thumb + index) / 2
    u = (index - thumb)[:2] / np.linalg.norm((index - thumb)[:2])
    c = g[:2] + CUP_R_M * u
    to_cam = np.load(demo / "hand_table.npz")["cam_in_table"][:2] - c
    a = np.arctan2(to_cam[1], to_cam[0]) + np.radians([-60, -20, 20, 60])  # camera-facing half
    ring = np.c_[c[0] + CUP_R_M * np.cos(a), c[1] + CUP_R_M * np.sin(a)]
    pts = np.r_[np.c_[ring, np.full(4, 0.8 * g[2])], np.c_[ring, np.full(4, 0.35 * g[2])]]
    pinch = (hand[1, :, :3] + hand[2, :, :3]) / 2
    shift = pinch[np.clip(np.arange(len(t)), i_grasp, i_release)] - g  # [N, 3]
    cup = np.ones((len(pts), len(t), 4), DTYPE)
    cup[:, :, :3] = pts[:, None] + shift[None]
    return cup


def draw_samples(demo, t, hand, cup, frame, starts, out):
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
    # BGR of dataviz slots 1-4: back of hand (wrist + knuckles), thumb tip, index tip, mug
    blue, orange, green, yellow = (214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237)
    trails = [(hand[i], c) for i, c in enumerate([blue, orange, green, blue, blue, blue])] + [(p, yellow) for p in cup]
    tiles = []
    for s in starts:
        img = frames[frame[s]].copy()
        for flow_i, c in trails:
            win = flow_i[s:s + T]
            uv = cv2.projectPoints(win[:, :3].astype(np.float64), rvec, tv, K, dist)[0].reshape(-1, 2).astype(int)
            for k in range(len(uv) - 1):
                cv2.line(img, tuple(uv[k]), tuple(uv[k + 1]), c, 3, cv2.LINE_AA)
            for k, (u, v) in enumerate(uv):
                cv2.circle(img, (u, v), 5, c, -1 if win[k, 3] else 1, cv2.LINE_AA)  # hidden: hollow
            cv2.circle(img, tuple(uv[0]), 9, (255, 255, 255), 2, cv2.LINE_AA)  # step 0 = now
        label = f"t0 = {t[s]:.1f} s, next {T} steps at {HZ} Hz"
        cv2.rectangle(img, (0, 0), (cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] + 24, 48), (0, 0, 0), -1)
        cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (960, 540)))
    gap = np.full((540, 8, 3), 255, np.uint8)
    row = np.hstack([x for tile in tiles for x in (tile, gap)][:-1])
    legend = np.full((70, row.shape[1], 3), 255, np.uint8)
    x = 20
    for c, text, hollow in [(blue, "wrist + knuckles", False), (orange, "thumb tip", False), (green, "index tip", False),
                            (yellow, "mug (stand-in, derived from the hand)", False), ((255, 255, 255), "now (step 0)", None),
                            ((90, 90, 90), "hidden point", True)]:
        if hollow is None:
            cv2.circle(legend, (x + 12, 35), 10, (90, 90, 90), 2, cv2.LINE_AA)
        else:
            cv2.line(legend, (x, 35), (x + 24, 35), c, 3, cv2.LINE_AA)
            cv2.circle(legend, (x + 12, 35), 6, c, 1 if hollow else -1, cv2.LINE_AA)
        cv2.putText(legend, text, (x + 36, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 30, 26), 2, cv2.LINE_AA)
        x += 36 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0] + 44
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([row, legend]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    demo = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((root / "data" / "demos").iterdir())[0]
    t, hand, frame = from_mediapipe(demo)
    cup = cup_from_hand(demo, t, hand)
    check(hand, len(HAND_POINTS))
    check(cup, N_CUP_POINTS)
    sample = hand[:, 0:T]
    check(sample, len(HAND_POINTS), T)
    print(f"{demo.name}: episode hand {hand.shape} {hand.dtype}, {len(t) / HZ:.1f} s; sample {sample.shape}; "
          f"visible {hand[..., 3].mean() * 100:.0f}% (per point: "
          + ", ".join(f"{n} {hand[i, :, 3].mean() * 100:.0f}%" for i, n in enumerate(HAND_POINTS)) + ")")
    rt = np.load(demo / "retarget.npz")
    i_grasp, i_release = (int(np.abs(t - rt["t"][int(rt[k])]).argmin()) for k in ("i_grasp", "i_release"))
    c0, c1 = cup[:, 0, :3].mean(0), cup[:, i_release, :3].mean(0)
    print(f"  cup at release: {(c1 - c0)[2] * 100:+.1f} cm above its start (on the ~1.5 cm saucer), centre xy "
          f"{np.round(c1[:2] * 100, 1)} cm; moved {np.linalg.norm(c1 - c0) * 100:.1f} cm")
    draw_samples(demo, t, hand, cup, frame, [max(0, i_grasp - 8), i_grasp + 12, max(0, i_release - 10)],
                 root / "media" / "phase3" / "task1" / "flow_samples.jpg")
