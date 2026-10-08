"""Hand landmarks (pixels + D405 depth) -> 3D points in the table (marker) frame.

  python replay/to_table.py [data/demos/<demo> ...]     (default: all demos)

Writes hand_table.npz to each demo folder:
  p      [T,21,3]  landmark positions in the table frame, metres (NaN where invalid)
  valid  [T,21]    hand detected and depth valid
  cam_in_table [3] camera position in the table frame
and a top-down + height plot of all demos to media/phase1/task5/hand_table.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
from markers import marker_pose  # noqa: E402
from track_hand import INDEX_TIP, THUMB_TIP, WRIST, depth_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media" / "phase1" / "task5"


def demo_marker_pose(demo, K, dist):
    """The camera is fixed during a demo, so use one pose: the per-axis median over all frames.
    (Per-frame poses jitter by 1-3 mm.) Returns R, t (table -> camera) and the per-frame spread."""
    cap = cv2.VideoCapture(str(demo / "color.mp4"))
    rvecs, tvecs = [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        pose = marker_pose(img, K, dist)
        if pose is not None:
            rvecs.append(pose[0].ravel())
            tvecs.append(pose[1].ravel())
    rvecs, tvecs = np.array(rvecs), np.array(tvecs)
    R, _ = cv2.Rodrigues(np.median(rvecs, 0))
    return R, np.median(tvecs, 0), tvecs.std(0)


def pixels_to_table(uv, z, K, dist, R, t):
    """uv [N,2] pixels, z [N] depth along the optical axis (metres) -> [N,3] table frame."""
    xy = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    p_cam = np.c_[xy * z[:, None], z]
    return (p_cam - t) @ R  # R^T (p - t), row-wise


def table_plane_check(demo, meta, K, dist, R, t):
    """Depth of bare table pixels, converted to the table frame, should give z ~ 0.
    Uses the first frame on a grid of pixels within 25 cm of the marker."""
    frames = depth_frames(demo / "depth.mkv", tuple(meta["image_size"]))
    d = next(frames).astype(np.float32) * meta["depth_scale"]
    frames.close()
    h, w = d.shape
    u, v = np.meshgrid(np.arange(0, w, 16), np.arange(0, h, 16))
    uv, z = np.c_[u.ravel(), v.ravel()], d[v.ravel(), u.ravel()]
    keep = z > 0
    p = pixels_to_table(uv[keep], z[keep], K, dist, R, t)
    near = np.linalg.norm(p[:, :2], axis=1) < 0.25
    return p[near & (np.abs(p[:, 2]) < 0.05), 2]  # drop objects / off-table points


def convert(demo):
    demo = Path(demo)
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    hand = np.load(demo / "hand.npz")
    R, t, spread = demo_marker_pose(demo, K, dist)

    valid = hand["detected"][:, None] & (hand["depth"] > 0)
    T = len(valid)
    p = np.full((T, 21, 3), np.nan)
    p[valid] = pixels_to_table(hand["uv"][valid], hand["depth"][valid], K, dist, R, t)
    cam = -R.T @ t
    np.savez(demo / "hand_table.npz", p=p, valid=valid, cam_in_table=cam)

    table_z = table_plane_check(demo, meta, K, dist, R, t)
    key = p[:, [WRIST, THUMB_TIP, INDEX_TIP]]
    zmin = np.nanmin(p[..., 2], axis=1)
    print(f"{demo.name}: marker pose spread (std, mm) {np.round(spread * 1000, 1)} | "
          f"table plane z: median {np.median(table_z) * 1000:+.1f} mm, IQR "
          f"{(np.percentile(table_z, 75) - np.percentile(table_z, 25)) * 1000:.1f} mm (N={len(table_z)}) | "
          f"lowest landmark per frame: min {np.nanmin(zmin) * 100:+.1f} cm, "
          f"{(zmin < -0.01).sum()} frames below -1 cm | wrist x {np.nanmin(key[:, 0, 0]) * 100:.0f}..{np.nanmax(key[:, 0, 0]) * 100:.0f} "
          f"y {np.nanmin(key[:, 0, 1]) * 100:.0f}..{np.nanmax(key[:, 0, 1]) * 100:.0f} "
          f"z {np.nanmin(key[:, 0, 2]) * 100:.0f}..{np.nanmax(key[:, 0, 2]) * 100:.0f} cm")
    return p, valid, cam, meta["fps"]


def plot(results, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz categorical slots 1-5
    ink, muted, grid = "#1a1e26", "#5d6573", "#e3e6eb"
    fig, (ax_top, ax_z) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (ax_top, ax_z):
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    h = 0.089 / 2
    ax_top.add_patch(plt.Rectangle((-h * 100, -h * 100), 2 * h * 100, 2 * h * 100, color="#1a1e26", alpha=0.85))
    ax_top.annotate("marker", (0, -h * 100 - 1.5), ha="center", va="top", fontsize=9, color=muted)
    for i, ((p, valid, cam, fps), name) in enumerate(zip(results, names)):
        w = p[:, WRIST] * 100
        ax_top.plot(w[:, 1], w[:, 0], color=colors[i], linewidth=2, label=f"demo {i + 1}")
        start = np.flatnonzero(~np.isnan(w[:, 0]))[0]
        ax_top.plot(w[start, 1], w[start, 0], "o", color=colors[i], markersize=8, markeredgecolor="white")
        ax_top.annotate(str(i + 1), (w[start, 1], w[start, 0]), xytext=(6, 4), textcoords="offset points",
                        fontsize=9, color=ink)
        tt = np.arange(len(p)) / fps
        ax_z.plot(tt, w[:, 2], color=colors[i], linewidth=2, label=f"demo {i + 1}")
        ax_z.plot(tt, np.nanmin(p[..., 2], axis=1) * 100, color=colors[i], linewidth=1, alpha=0.45)
    ax_top.set_xlabel("y (cm), table frame", color=muted, fontsize=9)
    ax_top.set_ylabel("x (cm), table frame", color=muted, fontsize=9)
    ax_top.invert_xaxis()  # x up, y to the left: the table seen from above with z towards the viewer
    ax_top.set_aspect("equal")
    ax_top.set_title("Wrist path, top-down (dot = start)", color=ink, fontsize=11, loc="left")
    ax_z.axhline(0, color=ink, linewidth=1)
    ax_z.annotate("table surface", (0.2, 0.4), fontsize=9, color=muted)
    ax_z.set_xlabel("time (s)", color=muted, fontsize=9)
    ax_z.set_ylabel("height above table (cm)", color=muted, fontsize=9)
    ax_z.set_title("Height: wrist (thick) and lowest hand point (thin)", color=ink, fontsize=11, loc="left")
    ax_z.legend(frameon=False, fontsize=9, labelcolor=ink, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "hand_table.png", dpi=150, facecolor="white")
    print("wrote", MEDIA / "hand_table.png")


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    results = [convert(d) for d in demos]
    plot(results, [Path(d).name for d in demos])
