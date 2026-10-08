"""CoTracker tracks (step 3.2) -> action flow in the table frame, in the flow/FORMAT.md format.

  python flow/to_flow.py [data/demos/<demo> ...]     (default: all demos)

Hand: tracked pixel + D405 depth at that pixel (median of a 7x7 patch), then the marker pose, as in step 1.5.
Mug: tracked pixel + the stand-in mug point's depth (flow/format.py; the D405 has no depth on the white mug), so
the mug's image position comes from tracking and only its distance from the camera from the hand.
Then sampled at 10 Hz: each step is the mean of the visible camera frames within +-50 ms (about 3 frames; taking
only the nearest frame left about twice the frame-to-frame jitter of the smoothed step 1.6 path). visible = the step
3.2 occlusion check (any visible frame in the window); hidden points interpolated.

Checks: pinch point (thumb/index midpoint) vs the step 1.6 MediaPipe trajectory (outside MediaPipe's gaps, where
step 1.6 only interpolates; in demo 5's 0.7 s gap the tracks show the hand swinging ~10 cm towards the camera, which
the interpolation misses), frame-to-frame jitter, points below the table, mug height at release.

Writes flow.npz to each demo folder and media/phase3/task3/flow_table.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT / "replay"))
from format import DTYPE, HAND_POINTS, HZ, N_CUP_POINTS, check, cup_from_hand, fill_hidden, from_mediapipe  # noqa: E402
from to_table import demo_marker_pose, pixels_to_table  # noqa: E402
from track_hand import depth_at, depth_frames  # noqa: E402

MEDIA = ROOT / "media" / "phase3" / "task3"


def to_flow(demo):
    demo = Path(demo)
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, tv, _ = demo_marker_pose(demo, K, dist)
    ct, hand_mp, rt = np.load(demo / "cotracker.npz"), np.load(demo / "hand.npz"), np.load(demo / "retarget.npz")
    n = len(ct["uv"])
    t_cam = (hand_mp["t_ms"][:n] - hand_mp["t_ms"][0]) / 1000

    d_hand = np.zeros((n, len(HAND_POINTS)))
    for i, d_raw in enumerate(depth_frames(demo / "depth.mkv", tuple(meta["image_size"]))):
        if i >= n:
            break
        d_m = d_raw.astype(np.float32) * meta["depth_scale"]
        d_hand[i] = [depth_at(d_m, p) for p in ct["uv"][i]]
    vis = ct["visible"] & (d_hand > 0)
    hand3d = pixels_to_table(ct["uv"].reshape(-1, 2), np.maximum(d_hand, 1e-3).ravel(), K, dist, R, tv).reshape(n, -1, 3)

    _, hand30, _ = from_mediapipe(demo, hz=None)
    cup_standin = cup_from_hand(demo, t_cam, hand30)[:, :n, :3]  # [8, T, 3]
    z_cam = (cup_standin.transpose(1, 0, 2) @ R.T + np.ravel(tv))[..., 2]  # [T, 8]
    cup3d = pixels_to_table(ct["cup_uv"].reshape(-1, 2), z_cam.ravel(), K, dist, R, tv).reshape(n, -1, 3)

    t = np.arange(0, t_cam[-1] + 1e-9, 1 / HZ)
    win = (np.abs(t_cam[None] - t[:, None]) <= 0.5 / HZ).astype(float)  # [N steps, T frames]
    out = {"t": t.astype(DTYPE)}
    for name, p3, v in (("hand", hand3d, vis), ("cup", cup3d, ct["cup_visible"])):
        w = win[:, :, None] * v[None]  # [N, T, P]
        cnt = w.sum(1)
        p3 = np.einsum("ntp,tpk->npk", w, p3) / np.maximum(cnt, 1)[..., None]
        v = cnt > 0
        flow = np.zeros((p3.shape[1], len(t), 4), DTYPE)
        for j in range(p3.shape[1]):
            flow[j, :, :3] = fill_hidden(p3[:, j], v[:, j]) if v[:, j].any() else p3[:, j]
            flow[j, :, 3] = v[:, j]
        out[name] = flow
    check(out["hand"], len(HAND_POINTS))
    check(out["cup"], N_CUP_POINTS)
    np.savez(demo / "flow.npz", **out)

    # Checks.
    hand, cup = out["hand"], out["cup"]
    pinch = (hand[1, :, :3] + hand[2, :, :3]) / 2
    both = (hand[1, :, 3] > 0) & (hand[2, :, 3] > 0)
    tr = np.load(demo / "traj.npz")
    ref = np.stack([np.interp(t, tr["t"], ((tr["thumb"] + tr["index"]) / 2)[:, k]) for k in range(3)], 1)
    filled = np.interp(t, tr["t"], tr["filled"].astype(float)) > 0  # step 1.6 interpolated over a MediaPipe gap
    dev = np.linalg.norm(pinch - ref, axis=1)[both & ~filled]
    jitter = np.linalg.norm(np.diff(hand[:, :, :3], 2, axis=1), axis=2)[:, :][hand[:, 1:-1, 3] > 0]
    below = (hand[..., 2] < -0.01) & (hand[..., 3] > 0)
    i_release = int(np.abs(t - rt["t"][int(rt["i_release"])]).argmin())
    cup_lift = cup[:, i_release, 2].mean() - cup[:, 0, 2].mean()
    gap = np.linalg.norm(pinch - ref, axis=1)[both & filled]
    print(f"{demo.name}: {len(t)} steps | hand visible {hand[..., 3].mean() * 100:.0f}%, mug {cup[..., 3].mean() * 100:.0f}% | "
          f"pinch vs step 1.6 (MediaPipe, smoothed): median {np.median(dev) * 1000:.0f} mm, p95 "
          f"{np.percentile(dev, 95) * 1000:.0f} mm; in {filled.sum()} steps where step 1.6 interpolated a gap: max "
          f"{gap.max() * 1000 if gap.size else 0:.0f} mm | 2nd difference median {np.median(jitter) * 1000:.1f} mm | visible "
          f"points below -1 cm: {below.sum()} | mug at release {cup_lift * 100:+.1f} cm above start")
    return out, ref


def plot(results, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz categorical slots 1-5
    ink, muted, grid = "#1a1e26", "#5d6573", "#e3e6eb"
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(len(results), 2, width_ratios=[1, 1.4], hspace=0.3, wspace=0.18)
    ax_top = fig.add_subplot(gs[:, 0])

    def style(ax):
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    style(ax_top)
    first = None
    for i, ((out, ref), name) in enumerate(zip(results, names)):
        c = colors[i]
        hand, cup, t = out["hand"], out["cup"], out["t"]
        pinch = (hand[1, :, :3] + hand[2, :, :3]) / 2
        cup_c = cup[:, :, :3].mean(0)
        ax_top.plot(pinch[:, 1] * 100, pinch[:, 0] * 100, color=c, linewidth=1.8, label=f"demo {i + 1}")
        ax_top.plot(cup_c[:, 1] * 100, cup_c[:, 0] * 100, color=c, linewidth=1.2, linestyle="--")
        ax = fig.add_subplot(gs[i, 1], sharex=first)
        first = first or ax
        style(ax)
        ax.plot(t, ref[:, 2] * 100, color=muted, linewidth=1, label="pinch, MediaPipe (step 1.6)")
        ax.plot(t, pinch[:, 2] * 100, color=c, linewidth=1.8, label="pinch, CoTracker")
        ax.plot(t, cup_c[:, 2] * 100, color=c, linewidth=1.2, linestyle="--", label="mug points, mean")
        ax.set_ylim(-2, 30)
        ax.set_ylabel(f"demo {i + 1}", color=ink, fontsize=9)
        if i == 0:
            ax.set_title("Height above the table (cm)", color=ink, fontsize=10, loc="left")
            ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=ink, ncol=3)
        if i < len(results) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (s)", color=muted, fontsize=9)
    ax_top.set_title("Top view, table frame (cm): solid = pinch, dashed = mug", color=ink, fontsize=10, loc="left")
    ax_top.set_xlabel("y (left +)", color=muted, fontsize=9)
    ax_top.set_ylabel("x (away from camera +)", color=muted, fontsize=9)
    ax_top.invert_xaxis()
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=ink)
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "flow_table.png", dpi=130, facecolor="white", bbox_inches="tight")
    print("wrote", MEDIA / "flow_table.png")


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    plot([to_flow(d) for d in demos], [Path(d).name for d in demos])
