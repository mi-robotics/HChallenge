"""Check the mug flow's depth against vision alone: per-frame PnP of the 8 tracked mug points.

  python flow/mug_check.py [data/demos/<demo> ...]     (default: all demos)

The mug flow (flow/to_flow.py) takes each mug point's image position from CoTracker and its depth from the stand-in
mug that moves with the hand (the D405 has no depth on the white mug). PnP needs no hand: the 8 points' layout on
the mug (as in the flow, first frame) + their tracked pixels -> mug pose, in frames with >= 5 visible mug points.

Result (see PROJECT_PLAN.md 3.4): PnP agrees with the stand-in while the mug stands still, but during the carry it
drifts up to ~11 cm along the camera's line of sight and jumps back at the release (with ~100 px between the
points, a small spread in the tracks reads as a large change in distance). So PnP checks the still phases and the
stand-in depth stays.

Writes media/phase3/task4/mug_check.png.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT / "replay"))
from to_flow import tracked_hand, tracked_mug  # noqa: E402

MEDIA = ROOT / "media" / "phase3" / "task4"
MIN_POINTS = 5


def check(demo):
    demo = Path(demo)
    rt = np.load(demo / "retarget.npz")
    t, hand, _, _, (K, dist, R, tv, ct) = tracked_hand(demo)
    tv = np.ravel(tv)
    n = len(t)
    cup = tracked_mug(demo, t, hand, K, dist, R, tv, ct)  # [T, 8, 3], what the flow uses
    standin, model = cup.mean(1), cup[0]  # mug-point centre [T, 3]; the 8 points' layout (first frame)
    i_grasp, i_release = (int(np.abs(t - rt["t"][int(rt[k])]).argmin()) for k in ("i_grasp", "i_release"))

    pnp = np.full((n, 3), np.nan)
    rvec, tvec = cv2.Rodrigues(R)[0], tv.copy()
    for i in range(n):
        v = ct["cup_visible"][i]
        if v.sum() < MIN_POINTS:
            continue
        _, rvec, tvec = cv2.solvePnP(model[v], ct["cup_uv"][i][v].astype(np.float64), K, dist, rvec, tvec,
                                     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        pnp[i] = ((model @ cv2.Rodrigues(rvec)[0].T + tvec.ravel() - tv) @ R).mean(0)
    diff = np.linalg.norm(pnp - standin, axis=1) * 1000
    still = (np.arange(n) < i_grasp) | (np.arange(n) > i_release)
    carry = ~still
    print(f"{demo.name}: PnP in {np.isfinite(diff).mean() * 100:.0f}% of frames | PnP vs flow, mug standing still: "
          f"median {np.nanmedian(diff[still]):.0f} mm, p95 {np.nanpercentile(diff[still], 95):.0f} mm | carrying: median "
          f"{np.nanmedian(diff[carry]):.0f} mm, max {np.nanmax(diff[carry]):.0f} mm")
    return t, standin, pnp, t[i_grasp], t[i_release]


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = "#2a78d6", "#eb6834"  # dataviz categorical slots 1-2
    ink, muted, grid = "#1a1e26", "#5d6573", "#e3e6eb"
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 1.7 * len(results)), sharex=True)
    for i, (ax, (t, standin, pnp, tg, tr)) in enumerate(zip(axes, results)):
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvspan(tg, tr, color=grid, alpha=0.6, linewidth=0)
        ax.plot(t, standin[:, 2] * 100, color=blue, linewidth=2, label="flow: tracked pixels, hand-derived depth")
        ax.plot(t, pnp[:, 2] * 100, color=orange, linewidth=1.5, label="PnP from the tracked mug points")
        ax.set_ylim(0, 25)
        ax.set_ylabel(f"demo {i + 1}", color=ink, fontsize=9)
        if i == 0:
            ax.set_title("Mug height above the table (cm), mean of the 8 points; shaded = carried", color=ink,
                         fontsize=10, loc="left")
            ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=ink, ncol=2)
    axes[-1].set_xlabel("time (s)", color=muted, fontsize=9)
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "mug_check.png", dpi=130, facecolor="white", bbox_inches="tight")
    print("wrote", MEDIA / "mug_check.png")


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    plot([check(d) for d in demos])
