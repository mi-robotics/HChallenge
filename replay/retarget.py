"""Map the cleaned hand trajectory to robot commands.

  python replay/retarget.py [data/demos/<demo> ...]     (default: all demos)

- Position: the pinch point (midpoint of thumb and index tips) drives the gripper's fingertip centre,
  relative to where the grasp happened. At replay time it is anchored at the sim rim-grasp point:
      target(t) = pinch(t) - pinch(t_grasp) + sim_grasp_point
  so the real mug (~9 cm) and sim mug (13.5 cm) heights cancel. No rotation or scale: the table
  frame's axes map onto the sim's (x away from the camera = away from the robot base).
- Gripper: one grasp per demo. Close at the first thumb-index distance < 2.5 cm; release at the first
  distance > 3.0 cm after the pinch has risen 3 cm above the grasp point (so a flicker while closing
  on the cup, or the hand relaxing after release, is ignored).

Writes retarget.npz to each demo folder:
  t [N], rel [N,3] pinch relative to the grasp point (m), gripper [N] (-1 open, +1 close),
  i_grasp, i_release, real_grasp [3], real_release [3] (table frame, m)
and media/phase1/task7/retarget.png.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media" / "phase1" / "task7"
CLOSE_BELOW, OPEN_ABOVE, LIFT_M = 0.025, 0.030, 0.03
OPEN, CLOSE = -1.0, 1.0


def grasp_events(aperture, pinch):
    i_grasp = int(np.flatnonzero(aperture < CLOSE_BELOW)[0])
    i_lift = i_grasp + int(np.flatnonzero(pinch[i_grasp:, 2] > pinch[i_grasp, 2] + LIFT_M)[0])
    i_release = i_lift + int(np.flatnonzero(aperture[i_lift:] > OPEN_ABOVE)[0])
    return i_grasp, i_release


def retarget(demo):
    demo = Path(demo)
    tr = np.load(demo / "traj.npz")
    pinch = (tr["thumb"] + tr["index"]) / 2
    i_grasp, i_release = grasp_events(tr["aperture"], pinch)
    gripper = np.full(len(pinch), OPEN)
    gripper[i_grasp:i_release] = CLOSE
    out = dict(t=tr["t"], rel=pinch - pinch[i_grasp], gripper=gripper, i_grasp=i_grasp, i_release=i_release,
               real_grasp=pinch[i_grasp], real_release=pinch[i_release])
    np.savez(demo / "retarget.npz", **out)
    t = tr["t"]
    print(f"{demo.name}: grasp {t[i_grasp]:.2f} s at {np.round(pinch[i_grasp] * 100, 1)} cm | release "
          f"{t[i_release]:.2f} s at {np.round(pinch[i_release] * 100, 1)} cm | release - grasp "
          f"{np.round((pinch[i_release] - pinch[i_grasp]) * 100, 1)} cm | max lift "
          f"{(pinch[i_grasp:i_release, 2].max() - pinch[i_grasp, 2]) * 100:.1f} cm")
    return out, tr["aperture"], pinch


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz categorical slots 1-5
    ink, muted, grid = "#1a1e26", "#5d6573", "#e3e6eb"
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(len(results), 2, width_ratios=[1, 1.3], hspace=0.25)
    ax_top = fig.add_subplot(gs[:, 0])
    ax_first = None

    def style(ax):
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    style(ax_top)
    for i, (out, aperture, pinch) in enumerate(results):
        c = colors[i]
        held = slice(out["i_grasp"], out["i_release"] + 1)
        ax_top.plot(pinch[:, 1] * 100, pinch[:, 0] * 100, color=c, linewidth=1, alpha=0.35)
        ax_top.plot(pinch[held, 1] * 100, pinch[held, 0] * 100, color=c, linewidth=2.2, label=f"demo {i + 1}")
        g, r = out["real_grasp"] * 100, out["real_release"] * 100
        ax_top.plot(g[1], g[0], "o", color=c, markersize=8, markeredgecolor="white")
        ax_top.plot(r[1], r[0], "X", color=c, markersize=9, markeredgecolor="white")
        ax_top.annotate(str(i + 1), (g[1], g[0]), xytext=(6, 4), textcoords="offset points", fontsize=9, color=ink)

        ax = fig.add_subplot(gs[i, 1], sharex=ax_first)
        ax_first = ax_first or ax
        style(ax)
        t = out["t"]
        ax.axvspan(t[out["i_grasp"]], t[out["i_release"]], color=c, alpha=0.15, linewidth=0)
        ax.plot(t, aperture * 100, color=c, linewidth=2)
        for y in (CLOSE_BELOW, OPEN_ABOVE):
            ax.axhline(y * 100, color=muted, linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 8)
        ax.set_ylabel(f"demo {i + 1}", color=ink, fontsize=9)
        if i == 0:
            ax.set_title("Thumb–index distance (cm); shaded = gripper closed; dashed = 2.5 / 3.0 cm",
                         color=ink, fontsize=10, loc="left")
        if i < len(results) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (s)", color=muted, fontsize=9)
    ax_top.invert_xaxis()
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("y (cm), table frame", color=muted, fontsize=9)
    ax_top.set_ylabel("x (cm), table frame", color=muted, fontsize=9)
    ax_top.set_title("Pinch point, top-down: ● grasp  ✕ release\n(bold = holding the cup)", color=ink, fontsize=10, loc="left")
    ax_top.legend(frameon=False, fontsize=8, labelcolor=ink, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "retarget.png", dpi=150, facecolor="white", bbox_inches="tight")
    print("wrote", MEDIA / "retarget.png")


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    results = [retarget(d) for d in demos]
    rel = np.array([r[0]["real_release"] for r in results]) * 100
    print(f"release points: spread (std) x {rel[:, 0].std():.1f} y {rel[:, 1].std():.1f} z {rel[:, 2].std():.1f} cm "
          "(the saucer did not move between demos)")
    plot(results)
