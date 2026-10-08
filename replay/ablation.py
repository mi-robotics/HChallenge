"""Which rim-grasp constraint fixes what: replay all 5 demos with every subset of the three constraints.

  python replay/ablation.py

Writes media/phase1/task9/ablation.json (success + cup-to-plate distance per subset and demo) and ablation.png
(one panel per constraint: the demo that fails without it, all three vs that one removed).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "replay"))
from replay import CLEAR_M, CONSTRAINTS, HZ, MEDIA, NEAR_M, make_env, replay, tag  # noqa: E402

DEMOS = sorted(p for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())


def run():
    subsets = [c for n in range(4) for c in itertools.combinations(CONSTRAINTS, n)]
    env = make_env(img_size=64)
    table = {"+".join(use) or "none": [replay(d, env, use) for d in DEMOS] for use in subsets}
    env.close()
    out = {k: [{"demo": d.name, "success": r["success"], "cup_final_xy_to_plate_cm": round(r["cup_final_xy_to_plate_cm"], 1),
                "cup_max_lift_cm": round(r["cup_max_lift_cm"], 1)} for d, r in zip(DEMOS, v)] for k, v in table.items()}
    (MEDIA / "ablation.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:22s} {sum(r['success'] for r in v)}/5  " + " ".join("✓" if r["success"] else "✗" for r in v))


def trace(demo, use):
    r = np.load(DEMOS[demo] / f"replay_trace{tag(use)}.npz")
    i_close = int(np.argmax(r["gripper"] > 0))
    i_release = i_close + int(np.argmax(r["gripper"][i_close:] < 0))
    return r, i_close, i_release


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = "#2a78d6", "#eb6834"  # dataviz categorical slots 1-2
    ink, muted, grid = "#1a1e26", "#5d6573", "#e3e6eb"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.subplots_adjust(wspace=0.36, top=0.76, bottom=0.14, left=0.05, right=0.99)

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, color=ink, fontsize=11, loc="left", pad=52)
        ax.set_xlabel(xlabel, color=muted, fontsize=9)
        ax.set_ylabel(ylabel, color=muted, fontsize=9)

    without = {c: tuple(x for x in CONSTRAINTS if x != c) for c in CONSTRAINTS}

    # (a) Approach, demo 3: side view of the path to the grasp, up to the fingers closing.
    ax = axes[0]
    for use, colour, label in [(CONSTRAINTS, blue, "all three"), (without["approach"], orange, "without approach")]:
        r, ic, _ = trace(2, use)
        off = r["eef"][:ic + 1] - r["grasp"]
        d, z = np.linalg.norm(off[:, :2], axis=1) * 100, off[:, 2] * 100
        ax.plot(d, z, color=colour, linewidth=2, label=label)
        moved = np.linalg.norm(r["cup"][:ic + 1, :2] - r["cup"][0, :2], axis=1) > 0.01
        if moved.any():
            k = int(np.argmax(moved))
            ax.plot(d[k], z[k], "X", color=colour, markersize=11, markeredgecolor="white")
            ax.annotate("mug pushed 1 cm\n(fingers still open)", (d[k], z[k]), xytext=(-10, 60), textcoords="offset points",
                        fontsize=9, color=ink, arrowprops={"arrowstyle": "-", "color": muted, "linewidth": 0.8})
    dd = np.linspace(0, 20, 200)
    ax.plot(dd, CLEAR_M * 100 * np.clip((dd / 100 - NEAR_M) / NEAR_M, 0, 1), color=muted, linewidth=1, linestyle="--",
            label="height floor (approach)")
    ax.plot(0, 0, "o", color=ink, markersize=6)
    ax.annotate("grasp point", (0, 0), xytext=(6, -14), textcoords="offset points", fontsize=9, color=ink)
    ax.set_xlim(20, -1)
    style(ax, "Approach · demo 3", "gripper's horizontal distance to the grasp point (cm)", "height above the grasp point (cm)")

    # (b) Settle, demo 5: gripper height around the fingers-close command.
    ax = axes[1]
    for use, colour, label in [(CONSTRAINTS, blue, "all three"), (without["settle"], orange, "without settle")]:
        r, ic, _ = trace(4, use)
        k = np.arange(max(0, ic - 30), ic + 40)
        lift = (r["cup"][:, 2] - r["cup"][0, 2]).max() * 100
        ax.plot((k - ic) / HZ, (r["eef"][k, 2] - r["grasp"][2]) * 100, color=colour, linewidth=2,
                label=f"{label}: cup lifted {lift:.1f} cm")
    ax.axvline(0, color=muted, linewidth=1, linestyle="--")
    ax.annotate("fingers close", (0, 14), xytext=(4, 0), textcoords="offset points", fontsize=9, color=ink)
    style(ax, "Settle · demo 5", "time from the fingers-close command (s)", "gripper height above the grasp point (cm)")

    # (c) Lift out, demo 2: cup distance to the plate centre after the fingers open.
    ax = axes[2]
    for use, colour, label in [(CONSTRAINTS, blue, "all three"), (without["lift"], orange, "without lift out")]:
        r, _, ir = trace(1, use)
        k = np.arange(ir - 10, len(r["cup"]))
        dist = np.linalg.norm(r["cup"][k, :2] - r["plate_xy"], axis=1) * 100
        final = json.loads((DEMOS[1] / f"replay{tag(use)}.json").read_text())["cup_final_xy_to_plate_cm"]
        ax.plot((k - ir) / HZ, dist, color=colour, linewidth=2, label=f"{label}: ends {final:.1f} cm off")
    ax.axvline(0, color=muted, linewidth=1, linestyle="--")
    ax.annotate("fingers open", (0, 3.6), xytext=(4, 0), textcoords="offset points", fontsize=9, color=ink)
    ax.set_ylim(0, 4.5)
    style(ax, "Lift out · demo 2", "time from the fingers-open command (s)", "mug's distance to the plate centre (cm)")

    for ax in axes:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, frameon=False, labelcolor=ink, ncol=1)
    fig.savefig(MEDIA / "ablation.png", dpi=130, facecolor="white")
    print("wrote", MEDIA / "ablation.png")


if __name__ == "__main__":
    run()
    plot()
