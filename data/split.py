"""Split the dataset by episode (step 4.5). Writes data/splits.json.

  python data/split.py

Cup position = mean xy of the 8 mug flow points at step 0 (flow.npz, table frame). Kept episodes with the cup more
than 61.3 cm to the right of the marker (y < -0.613 m; 10 episodes) form test_novel: a table region no training episode starts in.
The other kept episodes are shuffled (seed 0) into val (10), test (10) and train (the rest). Failed episodes (no sim
actions, flow only) are listed apart; the ones in the held-out region go to failed_novel and must not be trained on.
Also writes media/phase4/task5/splits.png (top view of the cup positions by split).
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent
MEDIA = DATA.parent / "media" / "phase4" / "task5"
NOVEL_Y_M = -0.613
N_VAL = N_TEST = 10
SEED = 0


def cup_xy(ep):
    return np.load(ep / "flow.npz")["cup"][:, 0, :2].mean(0)


def plot(splits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1a1e26", "#5d6573", "#e3e6eb"
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(NOVEL_Y_M * 100, color=muted, linewidth=1, linestyle="--")
    for key, folder, colour, filled in [("failed", "mug_pnp_failed", "#9aa1ad", False),
                                        ("failed_novel", "mug_pnp_failed", "#9aa1ad", False),
                                        ("train", "mug_pnp_dataset", "#2a78d6", True), ("val", "mug_pnp_dataset", "#1baf7a", True),
                                        ("test", "mug_pnp_dataset", "#eda100", True),
                                        ("test_novel", "mug_pnp_dataset", "#eb6834", True)]:
        xy = np.array([cup_xy(DATA / folder / e) for e in splits[key]]) * 100
        label = f"{key} ({len(xy)})" if key != "failed_novel" else None
        if key == "failed":
            label = f"failed, flow only ({len(splits['failed']) + len(splits['failed_novel'])})"
        ax.scatter(xy[:, 1], xy[:, 0], s=34, label=label, facecolor=colour if filled else "none",
                   edgecolor="white" if filled else colour, linewidth=1.2 if not filled else 0.8, zorder=3 if filled else 2)
    ax.scatter([0], [0], marker="s", s=60, color=ink, zorder=4, label="marker (origin)")
    ax.text(NOVEL_Y_M * 100 - 1, 8, "held out: y < -61.3 cm", color=muted, fontsize=8, ha="left")
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("y, cm (left +)", color=muted, fontsize=9)
    ax.set_ylabel("x, cm (away from camera +)", color=muted, fontsize=9)
    ax.set_title("Cup start position per episode, table frame (top view)", color=ink, fontsize=10, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False, labelcolor=ink)
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "splits.png", dpi=130, facecolor="white", bbox_inches="tight")


def episodes(folder, need):
    return sorted(p for p in (DATA / folder).iterdir() if (p / need).exists())


if __name__ == "__main__":
    kept = episodes("mug_pnp_dataset", "sim.npz")
    failed = episodes("mug_pnp_failed", "flow.npz")
    novel = [e.name for e in kept if cup_xy(e)[1] < NOVEL_Y_M]
    rest = [e.name for e in kept if e.name not in novel]
    rng = np.random.default_rng(SEED)
    rest = [rest[i] for i in rng.permutation(len(rest))]
    splits = {"train": sorted(rest[N_VAL + N_TEST:]), "val": sorted(rest[:N_VAL]),
              "test": sorted(rest[N_VAL:N_VAL + N_TEST]), "test_novel": novel,
              "failed": [e.name for e in failed if cup_xy(e)[1] >= NOVEL_Y_M],
              "failed_novel": [e.name for e in failed if cup_xy(e)[1] < NOVEL_Y_M],
              "rule": {"cup_xy": "mean xy of the mug flow points at step 0, table frame (m)",
                       "test_novel": f"cup y < {NOVEL_Y_M}", "seed": SEED}}
    (DATA / "splits.json").write_text(json.dumps(splits, indent=1))
    plot(splits)
    print({k: len(v) for k, v in splits.items() if k != "rule"})
