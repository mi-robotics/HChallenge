"""Split the dataset by episode into data/splits.json.

  python data/split.py

Kept episodes are shuffled (seed 0) into train (80), val (10) and test (10). Failed episodes are listed apart:
failed (replayed in sim, so they have robot data) and failed_no_robot (no usable hand track, flow only).
Also plots the cup start positions by split.
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent
MEDIA = DATA.parent / "media" / "phase4" / "task5"
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
    for key, folder, colour, filled in [("failed", "mug_pnp_failed", "#9aa1ad", False),
                                        ("failed_no_robot", "mug_pnp_failed", "#c9ced6", False),
                                        ("train", "mug_pnp_dataset", "#2a78d6", True), ("val", "mug_pnp_dataset", "#1baf7a", True),
                                        ("test", "mug_pnp_dataset", "#eda100", True)]:
        xy = np.array([cup_xy(DATA / folder / e) for e in splits[key]]) * 100
        label = f"{key} ({len(xy)})"
        ax.scatter(xy[:, 1], xy[:, 0], s=34, label=label, facecolor=colour if filled else "none",
                   edgecolor="white" if filled else colour, linewidth=1.2 if not filled else 0.8, zorder=3 if filled else 2)
    ax.scatter([0], [0], marker="s", s=60, color=ink, zorder=4, label="marker (origin)")
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
    kept = [e.name for e in episodes("mug_pnp_dataset", "sim.npz")]
    failed = episodes("mug_pnp_failed", "flow.npz")
    rng = np.random.default_rng(SEED)
    kept = [kept[i] for i in rng.permutation(len(kept))]
    splits = {"train": sorted(kept[N_VAL + N_TEST:]), "val": sorted(kept[:N_VAL]),
              "test": sorted(kept[N_VAL:N_VAL + N_TEST]),
              "failed": [e.name for e in failed if (e / "sim.npz").exists()],
              "failed_no_robot": [e.name for e in failed if not (e / "sim.npz").exists()],
              "rule": {"kept": "shuffled with seed 0: val, test, then train", "seed": SEED}}
    (DATA / "splits.json").write_text(json.dumps(splits, indent=1))
    plot(splits)
    print({k: len(v) for k, v in splits.items() if k != "rule"})
