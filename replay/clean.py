"""Clean the table-frame hand trajectory and resample it to 20 Hz.

  python replay/clean.py [<episode> ...]     (default: data/demos)

Per keypoint: drop spikes (> 3 cm from a rolling median, or below the table), fill gaps up to 1.5 s, smooth with a
zero-phase Gaussian, resample to 20 Hz. Writes traj.npz (t, wrist / thumb / index [N, 3], aperture, filled).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_hand import INDEX_TIP, THUMB_TIP, WRIST  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media" / "phase1" / "task6"
HZ = 20
KEYPOINTS = {"wrist": WRIST, "thumb": THUMB_TIP, "index": INDEX_TIP}
SPIKE_M, BELOW_TABLE_M, MAX_GAP_S, SIGMA_FRAMES = 0.03, -0.01, 1.5, 2.0


def rolling_median(x, w=7):
    """Median over a centred window, ignoring NaN. x [T,3]."""
    T, h = len(x), w // 2
    out = np.full_like(x, np.nan)
    for i in range(T):
        win = x[max(0, i - h):i + h + 1]
        if np.isfinite(win[:, 0]).any():
            out[i] = np.nanmedian(win, axis=0)
    return out


def gaussian_smooth(x, sigma):
    k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    k /= k.sum()
    pad = len(k) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def clean_keypoint(x, t, fps):
    """x [T,3] with NaN where invalid. Returns cleaned [T,3], filled mask [T], n spikes dropped."""
    x = x.copy()
    ok = np.isfinite(x[:, 0])
    med = rolling_median(x)
    spike = ok & ((np.linalg.norm(x - med, axis=1) > SPIKE_M) | (x[:, 2] < BELOW_TABLE_M))
    x[spike] = np.nan
    ok = np.isfinite(x[:, 0])

    # Gaps longer than MAX_GAP_S stay NaN (none in the current demos).
    idx = np.flatnonzero(ok)
    filled = ~ok
    for j in range(3):
        x[~ok, j] = np.interp(t[~ok], t[ok], x[ok, j])
    gap_starts = np.flatnonzero(np.diff(ok.astype(int)) == -1) + 1
    for s in gap_starts:
        e = idx[idx > s][0] if (idx > s).any() else len(x)
        if (e - s) / fps > MAX_GAP_S:
            x[s:e] = np.nan
    return gaussian_smooth(x, SIGMA_FRAMES), filled, int(spike.sum())


def clean(demo):
    demo = Path(demo)
    hand = np.load(demo / "hand.npz")
    p = np.load(demo / "hand_table.npz")["p"]
    t = (hand["t_ms"] - hand["t_ms"][0]) / 1000
    fps = (len(t) - 1) / t[-1]
    t20 = np.arange(0, t[-1], 1 / HZ)

    out, raw, filled_any = {"t": t20}, {}, np.zeros(len(t), bool)
    report = []
    for name, k in KEYPOINTS.items():
        xc, filled, n_spike = clean_keypoint(p[:, k], t, fps)
        filled_any |= filled
        raw[name] = p[:, k]
        out[name] = np.stack([np.interp(t20, t, xc[:, j]) for j in range(3)], 1)
        report.append(f"{name}: {n_spike} spikes, {filled.sum()} filled")
    out["aperture"] = np.linalg.norm(out["thumb"] - out["index"], axis=1)
    out["filled"] = np.interp(t20, t, filled_any.astype(float)) > 0.5
    np.savez(demo / "traj.npz", **out)

    # Jitter: RMS of the wrist's frame-to-frame acceleration, raw (valid frames) vs cleaned, at 30 Hz.
    def acc_rms(x):
        a = np.diff(x, 2, axis=0) * fps ** 2
        return np.sqrt(np.nanmean(np.sum(a ** 2, axis=1)))
    xc_w, _, _ = clean_keypoint(p[:, WRIST], t, fps)
    print(f"{demo.name}: {len(t)} frames -> {len(t20)} samples at {HZ} Hz | " + " | ".join(report) +
          f" | wrist accel RMS raw {acc_rms(p[:, WRIST]):.1f} -> clean {acc_rms(xc_w):.1f} m/s^2 | "
          f"aperture {out['aperture'].min() * 100:.1f}..{out['aperture'].max() * 100:.1f} cm")
    return t, raw, out


def plot(t, raw, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, c_raw, c_clean = "#1a1e26", "#5d6573", "#e3e6eb", "#9aa3b2", "#2a78d6"
    rows = [("wrist x", raw["wrist"][:, 0], out["wrist"][:, 0]), ("wrist y", raw["wrist"][:, 1], out["wrist"][:, 1]),
            ("wrist z", raw["wrist"][:, 2], out["wrist"][:, 2]), ("index tip z", raw["index"][:, 2], out["index"][:, 2]),
            ("thumb–index", np.linalg.norm(raw["thumb"] - raw["index"], axis=1), out["aperture"])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 8.5), sharex=True)
    for ax, (label, r, c) in zip(axes, rows):
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s, e in _runs(out["filled"]):
            ax.axvspan(out["t"][s], out["t"][min(e, len(out["t"]) - 1)], color="#eda100", alpha=0.15, linewidth=0)
        ax.plot(t, r * 100, ".", color=c_raw, markersize=3, label="raw 30 Hz")
        ax.plot(out["t"], c * 100, color=c_clean, linewidth=2, label="cleaned 20 Hz")
        ax.set_ylabel(f"{label} (cm)", color=muted, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink, ncol=2, loc="upper right")
    axes[0].set_title(f"{title}: raw vs cleaned (shaded = filled gap)", color=ink, fontsize=11, loc="left")
    axes[-1].set_xlabel("time (s)", color=muted, fontsize=9)
    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "clean.png", dpi=150, facecolor="white")
    print("wrote", MEDIA / "clean.png")


def _runs(mask):
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    results = [clean(d) for d in demos]
    # Plot the demo with the most filled samples, where cleaning matters most.
    i = int(np.argmax([r[2]["filled"].sum() for r in results]))
    plot(*results[i], f"demo {i + 1}")
