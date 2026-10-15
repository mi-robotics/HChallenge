"""Map the cleaned hand trajectory to robot commands.

  python replay/retarget.py [<episode> ...]     (default: data/demos)

Position: the pinch point (thumb / index midpoint) relative to the grasp; at replay it is anchored at the sim
rim-grasp point. Gripper: close at the first thumb-index distance < 2.5 cm, open at the first > 3 cm after the pinch
has risen 3 cm. Writes retarget.npz (t, rel, gripper, grasp / release indices and points).
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_table import demo_marker_pose  # noqa: E402
from track_hand import INDEX_TIP, THUMB_TIP  # noqa: E402

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


def render(demo, out):
    """Draw the signal that drives the robot on the colour video: the cleaned pinch point (projected from
    the table frame back into the image) with a 1 s trail, the raw thumb/index tips, and the gripper state."""
    demo = Path(demo)
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, t, _ = demo_marker_pose(demo, K, dist)
    hand = np.load(demo / "hand.npz")
    t_cam = (hand["t_ms"] - hand["t_ms"][0]) / 1000
    pinch = out["rel"] + out["real_grasp"]  # table frame, 20 Hz
    pinch30 = np.stack([np.interp(t_cam, out["t"], pinch[:, j]) for j in range(3)], 1)
    uv = cv2.projectPoints(pinch30, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
    closed = np.interp(t_cam, out["t"], (out["gripper"] > 0).astype(float)) > 0.5
    aperture = np.interp(t_cam, out["t"], np.linalg.norm(  # cleaned thumb-index distance
        np.load(demo / "traj.npz")["thumb"] - np.load(demo / "traj.npz")["index"], axis=1))

    w, h = meta["image_size"]
    cap = cv2.VideoCapture(str(demo / "color.mp4"))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", str(meta["fps"]), "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                           str(demo / "pinch_overlay.mp4")], stdin=subprocess.PIPE)
    green, red, white = (60, 200, 60), (40, 40, 230), (255, 255, 255)
    trail = int(meta["fps"])
    for i in range(len(t_cam)):
        ok, img = cap.read()
        if not ok:
            break
        c = red if closed[i] else green
        for j in range(max(0, i - trail), i):
            cv2.line(img, tuple(uv[j].astype(int)), tuple(uv[j + 1].astype(int)), c, 3, cv2.LINE_AA)
        if hand["detected"][i]:
            th, ix = hand["uv"][i, THUMB_TIP].astype(int), hand["uv"][i, INDEX_TIP].astype(int)
            cv2.line(img, tuple(th), tuple(ix), white, 2, cv2.LINE_AA)
            for q in (th, ix):
                cv2.circle(img, tuple(q), 6, white, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(uv[i].astype(int)), 13, c, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(uv[i].astype(int)), 13, white, 2, cv2.LINE_AA)
        label = f"thumb-index {aperture[i] * 100:4.1f} cm   gripper {'CLOSED' if closed[i] else 'OPEN'}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (12, 12), (36 + tw, 62), (0, 0, 0), -1)
        cv2.putText(img, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2, cv2.LINE_AA)
        if not hand["detected"][i]:
            cv2.putText(img, "hand not detected: interpolated", (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 255), 2, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()


def tile(demos, name, out_path):
    """5 demo videos -> one 3x2 grid (640x360 tiles)."""
    n = len(demos)
    f = "".join(f"[{i}:v]scale=640:360,drawtext=text='demo {i + 1}':x=10:y=330:fontsize=22:fontcolor=white:"
                f"box=1:boxcolor=black@0.5[v{i}];" for i in range(n))
    layout = "|".join(["0_0", "w0_0", "w0+w1_0", "0_h0", "w0_h0", "w0+w1_h0"][:n + 1])
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for d in demos:
        cmd += ["-i", str(Path(d) / name)]
    cmd += ["-f", "lavfi", "-i", "color=c=black:s=640x360:r=30", "-filter_complex",
            f"{f}[{n}:v]trim=duration=12[v{n}];{''.join(f'[v{i}]' for i in range(n + 1))}"
            f"xstack=inputs={n + 1}:layout={layout}:shortest=0,trim=duration=11.3[out]",
            "-map", "[out]", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", str(out_path)]
    subprocess.run(cmd, check=True)
    print("wrote", out_path)


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    results = [retarget(d) for d in demos]
    rel = np.array([r[0]["real_release"] for r in results]) * 100
    print(f"release points: spread (std) x {rel[:, 0].std():.1f} y {rel[:, 1].std():.1f} z {rel[:, 2].std():.1f} cm "
          "(the saucer did not move between demos)")
    plot(results)
    for d, r in zip(demos, results):
        render(d, r[0])
    tile(demos, "pinch_overlay.mp4", MEDIA / "pinch.mp4")
