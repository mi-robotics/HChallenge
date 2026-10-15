"""Replay kept episodes in sim and save what the robot did and saw per control step (20 Hz) to sim.npz.

  python replay/record_sim.py [--video out.mp4 [--flow]] [<episode> ...]   (default: all kept episodes)

sim.npz: action [N, 7], joint_pos / joint_vel, gripper_qpos, eef_pos / eef_quat, cup and plate state, agentview and
eye_in_hand images [N, 256, 256, 3], t_real (time in the real video each step follows), sim_minus_real (table frame
-> sim world offset), success. --video renders the real video next to both sim cameras; --flow draws the flow.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "replay"))
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.append(str(ROOT))
from dataset import Episode  # noqa: E402
from replay import HZ, make_env, replay  # noqa: E402

IMG = 256  # env camera size: both views are saved at this size


def project_third_person(pts, size=IMG):
    """World points [..., 3] -> pixels [..., 2] in the third-person image (sim/scene.py THIRD_PERSON)."""
    from scene import THIRD_PERSON, third_person_pose
    pos, R = third_person_pose()
    pc = (pts - pos) @ R
    f = size / 2 / np.tan(np.radians(THIRD_PERSON["fovy_deg"]) / 2)
    return f * pc[..., :2] / pc[..., 2:] + size / 2


def render(episodes, out_path, cell=256, flow=False):
    if flow:
        import json
        from format import HAND_POINTS, N_CUP_POINTS, T
        from sim_clock import SIM_HZ
        from to_table import demo_marker_pose
        from track_points import HAND_COLOUR, YELLOW
        from visualise import draw
    rows = []
    for k, ep in enumerate(episodes):
        ep = Path(ep)
        sim = np.load(ep / "sim.npz")
        t_cam = np.loadtxt(ep / "frames.csv", delimiter=",", skiprows=1)[:, 2]
        t_cam = (t_cam - t_cam[0]) / 1000
        cap, real = cv2.VideoCapture(str(ep / "color.mp4")), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            real.append(cv2.resize(f, (cell * 16 // 9 // 2 * 2, cell)))  # even width for H.264
        if flow:
            pts = Episode(ep).flow.transpose(1, 0, 2)  # [14, N_sim, 4], table frame, one row per sim step
            uv = project_third_person(pts[..., :3] + sim["sim_minus_real"]) * cell / IMG
            meta = json.loads((ep / "meta.json").read_text())
            K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
            R, tv, _ = demo_marker_pose(ep, K, dist)
            uv_real = cv2.projectPoints(pts[..., :3].reshape(-1, 3).astype(np.float64), cv2.Rodrigues(R)[0], tv, K,
                                        dist)[0].reshape(*pts.shape[:2], 2) * cell / meta["image_size"][1]
            colours, n_hand, N = HAND_COLOUR + [YELLOW] * N_CUP_POINTS, len(HAND_POINTS), pts.shape[1]
        frames = []
        for i in range(len(sim["action"])):
            r = real[min(int(np.searchsorted(t_cam, sim["t_real"][i])), len(real) - 1)].copy()
            views = [cv2.resize(np.ascontiguousarray(sim[v][i][:, :, ::-1]), (cell, cell), interpolation=cv2.INTER_NEAREST)
                     for v in ("agentview", "eye_in_hand")]
            if flow:
                st = np.minimum(i + np.arange(T) * SIM_HZ // 10, N - 1)  # 16 steps at 10 Hz of robot time
                draw(r, uv_real[:, st], pts[:, st, 3] > 0, colours, n_hand, scale=0.4)
                draw(views[0], uv[:, st], pts[:, st, 3] > 0, colours, n_hand, scale=0.4)
                if i and sim["t_real"][i] == sim["t_real"][i - 1]:  # a step the replay inserted: real time holds
                    cv2.putText(r, "real time paused", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
                g = project_third_person(sim["eef_pos"][i]) * cell / IMG  # robot gripper (eef site): white x
                cv2.drawMarker(views[0], tuple(int(v) for v in g), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
            for img, label in zip([r] + views, [f"episode {k + 1}: real", "sim: third person" + (" + flow" if flow else ""),
                                                "sim: gripper camera"]):
                cv2.putText(img, label, (8, cell - 10 if flow else 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(np.hstack([r] + views))
        rows.append(frames)
    n = max(len(f) for f in rows)
    h, w = rows[0][0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{w}x{h * len(rows)}", "-r", str(HZ), "-i", "-", "-c:v", "libx264", "-crf", "24",
                           "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(np.vstack([f[min(i, len(f) - 1)] for f in rows]).tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    video, flow = None, "--flow" in args
    if flow:
        args.remove("--flow")
    if "--video" in args:
        i = args.index("--video")
        video = Path(args[i + 1])
        del args[i:i + 2]
    if video and flow:  # render only, from the saved sim.npz and flow_sim.npz
        video.parent.mkdir(parents=True, exist_ok=True)
        render(args, video, flow=True)
        sys.exit()
    episodes = args or sorted(str(p) for p in (ROOT / "data" / "mug_pnp_dataset").iterdir() if p.is_dir())
    env = make_env(img_size=IMG)
    results = [replay(ep, env, record=True) for ep in episodes]
    env.close()
    print(f"success {sum(r['success'] for r in results)}/{len(results)}")
    if video:
        video.parent.mkdir(parents=True, exist_ok=True)
        render(episodes, video)
