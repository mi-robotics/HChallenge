"""Match the sim's starting scene to each demo.

  python replay/scene_match.py [<episode> ...]     (default: data/demos)

The mug and plate are white (no D405 depth), so they come from the hand: the pinch point at the grasp is on the cup,
at the release over the plate. sim cup = SIM_CUP_XY, sim plate = SIM_CUP_XY + (release - grasp)_xy. A sim camera
is placed at the D405's pose for side-by-side videos. Writes scene.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "replay"))
from scene import CUP, PLATE, make_env, place, render, reset, rim_grasp_point, set_camera  # noqa: E402
from to_table import demo_marker_pose  # noqa: E402

MEDIA = ROOT / "media" / "phase1" / "task8"
SIM_CUP_XY = np.array([-0.11, 0.0])
ROBOT_BASE = np.array([-0.51, 0.0, 0.42])
PANDA_REACH_M = 0.80  # spec 0.855 m; keep a margin
SAUCER_TOP_M = 0.015
SIM_CAMERA = "frontview"


def real_camera(demo):
    """D405 pose in the table frame (OpenCV convention) and its vertical field of view."""
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, t, _ = demo_marker_pose(demo, K, dist)
    fovy = np.degrees(2 * np.arctan(meta["image_size"][1] / 2 / K[1, 1]))
    return -R.T @ t, R.T, fovy


def first_frame_with_points(demo, points):
    meta = json.loads((demo / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, t, _ = demo_marker_pose(demo, K, dist)
    img = cv2.VideoCapture(str(demo / "color.mp4")).read()[1]
    uv = cv2.projectPoints(np.array(points), cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2).astype(int)
    for (u, v), label, colour in zip(uv, ["cup (grasp)", "plate (release xy)"], [(40, 40, 230), (230, 120, 40)]):
        cv2.drawMarker(img, (u, v), colour, cv2.MARKER_CROSS, 40, 5, cv2.LINE_AA)
        cv2.putText(img, label, (u + 25, v - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3, cv2.LINE_AA)
    return img


def match(demo, env):
    demo = Path(demo)
    rt = np.load(demo / "retarget.npz")
    grasp, release = rt["real_grasp"], rt["real_release"]
    cup_xy = SIM_CUP_XY
    plate_xy = SIM_CUP_XY + (release - grasp)[:2]

    reset(env)
    obs = place(env, cup_xy, plate_xy)
    sim_grasp = rim_grasp_point(obs)
    targets = rt["rel"] + sim_grasp
    reach = np.linalg.norm(targets - ROBOT_BASE, axis=1)
    table_z = obs[f"{CUP}_pos"][2]
    # Sim camera at the D405's pose relative to the cup grasp point (table axes = sim world axes).
    cam_pos, cam_R, fovy = real_camera(demo)
    sim_cam_pos = cam_pos - grasp + sim_grasp
    set_camera(env, SIM_CAMERA, sim_cam_pos, cam_R, fovy)
    info = {
        "sim_camera": {"name": SIM_CAMERA, "pos": sim_cam_pos.tolist(), "R_world_from_cv": cam_R.tolist(),
                       "fovy_deg": float(fovy)},
        "sim_cup_xy": obs[f"{CUP}_pos"][:2].tolist(),
        "sim_plate_xy": obs[f"{PLATE}_pos"][:2].tolist(),
        "sim_grasp_point": sim_grasp.tolist(),
        "plate_minus_cup_cm": ((release - grasp)[:2] * 100).round(1).tolist(),
        "max_reach_m": float(reach.max()),
        "min_target_height_above_table_m": float(targets[:, 2].min() - table_z),
    }
    (demo / "scene.json").write_text(json.dumps(info, indent=2))
    ok = reach.max() < PANDA_REACH_M and info["min_target_height_above_table_m"] > 0
    print(f"{demo.name}: plate - cup {info['plate_minus_cup_cm']} cm | sim cup {np.round(cup_xy, 3)} plate "
          f"{np.round(info['sim_plate_xy'], 3)} | max distance from robot base {reach.max():.2f} m | lowest target "
          f"{info['min_target_height_above_table_m'] * 100:+.1f} cm above table | {'OK' if ok else 'CHECK'}")
    real = first_frame_with_points(demo, [grasp, np.r_[release[:2], SAUCER_TOP_M]])
    sim = np.ascontiguousarray(render(env, SIM_CAMERA, 640, 360)[:, :, ::-1])  # RGB -> BGR
    return real, sim


def figure(rows):
    tiles = []
    for i, (real, sim) in enumerate(rows):
        h = 360
        real = cv2.resize(real, (640, h))
        cv2.putText(real, f"demo {i + 1}: real, first frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(sim, "sim start, camera at the D405's pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.hstack([real, np.full((h, 8, 3), 255, np.uint8), sim]))
    gap = np.full((8, tiles[0].shape[1], 3), 255, np.uint8)
    out = np.vstack([x for t in tiles for x in (t, gap)][:-1])
    MEDIA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MEDIA / "scene_match.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", MEDIA / "scene_match.jpg")


if __name__ == "__main__":
    demos = sys.argv[1:] or sorted(str(p) for p in (ROOT / "data" / "demos").iterdir() if p.is_dir())
    env = make_env(img_size=64)
    rows = [match(d, env) for d in demos]
    env.close()
    figure(rows)
