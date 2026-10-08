"""Intel RealSense D405: colour + aligned depth, a live viewer for aiming, and a table-frame check.

  python calibration/d405.py            live view; draws the table-marker axes when it is visible
                                        keys: s = save snapshot to data/calibration/, q = quit
  python calibration/d405.py check      1 s capture: marker detection rate, pose jitter, depth vs
                                        marker plane; saves media/phase1/task2/table_frame.jpg
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from markers import marker_pose
from make_targets import MARKER_MM_PRINTED

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "data" / "calibration"
MEDIA = ROOT / "media" / "phase1" / "task2"
# Depth colour scale for display. (The D405 is most accurate 7-50 cm from the scene.)
DISPLAY_RANGE_M = (0.2, 0.8)


class D405:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # Factory calibration: no checkerboard needed.
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5])
        self.size = (intr.width, intr.height)

    def read_raw(self):
        """Returns (bgr uint8 HxWx3, raw depth uint16 HxW aligned to colour, timestamp ms, frame number).
        Raw depth x depth_scale = metres."""
        frames = self.align.process(self.pipe.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth = np.asanyarray(frames.get_depth_frame().get_data()).copy()
        return color, depth, frames.get_timestamp(), frames.get_color_frame().get_frame_number()

    def read(self):
        """Returns (bgr uint8 HxWx3, depth in metres float32 HxW aligned to colour, timestamp ms)."""
        color, depth, t, _ = self.read_raw()
        return color, depth.astype(np.float32) * self.depth_scale, t

    def close(self):
        self.pipe.stop()


def colorize(depth):
    d = np.clip((depth - DISPLAY_RANGE_M[0]) / (DISPLAY_RANGE_M[1] - DISPLAY_RANGE_M[0]), 0, 1)
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth == 0] = 0  # no depth
    return img


def view():
    cam = D405()
    print(f"colour {cam.size[0]}x{cam.size[1]}  fx={cam.K[0, 0]:.1f} fy={cam.K[1, 1]:.1f} "
          f"cx={cam.K[0, 2]:.1f} cy={cam.K[1, 2]:.1f}")
    t0, n = time.time(), 0
    try:
        while True:
            color, depth, _ = cam.read()
            vis = color.copy()
            pose = marker_pose(color, cam.K, cam.dist)
            if pose is not None:
                rvec, tvec = pose
                cv2.drawFrameAxes(vis, cam.K, cam.dist, rvec, tvec, MARKER_MM_PRINTED / 1000, 3)
                R, _ = cv2.Rodrigues(rvec)
                cam_pos = (-R.T @ tvec).ravel() * 100
                msg = (f"marker {np.linalg.norm(tvec) * 100:.0f} cm away | camera in table frame "
                       f"x={cam_pos[0]:.0f} y={cam_pos[1]:.0f} z={cam_pos[2]:.0f} cm")
                colour = (0, 200, 0)
            else:
                msg, colour = "table marker not visible", (0, 0, 255)
            n += 1
            fps = n / (time.time() - t0)
            cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{fps:.0f} fps   s: snapshot   q: quit", (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("D405  colour | depth (red = near, blue = far, black = no depth)",
                       np.hstack([vis, colorize(depth)]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                SNAP_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%H%M%S")
                cv2.imwrite(str(SNAP_DIR / f"table_{stamp}.png"), color)
                np.save(SNAP_DIR / f"table_{stamp}_depth.npy", depth)
                print("saved", SNAP_DIR / f"table_{stamp}.png")
    finally:
        cam.close()
        cv2.destroyAllWindows()


def check(n_frames=30):
    cam = D405()
    for _ in range(15):  # let auto-exposure settle
        cam.read()
    frames = [cam.read()[:2] for _ in range(n_frames)]
    cam.close()
    poses = [(c, d, marker_pose(c, cam.K, cam.dist)) for c, d in frames]
    found = [p for p in poses if p[2] is not None]
    print(f"marker detected in {len(found)}/{n_frames} frames")
    if not found:
        raise SystemExit("Table marker not visible.")
    cam_pos = []
    for _, _, (rvec, tvec) in found:
        R, _ = cv2.Rodrigues(rvec)
        cam_pos.append((-R.T @ tvec).ravel())
    cam_pos = np.array(cam_pos) * 100
    print("camera in table frame (cm): x={:.1f} y={:.1f} z={:.1f}".format(*cam_pos.mean(0)),
          " jitter (std, mm): " + " ".join(f"{v:.1f}" for v in cam_pos.std(0) * 10))

    # Depth sensor vs the marker plane: points on a grid inside the marker, predicted from the pose.
    color, _, (rvec, tvec) = found[len(found) // 2]
    depth = np.median(np.stack([d for _, d, _ in found]), 0)
    R, _ = cv2.Rodrigues(rvec)
    h = MARKER_MM_PRINTED / 2000 * 0.8
    g = np.linspace(-h, h, 5)
    diffs = []
    for x in g:
        for y in g:
            p = R @ np.array([x, y, 0]) + tvec.ravel()
            u, v = (cam.K @ p)[:2] / p[2]
            patch = depth[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3]
            if (patch > 0).any():
                diffs.append(np.median(patch[patch > 0]) - p[2])
    diffs = np.array(diffs) * 1000
    print(f"depth sensor minus marker plane over {len(diffs)} points: mean {diffs.mean():+.1f} mm, "
          f"abs max {np.abs(diffs).max():.1f} mm (marker {np.linalg.norm(tvec) * 100:.1f} cm away)")

    vis = color.copy()
    cv2.drawFrameAxes(vis, cam.K, cam.dist, rvec, tvec, MARKER_MM_PRINTED / 1000, 4)
    MEDIA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MEDIA / "table_frame.jpg"), np.hstack([vis, colorize(depth)]))
    print("wrote", MEDIA / "table_frame.jpg", "(red = x, green = y, blue = z up)")


if __name__ == "__main__":
    import sys
    check() if sys.argv[1:] == ["check"] else view()
