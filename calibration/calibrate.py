"""Phone camera calibration and table-marker check.

  python calibration/calibrate.py intrinsics data/calibration/board.mp4
      ChArUco board video -> calibration/phone_camera.yaml (camera matrix, distortion, RMS error)

  python calibration/calibrate.py check data/calibration/table.mp4
      Photo or short clip (first frame is used) of the table. Finds the table marker, draws its axes, prints the camera pose relative to the marker.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from make_targets import (BOARD_DICT, BOARD_MARKER_MM, BOARD_SQUARES, MARKER_DICT, MARKER_ID,
                          MARKER_MM, SQUARE_MM)

ROOT = Path(__file__).resolve().parents[1]
CAMERA_YAML = Path(__file__).with_name("phone_camera.yaml")
MEDIA = ROOT / "media" / "phase1" / "task2"


def board():
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_MM / 1000, BOARD_MARKER_MM / 1000,
                                  cv2.aruco.getPredefinedDictionary(BOARD_DICT))


def read_frames(path, every=5):
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.iterdir()):
            img = cv2.imread(str(p))
            if img is not None:
                yield p.name, img
        return
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield f"frame {i}", img
        i += 1


def intrinsics(source, max_views=60, min_corners=10):
    b = board()
    detector = cv2.aruco.CharucoDetector(b)
    views, size = [], None
    for name, img in read_frames(source):
        size = img.shape[1], img.shape[0]
        corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is not None and len(ids) >= min_corners:
            obj, imgp = b.matchImagePoints(corners, ids)
            views.append((name, obj, imgp, len(ids)))
    total = (BOARD_SQUARES[0] - 1) * (BOARD_SQUARES[1] - 1)
    print(f"usable views: {len(views)} (>= {min_corners} of {total} corners)")
    if len(views) < 10:
        raise SystemExit("Need at least 10 usable views: record more of the board, closer and steadier.")
    # Spread the views evenly over the video rather than taking the first N.
    views = [views[i] for i in np.linspace(0, len(views) - 1, min(max_views, len(views))).astype(int)]
    rms, K, dist, _, _ = cv2.calibrateCamera([v[1] for v in views], [v[2] for v in views], size, None, None)
    print(f"views used: {len(views)}  image: {size[0]}x{size[1]}  RMS reprojection error: {rms:.3f} px")
    print("fx fy cx cy:", np.round([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], 1))
    print("distortion:", np.round(dist.ravel(), 4))
    fs = cv2.FileStorage(str(CAMERA_YAML), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", size[0])
    fs.write("image_height", size[1])
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("rms_px", rms)
    fs.write("views", len(views))
    fs.release()
    print("wrote", CAMERA_YAML)


def load_camera():
    fs = cv2.FileStorage(str(CAMERA_YAML), cv2.FILE_STORAGE_READ)
    K, dist = fs.getNode("camera_matrix").mat(), fs.getNode("dist_coeffs").mat()
    size = int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())
    fs.release()
    return K, dist, size


def marker_pose(img, K, dist):
    """Table frame (marker centre, x/y on the table, z up) -> camera. Returns (rvec, tvec) or None."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None or MARKER_ID not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2)
    h = MARKER_MM / 2000
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)  # TL, TR, BR, BL
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else None


def check(image):
    K, dist, size = load_camera()
    img = cv2.imread(str(image))
    if img is None:  # a video: use its first frame
        img = next(read_frames(image))[1]
    if (img.shape[1], img.shape[0]) != size:
        raise SystemExit(f"Image is {img.shape[1]}x{img.shape[0]} but calibration is {size[0]}x{size[1]}: "
                         "use the same camera mode for both.")
    pose = marker_pose(img, K, dist)
    if pose is None:
        raise SystemExit("Table marker not found.")
    rvec, tvec = pose
    R, _ = cv2.Rodrigues(rvec)
    cam_in_table = (-R.T @ tvec).ravel()
    print(f"camera position in table frame (cm): x={cam_in_table[0] * 100:.1f} "
          f"y={cam_in_table[1] * 100:.1f} z(height)={cam_in_table[2] * 100:.1f}")
    print(f"distance to marker: {np.linalg.norm(tvec) * 100:.1f} cm")
    cv2.drawFrameAxes(img, K, dist, rvec, tvec, MARKER_MM / 1000, 6)
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "table_frame_check.jpg"
    cv2.imwrite(str(out), img)
    print("wrote", out, "(red = x, green = y, blue = z up)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("intrinsics").add_argument("source", help="board video or folder of photos")
    sub.add_parser("check").add_argument("image", help="photo or clip of the table with the marker")
    a = ap.parse_args()
    intrinsics(a.source) if a.cmd == "intrinsics" else check(a.image)
