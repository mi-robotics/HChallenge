"""Geometry of the printed targets: table-marker pose and the ChArUco board."""
import cv2
import numpy as np

from make_targets import (BOARD_DICT, BOARD_MARKER_MM_PRINTED, BOARD_SQUARES, MARKER_DICT, MARKER_ID,
                          MARKER_MM_PRINTED, SQUARE_MM_PRINTED)

_marker_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT))


def board():
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_MM_PRINTED / 1000, BOARD_MARKER_MM_PRINTED / 1000,
                                  cv2.aruco.getPredefinedDictionary(BOARD_DICT))


def marker_pose(img, K, dist):
    """Table frame (marker centre, x/y on the table, z up) -> camera. Returns (rvec, tvec) or None."""
    corners, ids, _ = _marker_detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None or MARKER_ID not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2)
    h = MARKER_MM_PRINTED / 2000
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)  # TL, TR, BR, BL
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else None
