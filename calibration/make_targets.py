"""Printable calibration targets (A4 PDFs, content also fits US Letter).

- table_marker.pdf: ArUco marker that defines the table frame (taped to the table).
- charuco_board.pdf: ChArUco board for camera calibration (taped to something flat).

Print at 100% scale ("actual size", no fit-to-page), then measure with a ruler and put the
real sizes in calibration/targets.yaml if they differ.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT = Path(__file__).with_name("print")
DPI = 300
A4_MM = (210, 297)

# Table marker: its own dictionary so it can't be confused with the board's markers.
MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_MM = 100

# ChArUco board: 5 x 7 squares of 30 mm -> 150 x 210 mm.
BOARD_DICT = cv2.aruco.DICT_5X5_100
BOARD_SQUARES = (5, 7)
SQUARE_MM = 30
BOARD_MARKER_MM = 22


def mm(x):
    return int(round(x / 25.4 * DPI))


def page():
    return np.full((mm(A4_MM[1]), mm(A4_MM[0])), 255, np.uint8)


def text(img, s, x_mm, y_mm, scale=1.6, thick=3):
    cv2.putText(img, s, (mm(x_mm), mm(y_mm)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)


def ruler(img, x_mm, y_mm, length_mm=100):
    """A 100 mm scale bar with 10 mm ticks, to check the printer didn't scale the page."""
    x0, x1, y = mm(x_mm), mm(x_mm + length_mm), mm(y_mm)
    cv2.line(img, (x0, y), (x1, y), 0, 4)
    for i in range(0, length_mm + 1, 10):
        cv2.line(img, (mm(x_mm + i), y - mm(3 if i % 50 else 5)), (mm(x_mm + i), y), 0, 4)
    text(img, f"{length_mm} mm: measure this", x_mm, y_mm + 9, 1.2, 2)


def arrow(img, p0_mm, p1_mm, label, label_mm):
    p0, p1 = (mm(p0_mm[0]), mm(p0_mm[1])), (mm(p1_mm[0]), mm(p1_mm[1]))
    cv2.arrowedLine(img, p0, p1, 0, 8, cv2.LINE_AA, tipLength=0.2)
    text(img, label, *label_mm, 2.0, 5)


def save_pdf(img, name):
    OUT.mkdir(exist_ok=True)
    Image.fromarray(img).save(OUT / name, resolution=DPI)
    print("wrote", OUT / name)


def table_marker():
    img = page()
    side = mm(MARKER_MM)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(MARKER_DICT), MARKER_ID, side)
    x0, y0 = mm((A4_MM[0] - MARKER_MM) / 2), mm(70)
    img[y0:y0 + side, x0:x0 + side] = marker
    # Marker frame (OpenCV convention): origin at the centre, x to the right, y up, z out of the paper.
    cx, cy = (A4_MM[0]) / 2, 70 + MARKER_MM / 2
    arrow(img, (cx + MARKER_MM / 2 + 5, cy), (cx + MARKER_MM / 2 + 35, cy), "x", (cx + MARKER_MM / 2 + 38, cy + 3))
    arrow(img, (cx, 70 - 5), (cx, 70 - 30), "y", (cx + 5, 70 - 25))
    text(img, "Table marker  (ArUco 4x4_50, id 0)", 25, 22, 1.6, 3)
    text(img, f"Black square side = {MARKER_MM} mm. Tape flat to the table.", 25, 190, 1.2, 2)
    text(img, "x arrow points away from you, across the table", 25, 200, 1.2, 2)
    text(img, "(the direction the robot reaches, cup -> plate).", 25, 209, 1.2, 2)
    ruler(img, 55, 235)
    save_pdf(img, "table_marker.pdf")


def charuco_board():
    board = cv2.aruco.CharucoBoard(BOARD_SQUARES, SQUARE_MM / 1000, BOARD_MARKER_MM / 1000,
                                   cv2.aruco.getPredefinedDictionary(BOARD_DICT))
    w, h = mm(BOARD_SQUARES[0] * SQUARE_MM), mm(BOARD_SQUARES[1] * SQUARE_MM)
    img = page()
    x0, y0 = mm((A4_MM[0] - BOARD_SQUARES[0] * SQUARE_MM) / 2), mm(30)
    img[y0:y0 + h, x0:x0 + w] = board.generateImage((w, h), marginSize=0)
    text(img, f"ChArUco 5x5_100, {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]} squares, "
              f"square {SQUARE_MM} mm, marker {BOARD_MARKER_MM} mm", 30, 20, 1.1, 2)
    ruler(img, 55, 255)
    save_pdf(img, "charuco_board.pdf")


if __name__ == "__main__":
    table_marker()
    charuco_board()
