"""Draw a flow sample on video frames with the episode's marker pose, and the end-to-end eval videos.
"""
import json
import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
sys.path.insert(0, str(ROOT / "replay"))
from to_table import demo_marker_pose  # noqa: E402
from track_points import HAND_COLOUR, YELLOW  # noqa: E402
from visualise import draw  # noqa: E402


def camera(ep):
    meta = json.loads((ep / "meta.json").read_text())
    K, dist = np.array(meta["camera_matrix"]), np.array(meta["dist_coeffs"])
    R, t, _ = demo_marker_pose(ep, K, dist)
    return K, dist, cv2.Rodrigues(R)[0], t, meta["image_size"]


def overlay(frame, flow, cam):
    """frame [H, W, 3] RGB uint8, flow [16, 14, 4] -> a copy with the 14 trails drawn (hand on top)."""
    K, dist, rvec, t, (w0, h0) = cam
    h, w = frame.shape[:2]
    pts = flow.transpose(1, 0, 2)  # [14, 16, 4]
    uv = cv2.projectPoints(pts[..., :3].reshape(-1, 3).astype(np.float64), rvec, t, K, dist)[0].reshape(14, -1, 2)
    img = np.ascontiguousarray(frame[..., ::-1])
    draw(img, uv * [w / w0, h / h0], pts[..., 3] > 0, HAND_COLOUR + [YELLOW] * 8, 6, scale=0.35)
    return np.ascontiguousarray(img[..., ::-1])


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
                          stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
    ff.stdin.close()
    ff.wait()


FONT = "/usr/share/fonts/truetype/noto/NotoSans-{}.ttf"
PANELS = (("My recording + recorded flow", 448), ("Cosmos WAM: generated video + planned flow", 448),
          ("Robot wrist camera", 256), ("Robot, third person", 256))


def title_bars(title, width):
    """Header (title left, time drawn per frame on the right) and a caption bar over each panel, as RGB uint8."""
    from PIL import Image, ImageDraw, ImageFont
    bold, reg = ImageFont.truetype(FONT.format("Bold"), 17), ImageFont.truetype(FONT.format("Regular"), 14)
    img = Image.new("RGB", (width, 66), (24, 27, 34))
    d = ImageDraw.Draw(img)
    d.text((12, 20), title, font=bold, fill=(240, 242, 246), anchor="lm")
    x = 0
    for text, w in PANELS:
        d.rectangle([x, 40, x + w - 1, 65], fill=(38, 42, 52))
        d.text((x + w / 2, 53), text, font=reg, fill=(200, 205, 214), anchor="mm")
        x += w
    return np.asarray(img).copy(), reg


def write_panels(ep, plan, sim_frames, path, src=None, title=""):
    """One video per end-to-end episode at the robot's 20 Hz, under a title bar: real video + recorded flow | generated
    video + planned flow (wam/plan.py) | robot wrist camera | robot third-person camera. sim_frames: (third-person,
    wrist) per step. Each video shows the 1.6 s flow chunk its frame belongs to (the real video holds its last frame at
    the end). src: plan step per sim step when pauses were inserted into the plan (the generated video pauses with it)."""
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(ROOT / "wam"))
    from data import H, read_frames
    from dataset import Episode
    cam = camera(ep)
    real, real_flow = read_frames(ep / "video_sim.mp4"), Episode(ep).flow
    gen, gen_flow = read_frames(plan / "plans" / f"{ep.name}.mp4"), np.load(plan / "plans" / f"{ep.name}.npz")["flow"]
    bars, font = title_bars(title, sum(w for _, w in PANELS))
    out = []
    for i, (third, wrist) in enumerate(sim_frames):
        j = (i if src is None else src[min(i, len(src) - 1)]) // 2  # 10 Hz frame of the plan; chunk c covers frames 16 c + 1 ... 16 c + 16 (frame 0 = the real image)
        c = min(max(j - 1, 0) // 16, len(gen_flow) // 16 - 1)
        g = overlay(gen[min(j, len(gen) - 1)], gen_flow[16 * c:16 * c + 16], cam)
        r = min(i, len(real) - 1)
        rows = np.minimum(32 * (r // 32) + np.arange(0, 32, 2), len(real_flow) - 1)
        a = overlay(real[r], real_flow[rows], cam)
        head = Image.fromarray(bars)
        ImageDraw.Draw(head).text((head.width - 12, 20), f"t = {i / 20:4.1f} s", font=font, fill=(200, 205, 214),
                                  anchor="rm")
        out.append(np.concatenate([np.asarray(head), np.concatenate([a, g, cv2.resize(wrist, (H, H)),
                                                                     cv2.resize(third, (H, H))], 1)], 0))
    write_mp4(path, out, 20)
