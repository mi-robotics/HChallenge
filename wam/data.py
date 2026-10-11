"""Training samples for the world action model, all on the robot's clock (20 Hz, with the replay's pauses):
video_sim.mp4 (my real colour video, frame i = the camera frame sim step i follows) and flow_sim.npz (row i = the
hand + mug flow at that moment), see data/README.md.

A sample at step i:
  frames [17, H, W, 3] uint8   video frames i, i+2, ..., i+32 (now + 16 future frames at 10 Hz = 1.6 s)
  flow   [16, 14, 4] float   flow rows i, i+2, ..., i+30 (the S1 policy's flow input): 6 hand + 8 mug points,
                             x y z in the table frame (metres) + visible (0/1)
Rows past the end repeat the last one, as in the policy (FORMAT.md).
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H, W = 256, 448  # 1280 x 720 resized; both divisible by 16 (VAE 8x, patch 2x)
FLOW_STEPS, STRIDE = 16, 2
N_FRAMES = 1 + FLOW_STEPS  # now + 16 future frames
PROMPT = "pick up the mug and put it on the plate"


def episode_dirs(split):
    names = json.loads((ROOT / "data" / "splits.json").read_text())[split]
    return [ROOT / "data" / "mug_pnp_dataset" / n for n in names]


def read_frames(path, size=(W, H)):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, img = cap.read()
        if not ok:
            return np.stack(out)
        out.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA)[..., ::-1])  # RGB


def read_flow(ep):
    f = np.load(ep / "flow_sim.npz")
    return np.concatenate([f["hand"], f["cup"]]).transpose(1, 0, 2).astype(np.float32)  # [N, 14, 4]


class Episodes:
    """All episodes of one split in RAM (80 train episodes at 256 x 448: 5.1 GB)."""

    def __init__(self, split, limit=None):
        self.dirs = episode_dirs(split)[:limit]
        with ThreadPoolExecutor(16) as pool:
            self.frames = list(pool.map(lambda d: read_frames(d / "video_sim.mp4"), self.dirs))
        self.flow = [read_flow(d) for d in self.dirs]
        for d, v, f in zip(self.dirs, self.frames, self.flow):
            assert len(v) == len(f), (d, len(v), len(f))
        self.steps = [(e, i) for e, f in enumerate(self.flow) for i in range(len(f))]

    def sample(self, e, i):
        n = len(self.flow[e])
        rows = np.minimum(i + np.arange(0, STRIDE * FLOW_STEPS + 1, STRIDE), n - 1)  # 17 rows: i ... i+32
        return self.frames[e][rows], self.flow[e][rows[:-1]]

    def batch(self, rng, size):
        picks = [self.steps[k] for k in rng.integers(len(self.steps), size=size)]
        frames, flow = zip(*(self.sample(e, i) for e, i in picks))
        return np.stack(frames), np.stack(flow)


def flow_stats(episodes):
    """Per-axis mean/std of x y z over all points and rows (the visible flag is mapped to -1/+1 instead)."""
    xyz = np.concatenate([f[..., :3].reshape(-1, 3) for f in episodes.flow])
    return {"mean": xyz.mean(0).tolist(), "std": xyz.std(0).tolist()}


def normalise_flow(flow, stats):
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    return np.concatenate([(flow[..., :3] - mean) / std, 2 * flow[..., 3:] - 1], -1)


def denormalise_flow(flow, stats):
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    return np.concatenate([flow[..., :3] * std + mean, (flow[..., 3:] > 0).astype(np.float32)], -1)
