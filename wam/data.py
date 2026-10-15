"""World action model training samples: the real video and the flow, both on the robot's clock (dataset.py).

A sample at step i: frames [17, 256, 448, 3] (frames i, i+2, ..., i+32: now + 1.6 s at 10 Hz) and flow [16, 14, 4]
(rows i, i+2, ..., i+30, the policy's flow input). Rows past the end repeat the last one.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset import Episode, episode_dirs, read_video  # noqa: E402

H, W = 256, 448  # 1280 x 720 resized; both divisible by 16 (VAE 8x, patch 2x)
FLOW_STEPS, STRIDE = 16, 2
N_FRAMES = 1 + FLOW_STEPS  # now + 16 future frames
PROMPT = "pick up the mug and put it on the plate"


def read_frames(path, first=False):
    return read_video(path, (W, H), first)


class Episodes:
    """All episodes of one split in RAM (80 train episodes at 256 x 448: 5.1 GB)."""

    def __init__(self, split, limit=None):
        self.dirs = episode_dirs(split)[:limit]
        with ThreadPoolExecutor(16) as pool:
            self.frames = list(pool.map(lambda d: read_frames(d / "video_sim.mp4"), self.dirs))
        self.flow = [Episode(d).flow for d in self.dirs]
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
