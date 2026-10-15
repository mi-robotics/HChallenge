"""The recorded episodes, as the policy and the world action model read them (layout: data/README.md).

Everything is on the robot's clock: row (or video frame) i of every array belongs to sim control step i, 20 Hz.

  episode_dirs(split)          episode folders of a split in data/splits.json
  Episode(path).flow           [N, 14, 4] float32: 6 hand then 8 mug points, x y z (table frame, m) + visible (0/1)
  Episode(path).state          [N, 9] float32: gripper position, quaternion, finger joints
  Episode(path).action         [N, 4] float32: OSC dx dy dz, gripper
  read_video(path, size)       a video as RGB uint8 frames, optionally resized
"""
import json
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
STATE_KEYS = ("eef_pos", "eef_quat", "gripper_qpos")
ACTION_DIMS = [0, 1, 2, 6]  # the rotation dims are always 0


def episode_dirs(split):
    """train / val / test are kept episodes (data/mug_pnp_dataset), failed ones are in data/mug_pnp_failed."""
    names = json.loads((ROOT / "data" / "splits.json").read_text())[split]
    folder = "mug_pnp_failed" if split.startswith("failed") else "mug_pnp_dataset"
    return [ROOT / "data" / folder / n for n in names]


class Episode:
    def __init__(self, path):
        self.path = Path(path)

    @cached_property
    def sim(self):
        """sim.npz: what the robot did and saw during the replay (cameras, state, actions)."""
        return np.load(self.path / "sim.npz")

    @cached_property
    def flow(self):
        f = np.load(self.path / "flow_sim.npz")
        return np.concatenate([f["hand"], f["cup"]]).transpose(1, 0, 2).astype(np.float32)

    @property
    def state(self):
        n = len(self.sim["action"])
        return np.concatenate([np.asarray(self.sim[k], np.float32).reshape(n, -1) for k in STATE_KEYS], 1)

    @property
    def action(self):
        return self.sim["action"][:, ACTION_DIMS]

    @property
    def sim_minus_real(self):
        """Offset from the real table frame to the sim world (the sim grasp point minus the real one)."""
        return self.sim["sim_minus_real"]


def read_video(path, size=None, first=False):
    """[N, H, W, 3] RGB uint8; size (W, H) resizes; first: only frame 0."""
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, img = cap.read()
        if not ok or (first and out):
            return np.stack(out)
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        out.append(img[..., ::-1])
