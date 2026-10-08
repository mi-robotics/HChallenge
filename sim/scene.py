"""Cup-on-plate sim scene: Panda + table + mug + plate in LIBERO, and helpers to move the gripper.

Run `python sim/scene.py` to check the scene: free-space target accuracy, then a scripted
rim-grasp pick-and-place. Saves a GIF + contact sheet to media/.
"""
from pathlib import Path

import numpy as np
from libero.libero.envs import OffScreenRenderEnv

BDDL = Path(__file__).with_name("cup_on_plate.bddl")
CUP, PLATE = "red_coffee_mug_1", "plate_1"
CUP_HEIGHT = 0.135      # measured from the mesh; body radius ~4.9 cm, too wide for the 8 cm Panda grip
RIM_OFFSET = 0.04       # grasp the rim: one finger inside the mug, one outside (fingers close along world y)

# OSC_POSE delta controller: action[:3] in [-1, 1] maps to +-5 cm per control step (20 Hz).
OSC_MAX_DELTA = 0.05
OPEN, CLOSE = -1.0, 1.0


def make_env(img_size=256):
    env = OffScreenRenderEnv(
        bddl_file_name=str(BDDL),
        camera_heights=img_size,
        camera_widths=img_size,
    )
    return env


def reset(env, settle_steps=10):
    """Reset and let objects settle on the table (standard LIBERO practice)."""
    env.reset()
    obs = None
    for _ in range(settle_steps):
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, OPEN]))
    return obs


def step_towards(obs, target, gripper):
    """One action that moves the end-effector towards `target` (world xyz, metres), orientation held."""
    err = np.asarray(target) - obs["robot0_eef_pos"]
    action = np.zeros(7)
    action[:3] = np.clip(err / OSC_MAX_DELTA, -1, 1)
    action[6] = gripper
    return action


def move_to(env, obs, target, gripper=OPEN, tol=0.005, max_steps=100, frames=None):
    """Drive the end-effector to `target`. Returns (obs, steps taken, final error in metres)."""
    for t in range(max_steps):
        err = np.linalg.norm(np.asarray(target) - obs["robot0_eef_pos"])
        if err < tol:
            return obs, t, err
        obs, *_ = env.step(step_towards(obs, target, gripper))
        if frames is not None:
            frames.append(frame(obs))
    return obs, max_steps, np.linalg.norm(np.asarray(target) - obs["robot0_eef_pos"])


def set_gripper(env, obs, gripper, steps=15, frames=None):
    for _ in range(steps):
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, gripper]))
        if frames is not None:
            frames.append(frame(obs))
    return obs


def frame(obs):
    # LIBERO camera images come out upside down.
    return obs["agentview_image"][::-1]


if __name__ == "__main__":
    from PIL import Image

    env = make_env()
    obs = reset(env)
    print("instruction:", env.language_instruction)
    print("eef start:", obs["robot0_eef_pos"].round(3))
    cup, plate = obs[f"{CUP}_pos"], obs[f"{PLATE}_pos"]
    print("cup:", cup.round(3), " plate:", plate.round(3))

    frames = [frame(obs)]
    start = obs["robot0_eef_pos"].copy()

    # 1) Free-space accuracy: corners of a 20 x 20 cm square, 15 cm above the table, away from objects.
    table_z = cup[2]
    print("free-space targets:")
    for dx, dy in [(-0.1, 0.12), (0.1, 0.12), (0.1, 0.32), (-0.1, 0.32)]:
        tgt = np.array([dx, dy, table_z + 0.15])
        obs, steps, err = move_to(env, obs, tgt, frames=frames)
        print(f"  target {tgt.round(2)}  steps={steps:3d}  error={err * 1000:4.1f} mm")
    obs, *_ = move_to(env, obs, start, frames=frames)

    # 2) Scripted pick-and-place with a rim grasp.
    grasp = np.r_[cup[0], cup[1] + RIM_OFFSET, table_z + CUP_HEIGHT - 0.03]
    place = np.r_[plate[:2] + [0, RIM_OFFSET], grasp[2] + 0.025]
    plan = [
        ("above rim", grasp + [0, 0, 0.08], OPEN),
        ("down to rim", grasp, OPEN),
        ("close", None, CLOSE),
        ("lift", grasp + [0, 0, 0.12], CLOSE),
        ("over plate", place + [0, 0, 0.1], CLOSE),
        ("lower", place, CLOSE),
        ("release", None, OPEN),
        ("retreat", place + [0, 0, 0.1], OPEN),
    ]
    print("scripted pick-and-place:")
    for name, tgt, grip in plan:
        if tgt is None:
            obs = set_gripper(env, obs, grip, steps=20, frames=frames)
            print(f"  {name:12s} finger qpos={obs['robot0_gripper_qpos'].round(4)}")
            continue
        obs, steps, err = move_to(env, obs, tgt, gripper=grip, frames=frames)
        print(f"  {name:12s} steps={steps:3d}  error={err * 1000:4.1f} mm  cup z={(obs[f'{CUP}_pos'][2] - table_z) * 100:4.1f} cm")
    obs = set_gripper(env, obs, OPEN, frames=frames)
    print("task success:", env.check_success())

    env.close()

    out = Path(__file__).resolve().parents[1] / "media"
    out.mkdir(exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames[::2]]
    imgs[0].save(out / "1.1_scene.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)
    picks = np.linspace(0, len(frames) - 1, 8).astype(int)
    sheet = np.concatenate([frames[i] for i in picks], axis=1)
    Image.fromarray(sheet).save(out / "1.1_scene_sheet.png")
    print(f"saved {len(frames)} frames to {out}")
