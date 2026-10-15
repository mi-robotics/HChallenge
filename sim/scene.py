"""Cup-on-plate scene: Panda, table, mug and plate in LIBERO, and helpers to move the gripper.

  python sim/scene.py    checks target accuracy, then runs a scripted rim-grasp pick-and-place
"""
import subprocess
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


def place(env, cup_xy, plate_xy, settle_steps=10):
    """Move the cup and plate to the given world xy (keeps height and orientation), then let them settle."""
    sim = env.env.sim
    for name, xy in ((CUP, cup_xy), (PLATE, plate_xy)):
        joint = env.env.objects_dict[name].joints[-1]
        q = sim.data.get_joint_qpos(joint).copy()
        q[:2] = xy
        sim.data.set_joint_qpos(joint, q)
        sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()
    obs = None
    for _ in range(settle_steps):
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, OPEN]))
    return obs


def rim_grasp_point(obs):
    """Where the gripper's fingertip centre goes to grasp the mug rim (world frame)."""
    cup = obs[f"{CUP}_pos"]
    return np.array([cup[0], cup[1] + RIM_OFFSET, cup[2] + CUP_HEIGHT - 0.03])


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


def set_camera(env, name, pos, R_world_from_cv, fovy_deg):
    """Move a world-attached sim camera to an OpenCV-convention pose (x right, y down, looking along +z).
    MuJoCo cameras look along -z with y up, so flip the y and z axes."""
    from scipy.spatial.transform import Rotation
    m = env.env.sim.model
    cid = m.camera_name2id(name)
    x, y, z, w = Rotation.from_matrix(R_world_from_cv @ np.diag([1, -1, -1])).as_quat()
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = [w, x, y, z]
    m.cam_fovy[cid] = fovy_deg
    env.env.sim.forward()  # the renderer reads the camera's world pose from data, updated by forward()


# Third-person camera for the dataset: LIBERO's agentview moved back and up so every plate position in the
# recorded episodes (sim x -0.42..0.18, y -0.53..0.55) stays in view: looks at the workspace centre from 50 degrees
# above the table, 1.7 m away, from the side opposite the robot.
THIRD_PERSON = {"name": "agentview", "target": (-0.12, 0.0, 0.45), "elevation_deg": 50, "distance_m": 1.7, "fovy_deg": 45}


def third_person_pose(cam=THIRD_PERSON):
    """(position, R_world_from_cv) of the third-person camera, OpenCV convention."""
    e = np.radians(cam["elevation_deg"])
    back = np.array([np.cos(e), 0.0, np.sin(e)])
    z = -back  # OpenCV convention: looking along +z
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    return np.array(cam["target"]) + cam["distance_m"] * back, np.c_[x, np.cross(z, x), z]


def set_third_person_camera(env, cam=THIRD_PERSON):
    """Call after every reset (reset rebuilds the model and restores LIBERO's default camera)."""
    set_camera(env, cam["name"], *third_person_pose(cam), cam["fovy_deg"])


def render(env, name, width, height):
    """RGB image from a sim camera (right way up)."""
    return env.env.sim.render(width=width, height=height, camera_name=name)[::-1]


def frame(obs):
    # LIBERO camera images come out upside down.
    return obs["agentview_image"][::-1]


def save_mp4(frames, path, fps=20):
    """H.264 MP4 via system ffmpeg, so it plays in browsers."""
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)


if __name__ == "__main__":
    env = make_env(img_size=512)
    obs = reset(env)
    print("instruction:", env.language_instruction)
    print("eef start:", obs["robot0_eef_pos"].round(3))
    cup, plate = obs[f"{CUP}_pos"], obs[f"{PLATE}_pos"]
    print("cup:", cup.round(3), " plate:", plate.round(3))

    start = obs["robot0_eef_pos"].copy()

    # 1) Free-space accuracy: corners of a 20 x 20 cm square, 15 cm above the table, away from objects.
    table_z = cup[2]
    print("free-space targets:")
    for dx, dy in [(-0.1, 0.12), (0.1, 0.12), (0.1, 0.32), (-0.1, 0.32)]:
        tgt = np.array([dx, dy, table_z + 0.15])
        obs, steps, err = move_to(env, obs, tgt)
        print(f"  target {tgt.round(2)}  steps={steps:3d}  error={err * 1000:4.1f} mm")
    obs, *_ = move_to(env, obs, start)

    # 2) Scripted pick-and-place with a rim grasp.
    grasp = rim_grasp_point(obs)
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
    frames = [frame(obs)]
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

    out = Path(__file__).resolve().parents[1] / "media" / "phase1" / "task1"
    out.mkdir(parents=True, exist_ok=True)
    save_mp4(frames + [frames[-1]] * 20, out / "cup_on_plate.mp4")  # hold the last frame for 1 s
    print(f"saved {len(frames)} frames to {out / 'cup_on_plate.mp4'}")
