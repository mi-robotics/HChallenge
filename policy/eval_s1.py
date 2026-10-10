"""Closed-loop evaluation of an S1 checkpoint in LIBERO, on the scenes of a split's episodes.

  python policy/eval_s1.py <ckpt.pt> [--split test] [--execute 8] [--video out.mp4] [--video-episodes 4]

Per episode: the scene that episode was recorded in (sim cup and plate from scene.json, cameras as in step 4.3), the
gripper driven to where the recorded episode started (sim.npz eef_pos[0], fingers open; the replay starts there too),
then the policy runs: predict a 32-step chunk from the two cameras + state, execute the first --execute actions,
repeat. The episode gets the recorded length + 3 s, then 1 s of zero actions for the cup to settle, then LIBERO's
On(cup, plate) decides success. Writes <ckpt dir>/eval_<split>.json; --video tiles the third-person view of the
first --video-episodes episodes.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "replay"))
sys.path.insert(0, str(ROOT / "sim"))
from s1_model import S1Config, S1Model  # noqa: E402
from train_s1 import episode_dirs  # noqa: E402
from replay import HZ, snapshot  # noqa: E402
from scene import OPEN, make_env, move_to, place, reset, set_camera, set_third_person_camera  # noqa: E402

EXTRA_S = 3


def load_policy(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = S1Model(S1Config(**ck["config"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, {k: v.to(dev) for k, v in ck["norm"].items()}, ck


def run_episode(env, ep, model, norm, ck, execute, dev, frames=None):
    sim = np.load(ep / "sim.npz")
    sc = json.loads((ep / "scene.json").read_text())
    reset(env)
    obs = place(env, sc["sim_cup_xy"], sc["sim_plate_xy"])
    cam = sc["sim_camera"]
    set_camera(env, cam["name"], np.array(cam["pos"]), np.array(cam["R_world_from_cv"]), cam["fovy_deg"])
    set_third_person_camera(env)
    obs, *_ = move_to(env, obs, sim["eef_pos"][0], gripper=OPEN, max_steps=150)
    horizon, queue, last = len(sim["action"]) + EXTRA_S * HZ, [], np.zeros(7)
    for _ in range(horizon):
        s = snapshot(obs)
        if frames is not None:
            frames.append(s["agentview"])
        if not queue:
            images = [torch.from_numpy(np.ascontiguousarray(s[c]))[None].to(dev) for c in ck["cameras"]]
            state = torch.from_numpy(np.concatenate([np.asarray(s[k], np.float32).ravel() for k in ck["state_keys"]]))
            state = ((state.to(dev) - norm["state_mean"]) / norm["state_std"])[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = model.sample(images, state)[0].float()
            chunk = (chunk * norm["action_std"] + norm["action_mean"]).cpu().numpy()
            queue = list(chunk[:execute])
        a = queue.pop(0)
        last = np.zeros(7)
        last[ck["action_dims"]] = a
        last = np.clip(last, -1, 1)
        obs, *_ = env.step(last)
    for _ in range(HZ):
        obs, *_ = env.step(np.r_[np.zeros(6), last[6]])
        if frames is not None:
            frames.append(snapshot(obs)["agentview"])
    return bool(env.check_success())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--execute", type=int, default=8)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--video-episodes", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model, norm, ck = load_policy(args.ckpt, dev)
    env = make_env(img_size=256)
    results, videos = {}, []
    for k, ep in enumerate(episode_dirs(args.split)):
        frames = [] if args.video and k < args.video_episodes else None
        results[ep.name] = run_episode(env, ep, model, norm, ck, args.execute, dev, frames)
        print(f"{ep.name}: {'SUCCESS' if results[ep.name] else 'FAIL'}", flush=True)
        if frames is not None:
            videos.append((ep.name, frames))
    env.close()
    n = sum(results.values())
    print(f"{args.split}: {n}/{len(results)} succeeded")
    out = args.ckpt.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps({"ckpt": str(args.ckpt), "step": ck["step"], "execute": args.execute,
                               "success": n, "episodes": len(results), "per_episode": results}, indent=1))
    if args.video:
        write_grid(videos, args.video)


def write_grid(videos, path, cols=2):
    import cv2
    n = max(len(f) for _, f in videos)
    h, w = videos[0][1][0].shape[:2]
    rows = (len(videos) + cols - 1) // cols
    path.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{w * cols}x{h * rows}", "-r", str(HZ), "-i", "-", "-c:v", "libx264", "-crf", "23",
                           "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for i in range(n):
        grid = np.zeros((h * rows, w * cols, 3), np.uint8)
        for k, (name, f) in enumerate(videos):
            img = np.ascontiguousarray(f[min(i, len(f) - 1)])
            cv2.putText(img, name, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            grid[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = img
        ff.stdin.write(grid.tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", path)


if __name__ == "__main__":
    main()
