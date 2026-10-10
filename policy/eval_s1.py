"""Closed-loop evaluation of an S1 checkpoint in LIBERO, on the scenes of the dataset's episodes.

  python policy/eval_s1.py <ckpt.pt> [--splits train val test failed] [--seeds 0 1 2 3 4] [--workers 6]
                           [--execute 10] [--denoise 10] [--video out.mp4] [--video-episodes 10]

Per episode: the scene that episode was recorded in (sim cup and plate from scene.json, cameras as in step 4.3), the
gripper driven to where the recorded episode started (sim.npz eef_pos[0], fingers open; the replay starts there too),
then the policy runs: predict a 32-step chunk from the two cameras + state (--denoise Euler steps), execute the first
--execute actions, repeat. The episode gets the recorded length + 3 s, then 1 s of zero actions for the cup to settle,
then LIBERO's On(cup, plate) decides success. Every (split, seed, episode) is one run; seed s sets the flow-matching
noise (torch seed 1000 * s + episode index), the sim itself is deterministic. Runs are spread over --workers
processes, each with its own model and env. Per split: success rate per seed, then mean +- std over the seeds.
Writes <ckpt dir>/eval_<ckpt name>.json; --video tiles the third-person view of the first --video-episodes test
episodes, seed 0.
"""
import argparse
import json
import subprocess
import sys
import time
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


def run_episode(env, ep, model, norm, ck, execute, denoise, dev, frames=None):
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
                chunk = model.sample(images, state, num_steps=denoise)[0].float()
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


def worker(ckpt, execute, denoise, jobs, results, video_dir):
    dev = torch.device("cuda")
    model, norm, ck = load_policy(ckpt, dev)
    env = make_env(img_size=256)
    while True:
        job = jobs.get()
        if job is None:
            break
        split, seed, k, ep, keep_frames = job
        frames = [] if keep_frames else None
        torch.manual_seed(1000 * seed + k)
        ok = run_episode(env, Path(ep), model, norm, ck, execute, denoise, dev, frames)
        if frames is not None:
            np.save(video_dir / f"{k:02d}_{Path(ep).name}.npy", np.stack(frames))
        results.put((split, seed, Path(ep).name, ok))
    env.close()


def main():
    import torch.multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "failed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--execute", type=int, default=10)
    ap.add_argument("--denoise", type=int, default=10)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--video-episodes", type=int, default=10)
    args = ap.parse_args()
    video_dir = args.ckpt.parent / "eval_video_frames"
    if args.video:
        video_dir.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    n = 0
    for split in args.splits:
        for seed in args.seeds:
            for k, ep in enumerate(episode_dirs(split)):
                jobs.put((split, seed, k, str(ep), bool(args.video) and split == "test" and seed == args.seeds[0]
                          and k < args.video_episodes))
                n += 1
    for _ in range(args.workers):
        jobs.put(None)
    procs = [ctx.Process(target=worker, args=(args.ckpt, args.execute, args.denoise, jobs, results, video_dir))
             for _ in range(args.workers)]
    for p in procs:
        p.start()
    per, t0 = {}, time.time()
    for i in range(n):
        split, seed, name, ok = results.get()
        per.setdefault(split, {}).setdefault(str(seed), {})[name] = ok
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"{i + 1}/{n} runs, {(time.time() - t0) / 60:.1f} min", flush=True)
    for p in procs:
        p.join()
    summary = {}
    for split in args.splits:
        rates = [100 * np.mean(list(per[split][str(s)].values())) for s in args.seeds]
        summary[split] = {"episodes": len(per[split][str(args.seeds[0])]), "success_pct_per_seed": rates,
                          "mean": float(np.mean(rates)), "std": float(np.std(rates))}
        print(f"{split}: {summary[split]['mean']:.1f} +- {summary[split]['std']:.1f} % "
              f"({summary[split]['episodes']} episodes x {len(args.seeds)} seeds; per seed {[round(r) for r in rates]})")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    out = args.ckpt.parent / f"eval_{args.ckpt.stem}.json"
    out.write_text(json.dumps({"ckpt": str(args.ckpt), "step": ck["step"], "execute": args.execute,
                               "denoise": args.denoise, "seeds": args.seeds, "summary": summary, "per_episode": per},
                              indent=1))
    print("wrote", out)
    if args.video:
        files = sorted(video_dir.glob("*.npy"))
        write_grid([(f.stem.split("_", 1)[1], np.load(f)) for f in files], args.video)
        for f in files:
            f.unlink()
        video_dir.rmdir()


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
