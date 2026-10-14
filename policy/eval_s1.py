"""Closed-loop evaluation of an S1 checkpoint in LIBERO, on the scenes of the dataset's episodes.

  python policy/eval_s1.py <ckpt.pt> [--splits train val test failed] [--seeds 0 1 2 3 4] [--attempts 1] [--workers 6]
                           [--execute 10] [--denoise 10] [--video out.mp4] [--video-episodes 10] [--plan DIR]

Per episode: the scene that episode was recorded in (sim cup and plate from scene.json, cameras as in step 4.3), the
gripper driven to where the recorded episode started (sim.npz eef_pos[0], fingers open; the replay starts there too),
then the policy runs: predict a 32-step chunk from the two cameras + state (--denoise Euler steps), execute the first
--execute actions, repeat. The episode gets the recorded length + 3 s, then 1 s of zero actions for the cup to settle,
then LIBERO's On(cup, plate) decides success. Every (split, seed, attempt, episode) is one run; seed s, attempt a set
the flow-matching noise (torch seed 1000 * s + episode index + 1e6 * a, so attempt 0 is the 1-attempt eval), the sim
itself is deterministic. Runs are spread over --workers processes, each with its own model and env. Per split:
success rate per seed (over its episodes x attempts), then mean +- std over the seeds.
A flow checkpoint (--flow hand / full in training) gets, at every prediction at step i, the episode's recorded flow
(hand, or hand + mug) from flow_sim.npz rows i, i+2, .., i+30 (last row repeated past the end), indexed by sim time, not by progress.
Trail error (every model): median distance from the gripper to the flow's pinch point (thumb / index tip midpoint,
in the sim world) at the same step, over the recorded length; for the recorded replay itself it is printed as
"replay" (the gripper lags the target by ~0.2 s, step 4.4).
Writes <ckpt dir>/eval_<ckpt name>.json (eval_<ckpt name>_x<attempts>.json for --attempts > 1); --video tiles the third-person view of the first --video-episodes test
episodes, seed 0.
--plan DIR (end-to-end test, step 6.3): the flow comes from the WAM's plan DIR/plans/<episode>.npz (wam/plan.py,
10 Hz rows, row j = sim step 2 j; the policy is only queried at even steps) instead of the recorded flow; writes
DIR/eval_<policy run>.json, and --video writes DIR/videos/<episode>.mp4 for the first --video-episodes test episodes:
real video + recorded flow | generated video + planned flow | robot wrist camera | robot third-person camera
(wam/render.py).
--pauses (with --plan): insert the replay's pauses into the plan (add_pauses), DIR/eval_<policy run>_pauses.json and
DIR/videos_pauses/.
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
from train_s1 import FLOW_POINTS, episode_dirs, flow_features  # noqa: E402
from replay import ARRIVE_S, GRIP_S, HZ, LIFT_M, OPEN_HOLD_S, snapshot  # noqa: E402
from retarget import grasp_events  # noqa: E402
from scene import OPEN, make_env, move_to, place, reset, set_camera, set_third_person_camera  # noqa: E402

EXTRA_S = 3


def load_policy(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = dict(ck["config"])
    cfg.pop("flow_token_drop", None)  # per-token flow dropout, replaced by per-sample flow_drop before any run used it
    model = S1Model(S1Config(**cfg)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, {k: v.to(dev) for k, v in ck["norm"].items()}, ck


def pinch_sim(ep, sim):
    hand = np.load(ep / "flow_sim.npz")["hand"]
    return hand[1:3, :, :3].mean(0) + sim["sim_minus_real"]


def add_pauses(pts):
    """Plan [N, 14, 4] at sim steps -> (plan with the replay's pauses, source step per row), as the robot clock puts them
    into the recorded flow (replay.rim_grasp_constraints, flow/sim_clock.py): the flow holds still for the settle and
    close at the grasp (ARRIVE_S + GRIP_S) and for the hold and 1 cm/step rise after the release. Grasp and release
    from the plan's thumb / index tips with the step 1.7 rule; a plan without both gets no pauses."""
    try:
        g, r = grasp_events(np.linalg.norm(pts[:, 1, :3] - pts[:, 2, :3], axis=1), pts[:, 1:3, :3].mean(1))
    except IndexError:
        return pts, np.arange(len(pts))
    n_g, n_r = int((ARRIVE_S + GRIP_S) * HZ), int(OPEN_HOLD_S * HZ) + int(LIFT_M / 0.01)
    src = np.r_[np.arange(g), [g] * n_g, np.arange(g, r), [r] * n_r, np.arange(r, len(pts))]
    return pts[src], src


def run_episode(env, ep, model, norm, ck, execute, denoise, dev, frames=None, plan=None, pauses=False, src=None):
    """-> (success, trail error in mm). frames: list to append (third-person, wrist) images to; src: list that gets the
    plan step each row of a paused plan comes from."""
    sim = np.load(ep / "sim.npz")
    pinch, eef = pinch_sim(ep, sim), []
    if ck.get("flow"):
        fl = np.load(ep / "flow_sim.npz")
        pts = np.concatenate([fl["hand"], fl["cup"]]).transpose(1, 0, 2)
        if plan is not None:
            pts = np.repeat(np.load(plan / "plans" / f"{ep.name}.npz")["flow"], 2, axis=0)  # 10 Hz -> sim steps
            if pauses:
                pts, s = add_pauses(pts)
                if src is not None:
                    src.extend(s)
        pts = torch.from_numpy(pts.astype(np.float32)).to(dev)
        smr = torch.from_numpy(sim["sim_minus_real"].astype(np.float32)).to(dev)[None]
    sc = json.loads((ep / "scene.json").read_text())
    reset(env)
    obs = place(env, sc["sim_cup_xy"], sc["sim_plate_xy"])
    cam = sc["sim_camera"]
    set_camera(env, cam["name"], np.array(cam["pos"]), np.array(cam["R_world_from_cv"]), cam["fovy_deg"])
    set_third_person_camera(env)
    obs, *_ = move_to(env, obs, sim["eef_pos"][0], gripper=OPEN, max_steps=150)
    horizon, queue, last = len(sim["action"]) + EXTRA_S * HZ, [], np.zeros(7)
    for i in range(horizon):
        s = snapshot(obs)
        eef.append(np.asarray(s["eef_pos"]))
        if frames is not None:
            frames.append((s["agentview"], s["eye_in_hand"]))
        if not queue:
            images = [torch.from_numpy(np.ascontiguousarray(s[c]))[None].to(dev) for c in ck["cameras"]]
            raw = torch.from_numpy(np.concatenate([np.asarray(s[k], np.float32).ravel() for k in ck["state_keys"]])).to(dev)
            state = ((raw - norm["state_mean"]) / norm["state_std"])[None]
            flow = None
            if ck.get("flow"):
                rows = torch.clamp(i + torch.arange(0, model.cfg.flow_steps * 2, 2, device=dev), max=len(pts) - 1)[None]
                rel = raw[None, :3] if ck.get("flow_frame", "eef") == "eef" else torch.zeros_like(raw[None, :3])
                flow = (flow_features(pts, rows, smr, rel, FLOW_POINTS[ck["flow"]]) - norm["flow_mean"]) \
                    / norm["flow_std"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = model.sample(images, state, num_steps=denoise, flow=flow)[0].float()
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
            s = snapshot(obs)
            frames.append((s["agentview"], s["eye_in_hand"]))
    n = len(sim["action"])
    return bool(env.check_success()), float(1000 * np.median(np.linalg.norm(np.array(eef[:n]) - pinch[:n], axis=1)))


def worker(ckpt, execute, denoise, jobs, results, video_dir, plan, pauses, label):
    dev = torch.device("cuda")
    model, norm, ck = load_policy(ckpt, dev)
    env = make_env(img_size=256)
    while True:
        job = jobs.get()
        if job is None:
            break
        split, seed, attempt, k, ep, keep_frames = job
        frames = [] if keep_frames else None
        torch.manual_seed(1000 * seed + k + 1_000_000 * attempt)
        src = []
        ok, trail = run_episode(env, Path(ep), model, norm, ck, execute, denoise, dev, frames, plan, pauses, src)
        if frames is not None and plan is not None:
            sys.path.insert(0, str(ROOT / "wam"))
            from render import write_panels
            write_panels(Path(ep), plan, frames, video_dir / f"{Path(ep).name}.mp4", src or None,
                         f"{label}  ·  {split} episode {Path(ep).name}  ·  {'success' if ok else 'failure'}")
        elif frames is not None:
            np.save(video_dir / f"{k:02d}_{Path(ep).name}.npy", np.stack([f[0] for f in frames]))
        results.put((split, seed, attempt, Path(ep).name, ok, trail))
    env.close()


def main():
    import torch.multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "failed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--execute", type=int, default=10)
    ap.add_argument("--denoise", type=int, default=10)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--video-episodes", type=int, default=10)
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--pauses", action="store_true")
    ap.add_argument("--label", default="", help="--plan: title of the videos")
    args = ap.parse_args()
    tag = "_pauses" if args.pauses else ""
    video_dir = args.plan / f"videos{tag}" if args.plan else args.ckpt.parent / "eval_video_frames"
    if args.video or args.plan:
        video_dir.mkdir(exist_ok=True)
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    n = 0
    for split in args.splits:
        for seed in args.seeds:
            for attempt in range(args.attempts):
                for k, ep in enumerate(episode_dirs(split)):
                    jobs.put((split, seed, attempt, k, str(ep), bool(args.video or args.plan) and split == "test"
                              and seed == args.seeds[0] and attempt == 0 and k < args.video_episodes))
                    n += 1
    for _ in range(args.workers):
        jobs.put(None)
    procs = [ctx.Process(target=worker, args=(args.ckpt, args.execute, args.denoise, jobs, results, video_dir, args.plan, args.pauses, args.label))
             for _ in range(args.workers)]
    for p in procs:
        p.start()
    per, trail, t0 = {}, {}, time.time()
    for i in range(n):
        split, seed, attempt, name, ok, err = results.get()
        per.setdefault(split, {}).setdefault(str(seed), {}).setdefault(name, [None] * args.attempts)[attempt] = ok
        trail.setdefault(split, {}).setdefault(str(seed), {}).setdefault(name, [None] * args.attempts)[attempt] = err
        if (i + 1) % max(20, n // 100) == 0 or i + 1 == n:
            print(f"{i + 1}/{n} runs, {(time.time() - t0) / 60:.1f} min", flush=True)
    for p in procs:
        p.join()
    summary = {}
    for split in args.splits:
        rates = [100 * np.mean(list(per[split][str(s)].values())) for s in args.seeds]
        errs = [e for s in args.seeds for v in trail[split][str(s)].values() for e in v]
        replay = [np.median(np.linalg.norm(np.load(ep / "sim.npz")["eef_pos"] - pinch_sim(ep, np.load(ep / "sim.npz")),
                                           axis=1)) * 1000 for ep in episode_dirs(split)]
        summary[split] = {"episodes": len(per[split][str(args.seeds[0])]), "success_pct_per_seed": rates,
                          "mean": float(np.mean(rates)), "std": float(np.std(rates)),
                          "trail_mm_median": float(np.median(errs)), "replay_trail_mm_median": float(np.median(replay))}
        print(f"{split}: {summary[split]['mean']:.1f} +- {summary[split]['std']:.1f} % "
              f"({summary[split]['episodes']} episodes x {args.attempts} attempts x {len(args.seeds)} seeds; per seed {[round(r) for r in rates]}); "
              f"trail {summary[split]['trail_mm_median']:.0f} mm (replay {summary[split]['replay_trail_mm_median']:.0f} mm)")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    out = args.ckpt.parent / f"eval_{args.ckpt.stem}{f'_x{args.attempts}' if args.attempts > 1 else ''}.json"
    if args.plan:
        out = args.plan / f"eval_{args.ckpt.parent.name}{tag}.json"
    out.write_text(json.dumps({"ckpt": str(args.ckpt), "step": ck["step"], "execute": args.execute,
                               "denoise": args.denoise, "plan": str(args.plan), "pauses": args.pauses, "seeds": args.seeds, "attempts": args.attempts,
                               "summary": summary, "per_episode": per,
                               "trail_mm": trail},
                              indent=1))
    print("wrote", out)
    if args.video and not args.plan:
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
