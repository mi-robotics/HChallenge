"""Train S1: sim cameras + robot state (+ action flow with --flow) -> action chunk (flow matching).

  python policy/train_s1.py [--steps 20000] [--batch 64] [--cameras agentview eye_in_hand] [--img-aug]
                            [--flow hand|full] [--flow-frame eef|world] [--flow-drop 0.2] [--obs-pos-embed]
                            [--no-img-pos-embed] [--out outputs/s1/<name>]

Data: the train split of data/splits.json (kept episodes), from sim.npz, held on the GPU (uint8 images: 80 episodes
x ~185 steps x 2 cameras x 256 x 256 x 3 = 5.8 GB). One sample = the observation at step i and actions i .. i+31,
padded (and masked out of the loss) past the episode end.
  cameras  --cameras (default both): agentview (third person), eye_in_hand (gripper), 256 x 256 -> 224 for DINOv2;
           none = state (+ flow) only
  state    eef_pos [3], eef_quat [4], gripper_qpos [2]
  action   action[:, [0, 1, 2, 6]]: OSC delta xyz and gripper (the rotation dims are always 0 in the data)
  flow     flow_sim.npz rows i, i+2, .., i+30 (16 steps at 10 Hz = the 1.6 s of the action chunk, last row repeated
           past the episode end) of the 6 hand points (--flow hand) or hand + 8 mug points (--flow full), each
           (x, y, z, visible); xyz moved into the sim world by sim.npz sim_minus_real and taken relative to the
           current eef_pos (--flow-frame eef, default) or left in the sim world (--flow-frame world: the robot base
           frame up to a constant offset, which the z-scoring removes) -> 16 tokens of 24 / 56 values
State, actions and flow xyz are z-scored with train-split statistics (visible stays 0 / 1). --img-aug applies LeRobot's default image transforms
(lerobot/datasets/transforms.py ImageTransformsConfig, off by default there; HVLA's S1 training uses none) to each
training image independently: 3 of {brightness 0.8-1.2, contrast 0.8-1.2, saturation 0.5-1.5, hue +-0.05,
sharpness 0.5-1.5, affine +-5 deg / +-5 % shift}, drawn with equal weights, applied in that order. Written as
batched GPU ops with torchvision's formulas (per image in a loop, torchvision's own took 0.4 s per batch). Val loss on 1024 fixed val-split samples every
--val-every steps. Writes <out>/train.log, <out>/ckpt_<step>.pt and <out>/last.pt (model, config, norm stats).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.append(str(ROOT))
from dataset import ACTION_DIMS, STATE_KEYS, Episode, episode_dirs  # noqa: E402
from s1_model import S1Config, S1Model  # noqa: E402

CAMERAS = ("agentview", "eye_in_hand")
FLOW_POINTS = {"hand": 6, "full": 14}  # flow points: hand 0-5, then mug 6-13

def load(split, device, cameras=CAMERAS):
    """All steps of a split: images per camera [N, 256, 256, 3] uint8, state [N, 9], action [N, 4], episode end [N],
    flow points (hand, then mug) [N, 14, 4] (table frame), sim_minus_real [N, 3]."""
    imgs, state, action, end, pts, smr = {c: [] for c in cameras}, [], [], [], [], []
    n = 0
    for ep in map(Episode, episode_dirs(split)):
        steps = len(ep.action)
        assert len(ep.flow) == steps, ep.path
        for c in cameras:
            imgs[c].append(torch.from_numpy(ep.sim[c]).to(device))
        state.append(ep.state)
        action.append(ep.action)
        pts.append(ep.flow)
        smr.append(np.tile(ep.sim_minus_real.astype(np.float32), (steps, 1)))
        n += steps
        end.append(np.full(steps, n))
    return ({c: torch.cat(v) for c, v in imgs.items()},
            *(torch.from_numpy(np.concatenate(v)).to(device) for v in (state, action, end, pts, smr)))


def flow_features(pts, rows, smr, eef, n):
    """pts [M, 14, 4] table frame; rows [B, T] indices into it; smr, eef [B, 3]; first n points -> [B, T, 4 n]:
    xyz in the sim world relative to the gripper, then visible."""
    f = pts[rows][:, :, :n]
    xyz = f[..., :3] + (smr - eef)[:, None, None]
    return torch.cat([xyz, f[..., 3:]], -1).flatten(2)


N_AUGMENTS = 3  # LeRobot ImageTransformsConfig: max_num_transforms = 3, equal weights, fixed order


def _gray(x):
    return (0.2989 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _uniform(n, lo, hi, dev):
    return torch.empty(n, 1, 1, 1, device=dev).uniform_(lo, hi)


def _hue(x, shift):
    """torchvision adjust_hue, batched: RGB -> HSV, h += shift (per image), -> RGB."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, minc = x.max(1).values, x.min(1).values
    v, cr = maxc, maxc - minc
    s = cr / torch.where(maxc == 0, torch.ones_like(maxc), maxc)
    crd = torch.where(cr == 0, torch.ones_like(cr), cr)
    rc, gc, bc = (maxc - r) / crd, (maxc - g) / crd, (maxc - b) / crd
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.fmod(h / 6.0 + 1.0 + shift.view(-1, 1, 1), 1.0)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.long() % 6
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sel = [torch.stack(c, 1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))]
    out = torch.zeros_like(x)
    for k in range(6):
        out = torch.where((i == k).unsqueeze(1), sel[k], out)
    return out


def _sharpness(x, f):
    """torchvision adjust_sharpness, batched: blend with a 3x3 smoothed image (border pixels kept)."""
    k = torch.tensor([[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]], device=x.device) / 13
    blur = torch.nn.functional.conv2d(x.reshape(-1, 1, *x.shape[2:]), k.view(1, 1, 3, 3)).reshape(
        x.shape[0], 3, x.shape[2] - 2, x.shape[3] - 2)
    deg = x.clone()
    deg[:, :, 1:-1, 1:-1] = blur
    return (f * x + (1 - f) * deg).clamp(0, 1)


def _affine(x, deg, tx, ty):
    """RandomAffine(degrees, translate), batched: rotate about the centre, shift, nearest, fill 0."""
    a = torch.deg2rad(deg)
    cos, sin = torch.cos(a), torch.sin(a)
    theta = torch.stack([torch.stack([cos, -sin, -2 * tx], 1), torch.stack([sin, cos, -2 * ty], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def augment(images):
    """LeRobot's default image transforms on a batch: each image gets its own random 3 of the 6, applied in order.
    images: list of [B, H, W, 3] uint8 -> same."""
    out = []
    for cam in images:
        x = cam.permute(0, 3, 1, 2).float() / 255
        n, dev = len(x), x.device
        pick = torch.rand(n, 6, device=dev).argsort(1)[:, :N_AUGMENTS]
        use = torch.zeros(n, 6, dtype=torch.bool, device=dev).scatter_(1, pick, True).view(n, 6, 1, 1, 1)
        steps = [lambda y: (y * _uniform(n, 0.8, 1.2, dev)).clamp(0, 1),  # brightness
                 lambda y: (_uniform(n, 0.8, 1.2, dev) * (y - _gray(y).mean((2, 3), keepdim=True))
                            + _gray(y).mean((2, 3), keepdim=True)).clamp(0, 1),  # contrast
                 lambda y: (_uniform(n, 0.5, 1.5, dev) * (y - _gray(y)) + _gray(y)).clamp(0, 1),  # saturation
                 lambda y: _hue(y, torch.empty(n, device=dev).uniform_(-0.05, 0.05)),  # hue
                 lambda y: _sharpness(y, _uniform(n, 0.5, 1.5, dev)),  # sharpness
                 lambda y: _affine(y, torch.empty(n, device=dev).uniform_(-5, 5),
                                   torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                                   torch.empty(n, device=dev).uniform_(-0.05, 0.05))]  # affine
        for k, f in enumerate(steps):
            x = torch.where(use[:, k], f(x), x)
        out.append((x * 255).round().to(torch.uint8).permute(0, 2, 3, 1))
    return out


def flow_rows(idx, end, chunk):
    return torch.minimum(idx[:, None] + torch.arange(0, chunk, 2, device=idx.device), end[idx][:, None] - 1)


def batch(data, idx, chunk, norm, flow_kind=None, flow_frame="eef"):
    imgs, state, action, end, pts, smr = data
    steps = idx[:, None] + torch.arange(chunk, device=idx.device)
    pad = steps >= end[idx][:, None]
    steps = torch.minimum(steps, end[idx][:, None] - 1)
    flow = None
    if flow_kind:
        eef = state[idx, :3] if flow_frame == "eef" else torch.zeros_like(state[idx, :3])
        flow = (flow_features(pts, flow_rows(idx, end, chunk), smr[idx], eef, FLOW_POINTS[flow_kind])
                - norm["flow_mean"]) / norm["flow_std"]
    return ([v[idx] for v in imgs.values()], (state[idx] - norm["state_mean"]) / norm["state_std"],
            (action[steps] - norm["action_mean"]) / norm["action_std"], pad, flow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--lr-end", type=float, default=2.5e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "s1" / time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-aug", action="store_true")
    ap.add_argument("--cameras", nargs="*", default=list(CAMERAS), choices=CAMERAS)
    ap.add_argument("--flow", choices=list(FLOW_POINTS))
    ap.add_argument("--flow-frame", choices=["eef", "world"], default="eef")
    ap.add_argument("--flow-drop", type=float, default=0.0)
    ap.add_argument("--obs-pos-embed", action="store_true")
    ap.add_argument("--no-img-pos-embed", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    t0 = time.time()
    cameras = tuple(args.cameras)
    train, val = load("train", dev, cameras), load("val", dev, cameras)
    norm = {"state_mean": train[1].mean(0), "state_std": train[1].std(0).clamp(min=1e-3),
            "action_mean": train[2].mean(0), "action_std": train[2].std(0).clamp(min=1e-3)}
    if args.flow:
        idx = torch.arange(len(train[1]), device=dev)
        eef = train[1][:, :3] if args.flow_frame == "eef" else torch.zeros_like(train[1][:, :3])
        f = flow_features(train[4], flow_rows(idx, train[3], args.chunk), train[5], eef,
                          FLOW_POINTS[args.flow]).flatten(0, 1)
        visible = torch.arange(f.shape[1], device=dev) % 4 == 3
        norm["flow_mean"] = torch.where(visible, 0.0, f.mean(0))
        norm["flow_std"] = torch.where(visible, 1.0, f.std(0).clamp(min=1e-3))
    say(f"loaded train {len(train[1])} steps, val {len(val[1])} steps in {time.time() - t0:.0f} s; "
        f"GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    cfg = S1Config(action_dim=len(ACTION_DIMS), state_dim=train[1].shape[1], n_cameras=len(cameras), chunk_size=args.chunk,
                   flow_dim=4 * FLOW_POINTS[args.flow] if args.flow else 0, flow_steps=args.chunk // 2, obs_pos_embed=args.obs_pos_embed,
                   img_pos_embed=not args.no_img_pos_embed, flow_drop=args.flow_drop)
    model = S1Model(cfg).to(dev)
    say(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M; args {vars(args)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr_end + (args.lr - args.lr_end) * 0.5 * (1 + math.cos(math.pi * p))

    g = torch.Generator(device=dev).manual_seed(1234)
    val_idx = torch.randint(0, len(val[1]), (1024,), device=dev, generator=g)

    def val_loss():
        model.eval()
        torch.manual_seed(0)  # same noise and flow times each time
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            losses = [model(*batch(val, val_idx[i:i + args.batch], args.chunk, norm, args.flow, args.flow_frame)).item()
                      for i in range(0, len(val_idx), args.batch)]
        model.train()
        return float(np.mean(losses))

    def save(name, step):
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "norm": {k: v.cpu() for k, v in norm.items()},
                    "step": step, "img_aug": args.img_aug, "flow": args.flow, "flow_frame": args.flow_frame, "cameras": cameras, "state_keys": STATE_KEYS, "action_dims": ACTION_DIMS},
                   args.out / name)

    model.train()
    t0, losses = time.time(), []
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        idx = torch.randint(0, len(train[1]), (args.batch,), device=dev)
        images, state, actions, pad, flow = batch(train, idx, args.chunk, norm, args.flow, args.flow_frame)
        if args.img_aug and cameras:
            images = augment(images)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(images, state, actions, pad, flow)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            say(f"step {step + 1} loss {np.mean(losses[-100:]):.4f} grad {grad:.2f} lr {lr_at(step):.2e} "
                f"{(time.time() - t0) / (step + 1):.3f} s/step")
        if (step + 1) % args.val_every == 0:
            say(f"step {step + 1} val_loss {val_loss():.4f}")
        if (step + 1) % args.save_every == 0:
            save(f"ckpt_{step + 1}.pt", step + 1)
    save("last.pt", args.steps)
    say(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
