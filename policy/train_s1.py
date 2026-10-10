"""Train the image-only S1 baseline: two sim cameras + robot state -> action chunk (flow matching). No flow input.

  python policy/train_s1.py [--steps 20000] [--batch 64] [--out outputs/s1/<name>]

Data: the train split of data/splits.json (kept episodes), from sim.npz, held on the GPU (uint8 images: 80 episodes
x ~185 steps x 2 cameras x 256 x 256 x 3 = 5.8 GB). One sample = the observation at step i and actions i .. i+31,
padded (and masked out of the loss) past the episode end.
  cameras  agentview (third person), eye_in_hand (gripper), 256 x 256 -> 224 for DINOv2
  state    eef_pos [3], eef_quat [4], gripper_qpos [2]
  action   action[:, [0, 1, 2, 6]]: OSC delta xyz and gripper (the rotation dims are always 0 in the data)
State and actions are z-scored with train-split statistics. Val loss on 1024 fixed val-split samples every
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
from s1_model import S1Config, S1Model  # noqa: E402

CAMERAS = ("agentview", "eye_in_hand")
STATE_KEYS = ("eef_pos", "eef_quat", "gripper_qpos")
ACTION_DIMS = [0, 1, 2, 6]


def episode_dirs(split):
    names = json.loads((ROOT / "data" / "splits.json").read_text())[split]
    folder = "mug_pnp_failed" if split.startswith("failed") else "mug_pnp_dataset"
    return [ROOT / "data" / folder / n for n in names]


def state_of(sim):
    return np.concatenate([np.asarray(sim[k], np.float32).reshape(len(sim["action"]), -1) for k in STATE_KEYS], 1)


def load(split, device):
    """All steps of a split: images per camera [N, 256, 256, 3] uint8, state [N, 9], action [N, 4], episode end [N]."""
    imgs, state, action, end = {c: [] for c in CAMERAS}, [], [], []
    n = 0
    for ep in episode_dirs(split):
        sim = np.load(ep / "sim.npz")
        for c in CAMERAS:
            imgs[c].append(torch.from_numpy(sim[c]).to(device))
        state.append(state_of(sim))
        action.append(sim["action"][:, ACTION_DIMS])
        n += len(sim["action"])
        end.append(np.full(len(sim["action"]), n))
    return ({c: torch.cat(v) for c, v in imgs.items()}, torch.from_numpy(np.concatenate(state)).to(device),
            torch.from_numpy(np.concatenate(action)).to(device), torch.from_numpy(np.concatenate(end)).to(device))


def batch(data, idx, chunk, norm):
    imgs, state, action, end = data
    steps = idx[:, None] + torch.arange(chunk, device=idx.device)
    pad = steps >= end[idx][:, None]
    steps = torch.minimum(steps, end[idx][:, None] - 1)
    return ([imgs[c][idx] for c in CAMERAS], (state[idx] - norm["state_mean"]) / norm["state_std"],
            (action[steps] - norm["action_mean"]) / norm["action_std"], pad)


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
    train, val = load("train", dev), load("val", dev)
    norm = {"state_mean": train[1].mean(0), "state_std": train[1].std(0).clamp(min=1e-3),
            "action_mean": train[2].mean(0), "action_std": train[2].std(0).clamp(min=1e-3)}
    say(f"loaded train {len(train[1])} steps, val {len(val[1])} steps in {time.time() - t0:.0f} s; "
        f"GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    cfg = S1Config(action_dim=len(ACTION_DIMS), state_dim=train[1].shape[1], n_cameras=len(CAMERAS), chunk_size=args.chunk)
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
            losses = [model(*batch(val, val_idx[i:i + args.batch], args.chunk, norm)).item()
                      for i in range(0, len(val_idx), args.batch)]
        model.train()
        return float(np.mean(losses))

    def save(name, step):
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "norm": {k: v.cpu() for k, v in norm.items()},
                    "step": step, "cameras": CAMERAS, "state_keys": STATE_KEYS, "action_dims": ACTION_DIMS},
                   args.out / name)

    model.train()
    t0, losses = time.time(), []
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        idx = torch.randint(0, len(train[1]), (args.batch,), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(*batch(train, idx, args.chunk, norm))
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
