"""Plan whole episodes with the WAM from their first real frame, for the end-to-end eval.

  python wam/plan.py outputs/wam/v1/lora_20000.pt --out outputs/wam/v1/e2e [--given-state] [--seed 0]

Chained 1.6 s chunks (each starts from the previous chunk's last frame and flow row) until the policy's horizon is
covered. Writes <out>/plans/<episode>.npz (flow, 10 Hz) and .mp4 (generated video); policy/eval_s1.py --plan reads
them. --given-state: the first chunk also gets the ground truth flow row at frame 0.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft.tuners.lora import LoraLayer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FLOW_STEPS, STRIDE, read_frames  # noqa: E402
from dataset import Episode, episode_dirs  # noqa: E402
from model import Layout, load_transformer, load_vae  # noqa: E402
from render import write_mp4  # noqa: E402
from train import add_lora, prompt_embedding, rollout  # noqa: E402

HZ, EXTRA_S = 20, 3  # as policy/eval_s1.py


def load_run(ckpt):
    """A LoRA checkpoint -> (model, vae, prompt embedding, layout), on the GPU in bf16, for inference."""
    cfg = json.loads((Path(ckpt).parent / "config.json").read_text())
    vae = load_vae("cuda")
    model = load_transformer(cfg["base"]).cuda()
    add_lora(model, cfg["lora_rank"])
    missing = set(model.load_state_dict(torch.load(ckpt), strict=False).unexpected_keys)
    assert not missing, missing
    for m in model.modules():  # fold LoRA into the weights: 15 % faster sampling, flow within 0.1 mm
        if isinstance(m, LoraLayer):
            m.merge()
    model.bfloat16()  # inference only
    layout = Layout(flow=not cfg.get("video_only", False), state=cfg.get("state_p", 0) > 0)
    return model.eval(), vae, prompt_embedding(cfg["base"], "cuda"), layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "failed"])
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--given-state", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="sampling noise (a different plan per seed)")
    args = ap.parse_args()
    cfg = json.loads((args.ckpt.parent / "config.json").read_text())
    (args.out / "plans").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model, vae, text, layout = load_run(args.ckpt)
    eps = [ep for s in args.splits for ep in episode_dirs(s) if not (args.out / "plans" / f"{ep.name}.npz").exists()]
    horizon = {ep: len(np.load(ep / "sim.npz")["action"]) + EXTRA_S * HZ for ep in eps}
    eps.sort(key=horizon.get)  # similar lengths in a batch
    t0 = time.time()
    for k in range(0, len(eps), args.batch):
        batch = eps[k:k + args.batch]
        chunks = -(-max(horizon[ep] for ep in batch) // (STRIDE * FLOW_STEPS))
        frame0 = np.stack([read_frames(ep / "video_sim.mp4", first=True)[0] for ep in batch])
        state0 = np.stack([Episode(ep).flow[0] for ep in batch]) if args.given_state else None
        with torch.autocast("cuda", torch.bfloat16):
            frames, flow = rollout(model, vae, frame0, text, cfg["flow_stats"], layout, chunks,
                                   dict(steps=args.steps, sigma_min=args.sigma_min), state0)
        for j, ep in enumerate(batch):
            np.savez(args.out / "plans" / f"{ep.name}.npz", flow=flow[j])
            write_mp4(args.out / "plans" / f"{ep.name}.mp4", frames[j], 10)
        print(f"{k + len(batch)}/{len(eps)} episodes, {chunks} chunks, {(time.time() - t0) / 60:.1f} min", flush=True)
    (args.out / "plan_config.json").write_text(json.dumps(vars(args) | {"ckpt": str(args.ckpt), "out": str(args.out)}, indent=1))


if __name__ == "__main__":
    main()
