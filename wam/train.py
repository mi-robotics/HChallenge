"""Fine-tune Cosmos-Predict2-2B (LoRA) to predict the next 1.6 s of my real video and hand + mug flow from the
current frame and the prompt, on the robot's clock (wam/data.py, wam/model.py).

  python wam/train.py --out outputs/wam/v1
  python wam/train.py --base random-tiny --steps 300 --limit 2 --out outputs/wam/smoke   (no weights needed)

Every --val-every steps: flow error of sampled flows on val samples (mm, median over points and steps).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FLOW_STEPS, PROMPT, Episodes, denormalise_flow, flow_stats, normalise_flow  # noqa: E402
from model import (BASE_REPO, decode, encode, encode_prompt, load_transformer, load_vae, loss, sample,  # noqa: E402
                   tile_flow, untile_flow)

FLOW_DIM = FLOW_STEPS * 14 * 4


def to_latents(vae, frames, flow, stats, device):
    """uint8 frames [B, 17, H, W, 3], flow [B, 16, 14, 4] (table frame) -> x0 [B, 16, 6, h, w]."""
    z = encode(vae, torch.from_numpy(frames).to(device))
    f = torch.from_numpy(normalise_flow(flow, stats)).to(device).flatten(1)
    return torch.cat([z, tile_flow(f, (z.shape[1], *z.shape[3:]))], 2)


@torch.no_grad()
def predict(model, vae, frames_now, text, stats, steps=20, video=False):
    """frames_now [B, 1, H, W, 3] uint8 -> flow [B, 16, 14, 4] (table frame) and, if video, frames [B, 17, H, W, 3]."""
    cond = encode(vae, frames_now)
    x = sample(model, cond, text, 6, steps)
    flow = denormalise_flow(untile_flow(x[:, :, -1:], FLOW_DIM).view(-1, FLOW_STEPS, 14, 4).cpu().numpy(), stats)
    return flow, (decode(vae, x[:, :, :-1]) if video else None)


def flow_error_mm(pred, true):
    return float(np.median(np.linalg.norm(pred[..., :3] - true[..., :3], axis=-1)) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_REPO, help='diffusers repo/folder, or "random-2b" / "random-tiny"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--flow-weight", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N episodes of each split")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-samples", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--fixed-batch", action="store_true", help="train on one batch (overfit check)")
    args = ap.parse_args()
    out, device = Path(args.out), "cuda"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    t0 = time.time()
    train, val = Episodes("train", args.limit), Episodes("val", args.limit)
    stats = flow_stats(train)
    print(f"data: {len(train.steps)} train / {len(val.steps)} val steps, loaded in {time.time() - t0:.0f} s", flush=True)
    vae = load_vae(device)
    model = load_transformer(args.base).to(device)
    if args.base.startswith("random"):
        text = torch.randn(1, 512, 1024) * (torch.arange(512) < 12).view(1, -1, 1)  # stand-in for the T5 embedding
    else:
        prompt_file = out.parent / "prompt_t5.pt"
        if not prompt_file.exists():
            torch.save(encode_prompt(args.base, PROMPT, device), prompt_file)
        text = torch.load(prompt_file)
    text = text.to(device)
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, init_lora_weights=True,
                                 target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    model.enable_gradient_checkpointing()
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M",
          flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    vi = np.random.default_rng(1).choice(len(val.steps), min(args.val_samples, len(val.steps)), replace=False)
    val_frames, val_flow = map(np.stack, zip(*(val.sample(*val.steps[k]) for k in vi)))
    fixed = train.batch(rng, args.batch) if args.fixed_batch else None
    (out / "config.json").write_text(json.dumps({**vars(args), "flow_stats": stats}, indent=1))

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        frames, flow = fixed or train.batch(rng, args.batch)
        x0 = to_latents(vae, frames, flow, stats, device)
        with torch.autocast("cuda", torch.bfloat16):
            total, l_video, l_flow = loss(model, x0, text, args.flow_weight)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append((total.item(), l_video.item(), l_flow.item()))
        if step % 50 == 0:
            m = np.mean(log, 0)
            print(f"step {step} loss {m[0]:.4f} video {m[1]:.4f} flow {m[2]:.4f} {(time.time() - t0) / 50:.2f} s/step "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            log, t0 = [], time.time()
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            vf = fixed[0][:, :1] if args.fixed_batch else val_frames[:, :1]
            true = fixed[1] if args.fixed_batch else val_flow
            pred = np.concatenate([predict(model, vae, torch.from_numpy(vf[k:k + 8]).to(device), text, stats)[0]
                                   for k in range(0, len(vf), 8)])
            print(f"step {step} {'train-batch' if args.fixed_batch else 'val'} flow error {flow_error_mm(pred, true):.1f} mm "
                  f"(predicting the current position for all 16 steps: {flow_error_mm(true[:, :1].repeat(16, 1), true):.1f} mm)",
                  flush=True)
            model.train()
            t0 = time.time()
        if step % args.save_every == 0 or step == args.steps:
            torch.save({k: v for k, v in model.state_dict().items() if "lora" in k}, out / f"lora_{step}.pt")


if __name__ == "__main__":
    main()
