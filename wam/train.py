"""Fine-tune Cosmos-Predict2-2B (LoRA) to predict the next 1.6 s of my real video and hand + mug flow from the
current frame and the prompt, on the robot's clock (wam/data.py, wam/model.py).

  python wam/train.py --out outputs/wam/v1                       flow + video (v1)
  python wam/train.py --out outputs/wam/v2_state --state-p 0.5    + the flow's current row given half the time
  python wam/train.py --out outputs/wam/v3_video --video-only     video only
  python wam/train.py --base random-tiny --steps 300 --limit 2 --out outputs/wam/smoke   (no weights needed)

The VAE latents of every training sample are encoded once and cached (outputs/wam/cache_*.pt).
Every --sample-every steps, on val episodes (out/samples/, out/samples.jsonl):
  - flow error of sampled flows on --val-samples fixed val samples (mm, mean over points and steps),
    and a video of the first 4: predicted frames + predicted flow | real frames + recorded flow;
  - chained rollouts from the first frame of --rollouts val episodes, --chunks x 1.6 s (6 = 9.6 s): each chunk
    starts from the last frame the previous chunk predicted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from peft import LoraConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FLOW_STEPS, PROMPT, STRIDE, Episodes, denormalise_flow, flow_stats, normalise_flow  # noqa: E402
from model import (BASE_REPO, FLOW_DIM, Layout, decode, encode, encode_prompt, load_transformer,  # noqa: E402
                   load_vae, loss, sample, tile_flow, untile_flow)
from render import camera, overlay, write_mp4  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "outputs" / "wam"


@torch.no_grad()
def cache_latents(vae, eps, path, batch=16):
    """Latents of every sample's 17 frames, [n_steps, 16, 5, h, w] bf16 on the CPU (80 train episodes: 4.2 GB)."""
    if path.exists():
        return torch.load(path)
    out, t0 = [], time.time()
    for k in range(0, len(eps.steps), batch):
        frames = np.stack([eps.sample(*eps.steps[j])[0] for j in range(k, min(k + batch, len(eps.steps)))])
        out.append(encode(vae, torch.from_numpy(frames).cuda()).bfloat16().cpu())
    out = torch.cat(out)
    torch.save(out, path)
    print(f"cached {len(out)} latents in {time.time() - t0:.0f} s -> {path}", flush=True)
    return out


def prompt_embedding(base, device):
    if base.startswith("random"):
        return (torch.randn(1, 512, 1024) * (torch.arange(512) < 12).view(1, -1, 1)).to(device)  # stand-in for T5
    prompt_file = CACHE / "prompt_t5.pt"
    if not prompt_file.exists():
        torch.save(encode_prompt(base, PROMPT, device), prompt_file)
    return torch.load(prompt_file).to(device)


def add_lora(model, rank):
    """Freeze the model, add LoRA to attention + MLP; returns the trainable (float32) parameters."""
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights=True,
                                 target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    return params


def build(z, layout, flow=None, state=None):
    """Latents in `layout` from video latents z [B, 16, 5, h, w] (only frame 0 needed when sampling) and the
    normalised flow [B, FLOW_DIM] / state [B, STATE_DIM] (None -> zeros, to be predicted)."""
    B, C, _, h, w = z.shape
    x = z.new_zeros(B, C, layout.T, h, w)
    x[:, :, :z.shape[2]] = z
    for i, v in ((layout.flow_idx, flow), (layout.state_idx, state)):
        if i is not None and v is not None:
            x[:, :, i:i + 1] = tile_flow(v, (C, h, w))
    return x


@torch.no_grad()
def predict(model, vae, frames_now, text, stats, layout, state=None, sampler=None, video=False):
    """frames_now [B, 1, H, W, 3] uint8, state [B, 14, 4] table frame or None (not given) ->
    flow [B, 16, 14, 4] (table frame; None for a video-only layout) and, if video, frames [B, 17, H, W, 3].
    sampler: keyword arguments of model.sample (steps, sigma_min); default = Cosmos Policy's action sampling."""
    B = frames_now.shape[0]
    given = torch.zeros(B, layout.T, dtype=torch.bool, device=frames_now.device)
    given[:, 0] = True
    s = None
    if state is not None and layout.state_idx is not None:
        s = torch.from_numpy(normalise_flow(state, stats)).to(frames_now.device).flatten(1)
        given[:, layout.state_idx] = True
    x = sample(model, build(encode(vae, frames_now), layout, state=s), given, text, layout, **(sampler or {}))
    flow = None
    if layout.flow_idx is not None:
        i = layout.flow_idx
        flow = denormalise_flow(untile_flow(x[:, :, i:i + 1], FLOW_DIM).view(B, FLOW_STEPS, 14, 4).cpu().numpy(), stats)
    return flow, (decode(vae, x[:, :, :5]).cpu().numpy() if video else None)


@torch.no_grad()
def rollout(model, vae, frame0, text, stats, layout, chunks, sampler=None, state0=None):
    """frame0 [B, H, W, 3] -> frames [B, 1 + 16 chunks, H, W, 3], flow [B, 16 chunks, 14, 4] (None if video only).
    Each chunk starts from the last frame the previous one predicted; the first from the image alone, later ones (if
    the layout has a state slot) also from the previous chunk's last flow row (0.1 s before the chunk starts).
    state0 [B, 14, 4]: a flow row given to the first chunk too (e.g. the recorded one at frame0)."""
    frames, flows, now, state = [frame0[:, None]], [], frame0, state0
    for _ in range(chunks):
        flow, video = predict(model, vae, torch.from_numpy(now)[:, None].cuda(), text, stats, layout, state,
                              sampler, video=True)
        frames.append(video[:, 1:])
        now = video[:, -1]
        if flow is not None:
            flows.append(flow)
            state = flow[:, -1]
    return np.concatenate(frames, 1), (np.concatenate(flows, 1) if flows else None)


def flow_error_mm(pred, true):
    return float(np.mean(np.linalg.norm(pred[..., :3] - true[..., :3], axis=-1)) * 1000)


def side_by_side(pred_frames, pred_flow, real_frames, real_flow, cam, label=""):
    """Frame j shows the flow chunk it belongs to: frame 0 and frames 1-16 chunk 0, frames 17-32 chunk 1, ..."""
    out = []
    for j in range(len(pred_frames)):
        c = max(j - 1, 0) // FLOW_STEPS
        rows = slice(c * FLOW_STEPS, (c + 1) * FLOW_STEPS)
        a, b = pred_frames[j].copy(), real_frames[j].copy()
        if pred_flow is not None:
            a, b = overlay(a, pred_flow[rows], cam), overlay(b, real_flow[rows], cam)
        for img, text in ((a, f"predicted {label}"), (b, f"real  t = {j / 10:.1f} s")):
            cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(np.concatenate([a, b], 1))
    return out


def sharpness(frames):
    """Mean variance of the Laplacian of uint8 RGB frames [..., H, W, 3] (higher = more fine detail)."""
    f = frames.reshape(-1, *frames.shape[-3:])
    return float(np.mean([cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for x in f]))


class ValSet:
    """Fixed val samples (seed 1) and the val episodes rolled out from their first frame."""

    def __init__(self, val, n_samples, n_rollouts):
        self.val = val
        vi = np.random.default_rng(1).choice(len(val.steps), min(n_samples, len(val.steps)), replace=False)
        self.picks = [val.steps[k] for k in vi]
        self.frames, self.flow = map(np.stack, zip(*(val.sample(e, i) for e, i in self.picks)))
        self.rollouts = list(range(min(n_rollouts, len(val.dirs))))
        self.cams = {e: camera(val.dirs[e]) for e in set([e for e, _ in self.picks[:4]] + self.rollouts)}


def evaluate(model, vae, text, stats, vs, out, step, chunks, layout, sampler=None, batch=8):
    """Writes out/samples/<step>_chunks.mp4, <step>_rollout_<episode>.mp4 and a line in out/samples.jsonl.
    Flow errors (if the layout has flow): from the image alone (val_flow_mm) and, with a state slot, also with the
    recorded current row given (val_flow_given_mm)."""
    model.eval()
    (out / "samples").mkdir(parents=True, exist_ok=True)
    t1, res = time.time(), {"step": step, "sampler": sampler or {}}
    pred, pred_given, video = [], [], None
    for k in range(0, len(vs.frames), batch):
        now = torch.from_numpy(vs.frames[k:k + batch, :1]).cuda()
        f, v = predict(model, vae, now, text, stats, layout, None, sampler, video=k == 0)
        pred.append(f)
        video = v if k == 0 else video
        if layout.state_idx is not None:
            pred_given.append(predict(model, vae, now, text, stats, layout, vs.flow[k:k + batch, 0], sampler)[0])
    res["sharpness"] = sharpness(video[:4, 1:]) / sharpness(vs.frames[:4, 1:])
    msg = f"step {step} sharpness {res['sharpness']:.2f} of real"
    pred = np.concatenate(pred) if layout.flow_idx is not None else None
    if pred is not None:
        res["val_flow_mm"] = flow_error_mm(pred, vs.flow)
        res["hold_still_mm"] = flow_error_mm(vs.flow[:, :1].repeat(FLOW_STEPS, 1), vs.flow)
        msg += f", val flow error {res['val_flow_mm']:.1f} mm (hold still: {res['hold_still_mm']:.1f} mm)"
        if pred_given:
            res["val_flow_given_mm"] = flow_error_mm(np.concatenate(pred_given), vs.flow)
            msg += f", with current row given {res['val_flow_given_mm']:.1f} mm"
    clips = [side_by_side(video[j], None if pred is None else pred[j], vs.frames[j], vs.flow[j],
                          vs.cams[vs.picks[j][0]]) for j in range(4)]
    write_mp4(out / "samples" / f"{step:05d}_chunks.mp4", [np.concatenate(c, 0) for c in zip(*clips)], 10)
    res["rollout_flow_mm"] = {}
    val = vs.val
    for e in vs.rollouts:
        frames, flow = rollout(model, vae, val.frames[e][:1], text, stats, layout, chunks, sampler)
        frames, flow = frames[0], None if flow is None else flow[0]
        rows = np.minimum(np.arange(0, STRIDE * FLOW_STEPS * chunks + 1, STRIDE), len(val.flow[e]) - 1)
        real_frames, real_flow = val.frames[e][rows], val.flow[e][rows[:-1]]
        if flow is not None:
            res["rollout_flow_mm"][val.dirs[e].name] = flow_error_mm(flow, real_flow)
        write_mp4(out / "samples" / f"{step:05d}_rollout_{val.dirs[e].name}.mp4",
                  side_by_side(frames, flow, real_frames, real_flow, vs.cams[e], f"step {step}"), 10)
    if res["rollout_flow_mm"]:
        msg += f", 9.6 s rollouts {np.mean(list(res['rollout_flow_mm'].values())):.1f} mm"
    res["eval_s"] = round(time.time() - t1)
    print(f"{msg}, {res['eval_s']} s", flush=True)
    with open(out / "samples.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    model.train()


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
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--val-samples", type=int, default=32)
    ap.add_argument("--rollouts", type=int, default=4, help="val episodes rolled out from their first frame")
    ap.add_argument("--chunks", type=int, default=6, help="chunks per rollout (6 x 1.6 s = 9.6 s)")
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--state-p", type=float, default=0.0,
                    help="add a slot for the flow's current row, given clean with this probability (else predicted)")
    ap.add_argument("--video-only", action="store_true", help="no flow slot")
    args = ap.parse_args()
    out, device = Path(args.out), "cuda"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    layout = Layout(flow=not args.video_only, state=args.state_p > 0)

    t0 = time.time()
    train, val = Episodes("train", args.limit), Episodes("val", args.limit)
    stats = flow_stats(train)
    print(f"data: {len(train.steps)} train / {len(val.steps)} val steps, loaded in {time.time() - t0:.0f} s", flush=True)
    vae = load_vae(device)
    latents = cache_latents(vae, train, CACHE / f"cache_train_{len(train.steps)}.pt")
    train_flow = torch.from_numpy(normalise_flow(np.stack([train.sample(e, i)[1] for e, i in train.steps]), stats))
    model = load_transformer(args.base).to(device)
    text = prompt_embedding(args.base, device)
    params = add_lora(model, args.lora_rank)
    model.enable_gradient_checkpointing()
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M",
          flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    vs = ValSet(val, args.val_samples, args.rollouts)
    (out / "config.json").write_text(json.dumps({**vars(args), "flow_stats": stats}, indent=1))

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        k = torch.from_numpy(rng.integers(len(train.steps), size=args.batch))
        f = train_flow[k].to(device)
        x0 = build(latents[k].to(device).float(), layout, f.flatten(1), f[:, 0].flatten(1))
        given = torch.zeros(args.batch, layout.T, dtype=torch.bool, device=device)
        given[:, 0] = True
        if layout.state_idx is not None:
            given[:, layout.state_idx] = torch.rand(args.batch, device=device) < args.state_p
        with torch.autocast("cuda", torch.bfloat16):
            l = loss(model, x0, text, layout, given, args.flow_weight)
        opt.zero_grad(set_to_none=True)
        l["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append([l[n].item() for n in ("total", "video", "flow", "flow_hi", "state", "flow_free", "flow_given")])
        if step % 50 == 0:
            m = np.nanmean(log, 0)
            print(f"step {step} loss {m[0]:.4f} video {m[1]:.4f} flow {m[2]:.4f} flow_hi {m[3]:.4f} state {m[4]:.4f} "
                  f"flow_free {m[5]:.4f} flow_given {m[6]:.4f} "
                  f"{(time.time() - t0) / 50:.2f} s/step "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            log, t0 = [], time.time()
        if step % args.sample_every == 0 or step == args.steps:
            lora = {k: v for k, v in model.state_dict().items() if "lora" in k}
            torch.save(lora, out / "lora_latest.pt")  # overwritten each time (370 MB); wam/sample.py can re-sample it
            evaluate(model, vae, text, stats, vs, out, step, args.chunks, layout)
            t0 = time.time()
        if step % args.save_every == 0 or step == args.steps:
            torch.save({k: v for k, v in model.state_dict().items() if "lora" in k}, out / f"lora_{step}.pt")


if __name__ == "__main__":
    main()
