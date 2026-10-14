"""Plan whole episodes with the WAM for the end-to-end test (step 6.3): from the episode's first real frame, chained
1.6 s chunks (wam/train.py rollout) until the policy's horizon is covered (recorded length + 3 s, as policy/eval_s1.py).

  python wam/plan.py outputs/wam/v1/lora_20000.pt --out outputs/wam/v1/e2e      -> <out>/plans/<episode>.npz, .mp4

Per episode: <episode>.npz flow [16 chunks, 14, 4] (table frame, 10 Hz: row j is sim step 2 j) and <episode>.mp4,
the generated frames (10 Hz, frame 0 = the real first frame). policy/eval_s1.py --plan <out> reads them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))
from data import FLOW_STEPS, STRIDE, read_frames  # noqa: E402
from render import write_mp4  # noqa: E402
from sample import load_run  # noqa: E402
from train import rollout  # noqa: E402
from train_s1 import episode_dirs  # noqa: E402

HZ, EXTRA_S = 20, 3  # as policy/eval_s1.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "failed"])
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--sigma-min", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    cfg = json.loads((args.ckpt.parent / "config.json").read_text())
    (args.out / "plans").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model, vae, text, layout = load_run(args.ckpt)
    eps = [ep for s in args.splits for ep in episode_dirs(s) if not (args.out / "plans" / f"{ep.name}.npz").exists()]
    horizon = {ep: len(np.load(ep / "sim.npz")["action"]) + EXTRA_S * HZ for ep in eps}
    eps.sort(key=horizon.get)  # similar lengths in a batch
    t0 = time.time()
    for k in range(0, len(eps), args.batch):
        batch = eps[k:k + args.batch]
        chunks = -(-max(horizon[ep] for ep in batch) // (STRIDE * FLOW_STEPS))
        frame0 = np.stack([read_frames(ep / "video_sim.mp4", first=True)[0] for ep in batch])
        with torch.autocast("cuda", torch.bfloat16):
            frames, flow = rollout(model, vae, frame0, text, cfg["flow_stats"], layout, chunks,
                                   dict(steps=args.steps, sigma_min=args.sigma_min))
        for j, ep in enumerate(batch):
            np.savez(args.out / "plans" / f"{ep.name}.npz", flow=flow[j])
            write_mp4(args.out / "plans" / f"{ep.name}.mp4", frames[j], 10)
        print(f"{k + len(batch)}/{len(eps)} episodes, {chunks} chunks, {(time.time() - t0) / 60:.1f} min", flush=True)
    (args.out / "plan_config.json").write_text(json.dumps(vars(args) | {"ckpt": str(args.ckpt), "out": str(args.out)}, indent=1))


if __name__ == "__main__":
    main()
