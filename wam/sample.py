"""Generate the val samples and chained rollouts (as in training) from a saved LoRA checkpoint.

  python wam/sample.py outputs/wam/v1/lora_2500.pt --steps 5     -> outputs/wam/v1/resample/
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Episodes  # noqa: E402
from model import load_transformer, load_vae  # noqa: E402
from train import ValSet, add_lora, evaluate, prompt_embedding  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=5, help="sampler steps (model calls per chunk)")
    ap.add_argument("--out", default=None, help="default: <run>/resample")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    out = Path(args.out) if args.out else ckpt.parent / "resample"
    torch.manual_seed(0)
    vae = load_vae("cuda")
    model = load_transformer(cfg["base"]).cuda()
    add_lora(model, cfg["lora_rank"])
    missing = set(model.load_state_dict(torch.load(ckpt), strict=False).unexpected_keys)
    assert not missing, missing
    model.bfloat16()  # inference only
    text = prompt_embedding(cfg["base"], "cuda")
    vs = ValSet(Episodes("val", cfg["limit"]), cfg["val_samples"], cfg["rollouts"])
    if ckpt.stem == "lora_latest":  # saved just before the last sample round
        step = json.loads((ckpt.parent / "samples.jsonl").read_text().splitlines()[-1])["step"]
    else:
        step = int(ckpt.stem.split("_")[1])
    with torch.autocast("cuda", torch.bfloat16):
        evaluate(model, vae, text, cfg["flow_stats"], vs, out, step, cfg["chunks"], args.steps, args.batch)


if __name__ == "__main__":
    main()
