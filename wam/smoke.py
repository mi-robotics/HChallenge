"""Smoke tests for wam/ that need no Cosmos weights (the Wan2.1 VAE is downloaded, not gated).

  python wam/smoke.py

1. data: a sample's frames and flow are the right rows of video_sim.mp4 / flow_sim.npz; normalising round-trips.
2. flow tiling: tile -> untile gives the flow back exactly.
3. VAE: encode/decode of a 17-frame sample (latent shape, PSNR, time).
4. overlay: the recorded flow drawn on the sample's frames -> outputs/wam/smoke/overlay_gt.mp4 (check by eye).
Training-loop checks (overfit, memory of the full 2B model) are runs of wam/train.py, see PROJECT_PLAN.md.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import H, W, Episodes, denormalise_flow, flow_stats, normalise_flow, read_flow, read_frames  # noqa: E402
from model import decode, encode, load_vae, tile_flow, untile_flow  # noqa: E402
from render import camera, overlay, write_mp4  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "wam" / "smoke"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eps = Episodes("train", limit=2)
    ep = eps.dirs[1]
    n = len(eps.flow[1])
    i = n - 20  # near the end, so the last rows are clamped
    frames, flow = eps.sample(1, i)
    video, raw_flow = read_frames(ep / "video_sim.mp4"), read_flow(ep)
    rows = np.minimum(i + np.arange(0, 33, 2), n - 1)
    assert frames.shape == (17, H, W, 3) and flow.shape == (16, 14, 4)
    assert np.array_equal(frames, video[rows]) and np.array_equal(flow, raw_flow[rows[:-1]])
    stats = flow_stats(eps)
    back = denormalise_flow(normalise_flow(flow, stats), stats)
    assert np.abs(back - flow).max() < 1e-5
    print(f"1. data ok: {ep.name} has {n} steps, sample at {i} uses rows {rows.tolist()}; flow stats {stats}")

    f = torch.from_numpy(normalise_flow(np.stack([flow, flow[::-1]]), stats)).flatten(1)
    tiled = tile_flow(f, (16, H // 8, W // 8))
    err = (untile_flow(tiled, f.shape[1]) - f).abs().max().item()
    assert err < 1e-6
    print(f"2. flow tiling ok: {f.shape[1]} values -> latent frame {tuple(tiled.shape[1:])}, "
          f"{tiled[0].numel() // f.shape[1]} copies, round-trip error {err:.1e}")

    vae = load_vae("cuda")
    x = torch.from_numpy(frames[None]).cuda()
    torch.cuda.synchronize(); t0 = time.time()
    z = encode(vae, x)
    torch.cuda.synchronize(); t1 = time.time()
    rec = decode(vae, z)
    torch.cuda.synchronize(); t2 = time.time()
    mse = ((rec.float() - x.float()) ** 2).mean().item()
    print(f"3. VAE ok: latents {tuple(z.shape)} (mean {z.mean():.2f}, std {z.std():.2f}), "
          f"PSNR {10 * np.log10(255 ** 2 / mse):.1f} dB, encode {t1 - t0:.2f} s, decode {t2 - t1:.2f} s")
    write_mp4(OUT / "vae_roundtrip.mp4", np.concatenate([frames, rec[0].cpu().numpy()], 2), 10)

    cam = camera(ep)
    write_mp4(OUT / "overlay_gt.mp4", [overlay(f, flow, cam) for f in frames], 10)  # the hand should follow the trail
    print(f"4. overlay written: {OUT / 'overlay_gt.mp4'}")


if __name__ == "__main__":
    main()
