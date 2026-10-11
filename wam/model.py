"""World action model: Cosmos-Predict2-2B Video2World with the flow written into one extra latent frame (the
Cosmos Policy trick, arXiv 2601.16163: a small vector is tiled to fill a latent frame and averaged back out).

Latent sequence (Wan2.1 VAE: frame 0 -> 1 latent, then 4 frames -> 1 latent; 8x smaller in H and W):
  [0]    frame now                            given (clean)
  [1-4]  16 future frames at 10 Hz (1.6 s)    predicted
  [5]    flow, 16 steps x 14 points x 4       predicted
The flow frame is last so the VAE (causal in time) decodes the video latents without it.

Noise and parametrisation follow diffusers' Cosmos2VideoToWorldPipeline: x_t = (1 - t) x0 + t eps with
t = sigma / (1 + sigma); the transformer outputs v = eps - x0, so x0 = x_t - t v. Conditioning frames are passed
clean with t = 1e-4 / (1 + 1e-4) and condition_mask = 1. Training draws sigma ~ lognormal(1.386, 1.2) (Cosmos Policy).
"""
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, CosmosTransformer3DModel

VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # the Wan2.1 VAE that Cosmos-Predict2 uses; not gated
BASE_REPO = "nvidia/Cosmos-Predict2-2B-Video2World"  # gated (accept the licence on Hugging Face)
CONFIG_2B = dict(in_channels=17, out_channels=16, num_attention_heads=16, attention_head_dim=128, num_layers=28,
                 mlp_ratio=4.0, text_embed_dim=1024, adaln_lora_dim=256, max_size=(128, 240, 240),
                 patch_size=(1, 2, 2), rope_scale=(1.0, 3.0, 3.0), concat_padding_mask=True,
                 extra_pos_embed_type=None)  # diffusers scripts/convert_cosmos_to_diffusers.py, 2B Video2World
CONFIG_TINY = {**CONFIG_2B, "num_attention_heads": 4, "attention_head_dim": 64, "num_layers": 2, "adaln_lora_dim": 64}
FPS = 10
T_COND = 1e-4 / (1 + 1e-4)
N_COND = 1  # latent frames given


def load_transformer(base, dtype=torch.bfloat16):
    """base: a Hugging Face repo / folder in diffusers format, or "random-2b" / "random-tiny" (smoke tests)."""
    if base.startswith("random"):
        return CosmosTransformer3DModel(**(CONFIG_TINY if base == "random-tiny" else CONFIG_2B)).to(dtype)
    return CosmosTransformer3DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype)


def load_vae(device, dtype=torch.bfloat16):
    vae = AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder="vae", torch_dtype=dtype).to(device).eval()
    vae.requires_grad_(False)
    c = vae.config
    vae.lat_mean = torch.tensor(c.latents_mean, device=device).view(1, -1, 1, 1, 1)
    vae.lat_std = torch.tensor(c.latents_std, device=device).view(1, -1, 1, 1, 1)
    return vae


def encode_prompt(base, prompt, device):
    """T5 embedding of the prompt, [1, 512, 1024], zero past the prompt's tokens (as in the pipeline)."""
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    t5 = T5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device)
    ids = tok([prompt], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = t5(ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state
    emb[:, ids.attention_mask.sum():] = 0
    return emb.cpu()


@torch.no_grad()
def encode(vae, frames):
    """frames [B, T, H, W, 3] uint8 (T = 1 + 4k) -> normalised latents [B, 16, 1 + k, H/8, W/8]."""
    x = frames.permute(0, 4, 1, 2, 3).to(vae.dtype) / 127.5 - 1
    z = vae.encode(x).latent_dist.mode()
    return ((z - vae.lat_mean) / vae.lat_std).float()


@torch.no_grad()
def decode(vae, latents):
    """Normalised video latents [B, 16, 1 + k, h, w] -> frames [B, 1 + 4k, H, W, 3] uint8."""
    x = vae.decode((latents * vae.lat_std + vae.lat_mean).to(vae.dtype)).sample
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1)


def tile_flow(flow, shape):
    """flow [B, D] -> [B, C, 1, h, w]: the D values repeated in order until the latent frame is full."""
    C, h, w = shape
    n = C * h * w
    return flow.repeat(1, -(-n // flow.shape[1]))[:, :n].view(-1, C, 1, h, w)


def untile_flow(latent, D):
    """[B, C, 1, h, w] -> [B, D]: the mean over the whole copies."""
    flat = latent.flatten(1)
    k = flat.shape[1] // D
    return flat[:, :k * D].view(-1, k, D).mean(1)


def velocity(model, x_t, t, text):
    """x_t [B, C, T, h, w] with the first N_COND frames clean, t [B] noise level of the other frames."""
    B, _, T, h, w = x_t.shape
    timestep = t.view(B, 1, 1, 1, 1).expand(B, 1, T, 1, 1).clone()
    timestep[:, :, :N_COND] = T_COND
    cond_mask = torch.zeros(B, 1, T, h, w, device=x_t.device)
    cond_mask[:, :, :N_COND] = 1
    dtype = next(model.parameters()).dtype
    out = model(hidden_states=x_t.to(dtype), timestep=timestep.to(dtype), encoder_hidden_states=text.to(dtype).expand(B, -1, -1),
                fps=FPS, condition_mask=cond_mask.to(dtype), padding_mask=x_t.new_zeros(1, 1, h, w, dtype=dtype),
                return_dict=False)[0]
    return out.float()


def loss(model, x0, text, flow_weight=1.0):
    """x0 [B, C, T, h, w]: clean latents with the flow in the last frame. Returns (total, video, flow) losses."""
    B = x0.shape[0]
    sigma = torch.exp(1.386 + 1.2 * torch.randn(B, device=x0.device))
    t = sigma / (1 + sigma)
    eps = torch.randn_like(x0)
    x_t = (1 - t.view(B, 1, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1, 1) * eps
    x_t[:, :, :N_COND] = x0[:, :, :N_COND]
    err = (velocity(model, x_t, t, text) - (eps - x0)) ** 2
    video, flow = err[:, :, N_COND:-1].mean(), err[:, :, -1].mean()
    return video + flow_weight * flow, video, flow


@torch.no_grad()
def sample(model, cond, text, n_latents, steps=20, sigma_max=80.0):
    """cond [B, C, N_COND, h, w] clean latents -> [B, C, n_latents, h, w], Euler steps evenly spaced in t."""
    B, C, _, h, w = cond.shape
    ts = torch.linspace(sigma_max / (1 + sigma_max), 0, steps + 1, device=cond.device)
    x = torch.randn(B, C, n_latents, h, w, device=cond.device)
    for t, t_next in zip(ts[:-1], ts[1:]):
        x[:, :, :N_COND] = cond
        x = x + (t_next - t) * velocity(model, x, t.expand(B), text)
    x[:, :, :N_COND] = cond
    return x
