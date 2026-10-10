"""Flow-matching S1 action policy, extracted from HVLA (TheWisp/lerobot, src/lerobot/policies/hvla/s1/flow_matching,
commit 13f05f3, Apache-2.0), without S2.

  context = transformer encoder over [DINOv2 patch tokens of each camera, robot-state token, optional flow tokens]
  decoder = transformer decoder: noisy action chunk (+ flow time) cross-attends to the context, predicts the velocity

Flow matching (Lipman et al. 2023, as in pi0): x_t = t * noise + (1 - t) * actions, target v = noise - actions;
inference integrates from noise (t = 1) to actions (t = 0) with Euler steps.

Kept from HVLA: the architecture and sizes (hidden 768, 8 heads, 4 encoder + 6 decoder layers, DINOv2 ViT-S/14
fine-tuned with gradient checkpointing), action + time fusion by concat -> MLP(SiLU), Beta(1.5, 1) time sampling,
15 Euler steps. Removed: the S2 latent token and its age embedding, training-time RTC (the sim waits for the policy,
so there is no inference delay to train for), saliency, checkpoint feature contracts. Changed: images are normalised
with the ImageNet mean / std DINOv2 was trained with (HVLA feeds [0, 1] pixels).
Added (both off by default, so the image-only checkpoints load unchanged):
  flow_dim > 0     flow_steps tokens, one per future flow step, each a linear projection of that step's flow_dim values
  n_cameras = 0    no image tokens and no DINOv2 (state / flow only)
  obs_pos_embed    a learned embedding per context-token position, added before the obs encoder (HVLA has none, so
                   the encoder could not tell the two cameras apart except by content, nor order the flow steps)
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class S1Config:
    action_dim: int
    state_dim: int
    n_cameras: int = 2
    chunk_size: int = 32
    image_size: int = 224
    hidden_dim: int = 768
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    dino_model: str = "dinov2_vits14"
    backbone_dim: int = 384
    num_inference_steps: int = 15
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0
    time_min: float = 0.001
    time_max: float = 1.0
    flow_dim: int = 0
    flow_steps: int = 16
    obs_pos_embed: bool = False

    def to_dict(self):
        return asdict(self)


def sinusoidal_embedding(t, dim, min_period=4e-3, max_period=4.0):
    freqs = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), dim // 2, device=t.device))
    args = t[..., None] * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class S1Model(nn.Module):
    def __init__(self, cfg: S1Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        if cfg.n_cameras:
            self.backbone = torch.hub.load("facebookresearch/dinov2", cfg.dino_model)
            self.image_proj = nn.Linear(cfg.backbone_dim, d)
        self.state_proj = nn.Linear(cfg.state_dim, d)
        if cfg.flow_dim:
            self.flow_proj = nn.Linear(cfg.flow_dim, d)
        if cfg.obs_pos_embed:
            n_ctx = cfg.n_cameras * (cfg.image_size // 14) ** 2 + 1 + (cfg.flow_steps if cfg.flow_dim else 0)
            self.obs_pos_embed = nn.Embedding(n_ctx, d)
            nn.init.normal_(self.obs_pos_embed.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.num_heads, cfg.dim_feedforward, cfg.dropout, batch_first=True)
        self.obs_encoder = nn.TransformerEncoder(layer, cfg.num_encoder_layers)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d, cfg.num_heads, cfg.dim_feedforward, cfg.dropout, batch_first=True)
            for _ in range(cfg.num_decoder_layers)])
        self.action_in_proj = nn.Linear(cfg.action_dim, d)
        self.action_out_proj = nn.Linear(d, cfg.action_dim)
        self.action_time_mlp_in = nn.Linear(2 * d, d)
        self.action_time_mlp_out = nn.Linear(d, d)
        self.action_pos_embed = nn.Embedding(cfg.chunk_size, d)
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def preprocess(self, images):
        """images: list of [B, H, W, 3] uint8 (one per camera) -> [n_cams * B, 3, S, S] normalised float."""
        x = torch.cat(images).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.cfg.image_size, self.cfg.image_size), mode="bilinear", antialias=True,
                          align_corners=False)
        return (x - self.img_mean) / self.img_std

    def encode(self, images, state, flow=None):
        """images: list of [B, H, W, 3] uint8; state: [B, state_dim]; flow: [B, flow_steps, flow_dim] (normalised)
        -> context [B, N_ctx, D]."""
        B, tokens = state.shape[0], []
        if self.cfg.n_cameras:
            x = self.preprocess(images)
            if self.training:
                patches = torch.utils.checkpoint.checkpoint(
                    lambda y: self.backbone.forward_features(y)["x_norm_patchtokens"], x, use_reentrant=False)
            else:
                patches = self.backbone.forward_features(x)["x_norm_patchtokens"]
            patches = self.image_proj(patches)  # [n_cams * B, P, D]
            tokens = list(patches.reshape(len(images), B, *patches.shape[1:]))
        tokens.append(self.state_proj(state).unsqueeze(1))
        if self.cfg.flow_dim:
            tokens.append(self.flow_proj(flow))
        x = torch.cat(tokens, dim=1)
        if self.cfg.obs_pos_embed:
            x = x + self.obs_pos_embed.weight.unsqueeze(0)
        return self.obs_encoder(x)

    def denoise_step(self, x_t, context, t):
        """x_t: [B, T, A]; t: [B] flow time -> velocity [B, T, A]."""
        B, T, _ = x_t.shape
        t_emb = sinusoidal_embedding(t[:, None].expand(B, T), self.cfg.hidden_dim)
        x = torch.cat([self.action_in_proj(x_t), t_emb.to(x_t.dtype)], dim=-1)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))
        x = x + self.action_pos_embed(torch.arange(T, device=x_t.device)).unsqueeze(0)
        for layer in self.decoder_layers:
            x = layer(x, context)
        return self.action_out_proj(x)

    def forward(self, images, state, actions, action_pad, flow=None):
        """Flow-matching loss. actions: [B, T, A] normalised; action_pad: [B, T] True past the episode end."""
        B = actions.shape[0]
        context = self.encode(images, state, flow)
        t = torch.distributions.Beta(self.cfg.time_beta_alpha, self.cfg.time_beta_beta).sample((B,)).to(actions.device)
        t = t * (self.cfg.time_max - self.cfg.time_min) + self.cfg.time_min
        noise = torch.randn_like(actions)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        v = self.denoise_step(x_t, context, t)
        mask = (~action_pad).unsqueeze(-1).float()
        return ((v.float() - (noise - actions)) ** 2 * mask).sum() / mask.sum().clamp(min=1) / actions.shape[-1]

    @torch.no_grad()
    def sample(self, images, state, num_steps=None, flow=None):
        """-> normalised action chunk [B, T, A]."""
        num_steps = num_steps or self.cfg.num_inference_steps
        context = self.encode(images, state, flow)
        x = torch.randn(state.shape[0], self.cfg.chunk_size, self.cfg.action_dim, device=state.device)
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((state.shape[0],), 1.0 + i * dt, device=state.device)
            x = x + dt * self.denoise_step(x, context, t)
        return x
