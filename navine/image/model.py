import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Spatial self-attention used in open-source SD-style UNets."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = max(1, min(num_heads, channels // 32))
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = self.num_heads
        head = c // n
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, n, head, h * w).transpose(2, 3)
        k = k.view(b, n, head, h * w).transpose(2, 3)
        v = v.view(b, n, head, h * w).transpose(2, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        return x + self.proj(out)


class CrossAttention2d(nn.Module):
    """Cross-attention from spatial features to text tokens (SD conditioning pattern)."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = max(1, min(num_heads, channels // 32))
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.out_scale = nn.Parameter(torch.ones(1) * 2.0)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = self.num_heads
        head = c // n
        q = self.to_q(self.norm(x)).view(b, n, head, h * w).transpose(2, 3)
        k = self.to_k(context).view(b, -1, n, head).permute(0, 2, 1, 3)
        v = self.to_v(context).view(b, -1, n, head).permute(0, 2, 1, 3)
        valid = context.abs().sum(dim=-1) > 1e-6
        has_valid = valid.any(dim=1, keepdim=True)
        valid_eff = torch.where(has_valid, valid, torch.ones_like(valid))
        attn_mask = torch.zeros(
            b, 1, 1, context.size(1), device=x.device, dtype=q.dtype
        )
        attn_mask = attn_mask.masked_fill(~valid_eff.unsqueeze(1).unsqueeze(1), torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        return x + self.out_scale.to(dtype=out.dtype) * self.proj(out)


class AttnResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        context_dim: int,
        use_attn: bool,
    ):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.use_attn = use_attn
        if use_attn:
            self.self_attn = SelfAttention2d(out_ch)
            self.cross_attn = CrossAttention2d(out_ch, context_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.res(x, t_emb)
        if self.use_attn and context is not None:
            h = self.self_attn(h)
            h = self.cross_attn(h, context)
        return h


class ConditionedUNet(nn.Module):
    """
    Pixel-space UNet with SD-style self/cross-attention at deeper levels.
    Inspired by open-source from-scratch diffusion UNets; no external weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
        channel_mults: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        time_emb_dim: int = 384,
        text_emb_dim: int = 256,
        attn_resolutions: Optional[List[int]] = None,
        image_size: int = 128,
    ):
        super().__init__()
        channel_mults = list(channel_mults or [1, 2, 4, 4])
        num_res_blocks = max(1, int(num_res_blocks))
        self.num_levels = len(channel_mults)
        self.image_size = image_size
        attn_resolutions = list(attn_resolutions) if attn_resolutions is not None else [64, 32, 16]
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.text_pool_proj = nn.Linear(text_emb_dim, time_emb_dim)
        self.text_cond_scale = nn.Parameter(torch.ones(1) * 2.0)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        level_channels = [base_channels * mult for mult in channel_mults]

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = base_channels
        cur_res = image_size
        for i, ch in enumerate(level_channels):
            use_attn = cur_res in attn_resolutions
            blocks = nn.ModuleList()
            blocks.append(AttnResBlock(prev, ch, time_emb_dim, text_emb_dim, use_attn))
            for _ in range(num_res_blocks - 1):
                blocks.append(AttnResBlock(ch, ch, time_emb_dim, text_emb_dim, use_attn))
            self.down_blocks.append(blocks)
            if i < self.num_levels - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                cur_res = max(1, cur_res // 2)
            else:
                self.downsamples.append(None)
            prev = ch

        self.mid = AttnResBlock(prev, prev, time_emb_dim, text_emb_dim, True)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx in range(self.num_levels):
            i = self.num_levels - 1 - idx
            ch = level_channels[i]
            if idx == 0:
                self.upsamples.append(None)
                up_res = image_size // (2 ** (self.num_levels - 1))
            else:
                self.upsamples.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1))
                up_res = image_size // (2 ** (self.num_levels - 1 - idx))
            use_attn = up_res in attn_resolutions
            blocks = nn.ModuleList()
            blocks.append(AttnResBlock(ch * 2, ch, time_emb_dim, text_emb_dim, use_attn))
            for _ in range(num_res_blocks - 1):
                blocks.append(AttnResBlock(ch, ch, time_emb_dim, text_emb_dim, use_attn))
            self.up_blocks.append(blocks)
            prev = ch

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _run_block(self, block: nn.Module, h: torch.Tensor, t_emb: torch.Tensor, text_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            def _fn(x: torch.Tensor, te: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
                return block(x, te, ctx)

            return torch.utils.checkpoint.checkpoint(_fn, h, t_emb, text_ctx, use_reentrant=False)
        return block(h, t_emb, text_ctx)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_ctx: Optional[torch.Tensor] = None,
        text_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)
        if text_pool is not None:
            t_emb = t_emb + self.text_cond_scale.to(dtype=t_emb.dtype) * self.text_pool_proj(text_pool)
        h = self.in_conv(x)
        skips: List[torch.Tensor] = []
        for i in range(self.num_levels):
            for block in self.down_blocks[i]:
                h = self._run_block(block, h, t_emb, text_ctx)
            skips.append(h)
            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)
        h = self._run_block(self.mid, h, t_emb, text_ctx)
        for idx in range(self.num_levels):
            i = self.num_levels - 1 - idx
            if self.upsamples[idx] is not None:
                h = self.upsamples[idx](h)
            skip = skips[i]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in self.up_blocks[idx]:
                h = self._run_block(block, h, t_emb, text_ctx)
        return self.out_conv(F.silu(self.out_norm(h)))


class TokenTextEncoder(nn.Module):
    """
    Lightweight token transformer for prompts.
    Open-source SD uses CLIP; we train our own encoder from scratch (byte/char tokens).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        emb_dim: int = 256,
        max_len: int = 77,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_dim)
        nn.init.normal_(self.pos_emb, std=0.02)

    def tokenize(self, text: str, device: torch.device) -> torch.Tensor:
        chars = [min(ord(c), 255) for c in text[: self.max_len]]
        if not chars:
            chars = [0]
        if len(chars) < self.max_len:
            chars = chars + [0] * (self.max_len - len(chars))
        return torch.tensor(chars, dtype=torch.long, device=device)

    def encode(self, text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenize(text, device).unsqueeze(0)
        return self.encode_ids(ids)

    def encode_ids(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_mask = ids.eq(0)
        if pad_mask.all(dim=1).any():
            pad_mask = pad_mask.clone()
            for i in range(pad_mask.size(0)):
                if bool(pad_mask[i].all()):
                    pad_mask[i, 0] = False
        lengths = (~pad_mask).sum(dim=1)
        max_l = int(lengths.max().clamp(min=1).item())
        ids = ids[:, :max_l]
        pad_mask = pad_mask[:, :max_l]
        x = self.token_emb(ids) + self.pos_emb[:, : ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        keep = (~pad_mask).unsqueeze(-1).to(dtype=x.dtype)
        x = x * keep
        denom = keep.sum(dim=1).clamp(min=1.0)
        pooled = x.sum(dim=1) / denom
        return x, pooled

    def encode_texts(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.stack([self.tokenize(t, device) for t in texts], dim=0)
        return self.encode_ids(ids)


# Back-compat alias
TextConditionEncoder = TokenTextEncoder
SimpleUNet = ConditionedUNet


class NavineDiffusionModel(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        base_channels: int = 96,
        channel_mults: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        time_emb_dim: int = 384,
        text_emb_dim: int = 256,
        timesteps: int = 500,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        attn_resolutions: Optional[List[int]] = None,
        text_max_len: int = 77,
        text_layers: int = 4,
        text_heads: int = 4,
        min_snr_gamma: float = 5.0,
        arch_version: int = 2,
    ):
        super().__init__()
        channel_mults = list(channel_mults or [1, 2, 4, 4])
        self.image_size = image_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = str(beta_schedule or "cosine")
        self.attn_resolutions = list(attn_resolutions) if attn_resolutions is not None else [64, 32, 16]
        self.text_max_len = text_max_len
        self.text_layers = text_layers
        self.text_heads = text_heads
        self.min_snr_gamma = float(min_snr_gamma)
        self.arch_version = int(arch_version)
        self.unet = ConditionedUNet(
            in_channels,
            base_channels,
            channel_mults,
            num_res_blocks,
            time_emb_dim,
            text_emb_dim,
            self.attn_resolutions,
            image_size,
        )
        self.text_encoder = TokenTextEncoder(
            256, text_emb_dim, text_max_len, text_layers, text_heads
        )
        from navine.image.schedules import make_beta_schedule

        betas = make_beta_schedule(timesteps, beta_start, beta_end, self.beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def set_noise_schedule(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
    ) -> None:
        from navine.image.schedules import make_beta_schedule

        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = str(beta_schedule or "linear")
        betas = make_beta_schedule(self.timesteps, self.beta_start, self.beta_end, self.beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        device = self.betas.device if hasattr(self, "betas") else torch.device("cpu")
        dtype = self.betas.dtype if hasattr(self, "betas") else torch.float32
        self.register_buffer("betas", betas.to(device=device, dtype=dtype))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(device=device, dtype=dtype))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device=device, dtype=dtype))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).to(device=device, dtype=dtype),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def _split_text(self, text_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_emb.dim() == 2:
            return text_emb.unsqueeze(1), text_emb
        mag = text_emb.abs().sum(dim=-1)
        mask = (mag > 1e-6).to(dtype=text_emb.dtype)
        all_zero = mask.sum(dim=1, keepdim=True) < 0.5
        mask = torch.where(all_zero, torch.ones_like(mask), mask)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (text_emb * mask.unsqueeze(-1)).sum(dim=1) / denom
        return text_emb, pooled

    def forward(self, x0: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        ctx, pooled = self._split_text(text_emb)
        pred_noise = self.unet(x_noisy, t, ctx, pooled)
        loss = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=(1, 2, 3))
        if self.min_snr_gamma > 0:
            snr = self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t] + 1e-8)
            weight = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / (snr + 1e-8)
            loss = loss * weight
        return loss.mean()

    def _predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        ctx, pooled = self._split_text(text_emb)
        pred = self.unet(x, t, ctx, pooled)
        if guidance_scale > 1.0:
            uncond_ctx = torch.zeros_like(ctx)
            uncond_pool = torch.zeros_like(pooled)
            uncond = self.unet(x, t, uncond_ctx, uncond_pool)
            pred = uncond + guidance_scale * (pred - uncond)
        return pred

    def _sample_timesteps(self, num_steps: Optional[int], device: torch.device) -> torch.Tensor:
        steps = num_steps or self.timesteps
        if steps >= self.timesteps:
            return torch.arange(self.timesteps - 1, -1, -1, device=device, dtype=torch.long)
        return torch.linspace(self.timesteps - 1, 0, steps, device=device).round().long()

    def _predict_x0(self, x: torch.Tensor, t_int: int, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[t_int].to(dtype=x.dtype).clamp(min=1e-5)
        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar + 1e-8)
        return pred_x0.clamp(-1.0, 1.0)

    def _ddpm_step(
        self,
        x: torch.Tensor,
        t_int: int,
        t_prev_int: int,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[t_int]
        alpha_bar_prev = self.alphas_cumprod[t_prev_int] if t_prev_int >= 0 else self.alphas_cumprod[0] * 0 + 1.0
        pred_x0 = self._predict_x0(x, t_int, pred_noise)
        if t_prev_int < 0:
            return pred_x0
        alpha = 1.0 - self.betas[t_int]
        beta = self.betas[t_int]
        mean = (
            torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar + 1e-8) * pred_x0
            + torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar + 1e-8) * x
        )
        if t_prev_int == 0:
            return mean
        variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar + 1e-8)
        return mean + torch.sqrt(variance.clamp(min=1e-20)) * torch.randn_like(x)

    def _ddim_step(
        self,
        x: torch.Tensor,
        t_int: int,
        t_prev_int: int,
        pred_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[t_int]
        if t_prev_int < 0:
            alpha_bar_prev = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        else:
            alpha_bar_prev = self.alphas_cumprod[t_prev_int]
        ab_t = alpha_bar.to(dtype=x.dtype).clamp(min=1e-5)
        ab_p = alpha_bar_prev.to(dtype=x.dtype).clamp(min=1e-5)
        pred_x0 = (x - torch.sqrt(1.0 - ab_t) * pred_noise) / torch.sqrt(ab_t + 1e-8)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)
        return torch.sqrt(ab_p) * pred_x0 + torch.sqrt(1.0 - ab_p) * pred_noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        text: str,
        device: torch.device,
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        scheduler: Optional[str] = None,
        eta: float = 0.0,
        init_image=None,
        init_noise_strength: float = 0.55,
    ) -> torch.Tensor:
        self.eval()
        ctx, pooled = self.text_encoder.encode(text, device)
        ctx = ctx.expand(batch_size, -1, -1)
        pooled = pooled.expand(batch_size, -1)
        text_emb = ctx
        steps = num_steps or self.timesteps
        sched = str(scheduler or "ddim").lower()
        if sched in ("ddim", "dpm") and steps >= self.timesteps:
            steps = max(20, min(self.timesteps - 1, int(self.timesteps * 0.4)))
        timesteps = self._sample_timesteps(steps, device)
        # Honor explicit DDPM; only auto-switch to DDIM for ddim/dpm schedulers.
        use_ddim = sched in ("ddim", "dpm")
        if sched == "ddpm":
            use_ddim = False
            if steps < self.timesteps:
                timesteps = self._sample_timesteps(self.timesteps, device)
                steps = self.timesteps

        x = torch.randn(batch_size, 3, self.image_size, self.image_size, device=device)
        start_idx = 0
        if init_image is not None:
            try:
                from PIL import Image
                import torchvision.transforms as T

                if isinstance(init_image, Image.Image):
                    img = init_image.convert("RGB").resize(
                        (self.image_size, self.image_size), Image.Resampling.LANCZOS
                    )
                    tens = T.ToTensor()(img).unsqueeze(0).to(device)
                    x0 = tens * 2.0 - 1.0
                elif torch.is_tensor(init_image):
                    x0 = init_image.to(device)
                    if x0.dim() == 3:
                        x0 = x0.unsqueeze(0)
                    if x0.shape[-1] != self.image_size or x0.shape[-2] != self.image_size:
                        x0 = torch.nn.functional.interpolate(
                            x0, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
                        )
                    if x0.min() >= 0:
                        x0 = x0 * 2.0 - 1.0
                else:
                    x0 = None
                if x0 is not None:
                    if x0.size(0) == 1 and batch_size > 1:
                        x0 = x0.expand(batch_size, -1, -1, -1)
                    strength = float(max(0.15, min(0.95, init_noise_strength)))
                    start_idx = int((1.0 - strength) * max(len(timesteps) - 1, 1))
                    start_idx = max(0, min(start_idx, len(timesteps) - 2))
                    t0 = int(timesteps[start_idx].item())
                    t_tensor = torch.full((batch_size,), t0, device=device, dtype=torch.long)
                    x = self.q_sample(x0, t_tensor)
            except Exception:
                start_idx = 0
                x = torch.randn(batch_size, 3, self.image_size, self.image_size, device=device)

        for idx in range(start_idx, len(timesteps)):
            t_int = int(timesteps[idx].item())
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            pred = self._predict_noise(x, t, text_emb, guidance_scale)
            if idx == len(timesteps) - 1:
                x = self._predict_x0(x, t_int, pred)
                break
            t_prev = int(timesteps[idx + 1].item())
            if use_ddim:
                x = self._ddim_step(x, t_int, t_prev, pred)
            else:
                x = self._ddpm_step(x, t_int, t_prev, pred)
        return x.clamp(-1, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path) -> None:
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": self._get_config(),
                "parameters": int(self.count_parameters()),
            },
            path,
        )

    def _get_config(self) -> dict:
        return {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "channel_mults": self.channel_mults,
            "num_res_blocks": self.num_res_blocks,
            "time_emb_dim": self.time_emb_dim,
            "text_emb_dim": self.text_emb_dim,
            "timesteps": self.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "attn_resolutions": self.attn_resolutions,
            "text_max_len": self.text_max_len,
            "text_layers": self.text_layers,
            "text_heads": self.text_heads,
            "min_snr_gamma": self.min_snr_gamma,
            "arch_version": self.arch_version,
        }

    @staticmethod
    def infer_config_from_state(state: dict, model_defaults: dict) -> Optional[dict]:
        in_conv = state.get("unet.in_conv.weight")
        if in_conv is None or in_conv.dim() != 4:
            return None
        base_channels = int(in_conv.shape[0])
        in_channels = int(in_conv.shape[1])
        channel_mults: List[int] = []
        level = 0
        while f"unet.down_blocks.{level}.0.res.conv1.weight" in state or f"unet.down_blocks.{level}.0.conv1.weight" in state:
            key = (
                f"unet.down_blocks.{level}.0.res.conv1.weight"
                if f"unet.down_blocks.{level}.0.res.conv1.weight" in state
                else f"unet.down_blocks.{level}.0.conv1.weight"
            )
            ch = int(state[key].shape[0])
            channel_mults.append(max(1, ch // base_channels))
            level += 1
        num_res_blocks = 0
        while (
            f"unet.down_blocks.0.{num_res_blocks}.res.conv1.weight" in state
            or f"unet.down_blocks.0.{num_res_blocks}.conv1.weight" in state
        ):
            num_res_blocks += 1
        time_weight = state.get("unet.time_emb.mlp.0.weight")
        time_emb_dim = int(time_weight.shape[1]) if time_weight is not None else model_defaults.get("time_emb_dim", 384)
        text_weight = state.get("unet.text_pool_proj.weight") or state.get("unet.text_proj.weight")
        text_emb_dim = int(text_weight.shape[1]) if text_weight is not None else model_defaults.get("text_emb_dim", 256)
        cfg = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "channel_mults": channel_mults or model_defaults.get("channel_mults", [1, 2, 4, 4]),
            "num_res_blocks": num_res_blocks or model_defaults.get("num_res_blocks", 2),
            "time_emb_dim": time_emb_dim,
            "text_emb_dim": text_emb_dim,
            "image_size": model_defaults.get("image_size", 128),
            "arch_version": 2 if "text_encoder.token_emb.weight" in state else 1,
        }
        betas = state.get("betas")
        if betas is not None and betas.numel() > 0:
            cfg["timesteps"] = int(betas.numel())
            cfg["beta_start"] = float(betas[0])
            cfg["beta_end"] = float(betas[-1])
        return cfg

    @classmethod
    def load_checkpoint(
        cls,
        path,
        full_config: dict,
        device="cpu",
        *,
        prefer_saved_architecture: bool = False,
        allow_fresh_fallback: bool = True,
    ) -> "NavineDiffusionModel":
        import inspect

        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload.get("model_state") or payload.get("model") or payload
        if not isinstance(state, dict) or not any(hasattr(v, "numel") for v in state.values()):
            state = payload.get("model_state") or payload.get("model") or {}
        saved_cfg = payload.get("config") or payload.get("architecture") or {}
        if isinstance(saved_cfg, dict) and "model" in saved_cfg and isinstance(saved_cfg.get("model"), dict):
            saved_cfg = dict(saved_cfg.get("model") or {})
        model_defaults = full_config["model"]
        default_cfg = {**full_config["model"], **(full_config.get("diffusion") or {})}
        if saved_cfg:
            if prefer_saved_architecture:
                model_cfg = {**default_cfg, **saved_cfg}
            else:
                model_cfg = {**default_cfg, **saved_cfg}
                for key in (
                    "image_size",
                    "attn_resolutions",
                    "arch_version",
                    "base_channels",
                    "channel_mults",
                    "time_emb_dim",
                    "text_emb_dim",
                    "num_res_blocks",
                    "text_max_len",
                    "text_layers",
                    "text_heads",
                ):
                    if key in model_defaults:
                        model_cfg[key] = model_defaults[key]
        else:
            model_cfg = cls.infer_config_from_state(state, model_defaults) or default_cfg
        allowed = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}

        def _filter(cfg: dict) -> dict:
            return {k: v for k, v in cfg.items() if k in allowed}

        try:
            model = cls(**_filter(model_cfg))
            model.load_state_dict(state, strict=True)
            model.checkpoint_compatible = True
            return model
        except (RuntimeError, ValueError, TypeError, KeyError):
            try:
                native = {**default_cfg, **(saved_cfg or {})}
                if not saved_cfg:
                    native = cls.infer_config_from_state(state, model_defaults) or default_cfg
                model = cls(**_filter(native))
                model.load_state_dict(state, strict=True)
                model.checkpoint_compatible = True
                return model
            except (RuntimeError, ValueError, TypeError, KeyError):
                pass
            try:
                model = cls(**_filter(model_cfg))
                incompatible = model.load_state_dict(state, strict=False)
                missing = list(getattr(incompatible, "missing_keys", []) or [])
                if len(missing) > max(8, len(state) // 3):
                    raise RuntimeError("too many missing tensors for a compatible upgrade")
                print("Navine AI - Python: transferred compatible image weights into the upgraded geometry.")
                model.checkpoint_compatible = True
                return model
            except (RuntimeError, ValueError, TypeError, KeyError):
                print("Navine AI - Python warning: the saved image checkpoint does not match the current architecture.")
                print(f"Navine AI - Python: {path}")
                if not allow_fresh_fallback:
                    model = cls(**_filter(default_cfg))
                    model.checkpoint_compatible = False
                    return model
                print("Navine AI - Python: starting from a fresh image model so generation and training can continue.")
                print("Navine AI - Python: the next checkpoint will be saved in the self-describing format.")
                model = cls(**_filter(default_cfg))
                model.checkpoint_compatible = False
                return model
