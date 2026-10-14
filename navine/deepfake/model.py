from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from navine.utils.paths import get_checkpoint_dir, get_project_root


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class NavineDeepfakeModel(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        base_channels: int = 192,
        channel_mults: Optional[List[int]] = None,
        num_res_blocks: int = 3,
        latent_dim: int = 1024,
        arch_version: int = 1,
    ) -> None:
        super().__init__()
        channel_mults = list(channel_mults or [1, 2, 4, 4])
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = channel_mults
        self.num_res_blocks = int(num_res_blocks)
        self.latent_dim = int(latent_dim)
        self.arch_version = int(arch_version)

        chs = [self.base_channels * m for m in channel_mults]
        layers: List[nn.Module] = [nn.Conv2d(in_channels, chs[0], 3, padding=1)]
        prev = chs[0]
        spatial = image_size
        for i, ch in enumerate(chs):
            if prev != ch:
                layers.append(nn.Conv2d(prev, ch, 1))
                prev = ch
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(prev))
            if i < len(chs) - 1:
                nxt = chs[i + 1]
                layers.append(nn.Conv2d(prev, nxt, 4, stride=2, padding=1))
                prev = nxt
                spatial //= 2
        self.encoder = nn.Sequential(*layers)
        self.spatial = max(1, int(spatial))
        self.enc_out_ch = prev
        flat = prev * self.spatial * self.spatial
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)

        dec: List[nn.Module] = []
        rev = list(reversed(chs))
        prev = rev[0]
        for i, ch in enumerate(rev):
            if prev != ch:
                dec.append(nn.Conv2d(prev, ch, 1))
                prev = ch
            for _ in range(num_res_blocks):
                dec.append(ResidualBlock(prev))
            if i < len(rev) - 1:
                nxt = rev[i + 1]
                dec.append(nn.ConvTranspose2d(prev, nxt, 4, stride=2, padding=1))
                prev = nxt
        dec.extend(
            [
                nn.GroupNorm(8, prev),
                nn.GELU(),
                nn.Conv2d(prev, in_channels, 3, padding=1),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*dec)
        self.id_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.to_latent(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(z.shape[0], self.enc_out_ch, self.spatial, self.spatial)
        return self.decoder(h)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        src_z = self.encode(source)
        tgt_z = self.encode(target)
        blended = 0.55 * self.id_proj(src_z) + 0.45 * tgt_z
        return self.decode(blended)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            ptr = int(p.data_ptr()) if hasattr(p, "data_ptr") else id(p)
            if ptr in seen:
                continue
            seen.add(ptr)
            total += int(p.numel())
        return total

    def save(self, path: Optional[Path] = None) -> Path:
        out = Path(path) if path else get_checkpoint_dir("deepfake") / "latest.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.state_dict(),
            "parameters": int(self.count_parameters()),
            "architecture": {
                "image_size": self.image_size,
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "channel_mults": self.channel_mults,
                "num_res_blocks": self.num_res_blocks,
                "latent_dim": self.latent_dim,
                "arch_version": self.arch_version,
            },
        }
        torch.save(payload, out)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None, map_location: str = "cpu") -> "NavineDeepfakeModel":
        ckpt = Path(path) if path else get_checkpoint_dir("deepfake") / "latest.pt"
        payload = torch.load(ckpt, map_location=map_location, weights_only=False)
        arch = dict(payload.get("architecture") or {})
        keys = (
            "image_size",
            "in_channels",
            "base_channels",
            "channel_mults",
            "num_res_blocks",
            "latent_dim",
            "arch_version",
        )
        model = cls(**{k: arch[k] for k in keys if k in arch})
        state = payload.get("model") or payload.get("state_dict") or payload
        model.load_state_dict(state, strict=False)
        return model


def build_deepfake_model(config: Optional[Dict[str, Any]] = None) -> NavineDeepfakeModel:
    cfg = dict(config or {})
    model_cfg = dict(cfg.get("model") or cfg)
    keys = (
        "image_size",
        "in_channels",
        "base_channels",
        "channel_mults",
        "num_res_blocks",
        "latent_dim",
        "arch_version",
    )
    kwargs = {k: model_cfg[k] for k in keys if k in model_cfg}
    return NavineDeepfakeModel(**kwargs)


def load_deepfake_config() -> Dict[str, Any]:
    from navine.utils.config import load_config

    try:
        return load_config("deepfake_enterprise")
    except Exception:
        path = get_project_root() / "configs" / "deepfake_enterprise.yaml"
        if path.is_file():
            import yaml

            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {}
