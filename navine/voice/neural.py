from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavineVoiceModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 384,
        num_layers: int = 8,
        mel_bins: int = 80,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mel_bins = mel_bins
        self.max_seq_len = max_seq_len
        self.text_emb = nn.Embedding(vocab_size, hidden_dim)
        self.text_pos = nn.Embedding(max_seq_len, hidden_dim)
        self.encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.08 if num_layers > 1 else 0.0,
        )
        self.bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        dec_layers: List[nn.Module] = []
        ch = hidden_dim
        for _ in range(6):
            dec_layers.extend(
                [
                    nn.Conv1d(ch, ch, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.GroupNorm(8, ch),
                ]
            )
        self.mel_decoder = nn.Sequential(*dec_layers)
        self.mel_head = nn.Conv1d(hidden_dim, mel_bins, kernel_size=1)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        b, t = text_ids.shape
        pos = torch.arange(t, device=text_ids.device).unsqueeze(0).expand(b, -1)
        pos = pos.clamp(max=self.max_seq_len - 1)
        x = self.text_emb(text_ids) + self.text_pos(pos)
        encoded, _ = self.encoder(x)
        return self.bridge(encoded)

    def forward(self, text_ids: torch.Tensor, mel_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoded = self._encode_text(text_ids)
        mel_logits = self.mel_head(self.mel_decoder(encoded.transpose(1, 2))).transpose(1, 2)
        if mel_target is None:
            return mel_logits
        min_len = min(mel_logits.size(1), mel_target.size(1))
        return F.mse_loss(mel_logits[:, :min_len], mel_target[:, :min_len])

    @torch.no_grad()
    def predict_mel(self, text: str, device: torch.device) -> torch.Tensor:
        self.eval()
        chars = [min(ord(c), 255) for c in text[: self.max_seq_len - 1]] or [0]
        ids = torch.tensor([chars], dtype=torch.long, device=device)
        encoded = self._encode_text(ids)
        mel = self.mel_head(self.mel_decoder(encoded.transpose(1, 2))).transpose(1, 2)
        return mel.squeeze(0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "mel_bins": self.mel_bins,
            "max_seq_len": self.max_seq_len,
        }

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": self._get_config(),
                "parameters": int(self.count_parameters()),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Path, config: Optional[dict] = None, device: str = "cpu") -> "NavineVoiceModel":
        payload = torch.load(str(path), map_location=device, weights_only=False)
        model_cfg = dict(config or payload.get("config") or {})
        model = cls(
            vocab_size=int(model_cfg.get("vocab_size") or 256),
            hidden_dim=int(model_cfg.get("hidden_dim") or 384),
            num_layers=int(model_cfg.get("num_layers") or 8),
            mel_bins=int(model_cfg.get("mel_bins") or 80),
            max_seq_len=int(model_cfg.get("max_seq_len") or 512),
        )
        state = payload.get("model_state", payload)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
        return model


def build_voice_model(config: Optional[dict] = None) -> NavineVoiceModel:
    cfg = dict(config or {})
    return NavineVoiceModel(
        vocab_size=int(cfg.get("vocab_size") or 256),
        hidden_dim=int(cfg.get("hidden_dim") or 384),
        num_layers=int(cfg.get("num_layers") or 8),
        mel_bins=int(cfg.get("mel_bins") or 80),
        max_seq_len=int(cfg.get("max_seq_len") or 512),
    )
