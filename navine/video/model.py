from typing import List, Optional

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, text: str, device: torch.device) -> torch.Tensor:
        chars = [min(ord(c), 255) for c in text[:64]] or [0]
        ids = torch.tensor([chars], dtype=torch.long, device=device)
        emb = self.emb(ids).mean(dim=1)
        return self.fc(emb)


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class FrameDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int, frame_size: int):
        super().__init__()
        self.frame_size = frame_size
        self.fc = nn.Linear(hidden_dim, hidden_dim * 4 * 4)
        layers: List[nn.Module] = []
        ch = hidden_dim
        size = 4
        while size * 2 < frame_size:
            nxt = max(16, ch // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(ch, nxt, 4, stride=2, padding=1),
                    nn.ReLU(),
                ]
            )
            ch = nxt
            size *= 2
        layers.extend(
            [
                nn.ConvTranspose2d(ch, out_channels, 4, stride=2, padding=1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.fc(h).view(-1, h.size(-1), 4, 4)
        x = self.net(x)
        if x.size(-1) != self.frame_size:
            x = F.interpolate(x, size=(self.frame_size, self.frame_size), mode="bilinear", align_corners=False)
        return x


class NavineVideoModel(nn.Module):
    def __init__(
        self,
        frame_size: int = 64,
        num_frames: int = 8,
        in_channels: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.text_encoder = TextEncoder(256, hidden_dim)
        self.frame_encoder = FrameEncoder(in_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.frame_decoder = FrameDecoder(hidden_dim, in_channels, frame_size)

    def forward(self, frames: torch.Tensor, text: str) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        device = frames.device
        text_emb = self.text_encoder.encode(text, device).unsqueeze(1).expand(b, t, -1)
        frame_embs = []
        for i in range(t):
            frame_embs.append(self.frame_encoder(frames[:, i]))
        frame_embs = torch.stack(frame_embs, dim=1)
        combined = torch.cat([frame_embs, text_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        preds = []
        for i in range(t):
            preds.append(self.frame_decoder(lstm_out[:, i]))
        pred_frames = torch.stack(preds, dim=1)
        if pred_frames.shape[-2:] != frames.shape[-2:]:
            bt = frames.shape[0] * frames.shape[1]
            frames = F.interpolate(
                frames.reshape(bt, frames.shape[2], frames.shape[3], frames.shape[4]),
                size=pred_frames.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape_as(pred_frames)
        return F.mse_loss(pred_frames, frames)

    @torch.no_grad()
    def generate(
        self,
        text: str,
        device: torch.device,
        num_frames: Optional[int] = None,
        seed: Optional[int] = None,
        initial_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        n = num_frames or self.num_frames
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        text_emb = self.text_encoder.encode(text, device)
        h = torch.zeros(num_layers, 1, hidden_dim, device=device)
        c = torch.zeros(num_layers, 1, hidden_dim, device=device)
        frames = []
        if initial_frame is not None:
            prev_frame = initial_frame.to(device)
            if prev_frame.dim() == 3:
                prev_frame = prev_frame.unsqueeze(0)
            if prev_frame.size(-1) != self.frame_size or prev_frame.size(-2) != self.frame_size:
                prev_frame = F.interpolate(
                    prev_frame,
                    size=(self.frame_size, self.frame_size),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            prev_frame = torch.randn(1, 3, self.frame_size, self.frame_size, device=device)
        for i in range(n):
            frame_emb = self.frame_encoder(prev_frame).unsqueeze(1)
            text_seq = text_emb.unsqueeze(1)
            combined = torch.cat([frame_emb, text_seq], dim=-1)
            lstm_out, (h, c) = self.lstm(combined, (h, c))
            next_frame = self.frame_decoder(lstm_out[:, 0])
            frames.append(next_frame)
            prev_frame = next_frame
        return torch.stack(frames, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_config(self) -> dict:
        return {
            "frame_size": self.frame_size,
            "num_frames": self.num_frames,
            "in_channels": self.frame_decoder.net[-2].out_channels,
            "hidden_dim": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
        }

    def save_checkpoint(self, path) -> None:
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": self._get_config(),
                "parameters": int(self.count_parameters()),
            },
            path,
        )

    @staticmethod
    def infer_config_from_state(state: dict, model_defaults: dict) -> Optional[dict]:
        emb = state.get("text_encoder.emb.weight")
        if emb is None or emb.dim() != 2:
            for key, value in state.items():
                if key.endswith("text_encoder.emb.weight") and value.dim() == 2:
                    emb = value
                    break
        if emb is None or emb.dim() != 2:
            return None
        hidden_dim = int(emb.shape[1])
        first_conv = state.get("frame_encoder.net.0.weight")
        if first_conv is None:
            for key, value in state.items():
                if key.endswith("frame_encoder.net.0.weight") and value.dim() == 4:
                    first_conv = value
                    break
        in_channels = int(first_conv.shape[1]) if first_conv is not None else model_defaults.get("in_channels", 3)
        num_layers = 0
        while f"lstm.weight_ih_l{num_layers}" in state:
            num_layers += 1
        if num_layers == 0:
            for key in state:
                match = re.search(r"lstm\.weight_ih_l(\d+)$", key)
                if match:
                    num_layers = max(num_layers, int(match.group(1)) + 1)
        if hidden_dim <= 128:
            frame_size = 64
        elif hidden_dim <= 256:
            frame_size = 96
        elif hidden_dim <= 512:
            frame_size = 160
        else:
            frame_size = int(model_defaults.get("frame_size", 192))
        return {
            "frame_size": frame_size,
            "num_frames": model_defaults.get("num_frames", 16),
            "in_channels": in_channels,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers or model_defaults.get("num_layers", 3),
        }

    @classmethod
    def load_checkpoint(cls, path, config: dict, device="cpu") -> "NavineVideoModel":
        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload.get("model_state", payload)
        saved_cfg = payload.get("config") or {}
        model_defaults = config["model"]
        valid_keys = {"frame_size", "num_frames", "in_channels", "hidden_dim", "num_layers"}
        if saved_cfg:
            model_cfg = {**model_defaults, **saved_cfg}
            for key in ("hidden_dim", "num_layers", "frame_size", "num_frames"):
                if key in model_defaults:
                    model_cfg[key] = model_defaults[key]
        else:
            model_cfg = cls.infer_config_from_state(state, model_defaults) or model_defaults
        model_cfg = {k: v for k, v in model_cfg.items() if k in valid_keys}
        try:
            model = cls(**model_cfg)
            model.load_state_dict(state, strict=True)
            model.checkpoint_compatible = True
            return model
        except (RuntimeError, ValueError, TypeError, KeyError):
            try:
                model = cls(**model_cfg)
                incompatible = model.load_state_dict(state, strict=False)
                missing = list(getattr(incompatible, "missing_keys", []) or [])
                if len(missing) > max(6, len(state) // 3):
                    raise RuntimeError("too many missing tensors for a compatible upgrade")
                print("Navine AI - Python: transferred compatible video weights into the upgraded model.")
                model.checkpoint_compatible = True
                return model
            except (RuntimeError, ValueError, TypeError, KeyError):
                print("Navine AI - Python warning: the saved video checkpoint does not match the current architecture.")
                print(f"Navine AI - Python: {path}")
                print("Navine AI - Python: starting from a fresh video model so generation and training can continue.")
                model = cls(**{k: v for k, v in model_defaults.items() if k in valid_keys})
                model.checkpoint_compatible = False
                return model
