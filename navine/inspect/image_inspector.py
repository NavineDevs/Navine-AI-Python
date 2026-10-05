from pathlib import Path
from typing import Any, Dict, List

import torch

from navine.image.model import NavineDiffusionModel
from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir, get_project_root


def _logs_dir() -> Path:
    root = get_project_root()
    try:
        import yaml

        navine_cfg = yaml.safe_load((root / "configs" / "navine.yaml").read_text(encoding="utf-8")) or {}
        rel = str(navine_cfg.get("logs_dir") or "logs")
    except Exception:
        rel = "logs"
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def inspect_image_model(write_report: bool = True) -> Dict[str, Any]:
    config = load_config("image")
    ckpt_dir = get_checkpoint_dir("image")
    ckpt_path = ckpt_dir / "latest.pt"
    lines: List[str] = ["Navine AI - Python Image Model Inspector", "=" * 40]
    report: Dict[str, Any] = {"module": "image", "checkpoint": str(ckpt_path), "ok": False}
    if not ckpt_path.exists():
        lines.append(f"Missing checkpoint: {ckpt_path}")
        report["error"] = "missing_checkpoint"
        if write_report:
            _write_image_section(lines)
        return report
    device = torch.device("cpu")
    model = NavineDiffusionModel.load_checkpoint(ckpt_path, config, device)
    model_cfg = config.get("model") or {}
    unet = getattr(model, "unet", None)
    params = sum(p.numel() for p in model.parameters())
    channels = getattr(unet, "in_channels", model_cfg.get("in_channels"))
    image_size = int(model_cfg.get("image_size", 128))
    timesteps = int(getattr(model, "num_timesteps", model_cfg.get("timesteps", 1000)))
    text_encoder = type(getattr(model, "text_encoder", None)).__name__ if getattr(model, "unet", None) else "unknown"
    compatible = bool(getattr(model, "checkpoint_compatible", True))
    report.update(
        {
            "ok": True,
            "parameters": params,
            "image_size": image_size,
            "timesteps": timesteps,
            "in_channels": channels,
            "text_encoder": text_encoder,
            "checkpoint_compatible": compatible,
        }
    )
    lines.extend(
        [
            f"Checkpoint: {ckpt_path}",
            f"UNet parameters: {params:,}",
            f"image_size: {image_size}",
            f"timesteps: {timesteps}",
            f"in_channels: {channels}",
            f"text_encoder: {text_encoder}",
            f"checkpoint_compatible: {compatible}",
        ]
    )
    if write_report:
        _write_image_section(lines)
    return report


def _write_image_section(lines: List[str]) -> Path:
    out = _logs_dir() / "model_report.txt"
    text = "\n".join(lines) + "\n"
    if out.exists() and "Navine AI - Python Text Model Inspector" in out.read_text(encoding="utf-8"):
        existing = out.read_text(encoding="utf-8")
        if "Navine AI - Python Image Model Inspector" in existing:
            head, _, _tail = existing.partition("Navine AI - Python Image Model Inspector")
            out.write_text(head.rstrip() + "\n\n" + text, encoding="utf-8")
        else:
            out.write_text(existing.rstrip() + "\n\n" + text, encoding="utf-8")
    else:
        out.write_text(text, encoding="utf-8")
    return out
