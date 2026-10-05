from __future__ import annotations

from typing import Any, Dict, Tuple

from navine.device_manager import detect_hardware, get_device


def _is_auto(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"auto", "detect", "smart"}


def _model_score(model_cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> float:
    d_model = int(model_cfg.get("d_model") or 256)
    n_layers = int(model_cfg.get("n_layers") or 4)
    seq_len = int(
        train_cfg.get("train_max_seq_len")
        or train_cfg.get("train_seq_len")
        or model_cfg.get("max_seq_len")
        or 512
    )
    image_size = int(model_cfg.get("image_size") or 0)
    num_frames = int(model_cfg.get("num_frames") or 0)
    if image_size > 0 and num_frames > 0:
        return float(image_size * image_size * num_frames * d_model) / 1e7
    if image_size > 0:
        return float(image_size * image_size * d_model) / 1e6
    return float(d_model * n_layers * seq_len) / 1e6


def resolve_training_batch(
    train_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    modality: str = "text",
) -> Tuple[int, int]:
    configured_batch = train_cfg.get("batch_size")
    configured_accum = max(1, int(train_cfg.get("grad_accum_steps") or 1))
    if not _is_auto(configured_batch):
        return max(1, int(configured_batch or 1)), configured_accum

    hw = detect_hardware()
    vram = float(hw.get("cuda_vram_gb") or 0.0)
    device = get_device()
    score = _model_score(model_cfg, train_cfg)
    kind = str(modality or "text").lower()

    if kind in {"video"}:
        batch = 1
        accum = configured_accum if vram >= 10 else max(configured_accum, 4)
        return batch, accum

    if kind in {"image", "diffusion"}:
        if vram >= 24 or score < 2.0:
            batch = 4
        elif vram >= 16 or score < 3.5:
            batch = 2
        elif vram >= 10:
            batch = 1
            accum = max(configured_accum, 2)
            return batch, accum
        else:
            batch = 1
            accum = max(configured_accum, 4)
            return batch, accum
        accum = max(1, configured_accum)
        return batch, accum

    if device.type != "cuda" or vram <= 0:
        if score > 8:
            return 1, max(configured_accum, 8)
        if score > 3:
            return 2, max(configured_accum, 4)
        return 4, max(configured_accum, 2)

    if vram < 6:
        batch, accum = 1, max(configured_accum, 16)
    elif vram < 10:
        if score >= 5:
            batch, accum = 1, max(configured_accum, 8)
        elif score >= 3:
            batch, accum = 2, max(configured_accum, 4)
        else:
            batch, accum = 4, max(configured_accum, 2)
    elif vram < 16:
        if score >= 10:
            batch, accum = 1, max(configured_accum, 8)
        elif score >= 5:
            batch, accum = 2, max(configured_accum, 4)
        else:
            batch, accum = 4, max(configured_accum, 2)
    elif vram < 24:
        if score >= 10:
            batch, accum = 2, max(configured_accum, 4)
        else:
            batch, accum = 4, max(configured_accum, 2)
    else:
        if score >= 12:
            batch, accum = 4, max(configured_accum, 2)
        else:
            batch, accum = 8, max(configured_accum, 1)

    return max(1, batch), max(1, accum)


def apply_auto_batch(
    train_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    modality: str = "text",
) -> Dict[str, int]:
    batch_size, grad_accum = resolve_training_batch(train_cfg, model_cfg, modality=modality)
    if _is_auto(train_cfg.get("batch_size")):
        hw = detect_hardware()
        vram = hw.get("cuda_vram_gb", "n/a")
        score = round(_model_score(model_cfg, train_cfg), 2)
        print(
            f"Navine AI - Python: auto batch size {batch_size} | grad_accum {grad_accum} | "
            f"vram_gb={vram} model_score={score} modality={modality}",
            flush=True,
        )
    return {"batch_size": batch_size, "grad_accum_steps": grad_accum}
