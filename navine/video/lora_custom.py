from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from navine.utils.config import load_config
from navine.utils.lora_core import (
    adapter_state,
    inject_lora,
    load_adapter_state,
    merge_lora_into_module,
)
from navine.utils.paths import get_project_root
from navine.utils.tier import resolve_checkpoint_dir
from navine.video.model import NavineVideoModel

LORA_CONFIG_NAME = "video_lora"


def _lora_config() -> dict:
    return load_config(LORA_CONFIG_NAME)


def _base_config_name() -> str:
    return str(_lora_config().get("base_config") or "video_enterprise")


def _base_full_config() -> dict:
    return load_config(_base_config_name())


def _lora_dir() -> Path:
    cfg = _lora_config()
    ns = str(cfg.get("checkpoint_namespace") or "video_lora")
    path = get_project_root() / "checkpoints" / ns
    path.mkdir(parents=True, exist_ok=True)
    return path


def _adapter_path() -> Path:
    return _lora_dir() / "adapter.pt"


def _inject_video_lora(model: NavineVideoModel, rank: int, alpha: float) -> int:
    injected = inject_lora(model.frame_encoder, rank, alpha)
    injected += inject_lora(model.frame_decoder, rank, alpha)
    return injected


def _load_base_model(device: torch.device) -> NavineVideoModel:
    base_cfg = _base_full_config()
    ckpt_dir = resolve_checkpoint_dir("video", base_cfg)
    ckpt_path = ckpt_dir / "latest.pt"
    if not ckpt_path.exists():
        raise RuntimeError(
            f"No base video checkpoint at {ckpt_path}. Train the base model first: "
            f"python -m navine.cli train video --config {_base_config_name()}"
        )
    return NavineVideoModel.load_checkpoint(ckpt_path, base_cfg, device).to(device)


def _build_lora_model(device: torch.device, load_weights: bool = True) -> NavineVideoModel:
    cfg = _lora_config()
    lora_cfg = cfg.get("lora") or {}
    rank = int(lora_cfg.get("rank") or 8)
    alpha = float(lora_cfg.get("alpha") or 16)
    model = _load_base_model(device)
    for param in model.parameters():
        param.requires_grad_(False)
    _inject_video_lora(model, rank, alpha)
    model = model.to(device)
    adapter_path = _adapter_path()
    if load_weights and adapter_path.exists():
        payload = torch.load(adapter_path, map_location=device, weights_only=False)
        load_adapter_state(model, payload.get("adapter") or {})
    return model


def train_video_lora(
    steps: Optional[int] = None,
    progress=None,
) -> Dict[str, object]:
    from torch.utils.data import DataLoader

    from navine.device_manager import get_autocast_context, get_device, get_dataloader_kwargs
    from navine.image.ema import EMATracker
    from navine.utils.training_lock import training_lock
    from navine.video.train import VideoFrameDataset

    with training_lock("video_lora"):
        config = _lora_config()
        base_cfg = _base_full_config()
        lora_cfg = config.get("lora") or {}
        train_cfg = config.get("training") or {}
        data_cfg = config.get("data") or {}
        rank = int(lora_cfg.get("rank") or 8)
        alpha = float(lora_cfg.get("alpha") or 16)
        device = get_device()
        model = _load_base_model(device)
        for param in model.parameters():
            param.requires_grad_(False)
        injected = _inject_video_lora(model, rank, alpha)
        model = model.to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable)
        print(
            f"Navine AI - Python LoRA (video) | base={_base_config_name()} | "
            f"Adapters: {injected} | Trainable: {trainable_count:,}"
        )
        root = get_project_root()
        frame_size = int(base_cfg["model"]["frame_size"])
        num_frames = int(base_cfg["model"]["num_frames"])
        train_dir = root / str(data_cfg.get("train_dir") or base_cfg["data"]["train_dir"])
        extra_dirs = [root / rel for rel in (data_cfg.get("extra_dirs") or base_cfg["data"].get("extra_dirs") or [])]
        dataset = VideoFrameDataset(
            train_dir,
            frame_size,
            num_frames,
            extra_dirs=extra_dirs,
            oversample_categories=train_cfg.get("oversample_categories"),
            max_files_per_dir=int(train_cfg.get("max_files_per_dir", 0) or 0),
        )
        loader_kwargs = get_dataloader_kwargs()
        loader_kwargs = dict(loader_kwargs)
        loader_kwargs["num_workers"] = 0
        loader_kwargs["persistent_workers"] = False
        loader_kwargs.pop("prefetch_factor", None)
        batch_size = int(train_cfg.get("batch_size") or 1)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        lr = float(train_cfg.get("learning_rate") or 0.0002)
        max_steps = int(steps or train_cfg.get("max_steps") or 1200)
        save_interval = int(train_cfg.get("save_interval") or 250)
        grad_clip = float(train_cfg.get("grad_clip") or 1.0)
        optimizer = torch.optim.AdamW(trainable, lr=lr)
        ema = EMATracker(model, decay=float(train_cfg.get("ema_decay", 0.999)))
        adapter_path = _adapter_path()
        print(f"Training sequences: {len(dataset)} | Steps: {max_steps} | Device: {device}")
        model.train()
        step = 0
        from tqdm import tqdm

        pbar = tqdm(total=max_steps, desc="LoRA (video)")
        while step < max_steps:
            for batch, captions in loader:
                batch = batch.to(device)
                cap_list = list(captions) if isinstance(captions, (list, tuple)) else [captions]
                with get_autocast_context(device):
                    loss_total = None
                    for i, caption in enumerate(cap_list):
                        single = batch[i : i + 1]
                        loss = model(single, caption)
                        loss_total = loss if loss_total is None else loss_total + loss
                    loss = loss_total / len(cap_list)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                ema.update(model)
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                if step % save_interval == 0 or step >= max_steps:
                    torch.save(
                        {"adapter": adapter_state(model), "rank": rank, "alpha": alpha},
                        adapter_path,
                    )
                if step >= max_steps:
                    break
        pbar.close()
        ema.apply_shadow(model)
        torch.save(
            {"adapter": adapter_state(model), "rank": rank, "alpha": alpha},
            adapter_path,
        )
        report = {
            "adapters": injected,
            "trainable_params": trainable_count,
            "steps": step,
            "sequences": len(dataset),
            "adapter_path": str(adapter_path),
            "base_config": _base_config_name(),
        }
        print(f"Video LoRA training complete. Adapter saved to {adapter_path}")
        return report


def adapter_exists() -> bool:
    return _adapter_path().exists()


def load_lora_model(device: torch.device) -> NavineVideoModel:
    if not adapter_exists():
        raise RuntimeError(
            "No video LoRA adapter found. Train first: python -m navine.cli train video-lora"
        )
    model = _build_lora_model(device, load_weights=True)
    model.eval()
    return model


def evaluate_with_lora(config: Optional[dict] = None, device=None) -> dict:
    from navine.device_manager import get_device
    from navine.utils.tier import load_modality_config
    from navine.video.smoke_eval import run_video_smoke_eval

    config = config or load_modality_config("video")
    device = device or get_device()
    model = load_lora_model(device)
    return run_video_smoke_eval(model=model, device=device, config=config, use_loaded_model=True)


def merge_lora_into_base() -> Dict[str, object]:
    from navine.device_manager import get_device
    from navine.image.ema import EMATracker

    if not adapter_exists():
        raise RuntimeError("No video LoRA adapter to merge.")
    device = get_device()
    base_cfg = _base_full_config()
    ckpt_dir = resolve_checkpoint_dir("video", base_cfg)
    ckpt_path = ckpt_dir / "latest.pt"
    backup_path = ckpt_dir / f"pre_lora_merge_{_adapter_path().stat().st_mtime_ns}.pt"
    if ckpt_path.exists():
        shutil.copy2(ckpt_path, backup_path)
    model = _build_lora_model(device, load_weights=True)
    merged_layers = merge_lora_into_module(model.frame_encoder)
    merged_layers += merge_lora_into_module(model.frame_decoder)
    model.eval()
    ema = EMATracker(model, decay=0.999)
    ema.apply_shadow(model)
    model.save_checkpoint(ckpt_path)
    return {
        "merged_layers": merged_layers,
        "checkpoint_path": str(ckpt_path),
        "backup_path": str(backup_path) if backup_path.exists() else None,
        "base_config": _base_config_name(),
    }


def remove_adapter() -> bool:
    path = _adapter_path()
    if not path.exists():
        return False
    archive = _lora_dir() / f"merged_{path.stat().st_mtime_ns}.pt"
    shutil.move(str(path), str(archive))
    print(f"Video LoRA adapter archived to {archive}")
    return True
