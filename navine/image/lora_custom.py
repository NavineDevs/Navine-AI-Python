from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from torchvision.utils import save_image

from navine.image.model import NavineDiffusionModel
from navine.utils.config import load_config
from navine.utils.lora_core import (
    LoRAConv2d,
    LoRALinear,
    adapter_state,
    inject_lora,
    load_adapter_state,
    merge_lora_into_module,
)
from navine.utils.paths import get_project_root
from navine.utils.tier import resolve_checkpoint_dir

LORA_CONFIG_NAME = "image_lora"


def _lora_config() -> dict:
    return load_config(LORA_CONFIG_NAME)


def _base_config_name() -> str:
    return str(_lora_config().get("base_config") or "image_enterprise_v2")


def _base_full_config() -> dict:
    return load_config(_base_config_name())


def _lora_dir() -> Path:
    cfg = _lora_config()
    ns = str(cfg.get("checkpoint_namespace") or "image_lora")
    path = get_project_root() / "checkpoints" / ns
    path.mkdir(parents=True, exist_ok=True)
    return path


def _adapter_path() -> Path:
    return _lora_dir() / "adapter.pt"


def _load_base_model(device: torch.device) -> NavineDiffusionModel:
    base_cfg = _base_full_config()
    ckpt_dir = resolve_checkpoint_dir("image", base_cfg)
    ckpt_path = ckpt_dir / "latest.pt"
    if not ckpt_path.exists():
        raise RuntimeError(
            f"No base image checkpoint at {ckpt_path}. Train the base model first: "
            f"python -m navine.cli train image --config {_base_config_name()}"
        )
    return NavineDiffusionModel.load_checkpoint(ckpt_path, base_cfg, device).to(device)


def _build_lora_model(device: torch.device, load_weights: bool = True) -> NavineDiffusionModel:
    cfg = _lora_config()
    lora_cfg = cfg.get("lora") or {}
    rank = int(lora_cfg.get("rank") or 8)
    alpha = float(lora_cfg.get("alpha") or 16)
    model = _load_base_model(device)
    for param in model.parameters():
        param.requires_grad_(False)
    inject_lora(model.unet, rank, alpha)
    model = model.to(device)
    adapter_path = _adapter_path()
    if load_weights and adapter_path.exists():
        payload = torch.load(adapter_path, map_location=device, weights_only=False)
        load_adapter_state(model, payload.get("adapter") or {})
    return model


def train_custom_lora(
    steps: Optional[int] = None,
    progress=None,
) -> Dict[str, object]:
    from torch.utils.data import DataLoader

    from navine.device_manager import get_autocast_context, get_device
    from navine.image.ema import EMATracker
    from navine.image.train import ImageFolderDataset
    from navine.utils.training_lock import training_lock

    with training_lock("image_lora"):
        config = _lora_config()
        lora_cfg = config.get("lora") or {}
        train_cfg = config.get("training") or {}
        data_cfg = config.get("data") or {}
        rank = int(lora_cfg.get("rank") or 8)
        alpha = float(lora_cfg.get("alpha") or 16)
        device = get_device()
        base_cfg = _base_full_config()
        model = _load_base_model(device)
        for param in model.parameters():
            param.requires_grad_(False)
        injected = inject_lora(model.unet, rank, alpha)
        model = model.to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable)
        print(
            f"Navine AI - Python LoRA (image) | base={_base_config_name()} | "
            f"Adapters: {injected} | Trainable: {trainable_count:,}"
        )
        root = get_project_root()
        image_size = int(base_cfg["model"]["image_size"])
        data_dir = root / str(data_cfg.get("train_dir") or base_cfg["data"]["train_dir"])
        extra_dirs = [root / rel for rel in (data_cfg.get("extra_dirs") or base_cfg["data"].get("extra_dirs") or [])]
        dataset = ImageFolderDataset(data_dir, image_size, extra_dirs=extra_dirs)
        batch_size = int(train_cfg.get("batch_size") or 4)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        lr = float(train_cfg.get("learning_rate") or 0.0003)
        max_steps = int(steps or train_cfg.get("max_steps") or 1500)
        save_interval = int(train_cfg.get("save_interval") or 300)
        grad_clip = float(train_cfg.get("grad_clip") or 1.0)
        optimizer = torch.optim.AdamW(trainable, lr=lr)
        ema = EMATracker(model, decay=float(train_cfg.get("ema_decay", 0.999)))
        adapter_path = _adapter_path()
        print(f"Training samples: {len(dataset)} | Steps: {max_steps} | Device: {device}")
        model.train()
        step = 0
        from tqdm import tqdm

        pbar = tqdm(total=max_steps, desc="LoRA (image)")
        while step < max_steps:
            for batch, captions in loader:
                batch = batch.to(device)
                cap_list = list(captions) if isinstance(captions, (list, tuple)) else [captions]
                with get_autocast_context(device):
                    embs = [model.text_encoder.encode(c, device) for c in cap_list]
                    text_emb = torch.cat(embs, dim=0)
                    loss = model(batch, text_emb)
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
            "images": len(dataset),
            "adapter_path": str(adapter_path),
            "base_config": _base_config_name(),
        }
        print(f"LoRA training complete. Adapter saved to {adapter_path}")
        return report


def adapter_exists() -> bool:
    return _adapter_path().exists()


def load_lora_model(device: torch.device) -> NavineDiffusionModel:
    if not adapter_exists():
        raise RuntimeError(
            "No image LoRA adapter found. Train first: python -m navine.cli train image-lora"
        )
    model = _build_lora_model(device, load_weights=True)
    model.eval()
    return model


def evaluate_with_lora(config: Optional[dict] = None, device=None) -> dict:
    from navine.device_manager import get_device
    from navine.image.smoke_eval import run_image_smoke_eval
    from navine.utils.tier import load_modality_config

    config = config or load_modality_config("image")
    device = device or get_device()
    model = load_lora_model(device)
    return run_image_smoke_eval(model=model, device=device, config=config)


def merge_lora_into_base() -> Dict[str, object]:
    from navine.device_manager import get_device
    from navine.image.ema import EMATracker

    if not adapter_exists():
        raise RuntimeError("No image LoRA adapter to merge.")
    device = get_device()
    base_cfg = _base_full_config()
    ckpt_dir = resolve_checkpoint_dir("image", base_cfg)
    ckpt_path = ckpt_dir / "latest.pt"
    backup_path = ckpt_dir / f"pre_lora_merge_{_adapter_path().stat().st_mtime_ns}.pt"
    if ckpt_path.exists():
        shutil.copy2(ckpt_path, backup_path)
    model = _build_lora_model(device, load_weights=True)
    merged_layers = merge_lora_into_module(model.unet)
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
    print(f"LoRA adapter archived to {archive}")
    return True


def generate_custom_lora(
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    num_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> Path:
    from PIL import Image

    from navine.device_manager import get_device
    from navine.image.procedural import prompt_seed
    from navine.utils.paths import get_output_dir

    base_cfg = _base_full_config()
    infer_cfg = base_cfg.get("inference") or {}
    device = get_device()
    if seed is None:
        seed = prompt_seed(prompt)
    torch.manual_seed(int(seed))
    model = load_lora_model(device)
    steps = num_steps if num_steps is not None else int(infer_cfg.get("num_steps", 100))
    scale = guidance_scale if guidance_scale is not None else float(infer_cfg.get("guidance_scale", 2.0))
    samples = model.sample(
        batch_size=1,
        text=prompt,
        device=device,
        num_steps=steps,
        guidance_scale=scale,
    )
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = get_output_dir("image") / "generated_lora.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image((samples + 1) / 2, out_path)
    size = int(infer_cfg.get("output_size", 512))
    if size != model.image_size:
        img = Image.open(out_path).convert("RGB")
        if size > model.image_size:
            try:
                from navine.image.postprocess import postprocess_image

                target = max(model.image_size, min(size, model.image_size * 4))
                if target != model.image_size:
                    img = img.resize((target, target), Image.Resampling.LANCZOS)
                img = postprocess_image(img, upscale=False, sharpen=True)
                if target != size:
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
            except Exception:
                img = img.resize((size, size), Image.Resampling.LANCZOS)
        else:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        img.save(out_path, format="PNG")
    return out_path
