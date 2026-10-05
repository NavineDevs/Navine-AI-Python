import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from navine.utils.config import load_config
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]

CONFIG_NAME = "lora_sd"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
INSTALL_HINT = (
    "Navine AI - Python LoRA training needs extra packages:\n"
    "  pip install -r requirements-external.txt\n"
    "  pip install peft"
)


def _log_dir() -> Path:
    path = get_project_root() / "logs" / "lora_sd"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(message: str, progress: ProgressCallback = None) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    with (_log_dir() / "lora_sd.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if progress:
        progress(line)


def dependencies_available() -> Tuple[bool, Optional[str]]:
    try:
        import diffusers  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        return True, None
    except ImportError as exc:
        return False, str(exc)


def load_lora_config() -> Dict[str, Any]:
    return load_config(CONFIG_NAME)


def _caption_for_image(image_path: Path, default_caption: str) -> str:
    sidecar_json = image_path.with_suffix(".json")
    if sidecar_json.exists():
        try:
            data = json.loads(sidecar_json.read_text(encoding="utf-8"))
            caption = data.get("caption") or data.get("prompt")
            if caption:
                return str(caption).strip()
        except Exception:
            pass
    sidecar_txt = image_path.with_suffix(".txt")
    if sidecar_txt.exists():
        try:
            text = sidecar_txt.read_text(encoding="utf-8").strip()
            if text:
                return text
        except Exception:
            pass
    stem = image_path.stem.replace("_", " ").strip()
    if stem and not stem.isdigit():
        return stem
    return default_caption


def collect_training_images(config: Dict[str, Any]) -> List[Tuple[Path, str]]:
    root = get_project_root()
    data_cfg = config.get("data") or {}
    default_caption = str(data_cfg.get("default_caption") or "high quality detailed photo")
    max_images = int(data_cfg.get("max_images") or 0)
    pairs: List[Tuple[Path, str]] = []
    seen = set()
    for rel in data_cfg.get("train_dirs") or []:
        folder = root / str(rel)
        if not folder.exists():
            continue
        for image_path in sorted(folder.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = image_path.resolve()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((image_path, _caption_for_image(image_path, default_caption)))
    random.shuffle(pairs)
    if max_images and len(pairs) > max_images:
        pairs = pairs[:max_images]
    return pairs


def _build_dataset(pairs, tokenizer, resolution):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    class LoraImageDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            image_path, caption = self.samples[idx]
            image = Image.open(image_path).convert("RGB")
            side = min(image.size)
            left = (image.size[0] - side) // 2
            top = (image.size[1] - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize(
                (resolution, resolution), Image.Resampling.LANCZOS
            )
            import numpy as np

            arr = np.asarray(image, dtype="float32") / 127.5 - 1.0
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1)
            tokens = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            return {
                "pixel_values": pixel_values,
                "input_ids": tokens.input_ids[0],
            }

    return LoraImageDataset(pairs)


def train_lora(
    config_path: str = CONFIG_NAME,
    max_steps: Optional[int] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    ok, detail = dependencies_available()
    if not ok:
        raise RuntimeError(f"{INSTALL_HINT}\nImport error: {detail}")

    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import CLIPTextModel, CLIPTokenizer

    from navine.device_manager import get_device
    from navine.utils.training_lock import training_lock

    config = load_config(config_path)
    base_cfg = config.get("base_model") or {}
    lora_cfg = config.get("lora") or {}
    train_cfg = config.get("training") or {}
    data_cfg = config.get("data") or {}
    out_cfg = config.get("output") or {}

    model_id = str(base_cfg.get("model_id") or "runwayml/stable-diffusion-v1-5")
    resolution = int(data_cfg.get("resolution") or 512)
    steps_target = int(max_steps or train_cfg.get("max_steps") or 1000)
    batch_size = int(train_cfg.get("batch_size") or 1)
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps") or 1))
    lr = float(train_cfg.get("learning_rate") or 1e-4)
    warmup = int(train_cfg.get("lr_warmup_steps") or 0)
    grad_clip = float(train_cfg.get("grad_clip") or 1.0)
    save_every = int(train_cfg.get("save_every") or 200)
    seed = int(train_cfg.get("seed") or 42)
    caption_dropout = float(data_cfg.get("caption_dropout") or 0.0)

    device = get_device()
    torch.manual_seed(seed)
    random.seed(seed)

    with training_lock("lora_sd"):
        pairs = collect_training_images(config)
        if not pairs:
            raise RuntimeError(
                "No training images found. Add images to data/learn/image or run: "
                "python -m navine.cli learn external --count 5"
            )
        _log(f"Collected {len(pairs)} training images from local data", progress)

        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        lora_config = LoraConfig(
            r=int(lora_cfg.get("rank") or 8),
            lora_alpha=int(lora_cfg.get("alpha") or 16),
            lora_dropout=float(lora_cfg.get("dropout") or 0.0),
            init_lora_weights="gaussian",
            target_modules=list(lora_cfg.get("target_modules") or ["to_q", "to_k", "to_v", "to_out.0"]),
        )
        unet = get_peft_model(unet, lora_config)
        unet.to(device)
        vae.to(device)
        text_encoder.to(device)

        if bool(train_cfg.get("gradient_checkpointing", True)):
            try:
                unet.enable_gradient_checkpointing()
            except Exception:
                pass

        trainable = [p for p in unet.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable)
        _log(f"LoRA trainable parameters: {trainable_count:,}", progress)

        dataset = _build_dataset(pairs, tokenizer, resolution)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(trainable, lr=lr)

        def lr_at(step_idx: int) -> float:
            if warmup and step_idx < warmup:
                return lr * (step_idx + 1) / max(warmup, 1)
            progress_ratio = (step_idx - warmup) / max(steps_target - warmup, 1)
            return lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress_ratio, 0.0), 1.0)))

        out_dir = get_project_root() / str(out_cfg.get("dir") or "checkpoints/lora_sd")
        out_dir.mkdir(parents=True, exist_ok=True)
        weight_name = str(out_cfg.get("weight_name") or "navine_lora.safetensors")

        unet.train()
        step = 0
        accum_loss = 0.0
        _log(f"Starting LoRA training on {model_id} for {steps_target} steps (device={device})", progress)
        while step < steps_target:
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
                input_ids = batch["input_ids"].to(device)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    if caption_dropout > 0.0 and random.random() < caption_dropout:
                        input_ids = torch.zeros_like(input_ids)
                    encoder_hidden_states = text_encoder(input_ids)[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / grad_accum
                loss.backward()
                accum_loss += float(loss.item())
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_at(step)
                    optimizer.step()
                    optimizer.zero_grad()
                step += 1
                if step % 10 == 0:
                    _log(f"step {step}/{steps_target} loss={accum_loss:.4f}", progress)
                accum_loss = 0.0
                if step % save_every == 0 or step >= steps_target:
                    unet.save_pretrained(str(out_dir))
                    _log(f"Saved LoRA adapter to {out_dir} at step {step}", progress)
                if step >= steps_target:
                    break

        unet.save_pretrained(str(out_dir))
        report = {
            "base_model": model_id,
            "steps": step,
            "images": len(pairs),
            "trainable_params": trainable_count,
            "output_dir": str(out_dir),
            "weight_name": weight_name,
            "finished": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "lora_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _log(f"LoRA training complete: {report}", progress)
        return report


def lora_status() -> Dict[str, Any]:
    ok, detail = dependencies_available()
    config = load_lora_config()
    out_dir = get_project_root() / str((config.get("output") or {}).get("dir") or "checkpoints/lora_sd")
    trained = (out_dir / "adapter_config.json").exists() or (out_dir / "lora_report.json").exists()
    pairs = collect_training_images(config)
    return {
        "dependencies_installed": ok,
        "dependency_error": detail,
        "base_model": (config.get("base_model") or {}).get("model_id"),
        "training_images_available": len(pairs),
        "adapter_dir": str(out_dir),
        "adapter_trained": trained,
    }


if __name__ == "__main__":
    train_lora()
