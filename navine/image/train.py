import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from navine.image.model import NavineDiffusionModel
from navine.image.utils import quarantine_file, validate_image
from navine.utils.config import load_config
from navine.utils.paths import get_project_root
from navine.utils.tier import resolve_checkpoint_dir

logger = logging.getLogger(__name__)

PHOTOREAL_TRAIN_SUFFIX = (
    "photorealistic adult photo, realistic body, natural skin texture, natural lighting, sharp focus"
)
PORN_TRAIN_SUFFIX = (
    "photorealistic explicit adult photo, detailed anatomy, pussy boobs ass, natural lighting, sharp focus"
)
SFW_PERSON_SUFFIX = "sfw clothed person, non-explicit, natural lighting, clean photo"
SFW_OBJECT_SUFFIX = "sfw object still life, everyday item, sharp focus"
SKY_SUFFIX = "blue sky clouds atmosphere, wide open sky, natural daylight"
LANDSCAPE_SUFFIX = "outdoor landscape scenery, mountains fields horizon, natural lighting"
ANIME_SUFFIX = "nude hentai anime girl, detailed face and body, sharp lineart, vibrant colors"


def _category_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    part_set = set(parts)
    if "skies" in part_set or "sky" in part_set:
        return "sky"
    if "landscapes" in part_set or "landscape" in part_set:
        return "landscape"
    if "sfw" in part_set and "objects" in part_set:
        return "object"
    if "sfw" in part_set and ("people" in part_set or "person" in part_set):
        return "sfw"
    if "objects" in part_set or "object" in part_set:
        return "object"
    if "people" in part_set:
        return "people"
    for cat in (
        "detective",
        "mystery",
        "think",
        "thinking",
        "cipher",
        "noir",
        "hentai",
        "porn",
        "fetish",
        "cosplay",
        "anime",
        "animated",
        "real",
        "human",
        "mixed",
        "boy",
        "girl",
        "sfw",
        "object",
        "people",
        "sky",
        "skies",
        "landscape",
        "landscapes",
    ):
        if cat in part_set:
            if cat == "anime":
                return "hentai"
            if cat in ("boy", "girl"):
                return "people"
            if cat == "skies":
                return "sky"
            if cat == "landscapes":
                return "landscape"
            return cat
    for part in parts:
        for cat in (
            "hentai",
            "porn",
            "anime",
            "animated",
            "sfw",
            "object",
            "people",
            "human",
            "real",
            "mixed",
            "sky",
            "skies",
            "landscape",
            "landscapes",
        ):
            if cat in part:
                if cat == "anime":
                    return "hentai"
                if cat == "skies":
                    return "sky"
                if cat == "landscapes":
                    return "landscape"
                return cat
    return "general"


def _caption_style_for_category(category: str, caption: str) -> str:
    lower = caption.lower()
    if category in ("hentai", "animated"):
        if "anime" not in lower and "hentai" not in lower and "illustration" not in lower:
            return f"{caption}, {ANIME_SUFFIX}"
        if "nude" not in lower:
            return f"{caption}, {ANIME_SUFFIX}"
        return caption
    if category == "sky":
        if "sky" not in lower and "cloud" not in lower and "moon" not in lower:
            return f"{caption}, {SKY_SUFFIX}"
        return caption
    if category == "landscape":
        if "landscape" not in lower and "mountain" not in lower and "horizon" not in lower:
            return f"{caption}, {LANDSCAPE_SUFFIX}"
        return caption
    if category == "object":
        if "object" not in lower and "still life" not in lower:
            return f"{caption}, {SFW_OBJECT_SUFFIX}"
        return caption
    if category in ("sfw", "people"):
        if "sfw" not in lower and "clothed" not in lower:
            return f"{caption}, {SFW_PERSON_SUFFIX}"
        return caption
    if category in ("porn", "mixed"):
        if "explicit" not in lower and "pussy" not in lower and "photoreal" not in lower:
            return f"{caption}, {PORN_TRAIN_SUFFIX}"
        return caption
    if category in ("real", "human"):
        if "photoreal" not in lower:
            return f"{caption}, photorealistic adult photo, natural lighting, detailed skin"
        return caption
    if category == "general":
        if "photoreal" not in lower and "sfw" not in lower and "object" not in lower and "sky" not in lower:
            return f"{caption}, {PHOTOREAL_TRAIN_SUFFIX}"
    return caption


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int,
        extra_dirs: Optional[List[Path]] = None,
        photoreal_training: bool = False,
        oversample_categories: Optional[dict] = None,
        max_files_per_dir: int = 0,
    ):
        raw_samples: List[Tuple[Path, str]] = []
        roots = [root]
        if extra_dirs:
            roots.extend(extra_dirs)
        seen: set = set()
        skipped = 0
        self.photoreal_training = photoreal_training
        self.oversample_categories = {str(k).lower(): max(1, int(v)) for k, v in (oversample_categories or {}).items()}
        per_dir_cap = max(0, int(max_files_per_dir or 0))
        for data_root in roots:
            if not data_root.exists():
                continue
            dir_count = 0
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
                if per_dir_cap and dir_count >= per_dir_cap:
                    break
                for path in data_root.rglob(f"*{ext}"):
                    if per_dir_cap and dir_count >= per_dir_cap:
                        break
                    if not path.is_file():
                        continue
                    key = str(path.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    if not validate_image(path):
                        skipped += 1
                        quarantine_file(path, reason="invalid training image")
                        continue
                    caption = self._load_caption(data_root, path)
                    raw_samples.append((path, caption))
                    dir_count += 1
                    if per_dir_cap and dir_count >= per_dir_cap:
                        break
        self.samples: List[Tuple[Path, str]] = []
        for path, caption in raw_samples:
            repeat = self.oversample_categories.get(_category_from_path(path), 1)
            for _ in range(repeat):
                self.samples.append((path, caption))
        if skipped:
            logger.warning("ImageFolderDataset skipped %d invalid image(s)", skipped)
            print(f"Warning: skipped {skipped} invalid image(s) during dataset scan")
        print(f"ImageFolderDataset unique={len(raw_samples)} effective={len(self.samples)}")
        if not self.samples:
            self.samples = [(p, "photorealistic sample image") for p in self._create_synthetic(root, image_size, count=16)]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _apply_photoreal_caption(self, path: Path, caption: str) -> str:
        if not self.photoreal_training:
            return caption
        category = _category_from_path(path)
        return _caption_style_for_category(category, caption)

    def _load_caption(self, root: Path, path: Path) -> str:
        sidecar = path.parent / f"{path.stem}.json"
        if not sidecar.exists():
            sidecar = root / f"{path.stem}.json"
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if data.get("sfw") is True or str(data.get("rating") or "").lower() in ("sfw", "safe"):
                    cap = data.get("caption") or path.stem.replace("_", " ")
                    cat = _category_from_path(path)
                    if cat in (
                        "sky",
                        "landscape",
                        "object",
                        "people",
                        "human",
                        "hentai",
                        "porn",
                        "real",
                        "animated",
                    ):
                        return _caption_style_for_category(cat, str(cap))
                    return _caption_style_for_category("sfw", str(cap))
                if data.get("is_nsfw") is True:
                    cap = data.get("caption") or path.stem.replace("_", " ")
                    cat = _category_from_path(path)
                    if cat == "general":
                        cat = "porn"
                    return _caption_style_for_category(cat, str(cap))
                cap = data.get("caption", "")
                if cap:
                    return self._apply_photoreal_caption(path, cap)
            except Exception:
                pass
        stem = path.stem.replace("_", " ").replace("-", " ")
        caption = stem if stem else "image content"
        return self._apply_photoreal_caption(path, caption)

    def _create_synthetic(self, root: Path, size: int, count: int):
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(count):
            img = Image.new("RGB", (size, size))
            pixels = img.load()
            for y in range(size):
                for x in range(size):
                    r = int(128 + 127 * (x / size))
                    g = int(128 + 127 * (y / size))
                    b = int(128 + 127 * ((x + y) / (2 * size)))
                    pixels[x, y] = (r, g, b)
            path = root / f"synthetic_{i:03d}.png"
            img.save(path)
            paths.append(path)
        return paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        attempts = len(self.samples)
        for offset in range(attempts):
            path, caption = self.samples[(idx + offset) % attempts]
            try:
                if not validate_image(path):
                    continue
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                return self.transform(rgb), caption
            except Exception:
                continue
        raise RuntimeError("No valid images available in dataset")


def train(
    config_path: str = "image",
    finetune: bool = False,
    finetune_steps: Optional[int] = None,
    require_cuda: bool = False,
) -> None:
    from navine.utils.training_lock import training_lock

    with training_lock("image"):
        _train_impl(config_path, finetune, finetune_steps, require_cuda=require_cuda)


def _train_impl(
    config_path: str = "image",
    finetune: bool = False,
    finetune_steps: Optional[int] = None,
    require_cuda: bool = False,
) -> None:
    config = load_config(config_path)
    from navine.device_manager import (
        get_autocast_context,
        get_device,
        get_dataloader_kwargs,
        print_train_device_banner,
        to_device,
    )

    compute = print_train_device_banner(require_cuda=require_cuda)
    device = get_device()
    loader_kwargs = get_dataloader_kwargs()
    loader_kwargs = dict(loader_kwargs)
    try:
        from navine.device_manager import get_device_settings, detect_hardware

        use_all = bool(get_device_settings().get("use_all_system_ram", False))
        ram = float((detect_hardware() or {}).get("system_ram_gb") or 0)
        if not (use_all and ram >= 14):
            loader_kwargs["num_workers"] = 0
            loader_kwargs.pop("persistent_workers", None)
            loader_kwargs.pop("prefetch_factor", None)
    except Exception:
        loader_kwargs["num_workers"] = 0
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)
    root = get_project_root()
    data_dir = root / config["data"]["train_dir"]
    extra_dirs = [root / rel for rel in config.get("data", {}).get("extra_dirs") or []]
    image_size = config["model"]["image_size"]
    train_cfg = config.get("training") or {}
    dataset = ImageFolderDataset(
        data_dir,
        image_size,
        extra_dirs=extra_dirs,
        photoreal_training=bool(train_cfg.get("photoreal_training", False)),
        oversample_categories=train_cfg.get("oversample_categories"),
        max_files_per_dir=int(train_cfg.get("max_files_per_dir", 0) or 0),
    )
    from navine.train.batch import apply_auto_batch

    batch_info = apply_auto_batch(train_cfg, config.get("model") or {}, modality="image")
    batch_size = batch_info["batch_size"]
    drop_last = len(dataset) >= batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **loader_kwargs,
    )
    model_cfg = {**config["model"], **config["diffusion"]}
    ckpt_dir = resolve_checkpoint_dir("image", config)
    ckpt_path = ckpt_dir / "latest.pt"
    if bool(train_cfg.get("fresh_start", False)) and ckpt_path.exists() and not finetune:
        backup = ckpt_dir / "latest_collapsed.pt"
        if backup.exists():
            backup.unlink()
        ckpt_path.rename(backup)
        print(f"Navine AI - Python: backed up previous checkpoint to {backup}")
    if (finetune or (ckpt_path.exists() and not bool(train_cfg.get("fresh_start", False)))) and ckpt_path.exists():
        model = NavineDiffusionModel.load_checkpoint(ckpt_path, config, device)
        model = model.to(device)
        if finetune:
            lr = config["training"].get("finetune_lr", config["training"]["learning_rate"] * 0.5)
            max_steps = finetune_steps or config["training"].get("finetune_steps", 150)
            desc = "Fine-tuning Navine AI - Python Image"
        else:
            lr = config["training"]["learning_rate"]
            max_steps = config["training"]["max_steps"]
            desc = "Resuming Navine AI - Python Image"
    else:
        model = NavineDiffusionModel(**model_cfg).to(device)
        lr = config["training"]["learning_rate"]
        max_steps = config["training"]["max_steps"]
        desc = "Training Navine AI - Python Image"
    print(f"Navine AI - Python Image Model | Parameters: {model.count_parameters():,} | Device: {device}")
    from navine.utils.param_verify import assert_real_param_count

    assert_real_param_count(
        model,
        "image",
        size_tier=str((config.get("model") or {}).get("size_tier") or config.get("size_tier") or "gpt3_1b"),
    )
    if bool(train_cfg.get("gradient_checkpointing", False)) and hasattr(model, "unet"):
        if hasattr(model.unet, "set_gradient_checkpointing"):
            model.unet.set_gradient_checkpointing(True)
            print("Navine AI - Python: gradient checkpointing enabled for image UNet")
    print(f"Training samples: {len(dataset)} | Steps: {max_steps}")
    if compute.get("dataloader"):
        print(f"Hybrid compute | CPU threads: {compute.get('cpu_threads')} | DataLoader: {compute.get('dataloader')}")
    opt_name = str(train_cfg.get("optimizer") or "sgd").strip().lower()
    optimizer = None
    if opt_name in {"adamw8bit", "adam8bit", "bnb"}:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
            print("Navine AI - Python: using bitsandbytes AdamW8bit optimizer")
        except Exception as exc:
            print(f"Navine AI - Python: AdamW8bit unavailable ({exc}); falling back")
    if optimizer is None and opt_name in {"sgd", "sgd_momentum"}:
        optimizer = torch.optim.SGD(model.parameters(), lr=max(float(lr), 1e-4), momentum=0.9, weight_decay=1e-4)
        print("Navine AI - Python: using SGD optimizer (VRAM-safe for 800M+ on 8GB)")
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        print("Navine AI - Python: using AdamW optimizer")
    use_amp = device.type == "cuda" and bool(train_cfg.get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp)
    if bool(train_cfg.get("torch_compile", False)) and device.type == "cuda":
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("Navine AI - Python: torch.compile enabled for image model")
        except Exception as exc:
            print(f"Navine AI - Python: torch.compile skipped ({exc})")
    from navine.image.ema import EMATracker

    if bool(train_cfg.get("refresh_noise_schedule", False)):
        diff = config.get("diffusion") or {}
        model.set_noise_schedule(
            int(diff.get("timesteps", model.timesteps)),
            float(diff.get("beta_start", model.beta_start)),
            float(diff.get("beta_end", model.beta_end)),
            str(diff.get("beta_schedule", model.beta_schedule)),
        )
        print(
            "Navine AI - Python: refreshed noise schedule "
            f"T={model.timesteps} abar_T={float(model.alphas_cumprod[-1]):.6g}"
        )
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    if bool(train_cfg.get("disable_ema", False)):
        ema_decay = 0.0
    ema = EMATracker(model, decay=ema_decay)
    if not ema.enabled:
        print("Navine AI - Python: EMA disabled for this run")
    sample_interval = int(train_cfg.get("sample_interval", 0))
    overfit_prompt = str(train_cfg.get("overfit_sample_prompt") or "").strip()
    sample_dir = get_project_root() / "logs" / "image_samples"
    step = 0
    pbar = tqdm(total=max_steps, desc=desc)
    model.train()
    while step < max_steps:
        for batch, captions in loader:
            batch = to_device(batch, device)
            cap_list = list(captions) if isinstance(captions, (list, tuple)) else [captions]
            with get_autocast_context(device):
                if hasattr(model.text_encoder, "encode_texts"):
                    text_ctx, _pooled = model.text_encoder.encode_texts(cap_list, device)
                    text_emb = text_ctx
                else:
                    embs = [model.text_encoder.encode(c, device) for c in cap_list]
                    if isinstance(embs[0], tuple):
                        text_emb = torch.cat([e[0] for e in embs], dim=0)
                    else:
                        text_emb = torch.cat(embs, dim=0)
                dropout = float(train_cfg.get("cfg_dropout", 0.0))
                if dropout > 0:
                    drop = torch.rand(text_emb.size(0), device=device) < dropout
                    if text_emb.dim() == 3:
                        text_emb = text_emb.clone()
                        text_emb[drop] = 0
                    else:
                        mask = drop.float().unsqueeze(1)
                        text_emb = text_emb * (1.0 - mask)
                loss = model(batch, text_emb)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                optimizer.step()
            ema.update(model)
            step += 1
            pbar.update(1)
            loss_val = float(loss.item())
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            if step == 1 or step % 5 == 0 or step >= max_steps:
                print(f"PROGRESS steps={step}/{max_steps}", flush=True)
                try:
                    from navine.api.train_progress_helper import report_train_progress

                    report_train_progress(
                        stage="image",
                        steps=step,
                        steps_total=max_steps,
                        last_line=f"PROGRESS steps={step}/{max_steps}",
                        force=(step == 1 or step >= max_steps),
                    )
                except Exception:
                    pass
            if step == 1 or step % 50 == 0 or step == max_steps:
                try:
                    loss_log = get_project_root() / "logs" / "image_enterprise_v2_loss.tsv"
                    loss_log.parent.mkdir(parents=True, exist_ok=True)
                    with loss_log.open("a", encoding="utf-8") as fh:
                        fh.write(f"{step}\t{loss_val:.6f}\n")
                except Exception:
                    pass
            if sample_interval and step % sample_interval == 0:
                sample_dir.mkdir(parents=True, exist_ok=True)
                ema.apply_shadow(model)
                try:
                    from torchvision.utils import save_image

                    prompts = {
                        "sky": "clear blue sky with soft white clouds, wide open atmosphere, natural daylight",
                        "human": "photoreal adult woman portrait, natural skin texture, natural lighting, sharp focus",
                        "hentai": "nude hentai anime girl, detailed face and body, clean lineart, vibrant colors",
                        "porn": "photorealistic explicit adult woman nude, detailed boobs ass pussy, natural light",
                    }
                    if overfit_prompt:
                        prompts = {"overfit": overfit_prompt}
                    ns = str(config.get("checkpoint_namespace") or "image")
                    tag_dir = sample_dir / ns
                    tag_dir.mkdir(parents=True, exist_ok=True)
                    infer_cfg = config.get("inference") or {}
                    sample_sched = str(infer_cfg.get("scheduler") or "ddpm").lower()
                    sample_guidance = float(infer_cfg.get("guidance_scale") or 1.5)
                    t_total = int(config.get("diffusion", {}).get("timesteps", 200))
                    sample_steps = t_total if sample_sched == "ddpm" else min(40, t_total)
                    for tag, prompt in prompts.items():
                        sample = model.sample(
                            batch_size=1,
                            text=prompt,
                            device=device,
                            num_steps=sample_steps,
                            guidance_scale=sample_guidance,
                            scheduler=sample_sched,
                        )
                        save_image((sample + 1) / 2, tag_dir / f"{tag}_step_{step:05d}.png")
                        save_image((sample + 1) / 2, sample_dir / f"{tag}_step_{step:05d}.png")
                except Exception:
                    pass
                ema.restore(model)
            if step % config["training"]["save_interval"] == 0 or step == max_steps:
                ema.apply_shadow(model)
                model.save_checkpoint(ckpt_dir / "latest.pt")
                ema.restore(model)
            if step >= max_steps:
                break
    pbar.close()
    ema.apply_shadow(model)
    model.save_checkpoint(ckpt_dir / "latest.pt")
    print(f"Training complete. Checkpoint saved to {ckpt_dir / 'latest.pt'}")
    try:
        from navine.image.smoke_eval import run_image_smoke_eval

        run_image_smoke_eval(model=model, device=device, config=config)
    except Exception as exc:
        print(f"Navine AI - Python: image smoke eval skipped ({exc})")


if __name__ == "__main__":
    train()
