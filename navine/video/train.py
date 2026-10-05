import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from navine.image.train import (
    _caption_style_for_category,
    _category_from_path,
)
from navine.video.model import NavineVideoModel
from navine.utils.config import load_config
from navine.utils.paths import get_project_root
from navine.utils.tier import resolve_checkpoint_dir

VIDEO_PROMPTS = {
    "sky": "clear blue sky with soft white clouds, gentle cloud drift",
    "human": "photoreal adult woman portrait, natural lighting, subtle motion",
    "hentai": "nude hentai anime girl detailed face body, gentle motion",
    "porn": "photorealistic explicit adult anatomy, natural light, subtle camera drift",
    "think": "abstract mind map forming from particles, slow camera drift, glowing connections",
    "detective": "noir alley at night, fog, silhouette with flashlight, slow camera pan, mysterious atmosphere",
}


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root: Path,
        frame_size: int,
        num_frames: int,
        extra_dirs: Optional[List[Path]] = None,
        oversample_categories: Optional[dict] = None,
        max_files_per_dir: int = 0,
    ):
        raw_sequences: List[Tuple[List[Path], str]] = []
        roots = [root]
        if extra_dirs:
            roots.extend(extra_dirs)
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.oversample_categories = {
            str(k).lower(): max(1, int(v)) for k, v in (oversample_categories or {}).items()
        }
        per_dir_cap = max(0, int(max_files_per_dir or 0))
        seen: set = set()
        for data_root in roots:
            if not data_root.exists():
                continue
            self._load_root(data_root, raw_sequences, seen, per_dir_cap)
        self.sequences: List[Tuple[List[Path], str]] = []
        for paths, caption in raw_sequences:
            cat_path = paths[0] if paths else data_root
            repeat = self.oversample_categories.get(_category_from_path(cat_path), 1)
            for _ in range(repeat):
                self.sequences.append((paths, caption))
        print(f"VideoFrameDataset unique={len(raw_sequences)} effective={len(self.sequences)}")
        if not self.sequences:
            fallback = root if root.exists() else get_project_root() / "data" / "video" / "samples"
            self._load_root(fallback, raw_sequences, set(), 0, allow_synthetic=True)
            for paths, caption in raw_sequences:
                self.sequences.append((paths, caption))
        self.transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _load_root(
        self,
        root: Path,
        out: List[Tuple[List[Path], str]],
        seen: set,
        per_dir_cap: int,
        allow_synthetic: bool = False,
    ) -> None:
        dir_count = 0
        seq_dirs = [d for d in root.iterdir() if d.is_dir()] if root.exists() else []
        if seq_dirs:
            for seq_dir in seq_dirs:
                if per_dir_cap and dir_count >= per_dir_cap:
                    break
                images = sorted(seq_dir.glob("frame_*.png"))
                if not images:
                    images = sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg"))
                if len(images) < self.num_frames:
                    continue
                caption = self._load_caption(seq_dir)
                for i in range(max(1, len(images) - self.num_frames + 1)):
                    seq_paths = images[i : i + self.num_frames]
                    if len(seq_paths) != self.num_frames:
                        continue
                    key = "|".join(str(p.resolve()) for p in seq_paths)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((seq_paths, caption))
                    dir_count += 1
                    if per_dir_cap and dir_count >= per_dir_cap:
                        break
        images = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg")) + sorted(root.glob("*.jpeg"))
        if not images:
            images = (
                sorted(root.rglob("*.png"))
                + sorted(root.rglob("*.jpg"))
                + sorted(root.rglob("*.jpeg"))
                + sorted(root.rglob("*.webp"))
            )
        if images and len(images) >= self.num_frames:
            caption = self._load_caption(root)
            for i in range(max(1, len(images) - self.num_frames + 1)):
                seq_paths = images[i : i + self.num_frames]
                if len(seq_paths) != self.num_frames:
                    continue
                key = "|".join(str(p.resolve()) for p in seq_paths)
                if key in seen:
                    continue
                seen.add(key)
                out.append((seq_paths, caption))
                dir_count += 1
                if per_dir_cap and dir_count >= per_dir_cap:
                    break
        if allow_synthetic and not out:
            synth = self._create_synthetic(root, self.frame_size, self.num_frames)
            for i in range(max(1, len(synth) - self.num_frames + 1)):
                seq_paths = synth[i : i + self.num_frames]
                if len(seq_paths) == self.num_frames:
                    out.append((seq_paths, "motion sequence"))

    def _load_caption(self, seq_dir: Path) -> str:
        sidecars = [seq_dir / "meta.json"]
        for frame in sorted(seq_dir.glob("frame_*.png"))[:1]:
            sidecars.append(frame.with_suffix(".json"))
        for sidecar in sidecars:
            if not sidecar.exists():
                continue
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if data.get("sfw") is True or str(data.get("rating") or "").lower() in ("sfw", "safe"):
                    cap = data.get("caption") or "sfw motion clip"
                    return _caption_style_for_category("sfw", str(cap))
                if data.get("is_nsfw") is True:
                    cap = data.get("caption") or "motion clip"
                    cat = _category_from_path(seq_dir)
                    if cat == "general":
                        cat = "porn"
                    return _caption_style_for_category(cat, str(cap))
                cap = data.get("caption")
                if cap:
                    return _caption_style_for_category(_category_from_path(seq_dir), str(cap))
                url = data.get("url", "")
                if url:
                    from urllib.parse import urlparse

                    stem = Path(urlparse(url).path).stem.replace("_", " ").replace("-", " ")
                    if stem:
                        return _caption_style_for_category(_category_from_path(seq_dir), stem)
            except Exception:
                pass
        stem = seq_dir.name.replace("_", " ").replace("-", " ")
        category = _category_from_path(seq_dir)
        cap = stem if stem else "motion sequence"
        return _caption_style_for_category(category, cap)

    def _create_synthetic(self, root: Path, size: int, num_frames: int):
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in range(num_frames * 4):
            img = Image.new("RGB", (size, size))
            pixels = img.load()
            offset = f * 4
            for y in range(size):
                for x in range(size):
                    r = int(128 + 127 * ((x + offset) % size) / size)
                    g = int(128 + 127 * ((y + offset) % size) / size)
                    b = int(128 + 127 * (((x + y + offset) % size)) / size)
                    pixels[x, y] = (r, g, b)
            path = root / f"frame_{f:03d}.png"
            img.save(path)
            paths.append(path)
        return paths

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        paths, caption = self.sequences[idx % len(self.sequences)]
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames), caption


def train(
    config_path: str = "video",
    finetune: bool = False,
    finetune_steps: Optional[int] = None,
    require_cuda: bool = False,
) -> None:
    from navine.utils.training_lock import training_lock

    with training_lock("video"):
        _train_impl(config_path, finetune, finetune_steps, require_cuda=require_cuda)


def _train_impl(
    config_path: str = "video",
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

    if finetune_steps is not None and not finetune:
        finetune = True
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
        else:
            loader_kwargs["num_workers"] = min(int(loader_kwargs.get("num_workers") or 0), 2)
    except Exception:
        loader_kwargs["num_workers"] = 0
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)
    root = get_project_root()
    data_cfg = config["data"]
    train_dir = root / data_cfg["train_dir"]
    extra_dirs = [root / rel for rel in data_cfg.get("extra_dirs") or []]
    frame_size = int(config["model"]["frame_size"])
    num_frames = int(config["model"]["num_frames"])
    train_cfg = config.get("training") or {}
    ckpt_dir = resolve_checkpoint_dir("video", config)
    ckpt_path = ckpt_dir / "latest.pt"
    target_cfg = {k: config["model"][k] for k in ("frame_size", "num_frames", "in_channels", "hidden_dim", "num_layers")}
    target_params = NavineVideoModel(**target_cfg).count_parameters()

    def _load_video_model() -> NavineVideoModel:
        if ckpt_path.exists() and ckpt_path.suffix.lower() == ".pt" and not bool(train_cfg.get("fresh_start", False)):
            loaded = NavineVideoModel.load_checkpoint(ckpt_path, config, device)
            if int(loaded.count_parameters()) == int(target_params):
                return loaded.to(device)
            print(
                f"Navine AI - Python: video checkpoint has {loaded.count_parameters():,} params; "
                f"config expects {target_params:,}. Training a fresh model."
            )
        return NavineVideoModel(**target_cfg).to(device)

    if finetune and ckpt_path.exists() and not bool(train_cfg.get("fresh_start", False)):
        model = _load_video_model()
        frame_size = int(config["model"]["frame_size"])
        num_frames = int(config["model"]["num_frames"])
        lr = config["training"].get("finetune_lr", config["training"]["learning_rate"] * 0.5)
        max_steps = finetune_steps or config["training"].get("finetune_steps", 150)
        desc = "Fine-tuning Navine AI - Python Video"
    elif ckpt_path.exists() and not bool(train_cfg.get("fresh_start", False)):
        model = _load_video_model()
        frame_size = int(config["model"]["frame_size"])
        num_frames = int(config["model"]["num_frames"])
        lr = config["training"]["learning_rate"]
        max_steps = config["training"]["max_steps"]
        desc = "Resuming Navine AI - Python Video"
    else:
        model = NavineVideoModel(**target_cfg).to(device)
        frame_size = int(config["model"]["frame_size"])
        num_frames = int(config["model"]["num_frames"])
        lr = config["training"]["learning_rate"]
        max_steps = config["training"]["max_steps"]
        desc = "Training Navine AI - Python Video"
    dataset = VideoFrameDataset(
        train_dir,
        frame_size,
        num_frames,
        extra_dirs=extra_dirs,
        oversample_categories=train_cfg.get("oversample_categories"),
        max_files_per_dir=int(train_cfg.get("max_files_per_dir", 0) or 0),
    )
    train_cfg = config.get("training") or {}
    from navine.train.batch import apply_auto_batch

    batch_info = apply_auto_batch(train_cfg, config.get("model") or {}, modality="video")
    batch_size = batch_info["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    print(f"Navine AI - Python Video Model | Parameters: {model.count_parameters():,} | Device: {device}")
    from navine.utils.param_verify import assert_real_param_count

    assert_real_param_count(
        model,
        "video",
        size_tier=str((config.get("model") or {}).get("size_tier") or config.get("size_tier") or "gpt3_1b"),
    )
    print(f"Training sequences: {len(dataset)} | frame_size={frame_size} | Steps: {max_steps}")
    if compute.get("dataloader"):
        print(f"Hybrid compute | CPU threads: {compute.get('cpu_threads')} | DataLoader: {compute.get('dataloader')}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device.type == "cuda" and bool(train_cfg.get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp)
    perceptual_weight = float(config["training"].get("perceptual_loss_weight", 0.0))
    vgg = None
    if perceptual_weight > 0:
        try:
            from torchvision.models import vgg16

            vgg = vgg16(weights="DEFAULT").features[:16].to(device).eval()
            for param in vgg.parameters():
                param.requires_grad = False
        except Exception:
            perceptual_weight = 0.0
            vgg = None
    from navine.image.ema import EMATracker

    ema = EMATracker(model, decay=float(train_cfg.get("ema_decay", 0.999)))
    sample_interval = int(train_cfg.get("sample_interval", 0))
    sample_dir = get_project_root() / "logs" / "video_samples"
    step = 0
    pbar = tqdm(total=max_steps, desc=desc)
    model.train()
    while step < max_steps:
        for batch, captions in loader:
            batch = to_device(batch, device)
            cap_list = list(captions) if isinstance(captions, (list, tuple)) else [captions]
            with get_autocast_context(device):
                loss_total = None
                for i, caption in enumerate(cap_list):
                    single = batch[i : i + 1]
                    loss = model(single, caption)
                    loss_total = loss if loss_total is None else loss_total + loss
                loss = loss_total / len(cap_list)
                if perceptual_weight > 0 and vgg is not None:
                    try:
                        pred = batch[:, -1]
                        target = batch[:, 0]
                        pf = vgg(pred)
                        tf = vgg(target)
                        loss = loss + perceptual_weight * torch.nn.functional.l1_loss(pf, tf)
                    except Exception:
                        pass
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
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step == 1 or step % 5 == 0 or step >= max_steps:
                print(f"PROGRESS steps={step}/{max_steps}", flush=True)
                try:
                    from navine.api.train_progress_helper import report_train_progress

                    report_train_progress(
                        stage="video",
                        steps=step,
                        steps_total=max_steps,
                        last_line=f"PROGRESS steps={step}/{max_steps}",
                        force=(step == 1 or step >= max_steps),
                    )
                except Exception:
                    pass
            if sample_interval and step % sample_interval == 0:
                sample_dir.mkdir(parents=True, exist_ok=True)
                ema.apply_shadow(model)
                try:
                    from navine.video.infer import generate as generate_video

                    for tag, prompt in VIDEO_PROMPTS.items():
                        out = sample_dir / f"{tag}_step_{step:05d}.mp4"
                        generate_video(
                            prompt,
                            output_path=str(out),
                            config_name=config_path,
                            seed=42 + step,
                            auto_ingest=False,
                        )
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
    ema.restore(model)
    print(f"Training complete. Checkpoint saved to {ckpt_dir / 'latest.pt'}")
    try:
        from navine.video.smoke_eval import run_video_smoke_eval

        run_video_smoke_eval(model=model, device=device, config=config)
    except Exception as exc:
        print(f"Navine AI - Python: video smoke eval skipped ({exc})")


if __name__ == "__main__":
    import sys

    config_name = sys.argv[1] if len(sys.argv) > 1 else "video"
    train(config_name)
