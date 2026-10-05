import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from navine.image.external import (
    diffusers_available,
    external_learn_dir,
    generate_external,
    list_teacher_model_profiles,
    load_external_config,
    resolve_teacher_model,
    save_teaching_pair,
)
from navine.utils.config import load_config
from navine.utils.paths import get_project_root
from navine.video.external import (
    external_video_learn_dir,
    generate_external as generate_external_video,
    save_teaching_video,
)

DEFAULT_PROMPTS = [
    "photorealistic portrait of a woman, studio lighting, sharp focus, detailed skin texture",
    "photorealistic full body amateur photo, natural window light, realistic anatomy",
    "photorealistic candid portrait, soft daylight, shallow depth of field, natural skin",
    "professional headshot photograph, neutral background, high detail pores and skin",
    "photorealistic beach portrait, golden hour sunlight, natural skin tones, sharp focus",
    "anime hentai style explicit illustration, detailed character art, soft cel shading",
    "hentai ecchi anime girl, vibrant colors, sharp linework, detailed anatomy, illustration",
    "anime portrait illustration, blue hair, detailed eyes, soft shading, vibrant palette",
    "explicit hentai manga style full body character, vivid colors, clean linework",
    "anime girl summer beach scene, detailed illustration, bright aesthetic, sharp lines",
    "cinematic landscape mountain lake at sunset, photorealistic, dramatic clouds",
    "professional product photo of a coffee mug on a wooden table, studio lighting",
]


def _image_ingest_dir() -> Path:
    path = get_project_root() / "data" / "learn" / "image"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path() -> Path:
    path = external_learn_dir() / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_manifest() -> List[dict]:
    path = _manifest_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(entries: List[dict]) -> None:
    _manifest_path().write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _next_teacher_filename() -> str:
    learn_dir = external_learn_dir()
    max_idx = 0
    for path in learn_dir.glob("teacher_*.png"):
        suffix = path.stem[8:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return f"teacher_{max_idx + 1:03d}.png"


def generate_teaching_image(
    prompt: str,
    seed: Optional[int] = None,
    save: bool = True,
    model: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    learn_dir = external_learn_dir()
    out_path = learn_dir / _next_teacher_filename()
    image_path = generate_external(
        prompt,
        seed=seed,
        output_path=str(out_path),
        model_profile=model,
        cycle_index=cycle_index,
    )
    if save:
        meta_path = image_path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved = save_teaching_pair(prompt, image_path, caption=prompt.strip(), metadata=meta)
        manifest = _load_manifest()
        entry = {
            "id": hashlib.sha256(saved.read_bytes()).hexdigest()[:16],
            "prompt": prompt.strip(),
            "path": str(saved),
            "kind": "image",
            "model": meta.get("model_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if not any(row.get("path") == entry["path"] for row in manifest):
            manifest.append(entry)
            _save_manifest(manifest)
        return saved
    return image_path


def generate_teaching_video(
    prompt: str,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    fps: Optional[int] = None,
    save: bool = True,
    model: Optional[str] = None,
    cycle_index: int = 0,
) -> Path:
    video_path = generate_external_video(
        prompt,
        seed=seed,
        num_frames=num_frames,
        fps=fps,
        model_profile=model,
        cycle_index=cycle_index,
    )
    if save:
        meta_path = video_path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved = save_teaching_video(prompt, video_path, metadata=meta)
        manifest = _load_manifest()
        entry = {
            "id": hashlib.sha256(saved.read_bytes()).hexdigest()[:16],
            "prompt": prompt.strip(),
            "path": str(saved),
            "kind": "video",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if not any(row.get("path") == entry["path"] for row in manifest):
            manifest.append(entry)
            _save_manifest(manifest)
        return saved
    return video_path


def batch_generate(
    prompts: Optional[List[str]] = None,
    count: Optional[int] = None,
    include_video: bool = False,
    model: Optional[str] = None,
    cycle_index: int = 0,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    ok, detail = diffusers_available()
    if not ok:
        raise RuntimeError(
            "Navine AI - Python external diffusion is not available. "
            "pip install diffusers transformers accelerate safetensors"
            + (f"\n{detail}" if detail else "")
        )
    items = list(prompts or DEFAULT_PROMPTS)
    if count is not None and count > 0:
        expanded: List[str] = []
        idx = 0
        while len(expanded) < count:
            expanded.append(items[idx % len(items)])
            idx += 1
        items = expanded
    images_saved = 0
    videos_saved = 0
    for index, prompt in enumerate(items):
        item_cycle = cycle_index + index if model is None else cycle_index
        profile_key, _profile = resolve_teacher_model(model, item_cycle)
        if progress:
            progress(
                f"External teacher image {index + 1}/{len(items)} "
                f"[{profile_key}]: {prompt[:60]}"
            )
        generate_teaching_image(prompt, model=model, cycle_index=item_cycle)
        images_saved += 1
        if include_video:
            if progress:
                progress(f"External teacher video {index + 1}/{len(items)}")
            generate_teaching_video(prompt, num_frames=16, fps=12, model=model, cycle_index=item_cycle)
            videos_saved += 1
    return {"images": images_saved, "videos": videos_saved, "prompts": len(items)}


def ingest_external_for_training(image_size: Optional[int] = None) -> str:
    from PIL import Image

    if image_size is None:
        try:
            image_size = int(load_config("image_enterprise")["model"]["image_size"])
        except Exception:
            image_size = 256
    source = external_learn_dir()
    out_dir = _image_ingest_dir()
    count = 0
    for image_path in sorted(source.glob("*.png")):
        sidecar = source / f"{image_path.stem}.json"
        caption = image_path.stem.replace("_", " ")
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                caption = str(data.get("caption") or data.get("prompt") or caption)
            except Exception:
                pass
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
            out_name = f"ext_{digest}.png"
            out_path = out_dir / out_name
            img.save(out_path, format="PNG")
            meta = {
                "id": digest,
                "caption": caption,
                "prompt": caption,
                "source": "external_diffusion",
                "teacher_path": str(image_path),
            }
            (out_dir / f"ext_{digest}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            samples = get_project_root() / "data" / "image" / "samples"
            samples.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_path, samples / out_name)
            shutil.copy2(out_dir / f"ext_{digest}.json", samples / f"ext_{digest}.json")
            count += 1
        except Exception:
            continue
    video_source = external_video_learn_dir()
    video_out = get_project_root() / "data" / "learn" / "video" / "external"
    video_out.mkdir(parents=True, exist_ok=True)
    for video_path in sorted(video_source.glob("*.mp4")):
        dest = video_out / video_path.name
        if not dest.exists():
            shutil.copy2(video_path, dest)
        sidecar = video_source / f"{video_path.stem}.json"
        if sidecar.exists():
            dest_sidecar = video_out / sidecar.name
            if not dest_sidecar.exists():
                shutil.copy2(sidecar, dest_sidecar)
    return str(out_dir) if count else str(source)


def finetune_from_external(
    finetune_steps: Optional[int] = None,
    config_path: str = "image_enterprise",
) -> None:
    ingest_external_for_training()
    from navine.image.train import train

    config = load_config(config_path)
    steps = finetune_steps
    if steps is None:
        steps = int((config.get("training") or {}).get("finetune_steps", 400))
    train(config_path=config_path, finetune=True, finetune_steps=steps)


def external_status() -> Dict[str, object]:
    ok, detail = diffusers_available()
    image_dir = external_learn_dir()
    video_dir = external_video_learn_dir()
    image_cfg = load_external_config()
    diff_cfg = (image_cfg.get("inference") or {}).get("diffusers") or {}
    profiles = list_teacher_model_profiles()
    return {
        "diffusers_installed": ok,
        "import_detail": detail,
        "image_config": "image_external",
        "video_config": "video_external",
        "model_id": diff_cfg.get("model_id"),
        "teacher_profiles": list(profiles.keys()),
        "default_profile": (image_cfg.get("teacher_models") or {}).get("default"),
        "rotation": (image_cfg.get("teacher_models") or {}).get("rotation"),
        "image_teacher_dir": str(image_dir),
        "video_teacher_dir": str(video_dir),
        "image_pairs": len(list(image_dir.glob("*.png"))),
        "video_pairs": len(list(video_dir.glob("*.mp4"))),
        "train_ingest_dir": str(_image_ingest_dir()),
    }
