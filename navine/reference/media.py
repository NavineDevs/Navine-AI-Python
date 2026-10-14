import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

from navine.reference.config import (
    get_reference_settings,
    list_reference_entries,
    reference_strength_for_category,
)
from navine.utils.paths import get_project_root

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".gif"}


def _keyword_in_prompt(keyword: str, prompt: str) -> bool:
    kw = keyword.strip().lower()
    if not kw:
        return False
    lower = prompt.lower()
    if " " in kw:
        return kw in lower
    return bool(re.search(rf"\b{re.escape(kw)}\b", lower))


def list_reference_keywords() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for category, entry in list_reference_entries().items():
        mapping[category] = list(entry.get("keywords") or [])
    return mapping


def match_reference_keyword(prompt: str) -> Optional[Tuple[str, Path, str]]:
    text = prompt.strip()
    if not text:
        return None
    lower = text.lower()
    hentai_intent = any(
        term in lower
        for term in ("hentai", "anime", "waifu", "doujin", "manga", "2d", "ecchi", "nsfw anime")
    )
    photoreal = any(_keyword_in_prompt(term, text) for term in ("photoreal", "photorealistic", "real photo", "amateur", "webcam"))
    if not photoreal:
        photoreal = "realistic" in lower and not hentai_intent
    if not photoreal:
        photoreal = _keyword_in_prompt("photo", text) and not hentai_intent
    candidates: List[Tuple[int, str, Path, str]] = []
    root = get_project_root()
    for category, entry in list_reference_entries().items():
        folder = root / str(entry.get("dir") or "")
        for keyword in entry.get("keywords") or []:
            kw = str(keyword).strip().lower()
            if not kw:
                continue
            if _keyword_in_prompt(kw, text):
                candidates.append((len(kw), category, folder, kw))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if hentai_intent and not photoreal:
        preferred = ("hentai", "mixed", "animated", "porn", "human")
        for pref in preferred:
            for _, category, folder, keyword in candidates:
                if category != pref:
                    continue
                images, videos = _collect_media(folder)
                if images or videos:
                    return category, folder, keyword
    if photoreal and hentai_intent:
        preferred = ("hentai", "mixed", "real", "porn", "human")
        for pref in preferred:
            for _, category, folder, keyword in candidates:
                if category != pref:
                    continue
                images, videos = _collect_media(folder)
                if images or videos:
                    return category, folder, keyword
    if photoreal:
        preferred = ("real", "porn", "human", "mixed", "hentai")
        for pref in preferred:
            for _, category, folder, keyword in candidates:
                if category != pref:
                    continue
                images, videos = _collect_media(folder)
                if images or videos:
                    return category, folder, keyword
    for _, category, folder, keyword in candidates:
        images, videos = _collect_media(folder)
        if images or videos:
            return category, folder, keyword
    _, category, folder, keyword = candidates[0]
    return category, folder, keyword


_MEDIA_CACHE: Dict[str, Tuple[float, List[Path], List[Path]]] = {}


def _collect_media(folder: Path) -> Tuple[List[Path], List[Path]]:
    if not folder.exists():
        return [], []
    try:
        mtime = folder.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = str(folder.resolve())
    cached = _MEDIA_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    images: List[Path] = []
    videos: List[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(path)
        elif ext in {".mp4", ".webm", ".mov"}:
            videos.append(path)
    images.sort()
    videos.sort()
    _MEDIA_CACHE[key] = (mtime, images, videos)
    return images, videos


def count_folder_media(folder: Path) -> Tuple[int, int]:
    images, videos = _collect_media(folder)
    return len(images), len(videos)


def _sidecar_path(media_path: Path) -> Path:
    return media_path.with_suffix(media_path.suffix + ".json")


def _sidecar_is_nsfw(media_path: Path) -> bool:
    sidecar = _sidecar_path(media_path)
    if not sidecar.exists():
        alt = media_path.with_suffix(".json")
        sidecar = alt if alt.exists() else sidecar
    if not sidecar.exists():
        return False
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return bool(data.get("is_nsfw") or data.get("infini_is_nsfw"))
    except Exception:
        return False


def _filter_reference_pool(pool: List[Path], category: str) -> List[Path]:
    settings = get_reference_settings()
    cat = str(category).lower()
    filtered = list(pool)
    if settings.get("require_nsfw_sidecar") and cat in ("hentai", "porn", "fetish"):
        nsfw_pool = [path for path in filtered if _sidecar_is_nsfw(path)]
        if nsfw_pool:
            filtered = nsfw_pool
    if settings.get("prefer_infini_refs") and cat == "hentai":
        # Name-based first to avoid opening thousands of sidecars on every generate.
        infini_pool = [path for path in filtered if "infini" in path.name.lower()]
        if not infini_pool:
            infini_tagged = []
            for path in filtered[:400]:
                sidecar = _sidecar_path(path)
                if not sidecar.exists():
                    continue
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    if data.get("infini_tag") or data.get("site_name") == "infini-atomic":
                        infini_tagged.append(path)
                except Exception:
                    pass
            infini_pool = infini_tagged
        if infini_pool:
            filtered = infini_pool
    return filtered or pool


def _prompt_tokens(prompt: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", prompt.lower()) if len(t) > 2]


def _prompt_similarity_score(path: Path, prompt: str, category: str) -> float:
    tokens = _prompt_tokens(prompt)
    if not tokens:
        return 0.0
    stem = path.stem.lower().replace("_", " ").replace("-", " ")
    parent = path.parent.name.lower()
    score = 0.0
    for token in tokens:
        if token in stem:
            score += 2.5
        elif token in parent:
            score += 1.0
        if token == category:
            score += 1.5
    return score


def pick_reference_media(
    folder: Path,
    prompt: str,
    prompt_hash: int,
    prefer_video: bool = False,
    category: str = "",
) -> Optional[Path]:
    images, videos = _collect_media(folder)
    pool: List[Path] = []
    if prefer_video and videos:
        pool = videos
    elif images:
        pool = images
    elif videos:
        pool = videos
    if not pool:
        return None
    pool = _filter_reference_pool(pool, category)
    if str(category).lower() == "hentai":
        try:
            from navine.reference.sources.waifu import is_waifu_reference_path

            waifu_pool = [path for path in pool if is_waifu_reference_path(path)]
            if waifu_pool:
                pool = waifu_pool
        except Exception:
            pass
    settings = get_reference_settings()
    if settings.get("prompt_similarity", True) and len(pool) > 1:
        scored = [(path, _prompt_similarity_score(path, prompt, category)) for path in pool]
        best_score = max(s for _, s in scored)
        if best_score > 0:
            top = [path for path, score in scored if score >= best_score - 0.5]
            return top[prompt_hash % len(top)]
    return pool[prompt_hash % len(pool)]


def load_reference_image(path: Path, size: Optional[int] = None) -> Image.Image:
    ext = path.suffix.lower()
    if ext == ".gif":
        with Image.open(path) as img:
            img.seek(0)
            frame = img.convert("RGB")
    elif ext in {".mp4", ".webm", ".mov"}:
        frames = extract_video_frames(path, max_frames=1)
        if not frames:
            raise ValueError(f"Could not read video reference: {path}")
        frame = frames[0]
    else:
        frame = Image.open(path).convert("RGB")
    if size:
        frame = frame.resize((size, size), Image.Resampling.LANCZOS)
    return frame


def extract_video_frames(path: Path, max_frames: int = 8) -> List[Image.Image]:
    ext = path.suffix.lower()
    if ext == ".gif":
        frames: List[Image.Image] = []
        with Image.open(path) as img:
            try:
                while len(frames) < max_frames:
                    frames.append(img.convert("RGB").copy())
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
        return frames
    try:
        import imageio.v3 as iio

        frames = []
        meta = iio.immeta(path)
        total = int(meta.get("nframes") or meta.get("n_frames") or 0) if isinstance(meta, dict) else 0
        step = max(1, total // max(1, max_frames)) if total > 0 else 1
        for idx, arr in enumerate(iio.imiter(path)):
            if len(frames) >= max_frames:
                break
            if total > 0 and idx % step != 0 and idx < total - 1:
                continue
            frames.append(Image.fromarray(arr).convert("RGB"))
        if frames:
            return frames
    except Exception:
        pass
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return []
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // max(1, max_frames)) if total > 0 else 1
        frames = []
        index = 0
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb).convert("RGB"))
            index += 1
        capture.release()
        return frames
    except Exception:
        return []


def _region_mean(img: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    crop = img.crop(box)
    stat = crop.convert("RGB").resize((1, 1), Image.Resampling.LANCZOS).getpixel((0, 0))
    return float(stat[0]), float(stat[1]), float(stat[2])


def _apply_tone_mapping(
    base: Image.Image,
    reference: Image.Image,
    category: str,
    strength: float,
) -> Image.Image:
    w, h = base.size
    cx, cy = w // 2, h // 2
    box = (cx - w // 5, cy - h // 5, cx + w // 5, cy + h // 5)
    ref_mean = _region_mean(reference, box)
    base_mean = _region_mean(base, box)
    cat = str(category).lower()
    if cat in ("hentai", "animated", "cosplay") or cat in ("anime",):
        color = ImageEnhance.Color(base)
        sat_boost = 1.0 + strength * 0.35
        base = color.enhance(sat_boost)
        for channel, ref_val, base_val in zip(range(3), ref_mean, base_mean):
            if abs(ref_val - base_val) > 8:
                shift = (ref_val - base_val) * strength * 0.25
                plane = base.split()[channel]
                plane = ImageEnhance.Brightness(plane).enhance(1.0 + shift / 128.0)
                channels = list(base.split())
                channels[channel] = plane
                base = Image.merge("RGB", channels)
        return base
    if cat in ("real", "porn", "human", "fetish"):
        adjusted = base.copy()
        for channel, ref_val, base_val in zip(range(3), ref_mean, base_mean):
            delta = (ref_val - base_val) * strength * 0.45
            plane = adjusted.split()[channel]
            plane = ImageEnhance.Brightness(plane).enhance(1.0 + delta / 120.0)
            channels = list(adjusted.split())
            channels[channel] = plane
            adjusted = Image.merge("RGB", channels)
        warm = ImageEnhance.Color(adjusted).enhance(1.0 + strength * 0.08)
        contrast = ImageEnhance.Contrast(warm).enhance(1.0 + strength * 0.06)
        return contrast
    return base


def _edge_weight_map(reference: Image.Image, size: Tuple[int, int]) -> Image.Image:
    ref = reference.convert("L").resize(size, Image.Resampling.LANCZOS)
    edges = ref.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(2.2)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=1.2))
    return ImageOps.invert(edges)


def _structure_preserving_blend(
    generated: Image.Image,
    reference: Image.Image,
    strength: float,
) -> Image.Image:
    size = generated.size
    ref = reference.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    base = generated.convert("RGB")
    flat_blend = Image.blend(ref, base, 1.0 - strength)
    edge_map = _edge_weight_map(ref, size)
    structure_strength = min(0.92, strength + 0.12)
    structure = Image.composite(ref, base, edge_map)
    return Image.blend(flat_blend, structure, structure_strength * 0.55)


def _add_controlled_noise(image: Image.Image, amount: float, seed: int) -> Image.Image:
    import random

    rng = random.Random(seed)
    w, h = image.size
    noise = Image.effect_noise((w, h), amount * 64 + 8)
    noise = noise.convert("RGB")
    tinted = Image.new("RGB", (w, h))
    pixels = image.load()
    noise_pixels = noise.load()
    tinted_pixels = tinted.load()
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            nr, ng, nb = noise_pixels[x, y]
            mix = amount * (0.5 + rng.random() * 0.5)
            tinted_pixels[x, y] = (
                int(r * (1 - mix) + nr * mix),
                int(g * (1 - mix) + ng * mix),
                int(b * (1 - mix) + nb * mix),
            )
    return tinted


def reference_guided_img2img_lite(
    generated: Image.Image,
    reference: Image.Image,
    strength: float,
    seed: int,
    category: str,
    mode: str = "heavy",
) -> Image.Image:
    settings = get_reference_settings()
    if str(mode).lower() == "guide":
        return apply_reference_blend(
            generated, reference, strength=strength, seed=seed, category=category, mode="guide"
        )
    lite = settings.get("img2img_lite") or {}
    if not lite.get("enabled", True):
        return apply_reference_blend(
            generated, reference, strength=strength, seed=seed, category=category, mode=mode
        )
    noise = float(lite.get("noise_strength", 0.22))
    size = generated.size
    ref = reference.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    noised_ref = _add_controlled_noise(ref, noise, seed)
    anchor_strength = min(0.88, strength + 0.08)
    anchored = Image.blend(noised_ref, generated.convert("RGB"), 1.0 - anchor_strength)
    return apply_reference_blend(anchored, ref, strength=strength, seed=seed, category=category, mode=mode)


def apply_reference_blend(
    generated: Image.Image,
    reference: Image.Image,
    strength: float = 0.55,
    seed: int = 0,
    category: str = "",
    mode: str = "heavy",
) -> Image.Image:
    strength = max(0.10, min(0.90, strength))
    if str(mode).lower() == "guide":
        base = generated.convert("RGB")
        ref = reference.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)
        toned = _apply_tone_mapping(base, ref, category, strength)
        return Image.blend(base, toned, strength)
    blended = _structure_preserving_blend(generated, reference, strength)
    blended = _apply_tone_mapping(blended, reference, category, strength)
    brightness = 1.0 + ((seed % 7) - 3) * 0.012
    contrast = 1.0 + ((seed // 7) % 5 - 2) * 0.018
    blended = ImageEnhance.Brightness(blended).enhance(brightness)
    blended = ImageEnhance.Contrast(blended).enhance(contrast)
    if strength >= 0.42:
        blended = blended.filter(ImageFilter.UnsharpMask(radius=0.9, percent=75, threshold=3))
    return blended


def _usage_state_path() -> Path:
    path = get_project_root() / "data" / "nsfw" / "reference_usage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_usage_state() -> Dict[str, Any]:
    path = _usage_state_path()
    if not path.exists():
        return {"categories": {}, "total": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"categories": {}, "total": 0}


def _save_usage_state(state: Dict[str, Any]) -> None:
    _usage_state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def record_reference_use(category: str) -> None:
    state = _load_usage_state()
    cats = state.setdefault("categories", {})
    cat = str(category).lower()
    cats[cat] = int(cats.get(cat, 0)) + 1
    state["total"] = int(state.get("total", 0)) + 1
    _save_usage_state(state)
    settings = get_reference_settings()
    auto = settings.get("auto_train_on_reference") or {}
    if not auto.get("enabled", True):
        return
    threshold = int(auto.get("min_uses_per_category", 12))
    if int(cats.get(cat, 0)) >= threshold and int(cats.get(cat, 0)) % threshold == 0:
        # Never block image generation on a full finetune; training belongs in CLI/autolearn.
        return


def _maybe_train_reference_category(category: str) -> None:
    try:
        from navine.nsfw.config import load_nsfw_config

        cfg = load_nsfw_config()
        if not cfg.get("train_on_collect", True):
            return
        entry = list_reference_entries().get(category)
        if not entry:
            return
        from navine.autolearn.sources.nsfw import download_nsfw_images, download_nsfw_videos, fetch_local
        from navine.autolearn.engine import AutolearnEngine

        folder = str(entry.get("dir") or "")
        items = fetch_local(local_path=folder, config=cfg)
        if not items:
            return
        engine = AutolearnEngine()
        engine.process_items(items)
        if cfg.get("auto_train_image", True):
            download_nsfw_images(items)
        if cfg.get("auto_train_video", True):
            download_nsfw_videos(items)
        if cfg.get("auto_train_image", True):
            from navine.learn.image_learn import train_learned_image

            train_learned_image(finetune_steps=cfg.get("image_train_steps"))
    except Exception:
        pass


def get_reference_usage_stats() -> Dict[str, Any]:
    state = _load_usage_state()
    return {
        "total": int(state.get("total", 0)),
        "categories": dict(state.get("categories") or {}),
    }


def resolve_reference_for_prompt(
    prompt: str,
    prefer_video: bool = False,
    has_checkpoint: bool = False,
) -> Optional[Dict[str, Any]]:
    settings = get_reference_settings()
    mode = str(settings.get("mode") or "guide").lower()
    if mode == "off":
        return None
    matched = match_reference_keyword(prompt)
    if not matched:
        return None
    category, folder, keyword = matched
    from navine.image.procedural import prompt_seed

    media_path = pick_reference_media(
        folder,
        prompt,
        prompt_seed(prompt),
        prefer_video=prefer_video,
        category=category,
    )
    if not media_path:
        return None
    ext = media_path.suffix.lower()
    is_video = ext in {".mp4", ".webm", ".mov"} or (ext == ".gif" and prefer_video)
    strength = reference_strength_for_category(category, has_checkpoint=has_checkpoint)
    try:
        from navine.reference.sources.waifu import is_waifu_reference_path

        if is_waifu_reference_path(media_path):
            settings = get_reference_settings()
            waifu_strength = settings.get("waifu_im_strength")
            if waifu_strength is not None:
                strength = max(strength, float(waifu_strength))
            else:
                strength = min(0.90, strength + 0.12)
    except Exception:
        pass
    return {
        "category": category,
        "folder": str(folder),
        "keyword_matched": keyword,
        "reference_path": str(media_path),
        "reference_used": True,
        "is_video": is_video,
        "reference_strength": strength,
        "generation_mode": mode,
    }


def apply_reference_to_output(
    output_path: Path,
    prompt: str,
    size: int = 512,
    strength: Optional[float] = None,
    has_checkpoint: bool = False,
) -> Optional[Dict[str, Any]]:
    settings = get_reference_settings()
    mode = str(settings.get("mode") or "guide").lower()
    if mode == "off":
        return None
    info = resolve_reference_for_prompt(prompt, prefer_video=False, has_checkpoint=has_checkpoint)
    if not info:
        return None
    category = str(info.get("category") or "")
    blend_strength = strength if strength is not None else float(info.get("reference_strength", 0.58))
    ref = load_reference_image(Path(info["reference_path"]), size=size)
    generated = Image.open(output_path).convert("RGB")
    from navine.image.procedural import prompt_seed

    lite = settings.get("img2img_lite") or {}
    use_lite = False
    try:
        from navine.nsfw.prompts import analyze_prompt

        analyzed = analyze_prompt(prompt)
        use_lite = bool(analyzed.get("real")) or str(analyzed.get("category")) in ("real", "porn", "human")
    except Exception:
        use_lite = False
    if use_lite and lite.get("enabled", True) and category in ("real", "porn", "human", "mixed"):
        blended = reference_guided_img2img_lite(
            generated,
            ref,
            strength=blend_strength,
            seed=prompt_seed(prompt),
            category=category,
            mode="guide",
        )
    elif mode == "guide" or not lite.get("enabled", True):
        blended = apply_reference_blend(
            generated,
            ref,
            strength=blend_strength,
            seed=prompt_seed(prompt),
            category=category,
            mode=mode,
        )
    else:
        blended = reference_guided_img2img_lite(
            generated,
            ref,
            strength=blend_strength,
            seed=prompt_seed(prompt),
            category=category,
            mode=mode,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path, format="PNG")
    record_reference_use(category)
    info["reference_strength"] = blend_strength
    info["generation_mode"] = mode
    return info


def copy_reference_file(category: str, source_path: Path) -> Path:
    entries = list_reference_entries()
    cat = category.strip().lower()
    if cat not in entries:
        raise ValueError(f"Unknown reference category: {category}")
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Reference file not found: {source_path}")
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTENSIONS and ext not in {".mp4", ".webm", ".mov"}:
        raise ValueError(f"Unsupported reference media type: {ext}")
    dest_dir = get_project_root() / str(entries[cat]["dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return dest


def list_reference_summary() -> List[Dict[str, Any]]:
    root = get_project_root()
    usage = get_reference_usage_stats()
    rows: List[Dict[str, Any]] = []
    for category, entry in sorted(list_reference_entries().items()):
        folder = root / str(entry.get("dir") or "")
        image_count, video_count = count_folder_media(folder)
        rows.append(
            {
                "category": category,
                "dir": entry.get("dir"),
                "keywords": entry.get("keywords") or [],
                "images": image_count,
                "videos": video_count,
                "uses": int((usage.get("categories") or {}).get(category, 0)),
            }
        )
    return rows
