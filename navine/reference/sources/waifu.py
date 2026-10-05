import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode, urlparse

from navine.autolearn.http import learning_http_get
from navine.image.utils import validate_image_bytes
from navine.nsfw.config import load_nsfw_config
from navine.nsfw.media import enrich_media_item
from navine.policy import http_get
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]

API_BASE = "https://api.waifu.im"
WAIFU_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MIN_BYTES = 2048
MAX_BYTES = 15 * 1024 * 1024


def _report(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def _sleep(cfg: Dict[str, Any]) -> None:
    section = waifu_im_config(cfg)
    delay = float(section.get("sleep_between_requests") or cfg.get("sleep_between_requests", 1.5))
    time.sleep(delay)


def waifu_im_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = cfg or load_nsfw_config()
    section = merged.get("waifu_im") or {}
    if not isinstance(section, dict):
        return {"enabled": False}
    defaults = {
        "enabled": False,
        "tags": ["ecchi", "ero", "ass"],
        "max_per_run": 20,
        "is_nsfw": True,
        "media_category": "hentai",
        "reference_dir": "data/nsfw/local/hentai",
        "archive_dir": "data/nsfw/images/waifu",
        "gallery_urls": [],
        "sleep_between_requests": 1.5,
    }
    out = dict(defaults)
    out.update(section)
    return out


def _images_url(tag: str, is_nsfw: bool) -> str:
    params = [
        ("included_tags", tag.strip().lower()),
        ("is_nsfw", "true" if is_nsfw else "false"),
    ]
    return f"{API_BASE}/images?{urlencode(params)}"


def _search_url(tag: str, is_nsfw: bool, many: bool = False) -> str:
    return _images_url(tag, is_nsfw)


def _parse_tag_names(raw_tags: Any) -> List[str]:
    names: List[str] = []
    if isinstance(raw_tags, list):
        for entry in raw_tags:
            if isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("tag") or "").strip().lower()
            else:
                name = str(entry).strip().lower()
            if name:
                names.append(name)
    elif isinstance(raw_tags, str):
        names = [t.strip().lower() for t in raw_tags.replace("_", " ").split() if t.strip()]
    return names


def _image_record_to_item(
    record: Dict[str, Any],
    tag: str,
    category: str,
    media_category: str,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    file_url = str(record.get("url") or record.get("image_url") or "").strip()
    if not file_url:
        return None
    signature = str(record.get("signature") or record.get("id") or record.get("image_id") or "")
    tag_names = _parse_tag_names(record.get("tags"))
    if tag and tag not in tag_names:
        tag_names.insert(0, tag.lower())
    tag_names = list(dict.fromkeys(tag_names + ["waifu", "anime", media_category]))
    caption = " ".join(tag_names[:12]) or f"waifu {tag}"
    token = signature or format(hash(file_url) & 0xFFFFFFFF, "08x")
    item = {
        "source": "waifu_im",
        "source_id": f"waifu_im:{token}:{tag}",
        "title": caption[:120],
        "text": caption,
        "url": file_url,
        "category": category,
        "kind": "image",
        "image_url": file_url,
        "caption": caption,
        "media_category": media_category,
        "tags": tag_names,
        "waifu_signature": signature,
        "waifu_tag": tag,
        "is_nsfw": bool(record.get("is_nsfw", True)),
        "source_page": f"https://www.waifu.im/gallery?includedTags={tag}&orderBy=Random&isNsfw=All",
    }
    return enrich_media_item(item, cfg)


def _fetch_tag_batch(tag: str, cfg: Dict[str, Any], category: str, media_category: str) -> List[Dict[str, Any]]:
    section = waifu_im_config(cfg)
    is_nsfw = bool(section.get("is_nsfw", True))
    per_tag = max(1, int(section.get("max_per_tag") or 0))
    if per_tag <= 0:
        max_total = int(section.get("max_per_run", 20))
        tag_count = max(1, len(section.get("tags") or [tag]))
        per_tag = max(1, max_total // tag_count)
    headers = {"Accept": "application/json", "User-Agent": WAIFU_USER_AGENT}
    items: List[Dict[str, Any]] = []
    seen_ids: set = set()
    attempts = 0
    max_attempts = max(per_tag * 4, per_tag + 3)
    while len(items) < per_tag and attempts < max_attempts:
        attempts += 1
        url = _images_url(tag, is_nsfw=is_nsfw)
        try:
            response = learning_http_get(url, headers=headers)
        except Exception:
            break
        if response.status_code != 200:
            break
        try:
            payload = response.json()
        except Exception:
            break
        records = payload.get("items") if isinstance(payload, dict) else None
        if records is None and isinstance(payload, dict) and payload.get("url"):
            records = [payload]
        if not isinstance(records, list):
            records = []
        for record in records:
            if not isinstance(record, dict):
                continue
            image_id = record.get("id")
            if image_id is not None and image_id in seen_ids:
                continue
            if image_id is not None:
                seen_ids.add(image_id)
            item = _image_record_to_item(record, tag, category, media_category, cfg)
            if item:
                items.append(item)
                if len(items) >= per_tag:
                    break
        _sleep(cfg)
    return items


def fetch_waifu_im(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    section = waifu_im_config(cfg)
    if not section.get("enabled", False):
        return []
    tags = [str(t).strip().lower() for t in (section.get("tags") or []) if str(t).strip()]
    if not tags:
        return []
    media_category = str(section.get("media_category") or "hentai")
    max_total = int(section.get("max_per_run", 20))
    items: List[Dict[str, Any]] = []
    seen_urls: set = set()
    for tag in tags:
        if len(items) >= max_total:
            break
        _report(progress, f"Waifu.im API: searching tag '{tag}'...")
        try:
            batch = _fetch_tag_batch(tag, cfg, category, media_category)
        except Exception as exc:
            _report(progress, f"Warning: waifu.im tag '{tag}' failed: {exc}")
            batch = []
        if not batch:
            _report(progress, f"Warning: waifu.im tag '{tag}' returned 0 images.")
        for item in batch:
            url = item.get("image_url") or ""
            if url in seen_urls:
                continue
            seen_urls.add(url)
            items.append(item)
            if len(items) >= max_total:
                break
        _sleep(cfg)
    return items[:max_total]


def _guess_extension(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_EXT:
        return suffix
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    return ".jpg"


def _reference_dir(cfg: Dict[str, Any]) -> Path:
    section = waifu_im_config(cfg)
    rel = str(section.get("reference_dir") or "data/nsfw/local/hentai")
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def _archive_dir(cfg: Dict[str, Any]) -> Path:
    section = waifu_im_config(cfg)
    rel = str(section.get("archive_dir") or "data/nsfw/images/waifu")
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:48] or "waifu"


def _save_metadata(path: Path, meta: Dict[str, Any]) -> None:
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _download_image_bytes(url: str) -> tuple:
    response = http_get(
        url,
        headers={"User-Agent": WAIFU_USER_AGENT, "Accept": "image/*,*/*"},
        timeout=30,
        stream=True,
    )
    response.raise_for_status()
    content = response.content
    if len(content) < MIN_BYTES:
        raise ValueError("Downloaded waifu.im file too small")
    if len(content) > MAX_BYTES:
        raise ValueError("Downloaded waifu.im file too large")
    content_type = response.headers.get("Content-Type", "")
    if not validate_image_bytes(content):
        raise ValueError("Downloaded waifu.im file is not a valid image")
    return content, content_type


def download_waifu_references(
    items: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    progress: ProgressCallback = None,
) -> int:
    cfg = config or load_nsfw_config()
    section = waifu_im_config(cfg)
    if items is None:
        items = fetch_waifu_im(config=cfg, progress=progress)
    ref_dir = _reference_dir(cfg)
    archive_dir = _archive_dir(cfg)
    count = 0
    for item in items:
        if item.get("kind") != "image":
            continue
        url = str(item.get("image_url") or "").strip()
        if not url:
            continue
        remote_url = url
        tag = _safe_slug(str(item.get("waifu_tag") or "waifu"))
        signature = _safe_slug(str(item.get("waifu_signature") or f"{hash(url) & 0xFFFFFFFF:08x}"))
        try:
            content, content_type = _download_image_bytes(remote_url)
        except Exception as exc:
            _report(progress, f"Warning: waifu.im download failed: {exc}")
            continue
        ext = _guess_extension(remote_url, content_type)
        filename = f"waifu_{tag}_{signature}{ext}"
        dest = ref_dir / filename
        archive = archive_dir / filename
        if not dest.exists():
            dest.write_bytes(content)
        if not archive.exists():
            archive.write_bytes(content)
        item["local_reference_path"] = str(dest)
        item["image_url"] = str(dest)
        meta = {
            "source": "waifu.im",
            "source_url": remote_url,
            "tags": item.get("tags") or [],
            "waifu_tag": item.get("waifu_tag"),
            "waifu_signature": item.get("waifu_signature"),
            "media_category": item.get("media_category") or "hentai",
            "reference_path": str(dest.relative_to(get_project_root()).as_posix()),
            "gallery_url": item.get("source_page"),
        }
        _save_metadata(dest, meta)
        count += 1
        _report(progress, f"Waifu.im saved ({count}): {dest.name}")
        _sleep(cfg)
    return count


def fetch_and_download_waifu(
    config: Optional[Dict[str, Any]] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = config or load_nsfw_config()
    items = fetch_waifu_im(config=cfg, progress=progress)
    downloaded = download_waifu_references(items=items, config=cfg, progress=progress)
    return {
        "items": items,
        "items_fetched": len(items),
        "downloaded": downloaded,
        "reference_dir": str(_reference_dir(cfg)),
    }


def hentai_reference_has_waifu() -> bool:
    cfg = load_nsfw_config()
    ref_dir = _reference_dir(cfg)
    if not ref_dir.exists():
        return False
    for path in ref_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXT:
            continue
        if path.stem.lower().startswith("waifu_"):
            return True
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("source") == "waifu.im":
                    return True
            except Exception:
                pass
    return False


def is_waifu_reference_path(path: Path) -> bool:
    if path.stem.lower().startswith("waifu_"):
        return True
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("source") == "waifu.im"
        except Exception:
            return False
    return False
