import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

from bs4 import BeautifulSoup

from navine.autolearn.http import learning_http_get
from navine.image.utils import validate_image_bytes
from navine.nsfw.config import load_nsfw_config
from navine.nsfw.media import enrich_media_item
from navine.policy import http_get
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]

INFINI_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
WAIFU_API_BASE = "https://api.waifu.im"
DEFAULT_NSFW_TAGS = ["ecchi", "ero", "ass", "hentai", "milf", "oral", "paizuri"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MIN_BYTES = 2048
MAX_BYTES = 15 * 1024 * 1024


def _report(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def _sleep(cfg: Dict[str, Any]) -> None:
    section = infini_atomic_config(cfg)
    delay = float(section.get("sleep_between_requests") or cfg.get("sleep_between_requests", 1.5))
    time.sleep(delay)


def infini_atomic_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = cfg or load_nsfw_config()
    section = merged.get("infini_atomic") or {}
    if not isinstance(section, dict):
        return {"enabled": False}
    defaults = {
        "enabled": False,
        "base_url": "https://infini-atomic.w3spaces.com/",
        "max_images_per_run": 49,
        "max_per_tag": 7,
        "media_category": "hentai",
        "tags": [],
        "tag_allowlist": [],
        "use_all_site_tags": True,
        "nsfw_only": True,
        "reference_dir": "data/nsfw/local/hentai",
        "archive_dir": "data/nsfw/images/infini-atomic",
        "sleep_between_requests": 1.5,
    }
    out = dict(defaults)
    out.update(section)
    return out


def _base_url(cfg: Dict[str, Any]) -> str:
    return str(infini_atomic_config(cfg).get("base_url") or "https://infini-atomic.w3spaces.com/").rstrip("/") + "/"


def _site_domain(cfg: Dict[str, Any]) -> str:
    return urlparse(_base_url(cfg)).netloc.lower()


def _interstitial_cookie_value() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_site_html(cfg: Dict[str, Any]) -> Tuple[str, bool]:
    base = _base_url(cfg)
    domain = _site_domain(cfg)
    headers = {"User-Agent": INFINI_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    cookie_headers = {"Cookie": f"staticSpaceInterstitial={_interstitial_cookie_value()}"}
    try:
        first = learning_http_get(base, headers={**headers, **cookie_headers}, timeout=30)
        if first.status_code != 200:
            return "", False
        if "staticSpaceInterstitial" in first.text or "User-Created" in first.text:
            second = learning_http_get(base, headers={**headers, **cookie_headers}, timeout=30)
            if second.status_code != 200:
                return "", False
            return second.text, "Waifu.im Tag Viewer" in second.text or "tagSelector" in second.text
        return first.text, "Waifu.im Tag Viewer" in first.text or "tagSelector" in first.text
    except Exception:
        return "", False


def parse_site_tags(html: str) -> List[Dict[str, Any]]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    selector = soup.find("select", id="tagSelector") or soup.find("select")
    if not selector:
        return []
    tags: List[Dict[str, Any]] = []
    for optgroup in selector.find_all("optgroup"):
        label = str(optgroup.get("label") or "").strip().lower()
        is_nsfw = "nsfw" in label
        for option in optgroup.find_all("option"):
            value = str(option.get("value") or "").strip().lower()
            if value:
                tags.append({"tag": value, "is_nsfw": is_nsfw})
    if not tags:
        for option in selector.find_all("option"):
            value = str(option.get("value") or "").strip().lower()
            if value:
                tags.append({"tag": value, "is_nsfw": value in DEFAULT_NSFW_TAGS})
    return tags


def _filter_parsed_tags(parsed: List[Dict[str, Any]], section: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not parsed:
        return []
    if section.get("nsfw_only", True):
        nsfw = [entry for entry in parsed if entry.get("is_nsfw")]
        parsed = nsfw or parsed
    allowlist = [str(t).strip().lower() for t in (section.get("tag_allowlist") or []) if str(t).strip()]
    if allowlist:
        allow_set = set(allowlist)
        filtered = [entry for entry in parsed if str(entry.get("tag") or "").lower() in allow_set]
        return filtered or [{"tag": tag, "is_nsfw": tag in DEFAULT_NSFW_TAGS} for tag in allowlist]
    return parsed


def _configured_tags(cfg: Dict[str, Any], html: str) -> List[Dict[str, Any]]:
    section = infini_atomic_config(cfg)
    parsed = parse_site_tags(html)
    use_all = bool(section.get("use_all_site_tags", True))
    allowlist = [str(t).strip().lower() for t in (section.get("tag_allowlist") or []) if str(t).strip()]
    configured = [str(t).strip().lower() for t in (section.get("tags") or []) if str(t).strip()]
    if use_all and parsed:
        return _filter_parsed_tags(parsed, section)
    if allowlist:
        parsed_map = {entry["tag"]: entry for entry in parsed}
        tags: List[Dict[str, Any]] = []
        for tag in allowlist:
            entry = parsed_map.get(tag)
            if entry:
                tags.append(entry)
            else:
                tags.append({"tag": tag, "is_nsfw": tag in DEFAULT_NSFW_TAGS})
        return tags
    if configured:
        parsed_map = {entry["tag"]: entry for entry in parsed}
        tags = []
        for tag in configured:
            entry = parsed_map.get(tag)
            if entry:
                tags.append(entry)
            else:
                tags.append({"tag": tag, "is_nsfw": tag in DEFAULT_NSFW_TAGS})
        return tags
    if parsed:
        return _filter_parsed_tags(parsed, section)
    return [{"tag": tag, "is_nsfw": True} for tag in DEFAULT_NSFW_TAGS]


def list_infini_site_tags(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_nsfw_config()
    html, site_ok = _fetch_site_html(cfg)
    tags = _configured_tags(cfg, html)
    parsed_all = parse_site_tags(html)
    return {
        "site_ok": site_ok,
        "base_url": _base_url(cfg),
        "use_all_site_tags": bool(infini_atomic_config(cfg).get("use_all_site_tags", True)),
        "tags_active": [entry.get("tag") for entry in tags],
        "tags_parsed": [entry.get("tag") for entry in parsed_all],
        "tag_details": tags,
    }


def _waifu_api_url(tag: str, is_nsfw: bool) -> str:
    params = [
        ("included_tags", tag.strip().lower()),
        ("is_nsfw", "true" if is_nsfw else "false"),
    ]
    return f"{WAIFU_API_BASE}/images?{urlencode(params)}"


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


def _record_to_item(
    record: Dict[str, Any],
    tag: str,
    is_nsfw: bool,
    category: str,
    media_category: str,
    cfg: Dict[str, Any],
    site_url: str,
) -> Optional[Dict[str, Any]]:
    file_url = str(record.get("url") or record.get("image_url") or "").strip()
    if not file_url:
        return None
    parsed = urlparse(file_url)
    host = (parsed.netloc or "").lower()
    if host and "waifu.im" not in host and not host.endswith(".waifu.im"):
        return None
    signature = str(record.get("signature") or record.get("id") or record.get("image_id") or "")
    tag_names = _parse_tag_names(record.get("tags"))
    if tag and tag not in tag_names:
        tag_names.insert(0, tag.lower())
    tag_names = list(
        dict.fromkeys(tag_names + ["infini", "atomic", "anime", media_category, "waifu"])
    )
    caption = " ".join(tag_names[:12]) or f"infini atomic {tag}"
    token = signature or format(hash(file_url) & 0xFFFFFFFF, "08x")
    item = {
        "source": "infini_atomic",
        "source_id": f"infini_atomic:{token}:{tag}",
        "title": caption[:120],
        "text": caption,
        "url": file_url,
        "category": category,
        "kind": "image",
        "image_url": file_url,
        "caption": caption,
        "media_category": media_category,
        "tags": tag_names,
        "infini_tag": tag,
        "infini_is_nsfw": is_nsfw,
        "infini_signature": signature,
        "source_page": site_url,
        "site_name": "infini-atomic",
    }
    return enrich_media_item(item, cfg)


def _fetch_tag_image(
    tag_entry: Dict[str, Any],
    cfg: Dict[str, Any],
    category: str,
    media_category: str,
    site_url: str,
) -> Optional[Dict[str, Any]]:
    tag = str(tag_entry.get("tag") or "").strip().lower()
    if not tag:
        return None
    is_nsfw = bool(tag_entry.get("is_nsfw", True))
    url = _waifu_api_url(tag, is_nsfw=is_nsfw)
    headers = {"Accept": "application/json", "User-Agent": INFINI_USER_AGENT}
    try:
        response = learning_http_get(url, headers=headers, timeout=30)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    records = payload.get("items") if isinstance(payload, dict) else None
    if records is None and isinstance(payload, dict) and payload.get("url"):
        records = [payload]
    if not isinstance(records, list) or not records:
        return None
    record = records[0]
    if not isinstance(record, dict):
        return None
    return _record_to_item(record, tag, is_nsfw, category, media_category, cfg, site_url)


def fetch_infini_atomic(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    section = infini_atomic_config(cfg)
    if not section.get("enabled", False):
        return []
    site_url = _base_url(cfg)
    max_total = int(section.get("max_images_per_run", 49))
    max_per_tag = int(section.get("max_per_tag") or 0)
    media_category = str(section.get("media_category") or "hentai")
    html, ok = _fetch_site_html(cfg)
    if not ok:
        _report(progress, "Warning: infini-atomic site interstitial or HTML unavailable; using configured NSFW tags.")
    tag_entries = _configured_tags(cfg, html)
    if not tag_entries:
        _report(progress, "Warning: infini-atomic returned no tags.")
        return []
    if max_per_tag <= 0:
        per_tag = max(1, max_total // max(1, len(tag_entries)))
    else:
        per_tag = max_per_tag
    if max_total < per_tag * len(tag_entries):
        max_total = per_tag * len(tag_entries)
    items: List[Dict[str, Any]] = []
    seen_urls: set = set()
    per_tag_counts: Dict[str, int] = {}
    for tag_entry in tag_entries:
        tag = str(tag_entry.get("tag") or "")
        _report(progress, f"Infini-atomic: fetching tag '{tag}' via waifu.im API...")
        attempts = 0
        max_attempts = max(per_tag * 3, per_tag + 2)
        tag_count = 0
        while tag_count < per_tag and attempts < max_attempts:
            if len(items) >= max_total:
                break
            attempts += 1
            item = _fetch_tag_image(tag_entry, cfg, category, media_category, site_url)
            _sleep(cfg)
            if not item:
                continue
            url = item.get("image_url") or ""
            if url in seen_urls:
                continue
            seen_urls.add(url)
            items.append(item)
            tag_count += 1
            per_tag_counts[tag] = per_tag_counts.get(tag, 0) + 1
    if per_tag_counts:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(per_tag_counts.items()))
        _report(progress, f"Infini-atomic: fetched {len(items)} image(s) across {len(per_tag_counts)} tag(s): {summary}")
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
    section = infini_atomic_config(cfg)
    rel = str(section.get("reference_dir") or "data/nsfw/local/hentai")
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def _archive_dir(cfg: Dict[str, Any]) -> Path:
    section = infini_atomic_config(cfg)
    rel = str(section.get("archive_dir") or "data/nsfw/images/infini-atomic")
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:48] or "infini"


def _save_metadata(path: Path, meta: Dict[str, Any]) -> None:
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _download_image_bytes(url: str) -> tuple:
    response = http_get(
        url,
        headers={"User-Agent": INFINI_USER_AGENT, "Accept": "image/*,*/*"},
        timeout=30,
        stream=True,
    )
    response.raise_for_status()
    content = response.content
    if len(content) < MIN_BYTES:
        raise ValueError("Downloaded infini-atomic file too small")
    if len(content) > MAX_BYTES:
        raise ValueError("Downloaded infini-atomic file too large")
    content_type = response.headers.get("Content-Type", "")
    if not validate_image_bytes(content):
        raise ValueError("Downloaded infini-atomic file is not a valid image")
    return content, content_type


def download_infini_atomic_references(
    items: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    progress: ProgressCallback = None,
) -> int:
    cfg = config or load_nsfw_config()
    if items is None:
        items = fetch_infini_atomic(config=cfg, progress=progress)
    ref_dir = _reference_dir(cfg)
    archive_dir = _archive_dir(cfg)
    site_url = _base_url(cfg)
    count = 0
    for item in items:
        if item.get("kind") != "image":
            continue
        url = str(item.get("image_url") or "").strip()
        if not url:
            continue
        remote_url = url
        tag = _safe_slug(str(item.get("infini_tag") or "hentai"))
        signature = _safe_slug(str(item.get("infini_signature") or f"{hash(url) & 0xFFFFFFFF:08x}"))
        try:
            content, content_type = _download_image_bytes(remote_url)
        except Exception as exc:
            _report(progress, f"Warning: infini-atomic download failed: {exc}")
            continue
        ext = _guess_extension(remote_url, content_type)
        filename = f"infini_{tag}_{signature}{ext}"
        dest = ref_dir / filename
        archive = archive_dir / filename
        if not dest.exists():
            dest.write_bytes(content)
        if not archive.exists():
            archive.write_bytes(content)
        item["local_reference_path"] = str(dest)
        item["image_url"] = str(dest)
        meta = {
            "source": "infini-atomic.w3spaces.com",
            "source_url": remote_url,
            "tags": item.get("tags") or [],
            "infini_tag": item.get("infini_tag"),
            "infini_is_nsfw": bool(item.get("infini_is_nsfw", True)),
            "is_nsfw": bool(item.get("infini_is_nsfw", True)),
            "infini_signature": item.get("infini_signature"),
            "media_category": item.get("media_category") or "hentai",
            "reference_path": str(dest.relative_to(get_project_root()).as_posix()),
            "archive_path": str(archive.relative_to(get_project_root()).as_posix()),
            "gallery_url": site_url,
            "site_name": "infini-atomic",
        }
        _save_metadata(dest, meta)
        count += 1
        _report(progress, f"Infini-atomic saved ({count}): {dest.name}")
        _sleep(cfg)
    return count


def fetch_and_download_infini_atomic(
    config: Optional[Dict[str, Any]] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = config or load_nsfw_config()
    html, site_ok = _fetch_site_html(cfg)
    tags = _configured_tags(cfg, html)
    items = fetch_infini_atomic(config=cfg, progress=progress)
    downloaded = download_infini_atomic_references(items=items, config=cfg, progress=progress)
    return {
        "items": items,
        "items_fetched": len(items),
        "downloaded": downloaded,
        "reference_dir": str(_reference_dir(cfg)),
        "archive_dir": str(_archive_dir(cfg)),
        "site_ok": site_ok,
        "tags_found": [entry.get("tag") for entry in tags],
        "base_url": _base_url(cfg),
    }


def crawl_infini_atomic_gallery(
    start_url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    section = infini_atomic_config(cfg)
    if start_url:
        section = dict(section)
        section["base_url"] = start_url
        cfg = dict(cfg)
        cfg["infini_atomic"] = section
    return fetch_infini_atomic(config=cfg, category=category, progress=progress)


def is_infini_atomic_url(url: str) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower().split(":")[0]
    return host == "infini-atomic.w3spaces.com" or host.endswith(".infini-atomic.w3spaces.com")


def is_infini_reference_path(path: Path) -> bool:
    if path.stem.lower().startswith("infini_"):
        return True
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source = str(meta.get("source") or "").lower()
            return "infini" in source
        except Exception:
            return False
    return False
