from pathlib import Path
from typing import Dict, Tuple

from navine.nsfw.config import load_nsfw_config
from navine.utils.paths import get_project_root

_INGEST_MARKER = "data/nsfw/.auto_ingest_state"


def _scan_local_media() -> Tuple[Dict[str, int], float]:
    root = get_project_root()
    counts = {"images": 0, "videos": 0}
    latest_mtime = 0.0
    for rel in ("data/nsfw/local", "data/nsfw/images/local"):
        directory = root / rel
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
                counts["images"] += 1
            elif suffix in {".mp4", ".webm", ".mov"}:
                counts["videos"] += 1
            else:
                continue
            try:
                latest_mtime = max(latest_mtime, path.stat().st_mtime)
            except OSError:
                continue
    return counts, latest_mtime


def _count_local_media() -> Dict[str, int]:
    counts, _ = _scan_local_media()
    return counts


def _signature(counts: Dict[str, int], latest_mtime: float) -> str:
    return f"{counts['images']}:{counts['videos']}:{latest_mtime:.0f}"


def _read_marker() -> str:
    marker = get_project_root() / _INGEST_MARKER
    try:
        return marker.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_marker(signature: str) -> None:
    marker = get_project_root() / _INGEST_MARKER
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(signature, encoding="utf-8")
    except OSError:
        pass


def auto_ingest_before_generate(force: bool = False) -> Dict[str, int]:
    cfg = load_nsfw_config()
    if not cfg.get("enabled", True):
        return {"images": 0, "videos": 0, "skipped": True}
    if not cfg.get("auto_ingest_on_generate", True) and not force:
        return {"images": 0, "videos": 0, "skipped": True}
    counts, latest_mtime = _scan_local_media()
    if counts["images"] == 0 and counts["videos"] == 0:
        return {"images": 0, "videos": 0, "skipped": True}
    signature = _signature(counts, latest_mtime)
    if not force and signature == _read_marker():
        return {"images": 0, "videos": 0, "skipped": True}
    result = {"images": 0, "videos": 0, "skipped": False}
    try:
        from navine.autolearn.sources.nsfw import (
            download_nsfw_images,
            download_nsfw_videos,
            fetch_local,
        )

        items = fetch_local(config=cfg)
        if not items:
            _write_marker(signature)
            return result
        if cfg.get("include_images", True) and counts["images"] > 0:
            result["images"] = download_nsfw_images(items)
        if cfg.get("include_videos", True) and counts["videos"] > 0:
            result["videos"] = download_nsfw_videos(items)
        _write_marker(signature)
    except Exception:
        pass
    return result
