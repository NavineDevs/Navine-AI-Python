from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from navine.image.utils import validate_image
from navine.utils.paths import get_project_root


def aspect_bucket(width: int, height: int, buckets: Tuple[int, ...] = (128, 192, 256)) -> int:
    ratio = width / max(height, 1)
    if ratio > 1.2:
        return buckets[min(2, len(buckets) - 1)]
    if ratio < 0.8:
        return buckets[0]
    return buckets[min(1, len(buckets) - 1)]


def scan_image_folders(dirs: List[str]) -> List[Tuple[Path, str, int]]:
    root = get_project_root()
    seen: set = set()
    rows: List[Tuple[Path, str, int]] = []
    for rel in dirs:
        folder = root / rel
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            if not validate_image(path):
                continue
            seen.add(key)
            try:
                with Image.open(path) as img:
                    w, h = img.size
                bucket = aspect_bucket(w, h)
            except Exception:
                bucket = 128
            caption = _load_sidecar_caption(path)
            rows.append((path, caption, bucket))
    return rows


def _load_sidecar_caption(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            import json

            data = json.loads(sidecar.read_text(encoding="utf-8"))
            tags = data.get("tags") or []
            cap = data.get("caption") or ""
            if cap:
                return cap
            if tags:
                return ", ".join(str(t) for t in tags)
        except Exception:
            pass
    return path.stem.replace("_", " ")


def bucket_summary(dirs: List[str]) -> Dict[int, int]:
    rows = scan_image_folders(dirs)
    summary: Dict[int, int] = {}
    for _, _, bucket in rows:
        summary[bucket] = summary.get(bucket, 0) + 1
    return summary
