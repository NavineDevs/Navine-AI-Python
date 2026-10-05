import logging
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _magic_kind(data: bytes) -> Optional[str]:
    if len(data) < 12:
        return None
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"BM"):
        return "bmp"
    return None


def is_html_content(data: bytes) -> bool:
    head = data[:512].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or head.startswith(b"<?xml")


def validate_image_bytes(content: bytes) -> bool:
    if not content or len(content) < 64:
        return False
    if is_html_content(content):
        return False
    if _magic_kind(content) is None:
        return False
    try:
        with Image.open(BytesIO(content)) as img:
            img.verify()
        with Image.open(BytesIO(content)) as img:
            img.load()
        return True
    except Exception:
        return False


def validate_image(path: Union[str, Path]) -> bool:
    target = Path(path)
    try:
        if not target.is_file():
            return False
        return validate_image_bytes(target.read_bytes())
    except Exception:
        return False


def quarantine_path() -> Path:
    from navine.utils.paths import get_project_root

    path = get_project_root() / "data" / "nsfw" / "quarantine"
    path.mkdir(parents=True, exist_ok=True)
    return path


def quarantine_file(path: Union[str, Path], reason: str = "") -> Optional[Path]:
    source = Path(path)
    if not source.is_file():
        return None
    dest_dir = quarantine_path()
    dest = dest_dir / source.name
    if dest.exists():
        stem = source.stem
        suffix = source.suffix
        index = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{index}{suffix}"
            index += 1
    try:
        shutil.move(str(source), str(dest))
        meta = source.with_suffix(".json")
        if meta.exists():
            meta_dest = dest.with_suffix(".json")
            if not meta_dest.exists():
                shutil.move(str(meta), str(meta_dest))
        if reason:
            logger.warning("Quarantined invalid image %s: %s", source, reason)
        return dest
    except Exception:
        try:
            source.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def filter_valid_image_paths(
    paths: List[Path],
    quarantine: bool = False,
) -> Tuple[List[Path], int]:
    valid: List[Path] = []
    skipped = 0
    for path in paths:
        if validate_image(path):
            valid.append(path)
        else:
            skipped += 1
            if quarantine:
                quarantine_file(path, reason="invalid image")
    return valid, skipped


def scan_directory_for_invalid_images(
    root: Union[str, Path],
    quarantine: bool = True,
) -> Tuple[int, int, List[str]]:
    root_path = Path(root)
    if not root_path.exists():
        return 0, 0, []
    checked = 0
    invalid = 0
    quarantined: List[str] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.parent.name == "quarantine":
            continue
        checked += 1
        if not validate_image(path):
            invalid += 1
            if quarantine:
                moved = quarantine_file(path, reason="scan invalid")
                if moved:
                    quarantined.append(str(moved))
            else:
                quarantined.append(str(path))
    return checked, invalid, quarantined
