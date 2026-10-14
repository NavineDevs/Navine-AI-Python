import base64
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from navine.utils.paths import get_project_root


def _decode_base64_payload(data: str) -> bytes:
    raw = str(data or "").strip()
    if not raw:
        raise ValueError("Empty upload payload")
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw)
    except Exception as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc


def _upload_dir(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = get_project_root() / "outputs" / "uploads" / name / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload_image(data: str, prefix: str = "image", ext: str = ".png") -> Path:
    payload = _decode_base64_payload(data)
    if not payload:
        raise ValueError("Empty image payload")
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prefix)[:48] or "image"
    stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    out = _upload_dir("images") / f"{safe}_{stamp}{ext}"
    out.write_bytes(payload)
    return out


def save_upload_audio(data: str, prefix: str = "audio", ext: str = ".wav") -> Path:
    payload = _decode_base64_payload(data)
    if not payload:
        raise ValueError("Empty audio payload")
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prefix)[:48] or "audio"
    stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    out = _upload_dir("audio") / f"{safe}_{stamp}{ext}"
    out.write_bytes(payload)
    return out


def resolve_image_input(
    path: Optional[str] = None,
    data_base64: Optional[str] = None,
    *,
    label: str = "image",
) -> Path:
    if data_base64:
        return save_upload_image(data_base64, prefix=label)
    if path:
        candidate = Path(str(path).strip()).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"{label} file not found: {candidate}")
    raise ValueError(f"{label} is required")
