import os
from pathlib import Path
from typing import Optional, Tuple

from navine.utils.paths import get_project_root as _get_project_root

STATIC_TOP_DIRS = ("web", "configs", "scripts", "app")


def allowed_top_dirs() -> tuple:
    try:
        from navine.utils.brand import brand_package

        return (brand_package(),) + STATIC_TOP_DIRS
    except Exception:
        return ("navine",) + STATIC_TOP_DIRS

BLOCKED_BASENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
}

BLOCKED_EXTENSIONS = {".pem", ".key", ".p12", ".pfx"}

MAX_READ_BYTES = 120_000


def get_project_root() -> Path:
    return _get_project_root()


def _normalize_relative_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def resolve_safe_path(relative_path: str) -> Tuple[Optional[Path], Optional[str]]:
    root = get_project_root().resolve()
    normalized = _normalize_relative_path(relative_path)
    if not normalized:
        return None, "Path is empty"
    if ".." in normalized.split("/"):
        return None, "Path traversal is not allowed"
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, "Path is outside the project root"
    top = normalized.split("/")[0]
    allowed = allowed_top_dirs()
    if top not in allowed:
        return None, f"Reading is only allowed under: {', '.join(allowed)}"
    basename = candidate.name.lower()
    if basename in BLOCKED_BASENAMES or basename.startswith(".env"):
        return None, "Refusing to read secret or environment files"
    if candidate.suffix.lower() in BLOCKED_EXTENSIONS:
        return None, "Refusing to read key or certificate files"
    return candidate, None


def read_file(path: str, max_chars: Optional[int] = None) -> dict:
    resolved, error = resolve_safe_path(path)
    if error or resolved is None:
        return {"ok": False, "path": path, "error": error or "Invalid path"}
    if not resolved.exists():
        return {"ok": False, "path": path, "error": "File not found"}
    if not resolved.is_file():
        return {"ok": False, "path": path, "error": "Path is not a file"}
    try:
        size = resolved.stat().st_size
        if size > MAX_READ_BYTES:
            return {
                "ok": False,
                "path": path,
                "error": f"File too large ({size} bytes, max {MAX_READ_BYTES})",
            }
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "path": path, "error": str(exc)}
    limit = max_chars if max_chars is not None else 6000
    truncated = len(text) > limit
    if truncated:
        text = text[:limit]
    rel = _normalize_relative_path(str(resolved.relative_to(get_project_root().resolve())))
    return {
        "ok": True,
        "path": rel.replace("\\", "/"),
        "content": text,
        "truncated": truncated,
        "size": size,
    }


def is_searchable_dir(path: Path) -> bool:
    name = path.name.lower()
    if name in {".git", ".venv", "venv", "node_modules", "__pycache__", "checkpoints", "data"}:
        return False
    if name.startswith(".") and name not in {".github"}:
        return False
    return True


def iter_source_files(base: Path):
    root = get_project_root().resolve()
    for dirpath, dirnames, filenames in os.walk(base):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if is_searchable_dir(current / d)]
        for filename in filenames:
            full = current / filename
            try:
                full.relative_to(root)
            except ValueError:
                continue
            rel = _normalize_relative_path(str(full.relative_to(root)))
            top = rel.split("/")[0]
            if top not in allowed_top_dirs():
                continue
            lower = filename.lower()
            if lower in BLOCKED_BASENAMES or lower.startswith(".env"):
                continue
            if full.suffix.lower() in BLOCKED_EXTENSIONS:
                continue
            if full.suffix.lower() not in {
                ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
                ".toml", ".md", ".html", ".css", ".sh", ".bat", ".ps1", ".cfg", ".ini",
            }:
                continue
            yield full
