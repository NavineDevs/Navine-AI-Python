from __future__ import annotations

import base64
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from navine.utils.paths import get_project_root


def _safe_arcname(name: str) -> str:
    raw = (name or "file.txt").replace("\\", "/").split("/")[-1].strip()
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in "._- ")[:160].strip()
    return safe or "file.txt"


def create_zip_archive(
    files: List[Dict[str, str]],
    zip_name: Optional[str] = None,
) -> Tuple[Path, str]:
    if not files:
        raise ValueError("No files to zip")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = _safe_arcname(zip_name or "archive.zip")
    if not base.lower().endswith(".zip"):
        base = f"{base}.zip"
    out_dir = get_project_root() / "outputs" / "files"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{stamp}_{base}"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        used: set[str] = set()
        for row in files:
            arc = _safe_arcname(str(row.get("path") or row.get("name") or "file.txt"))
            if arc in used:
                stem = Path(arc).stem
                suffix = Path(arc).suffix or ".txt"
                n = 2
                while f"{stem}_{n}{suffix}" in used:
                    n += 1
                arc = f"{stem}_{n}{suffix}"
            used.add(arc)
            content = str(row.get("content") or "")
            zf.writestr(arc, content.encode("utf-8"))
    return dest, base


def zip_response_payload(path: Path, filename: str) -> Dict[str, Any]:
    data = path.read_bytes()
    return {
        "ok": True,
        "path": str(path),
        "filename": filename,
        "mime": "application/zip",
        "size": len(data),
        "download_url": f"/api/files/download?path={quote(path.as_posix())}",
        "data_base64": base64.b64encode(data).decode("ascii") if len(data) < 8_000_000 else None,
    }


def extract_code_files_from_text(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    for idx, match in enumerate(pattern.finditer(text or ""), start=1):
        lang = (match.group(1) or "txt").strip().lower() or "txt"
        code = match.group(2) or ""
        if not code.strip():
            continue
        ext_map = {
            "python": "py",
            "py": "py",
            "javascript": "js",
            "js": "js",
            "typescript": "ts",
            "ts": "ts",
            "html": "html",
            "css": "css",
            "json": "json",
            "shell": "sh",
            "bash": "sh",
            "sh": "sh",
            "markdown": "md",
            "md": "md",
        }
        ext = ext_map.get(lang, lang if lang.isalnum() and len(lang) <= 6 else "txt")
        rows.append({"path": f"file_{idx}.{ext}", "content": code})
    return rows


def parse_zip_request(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(?:make|create|build|export|download)\b.*\bzip\b", text):
        return True
    if re.search(r"\bzip\b.*\b(?:files?|project|folder|archive)\b", text):
        return True
    if re.search(r"^zip\s+(?:these|the|my)\b", text):
        return True
    return False
