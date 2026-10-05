from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DOWNLOADS = Path(r"C:\Users\hitbo\Downloads")
SRC = DOWNLOADS / "Navine AI - Python"

SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
}

SKIP_TOP_LEVEL_PYTHON = {"native", "services", "app"}

NON_PYTHON_SOURCE_SUFFIXES = {
    ".go",
    ".rs",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".hxx",
    ".java",
    ".kt",
    ".zig",
    ".swift",
}

NON_PYTHON_SOURCE_NAMES = {
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "CMakeLists.txt",
    "build.rs",
}

SKIP_CONFIGS = {"brand.yaml"}

CHECKPOINT_MODELS = [
    "text_enterprise",
    "text_code",
    "hitboyx23_ai",
    "hitboyx23_ai_python",
    "image_enterprise",
    "image_enterprise_v2",
    "video_enterprise",
    "image_lora",
    "voice",
    "deepfake",
]

CHECKPOINT_FILES = {
    "latest.pt",
    "best.pt",
    "tokenizer.json",
    "config.json",
    "train_report.json",
    "adapter.pt",
    ".gitkeep",
}

ROOT_FILES = (
    "requirements.txt",
    "requirements-external.txt",
    "requirements-voice.txt",
    "setup.py",
)

PROJECTS: List[Dict[str, Any]] = [
    {
        "dir": DOWNLOADS / "Navine AI - Python - Python",
        "display_name": "Navine AI - Python - Python",
        "engine_name": "Navine Engine",
        "package": "navine",
        "company": "Navine",
        "language_mode": "python",
        "python_only": True,
        "default_text_model": "text_enterprise",
        "port": 8766,
        "backend": "python",
        "bat": "Navine AI - Python - Python.bat",
        "pkg_src": "navine",
        "pkg_dst": "navine",
        "launcher_dst": "navine-launcher.ps1",
        "alias_bat": "navine.bat",
    },
    {
        "dir": DOWNLOADS / "Navuryx AI",
        "display_name": "Navuryx AI",
        "engine_name": "Navuryx Engine",
        "package": "navuryx",
        "company": "Navuryx",
        "language_mode": "multi",
        "python_only": False,
        "default_text_model": "text_enterprise",
        "port": 8767,
        "backend": "python",
        "bat": "Navuryx AI.bat",
        "pkg_src": "navine",
        "pkg_dst": "navuryx",
        "launcher_dst": "navuryx-launcher.ps1",
        "alias_bat": "navuryx.bat",
    },
    {
        "dir": DOWNLOADS / "Navuryx AI - Python",
        "display_name": "Navuryx AI - Python",
        "engine_name": "Navuryx Engine",
        "package": "navuryx",
        "company": "Navuryx",
        "language_mode": "python",
        "python_only": True,
        "default_text_model": "text_code",
        "port": 8768,
        "backend": "python",
        "bat": "Navuryx AI - Python.bat",
        "pkg_src": "navine",
        "pkg_dst": "navuryx",
        "launcher_dst": "navuryx-launcher.ps1",
        "alias_bat": "navuryx.bat",
    },
    {
        "dir": DOWNLOADS / "HitBoyXx23 AI",
        "display_name": "HitBoyXx23 AI",
        "engine_name": "HitBoyXx23 Engine",
        "package": "hitboyx23",
        "company": "none",
        "language_mode": "multi",
        "python_only": False,
        "default_text_model": "hitboyx23_ai",
        "port": 8769,
        "backend": "mixed",
        "bat": "HitBoyXx23 AI.bat",
        "pkg_src": "navine",
        "pkg_dst": "hitboyx23",
        "launcher_dst": "hitboyx23-launcher.ps1",
        "alias_bat": None,
    },
    {
        "dir": DOWNLOADS / "HitBoyXx23 AI - Python",
        "display_name": "HitBoyXx23 AI - Python",
        "engine_name": "HitBoyXx23 Engine",
        "package": "hitboyx23",
        "company": "none",
        "language_mode": "python",
        "python_only": True,
        "default_text_model": "hitboyx23_ai_python",
        "port": 8770,
        "backend": "python",
        "bat": "HitBoyXx23 AI - Python.bat",
        "pkg_src": "navine",
        "pkg_dst": "hitboyx23",
        "launcher_dst": "hitboyx23-launcher.ps1",
        "alias_bat": None,
    },
]

WEB_FILES = [
    "web/index.html",
    "web/assets/app.js",
    "web/assets/style.css",
    "web/api-docs.html",
]

SCRIPT_FILES = [
    "scripts/train_comprehensive.py",
    "scripts/run_24h_visual_improve.py",
    "scripts/sync_all_six.py",
    "scripts/propagate_six_ai_stack.py",
]


def _rewrite(text: str, spec: dict) -> str:
    src_pkg = spec["pkg_src"]
    dst_pkg = spec["pkg_dst"]
    display = spec["display_name"]
    engine = spec["engine_name"]
    if src_pkg != dst_pkg:
        text = text.replace(f"from {src_pkg} import", f"from {dst_pkg} import")
        text = text.replace(f"from {src_pkg}.", f"from {dst_pkg}.")
        text = text.replace(f"import {src_pkg} as", f"import {dst_pkg} as")
        text = text.replace(f"import {src_pkg}\n", f"import {dst_pkg}\n")
        text = text.replace(f"import {src_pkg}\r\n", f"import {dst_pkg}\r\n")
        text = text.replace(f"{src_pkg}.cli", f"{dst_pkg}.cli")
        text = text.replace(f"{src_pkg}.server", f"{dst_pkg}.server")
        text = text.replace(f"{src_pkg}.llm", f"{dst_pkg}.llm")
        text = text.replace(f"{src_pkg}.gguf_export", f"{dst_pkg}.gguf_export")
        text = text.replace(f"{src_pkg}.engine", f"{dst_pkg}.engine")
        text = text.replace(f"{src_pkg}.mcp", f"{dst_pkg}.mcp")
        text = text.replace(f"-m {src_pkg}.", f"-m {dst_pkg}.")
        text = text.replace(f'-m", "{src_pkg}.', f'-m", "{dst_pkg}.')
        text = text.replace(f"-m', '{src_pkg}.", f"-m', '{dst_pkg}.")
    if display != "Navine AI - Python":
        text = text.replace("Navine AI - Python", display)
    if engine != "Navine Engine":
        text = text.replace("Navine Engine", engine)
    return text


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def _copy_file(src: Path, dst: Path, spec: Optional[dict] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if spec and src.suffix in {".py", ".ps1", ".html", ".js", ".md", ".bat", ".yaml"}:
        text = src.read_text(encoding="utf-8", errors="replace")
        dst.write_text(_rewrite(text, spec), encoding="utf-8")
        return
    shutil.copy2(src, dst)


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _copy_with_retry(src: Path, dst: Path, retries: int = 3) -> bool:
    if _same_path(src, dst):
        return False
    for attempt in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except PermissionError:
            if attempt + 1 >= retries:
                return False
            time.sleep(0.25 * (attempt + 1))
        except OSError:
            return False
    return False


def _sync_tree(src: Path, dst: Path, spec: Optional[dict] = None) -> int:
    if not src.exists():
        return 0
    copied = 0
    skipped = 0
    for item in src.rglob("*"):
        if _should_skip(item):
            continue
        if _should_skip_python_only_file(item, spec):
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        newer = (
            not target.exists()
            or item.stat().st_size != target.stat().st_size
            or item.stat().st_mtime >= target.stat().st_mtime
        )
        if newer:
            if spec and item.suffix in {".py", ".ps1", ".html", ".js", ".md", ".bat", ".yaml"}:
                _copy_file(item, target, spec)
                copied += 1
            elif _copy_with_retry(item, target):
                copied += 1
            else:
                skipped += 1
    if skipped:
        print(f"    skipped {skipped} locked/unreadable files")
    return copied


def _brand_meta(spec: dict) -> dict:
    package = str(spec.get("package") or "navine").lower()
    display = str(spec.get("display_name") or "Navine AI - Python")
    company_override = spec.get("company")
    if "hitboy" in package or "hitboy" in display.lower():
        theme_id = "hitboy"
        company = "none"
    elif "navuryx" in package or "navuryx" in display.lower():
        theme_id = "navuryx"
        company = "Navuryx"
    else:
        theme_id = "navine"
        company = "Navine"
    if company_override is not None and str(company_override).strip() != "":
        company = str(company_override).strip()
    return {
        "theme_id": theme_id,
        "company": company,
        "creator": "HitBoyXx23",
    }


def _read_existing_public_url(dest: Path) -> str:
    path = dest / "configs" / "brand.yaml"
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.strip().startswith("public_url:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def write_brand_yaml(dest: Path, spec: dict) -> None:
    (dest / "configs").mkdir(parents=True, exist_ok=True)
    py_only = bool(spec.get("python_only")) or "- Python" in str(spec.get("display_name") or "")
    language_mode = str(spec.get("language_mode") or ("python" if py_only else "multi")).lower()
    backend = str(spec.get("backend") or "python")
    meta = _brand_meta(spec)
    public_url = str(spec.get("public_url") or _read_existing_public_url(dest) or "").strip()
    company_value = str(meta["company"]).strip() or "none"
    model_lines = [
        "  - text_enterprise",
        "  - text_code",
    ]
    if not py_only and backend == "mixed":
        model_lines.extend(["  - hitboyx23_ai", "  - hitboyx23_ai_python"])
    elif str(spec.get("package") or "") == "hitboyx23":
        model_lines.append("  - hitboyx23_ai_python" if py_only else "  - hitboyx23_ai")
        if not py_only:
            model_lines.append("  - hitboyx23_ai_python")
    model_lines.extend(
        [
            "  - image_enterprise_v2",
            "  - video_enterprise",
            "  - voice",
            "  - deepfake",
        ]
    )
    body = (
        f'display_name: "{spec["display_name"]}"\n'
        f'engine_name: "{spec["engine_name"]}"\n'
        f'package: {spec["package"]}\n'
        f'theme_id: {meta["theme_id"]}\n'
        f'creator: {meta["creator"]}\n'
        f'company: {company_value}\n'
        f"language_mode: {language_mode}\n"
        f'default_text_model: {spec["default_text_model"]}\n'
        f'port: {spec["port"]}\n'
        f"backend: {backend}\n"
    )
    if public_url:
        body += f'public_url: "{public_url}"\n'
    elif str(spec.get("display_name") or "") == "Navine AI - Python - Python":
        body += 'public_url: "https://pai.navinecord.dev"\n'
    elif str(spec.get("display_name") or "") == "Navine AI - Python":
        body += 'public_url: "https://ai.navinecord.dev"\n'
    else:
        body += 'public_url: ""\n'
    if py_only:
        body += "python_only: true\n"
        body += "ui_tabs:\n"
        body += "  - text\n"
        body += "  - voicechat\n"
        body += "  - image\n"
        body += "  - video\n"
        body += "  - osint\n"
        body += "  - deepfake\n"
        body += "  - voice\n"
        body += "  - music\n"
        body += "  - info\n"
        body += "  - train\n"
        body += "  - apigen\n"
        body += "  - settings\n"
        body += "ui_features:\n"
        body += "  train: true\n"
        body += "  apigen: true\n"
        body += "  train_strip: true\n"
        body += "  footer_ops: true\n"
        body += "  native_accelerators: false\n"
    else:
        body += "python_only: false\n"
    body += (
        "modes:\n"
        "  - chat\n"
        "  - code\n"
        "  - think\n"
        "  - detective\n"
        "  - analyze\n"
        "  - osint\n"
        "model_ids:\n"
        + "\n".join(model_lines)
        + "\n"
    )
    (dest / "configs" / "brand.yaml").write_text(body, encoding="utf-8")


def write_api_yaml(dest: Path, spec: dict) -> None:
    text = (SRC / "configs" / "api.yaml").read_text(encoding="utf-8")
    text = text.replace("port: 8765", f"port: {spec['port']}")
    keys = ".navine/api_keys.json"
    if spec["package"] == "navuryx":
        keys = ".navuryx/api_keys.json"
    elif spec["package"] == "hitboyx23":
        keys = ".hitboyx23/api_keys.json"
    text = text.replace(".navine/api_keys.json", keys)
    (dest / "configs" / "api.yaml").write_text(text, encoding="utf-8")


def write_bat(dest: Path, spec: dict) -> None:
    text = (SRC / "Navine AI - Python.bat").read_text(encoding="utf-8")
    text = text.replace('set "BRAND=Navine AI - Python"', f'set "BRAND={spec["display_name"]}"')
    text = text.replace('set "PKG=navine"', f'set "PKG={spec["package"]}"')
    text = text.replace('set "PORT=8765"', f'set "PORT={spec["port"]}"')
    (dest / spec["bat"]).write_text(text, encoding="utf-8")
    if spec.get("alias_bat"):
        alias = f'@echo off\r\ncall "%~dp0{spec["bat"]}" %*\r\n'
        (dest / spec["alias_bat"]).write_text(alias, encoding="utf-8")


def write_launch_md(dest: Path, spec: dict) -> None:
    name = spec["display_name"]
    bat = spec["bat"]
    text = f"""# Launch {name}

1. Double-click `{bat}` in this folder.
2. Click **Web UI** for chat, image, and video in the browser (`http://127.0.0.1:{spec["port"]}`).
3. Or use the desktop menu: Chat, Image, Video, Voice, Train.
4. Training:
   - GUI: **Train Everything** (full) or **Quick Train**
   - Command: `{bat} train`

## Sync with other AIs

From `Navine AI - Python`, run:

```text
python scripts/sync_all_six.py
```

This keeps code, data, checkpoints, configs, and model runners aligned across all six projects until you stop syncing.

## Modes

Chat modes: Auto, Chat, Code, Think, Detective, OSINT, Analyze. Image and video use prompt-only generation.

## First-time setup

```text
{bat} setup
```

## Troubleshooting

- `{bat} help`
- `python -m {spec["package"]}.cli doctor`
- Default text model: `{spec["default_text_model"]}`
- Port: `{spec["port"]}`
"""
    (dest / "LAUNCH.md").write_text(text, encoding="utf-8")


def write_hitboy_pyproject(dest: Path) -> None:
    text = """[build-system]
requires = ["setuptools>=61.0", "torch>=2.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "hitboyx23-ai"
version = "0.1.0"
description = "HitBoyXx23 AI - Local multimodal AI with text, image, and video models"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
    "PyYAML>=6.0",
    "tqdm>=4.65.0",
    "einops>=0.7.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.0.0",
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "opencv-python-headless>=4.8.0",
    "imageio>=2.31.0",
    "imageio-ffmpeg>=0.4.9",
]

[project.scripts]
hitboyx23 = "hitboyx23.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["hitboyx23*"]
"""
    (dest / "pyproject.toml").write_text(text, encoding="utf-8")


def patch_launcher(dest: Path, spec: dict) -> None:
    path = dest / "scripts" / spec["launcher_dst"]
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace('$BrandName = "Navine AI - Python"', f'$BrandName = "{spec["display_name"]}"')
    text = text.replace('$BrandPackage = "navine"', f'$BrandPackage = "{spec["package"]}"')
    text = text.replace("$BrandPort = 8765", f"$BrandPort = {spec['port']}")
    path.write_text(text, encoding="utf-8")


def cleanup_python_only_root(dest: Path) -> None:
    for name in SKIP_TOP_LEVEL_PYTHON:
        path = dest / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for root_name in ("navine", "navuryx", "hitboyx23"):
        root = dest / root_name
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            if item.name in NON_PYTHON_SOURCE_NAMES or item.suffix.lower() in NON_PYTHON_SOURCE_SUFFIXES:
                try:
                    item.unlink()
                except OSError:
                    pass


def _should_skip_python_only_file(path: Path, spec: Optional[dict]) -> bool:
    if not spec:
        return False
    py_only = bool(spec.get("python_only")) or "- Python" in str(spec.get("display_name") or "")
    if not py_only:
        return False
    if path.name in NON_PYTHON_SOURCE_NAMES:
        return True
    if path.suffix.lower() in NON_PYTHON_SOURCE_SUFFIXES:
        return True
    return False


def sync_code(dest: Path, spec: dict) -> None:
    pkg_src = SRC / spec["pkg_src"]
    pkg_dst = dest / spec["pkg_dst"]
    count = _sync_tree(pkg_src, pkg_dst, spec)
    init = pkg_dst / "__init__.py"
    init.write_text(
        f'__version__ = "0.2.2"\n__name__ = "{spec["display_name"]}"\n\n'
        f"from {spec['package']}.policy import enforce_local_only\n\nenforce_local_only()\n",
        encoding="utf-8",
    )
    for rel in WEB_FILES + SCRIPT_FILES:
        src = SRC / rel
        if src.exists():
            _copy_file(src, dest / rel, spec)
    launcher_src = SRC / "scripts" / "navine-launcher.ps1"
    if launcher_src.exists():
        _copy_file(launcher_src, dest / "scripts" / spec["launcher_dst"], spec)
    patch_launcher(dest, spec)
    write_brand_yaml(dest, spec)
    write_api_yaml(dest, spec)
    write_launch_md(dest, spec)
    write_bat(dest, spec)
    if spec["package"] == "hitboyx23":
        write_hitboy_pyproject(dest)
    else:
        src_py = SRC / "pyproject.toml"
        if src_py.exists():
            shutil.copy2(src_py, dest / "pyproject.toml")
    for name in ROOT_FILES:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, dest / name)
    docs = dest / "docs"
    docs.mkdir(exist_ok=True)
    live = SRC / "docs" / "LIVE_PARAM_COUNTS.md"
    if live.exists():
        shutil.copy2(live, docs / "LIVE_PARAM_COUNTS.md")
    print(f"  code: {count} files updated in {spec['display_name']}")


def sync_configs(dest: Path, spec: dict) -> None:
    cfg_src = SRC / "configs"
    cfg_dst = dest / "configs"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in cfg_src.glob("*.yaml"):
        if src.name in SKIP_CONFIGS:
            continue
        if src.name == "api.yaml":
            write_api_yaml(dest, spec)
            copied += 1
            continue
        dst = cfg_dst / src.name
        _copy_file(src, dst, spec if spec["pkg_src"] != spec["pkg_dst"] else None)
        copied += 1
    train_src = cfg_src / "train"
    train_dst = cfg_dst / "train"
    if train_src.exists():
        copied += _sync_tree(train_src, train_dst, spec if spec["pkg_src"] != spec["pkg_dst"] else None)
    write_brand_yaml(dest, spec)
    print(f"  configs: {copied} items -> {spec['display_name']}")


def _robocopy_mirror(src: Path, dst: Path) -> int:
    if os.name != "nt":
        return _sync_tree(src, dst)
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "robocopy",
        str(src),
        str(dst),
        "/E",
        "/XO",
        "/XJ",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode >= 8:
        print(f"    robocopy warning ({result.returncode}): {(result.stderr or result.stdout or '').strip()[:200]}")
    try:
        return sum(1 for _ in dst.rglob("*") if _.is_file())
    except Exception:
        return 0


LIGHT_DATA = ("text", "train", "learn", "autolearn", "conversations", "unrestricted", "corpus")
SHARED_DATA = ("nsfw", "image", "video")


def _junction(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)


def sync_data(dest: Path, spec: dict) -> None:
    src_data = SRC / "data"
    dst_data = dest / "data"
    dst_data.mkdir(parents=True, exist_ok=True)
    copied = 0
    for folder in LIGHT_DATA:
        src = src_data / folder
        if src.exists():
            copied += _robocopy_mirror(src, dst_data / folder)
    for folder in SHARED_DATA:
        src = src_data / folder
        dst = dst_data / folder
        if not src.exists():
            continue
        if dst.exists() and not dst.is_symlink() and dst.resolve() != src.resolve():
            shutil.rmtree(dst, ignore_errors=True)
        _junction(dst, src)
    print(f"  data: light copy ~{copied} files; nsfw/image/video linked -> {spec['display_name']}")


def _hardlink_or_copy(src: Path, dst: Path) -> bool:
    if _same_path(src, dst):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if dst.stat().st_ino == src.stat().st_ino and dst.stat().st_dev == src.stat().st_dev:
                return False
        except OSError:
            pass
        try:
            if _same_path(src, dst):
                return False
            dst.unlink()
        except OSError:
            pass
    try:
        os.link(str(src), str(dst))
        return True
    except OSError:
        return _copy_with_retry(src, dst)


def sync_checkpoints(dest: Path, spec: dict) -> None:
    copied = 0
    src_ck = SRC / "checkpoints"
    dst_ck = dest / "checkpoints"
    if _same_path(src_ck, dst_ck):
        print(f"  checkpoints: skip (junction/same path) -> {spec['display_name']}")
        return
    dst_ck.mkdir(parents=True, exist_ok=True)
    for model in CHECKPOINT_MODELS:
        src_dir = src_ck / model
        if not src_dir.exists():
            continue
        dst_dir = dst_ck / model
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_file in src_dir.iterdir():
            if src_file.is_dir():
                continue
            if src_file.name not in CHECKPOINT_FILES and src_file.suffix not in {".pt", ".json"}:
                continue
            dst_file = dst_dir / src_file.name
            if src_file.suffix == ".pt":
                if _hardlink_or_copy(src_file, dst_file):
                    copied += 1
            elif not dst_file.exists() or src_file.stat().st_mtime >= dst_file.stat().st_mtime:
                if _copy_with_retry(src_file, dst_file):
                    copied += 1
    print(f"  checkpoints: {copied} files -> {spec['display_name']}")


def _python_for(dest: Path) -> List[str]:
    venv = dest / "venv" / "Scripts" / "python.exe"
    if venv.exists():
        return [str(venv)]
    return [sys.executable]


def pack_models(dest: Path, spec: dict) -> None:
    for py in _python_for(dest):
        try:
            result = subprocess.run(
                [py, "-m", f"{spec['package']}.cli", "llm", "pack"],
                cwd=str(dest),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"  models: packed -> {spec['display_name']}")
                return
            err = (result.stderr or result.stdout or "").strip()
            print(f"  models: pack failed for {spec['display_name']}: {err[:200]}")
            return
        except Exception as exc:
            print(f"  models: pack error for {spec['display_name']}: {exc}")
            return


def sync_project(spec: dict, *, code: bool, configs: bool, data: bool, checkpoints: bool, models: bool) -> dict:
    dest = spec["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n== {spec['display_name']} ==")
    if code:
        sync_code(dest, spec)
        if "- Python" in str(spec.get("display_name") or "") or spec.get("python_only"):
            cleanup_python_only_root(dest)
    if configs:
        sync_configs(dest, spec)
    if data:
        sync_data(dest, spec)
    if checkpoints:
        sync_checkpoints(dest, spec)
    if models:
        pack_models(dest, spec)
    ready = 0
    for model in CHECKPOINT_MODELS[:6]:
        if (dest / "checkpoints" / model / "latest.pt").exists():
            ready += 1
    runners = len(list((dest / "models").glob("*.py"))) if (dest / "models").exists() else 0
    return {
        "name": spec["display_name"],
        "dir": str(dest),
        "checkpoints_ready": ready,
        "model_runners": runners,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync all six AI projects from Navine AI - Python")
    parser.add_argument("--code", action="store_true", help="Sync Python package, web, scripts, launchers")
    parser.add_argument("--configs", action="store_true", help="Sync configs and train configs")
    parser.add_argument("--data", action="store_true", help="Sync data directory")
    parser.add_argument("--checkpoints", action="store_true", help="Sync checkpoint weights")
    parser.add_argument("--models", action="store_true", help="Run llm pack on each project")
    parser.add_argument("--all", action="store_true", help="Sync everything (default)")
    args = parser.parse_args()

    if not any((args.code, args.configs, args.data, args.checkpoints, args.models, args.all)):
        args.all = True

    do_code = args.all or args.code
    do_configs = args.all or args.configs
    do_data = args.all or args.data
    do_checkpoints = args.all or args.checkpoints
    do_models = args.all or args.models

    started = datetime.now(timezone.utc).isoformat()
    print(f"Sync source: {SRC}")
    print(f"Targets: {len(PROJECTS)}")

    rows = [
        sync_project(
            spec,
            code=do_code,
            configs=do_configs,
            data=do_data,
            checkpoints=do_checkpoints,
            models=do_models,
        )
        for spec in PROJECTS
    ]

    if do_models:
        pack_models(SRC, {
            "display_name": "Navine AI - Python",
            "package": "navine",
            "dir": SRC,
        })

    summary = {
        "started": started,
        "ended": datetime.now(timezone.utc).isoformat(),
        "source": str(SRC),
        "projects": rows,
    }
    log_dir = SRC / "logs" / "sync_all_six"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = log_dir / f"sync_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {out}")
    for row in rows:
        print(f"  {row['name']}: {row['checkpoints_ready']}/6 checkpoints, {row['model_runners']} runners")


if __name__ == "__main__":
    main()
