from __future__ import annotations

import shutil
from pathlib import Path

DOWNLOADS = Path(r"C:\Users\hitbo\Downloads")
SRC = DOWNLOADS / "Navine AI - Python"

PROJECTS = [
    {
        "dir": DOWNLOADS / "Navine AI - Python - Python",
        "display_name": "Navine AI - Python - Python",
        "engine_name": "Navine Engine",
        "package": "navine",
        "default_text_model": "text_code",
        "port": 8766,
        "backend": "python",
        "bat": "Navine AI - Python - Python.bat",
        "pkg_src": "navine",
        "pkg_dst": "navine",
        "launcher_dst": "navine-launcher.ps1",
        "copy_full_pkg": False,
        "alias_bat": "navine.bat",
    },
    {
        "dir": DOWNLOADS / "Navuryx AI",
        "display_name": "Navuryx AI",
        "engine_name": "Navuryx Engine",
        "package": "navuryx",
        "default_text_model": "text_enterprise",
        "port": 8767,
        "backend": "python",
        "bat": "Navuryx AI.bat",
        "pkg_src": "navine",
        "pkg_dst": "navuryx",
        "launcher_dst": "navuryx-launcher.ps1",
        "copy_full_pkg": False,
        "alias_bat": "navuryx.bat",
    },
    {
        "dir": DOWNLOADS / "Navuryx AI - Python",
        "display_name": "Navuryx AI - Python",
        "engine_name": "Navuryx Engine",
        "package": "navuryx",
        "default_text_model": "text_code",
        "port": 8768,
        "backend": "python",
        "bat": "Navuryx AI - Python.bat",
        "pkg_src": "navine",
        "pkg_dst": "navuryx",
        "launcher_dst": "navuryx-launcher.ps1",
        "copy_full_pkg": False,
        "alias_bat": "navuryx.bat",
    },
    {
        "dir": DOWNLOADS / "HitBoyXx23 AI",
        "display_name": "HitBoyXx23 AI",
        "engine_name": "HitBoyXx23 Engine",
        "package": "hitboyx23",
        "default_text_model": "hitboyx23_ai",
        "port": 8769,
        "backend": "mixed",
        "bat": "HitBoyXx23 AI.bat",
        "pkg_src": "navine",
        "pkg_dst": "hitboyx23",
        "launcher_dst": "hitboyx23-launcher.ps1",
        "copy_full_pkg": True,
        "alias_bat": None,
    },
    {
        "dir": DOWNLOADS / "HitBoyXx23 AI - Python",
        "display_name": "HitBoyXx23 AI - Python",
        "engine_name": "HitBoyXx23 Engine",
        "package": "hitboyx23",
        "default_text_model": "hitboyx23_ai_python",
        "port": 8770,
        "backend": "python",
        "bat": "HitBoyXx23 AI - Python.bat",
        "pkg_src": "navine",
        "pkg_dst": "hitboyx23",
        "launcher_dst": "hitboyx23-launcher.ps1",
        "copy_full_pkg": True,
        "alias_bat": None,
    },
]

REBRAND_FILES = [
    "utils/brand.py",
    "api/settings.py",
    "api/routes.py",
    "api/schemas.py",
    "api/chat_utils.py",
    "server.py",
    "text/infer.py",
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
]


def _rewrite(text: str, src_pkg: str, dst_pkg: str, display: str, engine: str) -> str:
    if src_pkg == dst_pkg:
        return text
    text = text.replace(f"from {src_pkg}.", f"from {dst_pkg}.")
    text = text.replace(f"import {src_pkg}", f"import {dst_pkg}")
    text = text.replace(f"{src_pkg}.cli", f"{dst_pkg}.cli")
    text = text.replace(f"{src_pkg}.server", f"{dst_pkg}.server")
    text = text.replace(f"{src_pkg}.llm", f"{dst_pkg}.llm")
    text = text.replace(f"{src_pkg}.gguf_export", f"{dst_pkg}.gguf_export")
    text = text.replace("Navine Engine", engine)
    text = text.replace("Navine AI - Python", display)
    text = text.replace('"package": "navine"', f'"package": "{dst_pkg}"')
    text = text.replace("package: navine", f"package: {dst_pkg}")
    return text


def _copy_file(src: Path, dst: Path, spec: dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8", errors="replace")
    if src.suffix in {".py", ".ps1", ".html", ".js", ".md", ".bat", ".yaml"}:
        text = _rewrite(text, spec["pkg_src"], spec["pkg_dst"], spec["display_name"], spec["engine_name"])
    dst.write_text(text, encoding="utf-8")


def _copy_tree_py(src_dir: Path, dst_dir: Path, spec: dict) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.rglob("*"):
        if "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix == ".py":
            _copy_file(src, dst, spec)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_brand_yaml(dest: Path, spec: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "configs").mkdir(exist_ok=True)
    body = (
        f'display_name: "{spec["display_name"]}"\n'
        f'engine_name: "{spec["engine_name"]}"\n'
        f'package: {spec["package"]}\n'
        f'default_text_model: {spec["default_text_model"]}\n'
        f'port: {spec["port"]}\n'
        f'backend: {spec["backend"]}\n'
        "modes:\n"
        "  - chat\n"
        "  - code\n"
        "  - think\n"
        "  - detective\n"
        "  - analyze\n"
        "model_ids:\n"
        "  - text_enterprise\n"
        "  - text_code\n"
        "  - hitboyx23_ai\n"
        "  - hitboyx23_ai_python\n"
        "  - image_enterprise\n"
        "  - video_enterprise\n"
    )
    (dest / "configs" / "brand.yaml").write_text(body, encoding="utf-8")


def write_api_yaml(dest: Path, spec: dict) -> None:
    src = SRC / "configs" / "api.yaml"
    text = src.read_text(encoding="utf-8")
    text = text.replace("port: 8765", f"port: {spec['port']}")
    keys = ".navine/api_keys.json"
    if spec["package"] == "navuryx":
        keys = ".navuryx/api_keys.json"
    elif spec["package"] == "hitboyx23":
        keys = ".hitboyx23/api_keys.json"
    text = text.replace(".navine/api_keys.json", keys)
    (dest / "configs").mkdir(exist_ok=True)
    (dest / "configs" / "api.yaml").write_text(text, encoding="utf-8")


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

## Modes

Chat, Code, Think, Detective (puzzles/ciphers), Analyze. Image and video have the same mode dropdown.

## First-time setup

```text
{bat} setup
```

Creates `venv` and installs `requirements.txt`. Then double-click `{bat}`.

## Troubleshooting

- `{bat} help` prints launcher commands
- `python -m {spec["package"]}.cli doctor` runs health checks
- Default text model: `{spec["default_text_model"]}`
- Port: `{spec["port"]}` (see `configs/brand.yaml`)
"""
    (dest / "LAUNCH.md").write_text(text, encoding="utf-8")


def write_bat(dest: Path, spec: dict) -> None:
    src_bat = SRC / "Navine AI - Python.bat"
    text = src_bat.read_text(encoding="utf-8")
    text = text.replace("set \"BRAND=Navine AI - Python\"", f"set \"BRAND={spec['display_name']}\"")
    text = text.replace("set \"PKG=navine\"", f"set \"PKG={spec['package']}\"")
    text = text.replace("set \"PORT=8765\"", f"set \"PORT={spec['port']}\"")
    (dest / spec["bat"]).write_text(text, encoding="utf-8")
    if spec.get("alias_bat"):
        alias = f'@echo off\r\ncall "%~dp0{spec["bat"]}" %*\r\n'
        (dest / spec["alias_bat"]).write_text(alias, encoding="utf-8")


def copy_data_if_missing(dest: Path) -> None:
    src_data = SRC / "data"
    dst_data = dest / "data"
    if not src_data.exists():
        return
    if not dst_data.exists():
        shutil.copytree(src_data, dst_data, dirs_exist_ok=True)
        return
    for sub in ("train", "text", "learn"):
        s = src_data / sub
        d = dst_data / sub
        if s.exists() and not d.exists():
            shutil.copytree(s, d, dirs_exist_ok=True)


def copy_requirements(dest: Path) -> None:
    for name in ("requirements.txt", "requirements-external.txt", "requirements-voice.txt", "pyproject.toml"):
        src = SRC / name
        if src.exists() and not (dest / name).exists():
            shutil.copy2(src, dest / name)


def apply_project(spec: dict) -> None:
    dest: Path = spec["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    pkg_dst = dest / spec["pkg_dst"]
    pkg_src = SRC / spec["pkg_src"]

    if spec["copy_full_pkg"]:
        _copy_tree_py(pkg_src, pkg_dst, spec)
        init = pkg_dst / "__init__.py"
        init.write_text(
            f'__version__ = "0.1.0"\n__name__ = "{spec["display_name"]}"\n\n'
            f"from {spec['package']}.policy import enforce_local_only\n\nenforce_local_only()\n",
            encoding="utf-8",
        )
    else:
        for rel in REBRAND_FILES:
            src = pkg_src / rel
            dst = pkg_dst / rel
            if src.exists():
                _copy_file(src, dst, spec)
        for extra in ("api/ollama_routes.py", "api/openai_routes.py", "llm.py", "gguf_export.py"):
            src = pkg_src / extra
            dst = pkg_dst / extra
            if src.exists() and (spec["copy_full_pkg"] or not dst.exists() or extra.startswith("api/")):
                _copy_file(src, dst, spec)
        src = pkg_src / "utils" / "brand.py"
        if src.exists():
            _copy_file(src, pkg_dst / "utils" / "brand.py", spec)

    for rel in WEB_FILES:
        src = SRC / rel
        if src.exists():
            _copy_file(src, dest / rel, spec)

    (dest / "scripts").mkdir(exist_ok=True)
    for rel in SCRIPT_FILES:
        src = SRC / rel
        if src.exists():
            _copy_file(src, dest / rel, spec)

    launcher_src = SRC / "scripts" / "navine-launcher.ps1"
    _copy_file(launcher_src, dest / "scripts" / spec["launcher_dst"], spec)

    write_brand_yaml(dest, spec)
    write_api_yaml(dest, spec)
    write_launch_md(dest, spec)
    write_bat(dest, spec)
    copy_requirements(dest)
    if spec["copy_full_pkg"]:
        copy_data_if_missing(dest)
        for cfg in ("assistant.yaml", "prompts.yaml", "navine.yaml", "voice.yaml"):
            s = SRC / "configs" / cfg
            d = dest / "configs" / cfg
            if s.exists() and not d.exists():
                shutil.copy2(s, d)
        docs = dest / "docs"
        docs.mkdir(exist_ok=True)
        live = SRC / "docs" / "LIVE_PARAM_COUNTS.md"
        if live.exists():
            shutil.copy2(live, docs / "LIVE_PARAM_COUNTS.md")

    print(f"OK {dest} -> {spec['bat']} :{spec['port']}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Propagate Navine AI - Python stack to sibling projects")
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help="Run scripts/sync_all_six.py (code + data + checkpoints + configs + models)",
    )
    args = parser.parse_args()
    if args.full_sync:
        sync_script = SRC / "scripts" / "sync_all_six.py"
        import subprocess
        import sys

        raise SystemExit(subprocess.call([sys.executable, str(sync_script), "--all"], cwd=str(SRC)))
    for spec in PROJECTS:
        apply_project(spec)


if __name__ == "__main__":
    main()
