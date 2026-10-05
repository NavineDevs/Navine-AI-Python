import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine import __version__
from navine.utils.brand import brand_name
from navine.utils.paths import get_project_root


def downloads_dir() -> Path:
    path = get_project_root() / "web" / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_tauri_exe() -> Optional[Path]:
    root = get_project_root()
    candidates = [
        root / "app" / "src-tauri" / "target" / "release" / "navine-ai.exe",
        root / "app" / "src-tauri" / "target" / "release" / "bundle" / "nsis" / "Navine AI - Python_0.1.0_x64-setup.exe",
        root / "dist" / "NavineAI.exe",
        root / "web" / "downloads" / "NavineAI.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    release_dir = root / "app" / "src-tauri" / "target" / "release" / "bundle"
    if release_dir.is_dir():
        for path in sorted(release_dir.rglob("*.exe")):
            if path.is_file() and path.stat().st_size > 100_000:
                return path
    return None


def _cli_readme() -> str:
    name = brand_name()
    return (
        f"{name} CLI\n"
        f"Version {__version__}\n\n"
        "Works like Claude / ChatGPT CLI:\n"
        "  navine                  open interactive chat\n"
        "  navine \"hello\"          one-shot prompt\n"
        "  navine chat             same as bare navine\n"
        "  navine train list       tool commands\n\n"
        "Install (Windows):\n"
        "  1. Extract this zip\n"
        "  2. Right-click Install.ps1 -> Run with PowerShell\n"
        "     or: powershell -ExecutionPolicy Bypass -File Install.ps1\n"
        "  3. Open a new terminal and type: navine\n\n"
        "Manual:\n"
        "  Add the extracted bin folder to your PATH, or run navine.cmd from that folder.\n"
        "  Requires this Navine AI - Python project + venv on the machine.\n"
    )


def _app_readme() -> str:
    name = brand_name()
    return (
        f"{name} Desktop App (Rust / Tauri)\n"
        f"Version {__version__}\n\n"
        "This package contains the native Windows desktop app built with Rust (Tauri).\n"
        "It starts the local Navine server and opens the UI in a desktop window.\n\n"
        "1. Extract the zip\n"
        "2. Double-click NavineAI.exe\n"
        "3. Keep the Navine AI - Python project folder nearby (same PC) so models/venv resolve\n\n"
        "You can also set NAVINE_ROOT to your project path before launching.\n"
    )


def _cli_cmd_script() -> str:
    return (
        "@echo off\n"
        "setlocal EnableExtensions\n"
        "set \"BIN=%~dp0\"\n"
        "set \"ROOT=%NAVINE_ROOT%\"\n"
        "if not defined ROOT if exist \"%BIN%..\\..\\venv\\Scripts\\python.exe\" set \"ROOT=%BIN%..\\..\\\"\n"
        "if not defined ROOT if exist \"%BIN%..\\venv\\Scripts\\python.exe\" set \"ROOT=%BIN%..\\\"\n"
        "if not defined ROOT if exist \"%CD%\\venv\\Scripts\\python.exe\" set \"ROOT=%CD%\\\"\n"
        "if defined ROOT (\n"
        "  if exist \"%ROOT%venv\\Scripts\\python.exe\" (\n"
        "    set \"PY=%ROOT%venv\\Scripts\\python.exe\"\n"
        "  ) else (\n"
        "    set \"PY=python\"\n"
        "  )\n"
        "  cd /d \"%ROOT%\"\n"
        ") else (\n"
        "  set \"PY=python\"\n"
        ")\n"
        "\"%PY%\" -m navine %*\n"
        "exit /b %ERRORLEVEL%\n"
    )


def _cli_install_ps1() -> str:
    return r"""$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bin = Join-Path $Here "bin"
New-Item -ItemType Directory -Force -Path $Bin | Out-Null
Copy-Item -Force (Join-Path $Here "navine.cmd") (Join-Path $Bin "navine.cmd")
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $UserPath) { $UserPath = "" }
$parts = $UserPath -split ";" | Where-Object { $_ -and $_.Trim() -ne "" }
if ($parts -notcontains $Bin) {
  $parts += $Bin
  [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
  $env:Path = "$Bin;$env:Path"
}
Write-Host "Installed navine CLI."
Write-Host "Open a NEW terminal and type: navine"
Write-Host "Optional: set NAVINE_ROOT to your Navine AI - Python project folder."
"""


def _write_cli_zip(dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("NavineAI-CLI/README.txt", _cli_readme())
        zf.writestr("NavineAI-CLI/navine.cmd", _cli_cmd_script())
        zf.writestr("NavineAI-CLI/bin/navine.cmd", _cli_cmd_script())
        zf.writestr("NavineAI-CLI/Install.ps1", _cli_install_ps1())
    return dest


def _write_app_zip(dest: Path) -> Path:
    root = get_project_root()
    exe = _find_tauri_exe()
    nsis = (
        root
        / "app"
        / "src-tauri"
        / "target"
        / "release"
        / "bundle"
        / "nsis"
        / "Navine AI - Python_0.1.0_x64-setup.exe"
    )
    msi = (
        root
        / "app"
        / "src-tauri"
        / "target"
        / "release"
        / "bundle"
        / "msi"
        / "Navine AI - Python_0.1.0_x64_en-US.msi"
    )
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("NavineAI-App/README.txt", _app_readme())
        if exe is not None:
            zf.write(exe, "NavineAI-App/NavineAI.exe")
        if nsis.is_file():
            zf.write(nsis, "NavineAI-App/NavineAI-Setup.exe")
        if msi.is_file():
            zf.write(msi, "NavineAI-App/NavineAI.msi")
        if exe is None and not nsis.is_file() and not msi.is_file():
            zf.writestr(
                "NavineAI-App/BUILD_PENDING.txt",
                (
                    "Native Rust app binary was not found yet.\n"
                    "Build it with:\n"
                    "  cd app\n"
                    "  npm run build\n"
                    "Then re-open Settings or restart the server to refresh this package.\n"
                ),
            )
            stub = (
                "@echo off\n"
                "echo Navine AI - Python desktop app is not built yet.\n"
                "echo Run: cd app ^&^& npm run build\n"
                "pause\n"
            )
            zf.writestr("NavineAI-App/Build-Required.bat", stub)
        else:
            marker = (
                f"Rust Tauri desktop app for {brand_name()}\n"
                f"Version {__version__}\n"
                "Use NavineAI.exe for portable, or NavineAI-Setup.exe / NavineAI.msi to install.\n"
            )
            zf.writestr("NavineAI-App/BUILD.txt", marker)
    return dest


def ensure_download_packages(force: bool = False) -> List[Dict[str, Any]]:
    out_dir = downloads_dir()
    version = __version__.replace(".", "-")
    items = [
        {
            "id": "cli",
            "label": "CLI",
            "filename": f"NavineAI-CLI-{version}.zip",
            "description": "Type navine in a terminal for interactive chat (Claude-style CLI)",
            "builder": _write_cli_zip,
        },
        {
            "id": "app",
            "label": "App",
            "filename": f"NavineAI-App-{version}.zip",
            "description": "Native Rust (Tauri) desktop app",
            "builder": _write_app_zip,
        },
    ]
    results: List[Dict[str, Any]] = []
    for item in items:
        path = out_dir / item["filename"]
        if force or not path.is_file() or (item["id"] == "app" and _find_tauri_exe() and force):
            item["builder"](path)
        elif item["id"] == "app":
            exe = _find_tauri_exe()
            if exe is not None:
                item["builder"](path)
            elif force or not path.is_file():
                item["builder"](path)
        results.append(
            {
                "id": item["id"],
                "label": item["label"],
                "filename": item["filename"],
                "description": item["description"],
                "url": f"/downloads/{item['filename']}",
                "size": path.stat().st_size if path.is_file() else 0,
                "version": __version__,
                "ready": item["id"] != "app" or _find_tauri_exe() is not None,
            }
        )
    return results


def find_download(kind: str) -> Optional[Dict[str, Any]]:
    kind = (kind or "").strip().lower()
    for row in ensure_download_packages():
        if row["id"] == kind or row["filename"] == kind:
            return row
    return None
