from __future__ import annotations

import platform
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional


@lru_cache(maxsize=1)
def detect_host_form_factor() -> str:
    system = platform.system().lower()
    try:
        if system == "windows":
            return _windows_form_factor()
        if system == "darwin":
            return "laptop" if _macos_is_laptop() else "desktop"
        if system == "linux":
            return _linux_form_factor()
    except Exception:
        pass
    return "host"


def _windows_form_factor() -> str:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance -ClassName Win32_SystemEnclosure).ChassisTypes"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        raw = (completed.stdout or "").strip()
        digits = [int(part) for part in raw.replace("{", "").replace("}", "").split() if part.isdigit()]
        laptop_types = {8, 9, 10, 11, 12, 14, 18, 21, 30, 31, 32}
        desktop_types = {3, 4, 5, 6, 7, 15, 16}
        if any(value in laptop_types for value in digits):
            return "laptop"
        if any(value in desktop_types for value in digits):
            return "desktop"
    except Exception:
        pass
    if shutil.which("wmic"):
        try:
            completed = subprocess.run(
                ["wmic", "path", "Win32_Battery", "get", "BatteryStatus"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            if "BatteryStatus" in (completed.stdout or "") and any(ch.isdigit() for ch in completed.stdout or ""):
                return "laptop"
        except Exception:
            pass
    return "host"


def _macos_is_laptop() -> bool:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        model = (completed.stdout or "").lower()
        return "macbook" in model or "book" in model
    except Exception:
        return False


def _linux_form_factor() -> str:
    try:
        chassis = open("/sys/class/dmi/id/chassis_type", encoding="utf-8").read().strip()
        if chassis in {"8", "9", "10", "11", "14", "30", "31", "32"}:
            return "laptop"
        if chassis in {"3", "4", "5", "6", "7"}:
            return "desktop"
    except Exception:
        pass
    return "host"


def runtime_split_summary(*, client_local: Optional[bool] = None) -> Dict[str, Any]:
    host = detect_host_form_factor()
    host_label = host if host in ("laptop", "desktop") else "host"
    return {
        "ui": "browser",
        "ui_on_visitor_device": True,
        "inference_host": host_label,
        "inference_on_visitor_device": False,
        "client_local": bool(client_local) if client_local is not None else True,
        "how_host_detected": (
            "chassis/battery probe on the machine running the Navine server"
            if host_label in ("laptop", "desktop")
            else "form factor unknown; using generic host label"
        ),
        "summary": (
            "The chat UI runs in your browser on your device. "
            f"Image, video, and chat models run on the Navine {host_label} that hosts this site - "
            "not on third-party cloud generators."
        ),
    }


def runtime_identity_line() -> str:
    host = detect_host_form_factor()
    host_label = host if host in ("laptop", "desktop") else "host"
    return (
        "The chat UI runs in your browser on your device. "
        f"My models run on the Navine {host_label} hosting this service - "
        "not on third-party cloud model APIs."
    )


def runtime_welcome_line() -> str:
    host = detect_host_form_factor()
    host_label = host if host in ("laptop", "desktop") else "host"
    return (
        f"UI in your browser | models on the Navine {host_label} | "
        "image/video use Navine-trained weights only"
    )
