from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from typing import Any, Dict, Optional

_CACHE: Dict[str, Any] = {"at": 0.0, "specs": {}}
_CACHE_TTL = 45.0


def _wmic_value(query: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["wmic"] + query.split(),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        lines = [ln.strip() for ln in (completed.stdout or "").splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines[-1]
    except Exception:
        return None
    return None


def _powershell_value(script: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        value = (completed.stdout or "").strip()
        return value or None
    except Exception:
        return None


def collect_host_specs(force: bool = False) -> Dict[str, Any]:
    now = time.time()
    if not force and _CACHE["specs"] and (now - float(_CACHE["at"])) < _CACHE_TTL:
        return dict(_CACHE["specs"])

    specs: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }

    try:
        import psutil

        mem = psutil.virtual_memory()
        specs["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        specs["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
        specs["ram_used_percent"] = round(float(mem.percent), 1)
        specs["cpu_count_physical"] = psutil.cpu_count(logical=False)
        specs["cpu_percent"] = round(float(psutil.cpu_percent(interval=0.2)), 1)
        disk = psutil.disk_usage(os.path.splitdrive(os.getcwd())[0] + os.sep if os.name == "nt" else "/")
        specs["disk_total_gb"] = round(disk.total / (1024 ** 3), 1)
        specs["disk_free_gb"] = round(disk.free / (1024 ** 3), 1)
        specs["disk_used_percent"] = round(float(disk.percent), 1)
    except Exception:
        pass

    try:
        from navine.device_manager import detect_hardware

        hw = detect_hardware()
        if hw.get("cuda_available"):
            specs["gpu"] = hw.get("cuda_name")
            specs["vram_gb"] = hw.get("cuda_vram_gb")
            specs["cuda"] = True
        elif hw.get("mps_available"):
            specs["gpu"] = hw.get("mps_name")
            specs["cuda"] = False
        else:
            specs["cuda"] = False
        if hw.get("system_ram_gb") and "ram_total_gb" not in specs:
            specs["ram_total_gb"] = hw.get("system_ram_gb")
            specs["ram_available_gb"] = hw.get("system_ram_available_gb")
    except Exception:
        pass

    if platform.system() == "Windows":
        cpu_name = _powershell_value("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
        if cpu_name:
            specs["cpu_name"] = cpu_name
        else:
            cpu_name = _wmic_value("cpu get name")
            if cpu_name:
                specs["cpu_name"] = cpu_name
        if not specs.get("gpu"):
            gpu_name = _powershell_value(
                "(Get-CimInstance Win32_VideoController | Sort-Object -Property AdapterRAM -Descending | Select-Object -First 1 -ExpandProperty Name)"
            )
            if gpu_name:
                specs["gpu"] = gpu_name
        board = _powershell_value(
            "$b=Get-CimInstance Win32_BaseBoard; if($b){ '{0} {1}' -f $b.Manufacturer, $b.Product }"
        )
        if board:
            specs["motherboard"] = board

    _CACHE["at"] = now
    _CACHE["specs"] = dict(specs)
    return specs


def format_host_specs_block(specs: Optional[Dict[str, Any]] = None) -> str:
    data = specs or collect_host_specs()
    parts = [
        f"Host PC hostname={data.get('hostname')}",
        f"OS={data.get('os')}",
    ]
    if data.get("cpu_name"):
        parts.append(f"CPU={data.get('cpu_name')}")
    elif data.get("processor"):
        parts.append(f"CPU={data.get('processor')}")
    if data.get("cpu_count_physical") or data.get("cpu_count_logical"):
        parts.append(
            f"cores={data.get('cpu_count_physical') or '?'}p/{data.get('cpu_count_logical') or '?'}l"
        )
    if data.get("ram_total_gb") is not None:
        parts.append(
            f"RAM={data.get('ram_total_gb')}GB total ({data.get('ram_available_gb')}GB free, {data.get('ram_used_percent')}% used)"
        )
    if data.get("gpu"):
        vram = data.get("vram_gb")
        parts.append(f"GPU={data.get('gpu')}" + (f" ({vram}GB VRAM)" if vram is not None else ""))
    if data.get("disk_total_gb") is not None:
        parts.append(
            f"Disk={data.get('disk_free_gb')}GB free / {data.get('disk_total_gb')}GB ({data.get('disk_used_percent')}% used)"
        )
    if data.get("motherboard"):
        parts.append(f"Board={data.get('motherboard')}")
    parts.append("Use these host PC facts when the user asks about their computer, hardware, GPU, RAM, or performance.")
    return " ".join(parts)


def format_host_specs_reply(specs: Optional[Dict[str, Any]] = None) -> str:
    data = specs or collect_host_specs()
    lines = [
        f"Host PC = {data.get('hostname')}",
        f"OS = {data.get('os')}",
        f"CPU = {data.get('cpu_name') or data.get('processor') or 'unknown'}",
        f"Cores = {data.get('cpu_count_physical') or '?'} physical / {data.get('cpu_count_logical') or '?'} logical",
    ]
    if data.get("ram_total_gb") is not None:
        lines.append(
            f"RAM = {data.get('ram_total_gb')} GB total, {data.get('ram_available_gb')} GB free ({data.get('ram_used_percent')}% used)"
        )
    if data.get("gpu"):
        extra = f", {data.get('vram_gb')} GB VRAM" if data.get("vram_gb") is not None else ""
        lines.append(f"GPU = {data.get('gpu')}{extra}")
    if data.get("disk_total_gb") is not None:
        lines.append(
            f"Disk = {data.get('disk_free_gb')} GB free of {data.get('disk_total_gb')} GB"
        )
    if data.get("motherboard"):
        lines.append(f"Motherboard = {data.get('motherboard')}")
    if data.get("cpu_percent") is not None:
        lines.append(f"CPU load now = {data.get('cpu_percent')}%")
    return "\n".join(lines)
