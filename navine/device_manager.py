from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import yaml

from navine.utils.paths import get_project_root

_config_cache: Optional[Dict[str, Any]] = None


def _load_Navine_config() -> Dict[str, Any]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    root = get_project_root()
    path = root / "configs" / "navine.yaml"
    if not path.exists():
        path = root / "configs" / "Navine.yaml"
    data: Dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    _config_cache = data
    return data


def reload_device_config() -> Dict[str, Any]:
    global _config_cache
    _config_cache = None
    return _load_Navine_config()


def get_device_settings() -> Dict[str, Any]:
    cfg = _load_Navine_config()
    device_cfg = dict(cfg.get("device") or {})
    device_cfg.setdefault("mode", "auto")
    device_cfg.setdefault("dtype", "auto")
    return device_cfg


def detect_hardware() -> Dict[str, Any]:
    import os

    logical = int(os.cpu_count() or 4)
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu_count": logical,
        "logical_cpus": logical,
    }
    ram_ok = False
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["system_ram_gb"] = round(float(mem.total) / (1024 ** 3), 2)
        info["system_ram_available_gb"] = round(float(mem.available) / (1024 ** 3), 2)
        info["system_ram_used_gb"] = round(float(mem.used) / (1024 ** 3), 2)
        info["system_ram_percent"] = float(mem.percent)
        ram_ok = True
    except Exception:
        pass
    if not ram_ok:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["system_ram_gb"] = round(float(stat.ullTotalPhys) / (1024 ** 3), 2)
            info["system_ram_available_gb"] = round(float(stat.ullAvailPhys) / (1024 ** 3), 2)
            info["system_ram_percent"] = float(stat.dwMemoryLoad)
            ram_ok = True
        except Exception:
            info["system_ram_gb"] = 16.0
            info["system_ram_available_gb"] = 8.0
    if info["cuda_available"]:
        idx = torch.cuda.current_device()
        info["cuda_device"] = idx
        info["cuda_name"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        info["cuda_vram_gb"] = round(float(getattr(props, "total_memory", 0)) / (1024 ** 3), 2)
    if info["mps_available"]:
        info["mps_name"] = "Apple Metal (MPS)"
    return info


def resolve_device_mode(mode: Optional[str] = None) -> str:
    selected = str(mode or get_device_settings().get("mode") or "auto").lower()
    if selected in ("hybrid", "gpu+cpu", "both"):
        hw = detect_hardware()
        if hw["cuda_available"]:
            return "cuda"
        if hw["mps_available"]:
            return "mps"
        return "cpu"
    if selected not in ("cpu", "gpu", "auto"):
        selected = "auto"
    hw = detect_hardware()
    if selected == "cpu":
        return "cpu"
    if selected == "gpu":
        if hw["cuda_available"]:
            return "cuda"
        if hw["mps_available"]:
            return "mps"
        return "cpu"
    if hw["cuda_available"]:
        return "cuda"
    if hw["mps_available"]:
        return "mps"
    return "cpu"


def get_device(mode: Optional[str] = None) -> torch.device:
    import os

    env_mode = os.environ.get("NAVINE_DEVICE_MODE") or os.environ.get("Navine_DEVICE_MODE")
    if env_mode:
        mode = env_mode
    resolved = resolve_device_mode(mode)
    if resolved == "cuda":
        return torch.device("cuda")
    if resolved == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def print_train_device_banner(require_cuda: bool = False) -> Dict[str, Any]:
    compute = configure_compute_environment()
    dev = get_device()
    hw = compute.get("hardware") or detect_hardware()
    lines = [
        f"Navine train device: {dev}",
        f"mode={compute.get('mode')} dtype={compute.get('dtype')}",
        f"cuda_available={hw.get('cuda_available')} name={hw.get('cuda_name', 'n/a')} "
        f"vram_gb={hw.get('cuda_vram_gb', 'n/a')}",
        f"cpu_threads={compute.get('cpu_threads')} dataloader={compute.get('dataloader')} "
        f"ram={hw.get('system_ram_gb', '?')}GB usable={compute.get('usable_system_ram_gb', '?')}GB",
    ]
    msg = " | ".join(str(x) for x in lines)
    print(msg, flush=True)
    if bool(get_device_settings().get("prefer_cuda", True)) and dev.type == "cpu" and hw.get("cuda_available"):
        print("Warning: prefer_cuda true but resolved device is CPU — check device.mode / env", flush=True)
    if require_cuda and not bool(hw.get("cuda_available")):
        raise RuntimeError(
            "CUDA GPU required for this train run but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or attach a GPU."
        )
    if require_cuda and dev.type != "cuda":
        raise RuntimeError(f"Expected CUDA device for training, got {dev}. Check device.mode in configs/navine.yaml.")
    return compute


def get_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    settings = get_device_settings()
    dtype_mode = str(settings.get("dtype") or "auto").lower()
    dev = device or get_device()
    if dtype_mode == "float32":
        return torch.float32
    if dtype_mode == "float16":
        return torch.float16
    if dev.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32


@contextmanager
def get_autocast_context(device: Optional[torch.device] = None) -> Iterator[None]:
    dev = device or get_device()
    if dev.type == "cuda":
        dtype = get_dtype(dev)
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
        return
    yield


def save_device_settings(mode: str, dtype: Optional[str] = None) -> Path:
    path = get_project_root() / "configs" / "Navine.yaml"
    data: Dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    device_cfg = dict(data.get("device") or {})
    device_cfg["mode"] = str(mode).lower()
    if dtype:
        device_cfg["dtype"] = str(dtype).lower()
    data["device"] = device_cfg
    if "logs_dir" not in data:
        data["logs_dir"] = "logs"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    reload_device_config()
    return path


def device_summary() -> Tuple[torch.device, Dict[str, Any]]:
    dev = get_device()
    hw = detect_hardware()
    settings = get_device_settings()
    summary = {
        "device": str(dev),
        "mode": settings.get("mode", "auto"),
        "dtype": str(get_dtype(dev)),
        "hardware": hw,
    }
    return dev, summary


_CONFIGURED = False


def configure_compute_environment() -> Dict[str, Any]:
    global _CONFIGURED
    import os

    settings = get_device_settings()
    hw = detect_hardware()
    use_all_ram = bool(settings.get("use_all_system_ram", False))
    target_ram = float(settings.get("target_system_ram_gb") or hw.get("system_ram_gb") or 16)
    reserve = float(settings.get("ram_reserve_gb") or 1.25)
    usable_ram = max(4.0, min(target_ram, float(hw.get("system_ram_gb") or target_ram)) - reserve)

    cpu_threads = int(settings.get("cpu_threads") or 0)
    logical = int(hw.get("logical_cpus") or os.cpu_count() or 4)
    if cpu_threads <= 0:
        cpu_threads = logical if use_all_ram else max(1, logical - 1)
    cpu_threads = max(1, min(cpu_threads, logical))

    if not _CONFIGURED:
        # Cap BLAS threads: cpu_adamw + full OpenBLAS on 16GB caused alloc failures.
        blas_threads = max(2, min(cpu_threads, 8 if use_all_ram else cpu_threads))
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(blas_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
        torch.set_num_threads(cpu_threads)
        try:
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max(1, min(4, max(2, cpu_threads // 4))))
        except RuntimeError:
            pass
        _CONFIGURED = True
    if torch.cuda.is_available() and bool(settings.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    dev, summary = device_summary()
    summary["cpu_threads"] = cpu_threads
    summary["dataloader"] = get_dataloader_kwargs()
    summary["target_system_ram_gb"] = target_ram
    summary["usable_system_ram_gb"] = round(usable_ram, 2)
    summary["use_all_system_ram"] = use_all_ram
    return summary


def get_dataloader_kwargs() -> Dict[str, Any]:
    import os

    settings = get_device_settings()
    hw = detect_hardware()
    use_all_ram = bool(settings.get("use_all_system_ram", False))
    parallel = os.environ.get("Navine_PARALLEL_TRAIN") == "1"
    raw_workers = settings.get("dataloader_workers", "auto")
    if str(raw_workers).strip().lower() in ("auto", "detect", "smart", ""):
        ram = float(hw.get("system_ram_gb") or 16)
        if use_all_ram and ram >= 14:
            workers = 3
        elif ram >= 12:
            workers = 2
        elif torch.cuda.is_available():
            workers = 2
        else:
            workers = 0
    else:
        workers = int(raw_workers)
    workers = max(0, min(workers, 4))
    if parallel:
        workers = int(settings.get("parallel_dataloader_workers") or max(0, workers // 2))
    prefetch = int(settings.get("prefetch_factor") or (2 if use_all_ram else 2))
    prefetch = max(1, min(prefetch, 4))
    kwargs: Dict[str, Any] = {
        "num_workers": max(0, workers),
        "pin_memory": bool(settings.get("pin_memory", True)) and torch.cuda.is_available(),
        "persistent_workers": bool(workers > 0),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
    return kwargs


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    pin_memory = bool(get_device_settings().get("pin_memory", True)) and device.type == "cuda"
    return tensor.to(device, non_blocking=pin_memory)
