import os
from typing import Literal, Optional


Tier = Literal["gpu", "cloud", "cpu"]


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cloud_available() -> bool:
    return bool(
        os.getenv("NAVINE_CLOUD_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("HF_TOKEN")
    )


def pick_tier(prefer_cloud_when_available: bool = True) -> Tier:
    if cuda_available():
        return "gpu"
    if prefer_cloud_when_available and cloud_available():
        return "cloud"
    return "cpu"


def detect_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def cloud_provider() -> Optional[str]:
    if os.getenv("NAVINE_CLOUD_API_KEY"):
        return "navine_cloud"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("HF_TOKEN"):
        return "huggingface"
    return None

