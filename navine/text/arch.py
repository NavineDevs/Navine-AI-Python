from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from navine.text.model import NavineTextModel, build_text_model

ARCH_COMPARE_KEYS = (
    "vocab_size",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "max_seq_len",
    "use_rope",
    "use_swiglu",
    "use_rms_norm",
    "tie_embeddings",
)

def resolve_size_tier(config: Dict[str, Any]) -> str:
    tier = str(config.get("size_tier") or config.get("model_size") or "medium").strip().lower()
    if tier in (
        "150m",
        "150",
        "149m",
        "gpt2",
        "gpt2_small",
        "gpt-2",
        "124m",
        "gpt2small",
        "gpt3_125m",
        "125m",
    ):
        return "150m"
    if tier in ("small", "sm", "58m", "standard"):
        return "small"
    if tier in ("compact", "13m", "micro", "tiny", "13M"):
        return "compact"
    if tier in ("medium", "med", "mid", "317m", "304m", "266m", "350m", "300m", "300", "gpt3_350m"):
        return "medium"
    if tier in ("gpt3_760m", "760m", "760", "gpt3-medium", "gpt3_medium"):
        return "gpt3_760m"
    if tier in ("gpt3_800m", "800m", "800", "gpt3-medium-plus", "gpt3_medium_plus"):
        return "gpt3_800m"
    if tier in ("gpt3_1b", "1b", "1000m", "960m", "gpt3-1b", "gpt3_1b_near"):
        return "gpt3_1b"
    if tier in ("gpt3_1.3b", "gpt3_1p3b", "1.3b", "1p3b", "1300m", "gpt3-large", "gpt3_large"):
        return "gpt3_1p3b"
    if tier in ("gpt3_1.5b", "gpt3_1p5b", "1.5b", "1p5b", "1500m", "1440m"):
        return "gpt3_1p5b"
    if tier in ("gpt3_1.8b", "gpt3_1p8b", "1.8b", "1p8b", "1800m", "1812m"):
        return "gpt3_1p8b"
    if tier in ("large", "lg"):
        return "large"
    return tier


def apply_size_tier(config: Dict[str, Any], tier: Optional[str] = None) -> Dict[str, Any]:
    out = dict(config)
    profiles = dict(out.get("profiles") or {})
    chosen = resolve_size_tier(out) if tier is None else resolve_size_tier({"size_tier": tier})
    out["size_tier"] = chosen
    profile = dict(profiles.get(chosen) or {})
    if not profile and chosen == "150m":
        profile = dict(profiles.get("gpt2") or {})
    if not profile and chosen == "gpt2":
        profile = dict(profiles.get("150m") or {})
    if not profile:
        return out
    if profile.get("model"):
        base_model = dict(out.get("model") or {})
        base_model.update(dict(profile["model"]))
        out["model"] = base_model
    if profile.get("training"):
        base_train = dict(out.get("training") or {})
        base_train.update(dict(profile["training"]))
        out["training"] = base_train
    if profile.get("tokenizer"):
        base_tok = dict(out.get("tokenizer") or {})
        base_tok.update(dict(profile["tokenizer"]))
        out["tokenizer"] = base_tok
    if profile.get("inference"):
        base_inf = dict(out.get("inference") or {})
        base_inf.update(dict(profile["inference"]))
        out["inference"] = base_inf
    return out


def architecture_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model = dict(cfg.get("model") or cfg)
    sig: Dict[str, Any] = {}
    for key in ARCH_COMPARE_KEYS:
        if key in model:
            sig[key] = model[key]
    return sig


def architectures_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    left = architecture_signature(a if "model" in a else {"model": a})
    right = architecture_signature(b if "model" in b else {"model": b})
    keys = set(left) & set(right)
    if not keys:
        return False
    for key in keys:
        if left.get(key) != right.get(key):
            return False
    for required in ("d_model", "n_heads", "n_layers"):
        if required in left and required in right and left[required] != right[required]:
            return False
    return True


def archive_text_checkpoint(ckpt_dir: Path, label: str = "archive_prev") -> Optional[Path]:
    ckpt_dir = Path(ckpt_dir)
    archive = ckpt_dir / label
    archive.mkdir(parents=True, exist_ok=True)
    moved = False
    for name in ("latest.pt", "best.pt", "tokenizer.json", "config.json"):
        src = ckpt_dir / name
        if not src.is_file():
            continue
        dst = archive / name
        if not dst.exists():
            shutil.copy2(src, dst)
            moved = True
    return archive if moved else None


def _slice_copy(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if not isinstance(src, torch.Tensor) or src.dim() != dst.dim():
        return False
    if src.shape == dst.shape:
        dst.copy_(src)
        return True
    slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, dst.shape))
    if any(s.stop == 0 for s in slices):
        return False
    dst[slices].copy_(src[slices])
    return True


def _copy_compatible_tensors(
    target: NavineTextModel,
    state: Dict[str, torch.Tensor],
) -> Tuple[int, int, List[str]]:
    target_state = target.state_dict()
    transferred = 0
    skipped = 0
    missing: List[str] = []
    new_state: Dict[str, torch.Tensor] = {}
    for key, tensor in target_state.items():
        if key not in state:
            missing.append(key)
            new_state[key] = tensor
            skipped += 1
            continue
        src = state[key]
        if not isinstance(src, torch.Tensor):
            missing.append(key)
            new_state[key] = tensor
            skipped += 1
            continue
        out = tensor.clone()
        if _slice_copy(out, src):
            new_state[key] = out
            transferred += 1
        else:
            missing.append(key)
            new_state[key] = tensor
            skipped += 1
    target.load_state_dict(new_state, strict=True)
    return transferred, skipped, missing


def build_or_upgrade_text_model(
    target_cfg: Dict[str, Any],
    vocab_size: int,
    checkpoint_path: Optional[Path] = None,
    device: str | torch.device = "cpu",
    force_fresh: bool = False,
    max_missing_ratio: float = 0.35,
) -> Tuple[NavineTextModel, Dict[str, Any]]:
    model_cfg = dict(target_cfg.get("model") or target_cfg)
    model_cfg["vocab_size"] = int(vocab_size)
    target = build_text_model(model_cfg, vocab_size)
    meta: Dict[str, Any] = {
        "upgraded": False,
        "fresh": True,
        "transferred": 0,
        "skipped": 0,
        "source": None,
    }
    if force_fresh or checkpoint_path is None or not Path(checkpoint_path).exists():
        return target.to(device), meta

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return target.to(device), meta
    saved_cfg = dict(payload.get("config") or {})
    state = payload.get("model_state") or payload
    if not isinstance(state, dict):
        return target.to(device), meta

    meta["source"] = str(checkpoint_path)
    if architectures_match(saved_cfg, model_cfg):
        try:
            target.load_state_dict(state, strict=True)
            meta["fresh"] = False
            meta["transferred"] = len(state)
            return target.to(device), meta
        except RuntimeError:
            pass

    transferred, skipped, missing = _copy_compatible_tensors(target, state)
    total = max(transferred + skipped, 1)
    missing_ratio = skipped / total
    meta["transferred"] = transferred
    meta["skipped"] = skipped
    meta["missing_keys"] = len(missing)
    meta["missing_ratio"] = round(missing_ratio, 4)
    if transferred < 8:
        fresh = build_text_model(model_cfg, vocab_size)
        meta["fresh"] = True
        meta["upgraded"] = False
        meta["note"] = "too few tensors transferred; fresh init"
        return fresh.to(device), meta

    meta["fresh"] = False
    meta["upgraded"] = True
    if missing_ratio > max_missing_ratio:
        meta["note"] = (
            f"large architecture bump with partial transfer "
            f"(missing_ratio={missing_ratio:.2f})"
        )
    else:
        meta["note"] = "partial weight transfer into larger architecture"
    return target.to(device), meta


def checkpoint_looks_readable(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_file():
            return False
        size = path.stat().st_size
        if size < 1024:
            return False
        with path.open("rb") as handle:
            header = handle.read(4)
            handle.seek(max(0, size - 65557))
            tail = handle.read()
        if header == b"PK\x03\x04":
            return b"PK\x05\x06" in tail
        if header[:2] == b"\x80\x02" or header[:1] == b"\x80":
            return True
        return size > 1024 * 1024
    except OSError:
        return False


def preferred_text_checkpoint(
    namespace: str,
    prefer_medium: bool = True,
) -> Optional[Path]:
    from navine.utils.paths import get_checkpoint_dir

    root = get_checkpoint_dir(namespace)
    candidates = []
    if prefer_medium:
        candidates.extend(
            [
                root / "medium" / "latest.pt",
                root / "latest_medium.pt",
            ]
        )
    candidates.extend(
        [
            root / "latest.pt",
            root / "best.pt",
            root / "archive_small_58m" / "latest.pt",
            root / "archive_medium" / "latest.pt",
        ]
    )
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and checkpoint_looks_readable(path):
            return path
    return None
