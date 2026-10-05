from __future__ import annotations

from typing import Optional, Tuple

TIER_RANGES = {
    "compact": (10_000_000, 20_000_000),
    "small": (40_000_000, 80_000_000),
    "150m": (80_000_000, 180_000_000),
    "medium": (280_000_000, 360_000_000),
    "gpt3_350m": (280_000_000, 360_000_000),
    "gpt3_760m": (740_000_000, 780_000_000),
    "gpt3_800m": (800_000_000, 900_000_000),
    "gpt3_1b": (1_000_000_000, 1_080_000_000),
    "gpt3_1p3b": (1_100_000_000, 1_350_000_000),
    "gpt3_1p5b": (1_350_000_000, 1_550_000_000),
    "gpt3_1p8b": (1_700_000_000, 1_900_000_000),
    "large": (650_000_000, 820_000_000),
}

TARGET_MIN = 780_000_000
TARGET_MAX = 820_000_000


def range_for_tier(tier: Optional[str] = None) -> Tuple[int, int]:
    key = str(tier or "").strip().lower()
    if key in TIER_RANGES:
        return TIER_RANGES[key]
    if key in ("800m", "800", "gpt3-medium-plus", "gpt3_medium_plus"):
        return TIER_RANGES["gpt3_800m"]
    if key in ("1b", "1000m", "960m", "gpt3_1b", "gpt3-1b"):
        return TIER_RANGES["gpt3_1b"]
    if key in ("760m", "760", "gpt3-medium", "gpt3_medium", ""):
        return TIER_RANGES["gpt3_800m"]
    if key in ("1.3b", "1p3b", "1300m", "gpt3-large", "gpt3_large", "gpt3_1.3b"):
        return TIER_RANGES["gpt3_1p3b"]
    if key in ("1.5b", "1p5b", "1500m", "gpt3_1.5b", "gpt3_1p5b"):
        return TIER_RANGES["gpt3_1p5b"]
    if key in ("1.8b", "1p8b", "1800m", "gpt3_1.8b", "gpt3_1p8b"):
        return TIER_RANGES["gpt3_1p8b"]
    if key in ("350m", "300m", "300", "317m"):
        return TIER_RANGES["medium"]
    return TARGET_MIN, TARGET_MAX


def unique_trainable_params(model) -> int:
    seen = set()
    unique = 0
    for p in model.parameters():
        if not getattr(p, "requires_grad", True):
            continue
        ptr = int(p.data_ptr()) if hasattr(p, "data_ptr") else id(p)
        if ptr in seen:
            continue
        seen.add(ptr)
        unique += int(p.numel())
    return unique


def assert_real_param_count(
    model,
    label: str,
    min_params: Optional[int] = None,
    max_params: Optional[int] = None,
    size_tier: Optional[str] = None,
) -> int:
    unique = unique_trainable_params(model)
    api = int(model.count_parameters()) if hasattr(model, "count_parameters") else unique
    if unique != api:
        raise RuntimeError(
            f"{label}: unique storage params {unique:,} != count_parameters() {api:,} "
            f"(possible fake/double-count)"
        )
    if min_params is None or max_params is None:
        lo, hi = range_for_tier(size_tier or "gpt3_800m")
        if min_params is None:
            min_params = lo
        if max_params is None:
            max_params = hi
    if unique < int(min_params) or unique > int(max_params):
        raise RuntimeError(
            f"{label}: real params {unique:,} outside {int(min_params):,}-{int(max_params):,} "
            f"(tier={size_tier or 'gpt3_800m'}). Refusing to train a mismatched architecture."
        )
    print(f"Navine AI - Python: verified real params for {label}: {unique:,} (unique storage)")
    return unique
