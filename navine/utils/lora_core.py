from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        rank = max(1, min(rank, base.in_channels, base.out_channels))
        self.down = nn.Conv2d(
            base.in_channels,
            rank,
            base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            bias=False,
        )
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        rank = max(1, min(rank, base.in_features, base.out_features))
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))


def inject_lora(module: nn.Module, rank: int, alpha: float) -> int:
    injected = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank, alpha))
            injected += 1
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            injected += 1
        else:
            injected += inject_lora(child, rank, alpha)
    return injected


def adapter_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if ".down." in key or ".up." in key
    }


def load_adapter_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=False)


def merge_lora_conv(lora: LoRAConv2d) -> nn.Conv2d:
    with torch.no_grad():
        up_w = lora.up.weight.squeeze(-1).squeeze(-1)
        down_w = lora.down.weight
        delta = torch.einsum("or, rijk -> oijk", up_w, down_w) * lora.scale
        lora.base.weight.add_(delta)
    return lora.base


def merge_lora_linear(lora: LoRALinear) -> nn.Linear:
    with torch.no_grad():
        delta = (lora.up.weight @ lora.down.weight) * lora.scale
        lora.base.weight.add_(delta)
    return lora.base


def merge_lora_into_module(module: nn.Module) -> int:
    merged = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            setattr(module, name, merge_lora_conv(child))
            merged += 1
        elif isinstance(child, LoRALinear):
            setattr(module, name, merge_lora_linear(child))
            merged += 1
        else:
            merged += merge_lora_into_module(child)
    return merged


def smoke_score(results: dict) -> Tuple[float, int, int]:
    items = results.get("items") or {}
    if not items:
        return 0.0, 0, 0
    valid = sum(1 for item in items.values() if item.get("valid"))
    total = len(items)
    return valid / total, valid, total
