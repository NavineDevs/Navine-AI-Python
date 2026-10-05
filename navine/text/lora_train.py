from pathlib import Path
from typing import Any, Dict, Optional

import torch

from navine.text.model import NavineTextModel
from navine.text.tokenizer import NavineTokenizer
from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir


def train_lora(
    config_path: str = "text",
    rank: int = 8,
    steps: int = 200,
) -> Dict[str, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        return {"ok": False, "skipped": True, "reason": "peft not installed"}
    config = load_config(config_path)
    from navine.device_manager import get_device

    device = get_device()
    ckpt_dir = get_checkpoint_dir("text")
    ckpt_path = ckpt_dir / "latest.pt"
    tok_path = ckpt_dir / "tokenizer.json"
    if not ckpt_path.exists() or not tok_path.exists():
        return {"ok": False, "error": "missing_checkpoint"}
    model, _ = NavineTextModel.load_checkpoint(ckpt_path, device)
    model = model.to(device)
    lora_cfg = LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["qkv", "out"], lora_dropout=0.05)
    model = get_peft_model(model, lora_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"] * 0.25)
    tokenizer = NavineTokenizer.load(tok_path)
    sample = "Navine AI - Python local assistant"
    ids = torch.tensor([tokenizer.encode(sample)], dtype=torch.long, device=device)
    model.train()
    for step in range(steps):
        logits = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    out = ckpt_dir / "lora_adapter.pt"
    torch.save({"lora_state": model.state_dict(), "rank": rank}, out)
    return {"ok": True, "path": str(out), "steps": steps}
