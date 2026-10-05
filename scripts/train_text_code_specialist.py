from __future__ import annotations

import json
import math
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navine.device_manager import (
    get_device,
    get_dataloader_kwargs,
    print_train_device_banner,
    to_device,
)
from navine.text.arch import apply_size_tier, build_or_upgrade_text_model
from navine.text.model import NavineTextModel
from navine.text.tokenizer import NavineTokenizer
from navine.text.train import TextDataset, collate_fn
from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir, get_project_root


def _is_junk(prompt: str, code: str) -> bool:
    blob = f"{prompt}\n{code}".lower()
    markers = ("<!doctype", "<html", "wikipedia", "from wikipedia", "4chan.org")
    return any(m in blob for m in markers)


def load_code_sft_texts(limit: Optional[int] = None) -> List[str]:
    root = get_project_root()
    coding = root / "data" / "train" / "coding"
    texts: List[str] = []
    for path in sorted(coding.glob("*.jsonl")):
        if path.name in ("no_run.jsonl", "text.jsonl"):
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = str(obj.get("prompt") or "").strip()
                code = str(obj.get("code") or "").strip()
                lang = str(obj.get("language") or "python").strip() or "python"
                if len(prompt) < 8 or len(code) < 12 or _is_junk(prompt, code):
                    continue
                texts.append(
                    "### User:\n"
                    f"Write complete runnable {lang} code for this task. "
                    "Return only a markdown code block.\n"
                    f"{prompt}\n"
                    "### Assistant:\n"
                    f"```{lang}\n{code}\n```"
                )
                if limit and len(texts) >= limit:
                    return texts
    return texts


def _safe_save(model: NavineTextModel, path: Path, device) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    model.save_checkpoint(tmp)
    tmp.replace(path)


def train_text_code(lm_steps: int = 8000, sft_steps: int = 4000, rebuild_tokenizer: bool = True) -> None:
    print_train_device_banner(require_cuda=True)
    device = get_device()
    loader_kwargs = get_dataloader_kwargs()
    config = apply_size_tier(load_config("text_code"))
    ckpt_dir = get_checkpoint_dir("text_code")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ent_dir = get_checkpoint_dir("text_enterprise")
    source = ent_dir / "latest.pt"
    if not source.exists():
        source = ckpt_dir / "latest.pt"
    texts = load_code_sft_texts(limit=2500)
    if len(texts) < 100:
        raise RuntimeError(f"Need more coding rows, got {len(texts)}. Run learn hf-code first.")
    random.shuffle(texts)
    split = max(1, int(len(texts) * 0.95))
    train_texts = texts[:split]
    val_texts = texts[split:] or texts[:32]
    print(f"Code SFT corpus: {len(texts)} examples")

    tok_path = ckpt_dir / "tokenizer.json"
    use_bpe = True
    vocab_target = int(config["model"]["vocab_size"])
    if rebuild_tokenizer or not tok_path.exists():
        tokenizer = NavineTokenizer(vocab_target, use_bpe=use_bpe)
        corpus = train_texts[:15000]
        print(f"Building BPE tokenizer vocab={vocab_target} on {len(corpus)} samples...")
        tokenizer.build(corpus, use_bpe=use_bpe)
        tokenizer.save(tok_path)
    else:
        tokenizer = NavineTokenizer.load(tok_path)
    actual_vocab = len(tokenizer.token_to_id)
    print(f"Tokenizer vocab={actual_vocab}")

    model, meta = build_or_upgrade_text_model(
        config,
        vocab_size=actual_vocab,
        checkpoint_path=source if source.exists() else None,
        device=device,
        force_fresh=False,
    )
    print(
        f"text_code model params={model.count_parameters():,} "
        f"fresh={meta.get('fresh')} transferred={meta.get('transferred')} skipped={meta.get('skipped')}"
    )
    tokenizer.save(tok_path)

    train_cfg = config["training"]
    train_seq = int(train_cfg.get("train_seq_len") or 256)
    sft_mask = bool(train_cfg.get("sft_mask", True))
    print(f"Encoding train sequences seq={train_seq} n={len(train_texts)}...")
    train_ds = TextDataset(train_texts, tokenizer, train_seq, sft_mask=sft_mask)
    print(f"Encoded train={len(train_ds)}")
    val_ds = TextDataset(val_texts[:64], tokenizer, train_seq, sft_mask=sft_mask)
    pad_collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=pad_collate,
        **loader_kwargs,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    steps_limit = int(lm_steps) + int(sft_steps)
    warmup = int(train_cfg.get("warmup_steps", 200))
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 16)))
    use_amp = device.type == "cuda" and bool(train_cfg.get("mixed_precision", True))
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def lr_at(step_idx: int) -> float:
        base = float(train_cfg["learning_rate"])
        if step_idx < warmup:
            return base * (step_idx + 1) / max(warmup, 1)
        progress = (step_idx - warmup) / max(steps_limit - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    from contextlib import nullcontext
    from navine.device_manager import get_autocast_context


    model.train()
    step = 0
    pbar = tqdm(total=steps_limit, desc="Training text_code")
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < steps_limit:
            for x, y in train_loader:
                x, y = to_device(x, device), to_device(y, device)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_at(step)
                with get_autocast_context(device) if use_amp else nullcontext():
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )
                    loss = loss / grad_accum
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_loss += float(loss.item()) * grad_accum
                if (step + 1) % grad_accum == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{accum_loss:.4f}")
                accum_loss = 0.0
                if step % int(train_cfg["save_interval"]) == 0 or step == steps_limit:
                    _safe_save(model, ckpt_dir / "latest.pt", device)
                    shutil.copy2(tok_path, ckpt_dir / "tokenizer.json")
                if step >= steps_limit:
                    break
    except KeyboardInterrupt:
        _safe_save(model, ckpt_dir / "latest.pt", device)
        print("Interrupted. Checkpoint saved.")
    pbar.close()
    _safe_save(model, ckpt_dir / "latest.pt", device)
    _safe_save(model, ckpt_dir / "best.pt", device)
    print(f"text_code training complete -> {ckpt_dir / 'latest.pt'}")


if __name__ == "__main__":
    steps = 12000
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
    lm = max(1, int(steps * 0.67))
    sft = max(1, steps - lm)
    train_text_code(lm_steps=lm, sft_steps=sft, rebuild_tokenizer=False)
