import json
import random
from pathlib import Path
from typing import List

from navine.text.train import TextDataset, collate_fn, train as base_train
from navine.text.model import NavineTextModel
from navine.text.tokenizer import NavineTokenizer
from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir, get_project_root

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_chat_texts(config) -> List[str]:
    root = get_project_root()
    texts: List[str] = []
    tokenizer = NavineTokenizer(config["model"]["vocab_size"])
    for key in ("chat_file", "train_file", "learned_file", "code_file"):
        rel = config["data"].get(key)
        if not rel:
            continue
        path = root / rel
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                texts.append(entry.get("prompt", ""))
                texts.append(entry.get("code", ""))
        elif path.suffix == ".txt":
            content = path.read_text(encoding="utf-8")
            if "### User:" in content:
                blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
                texts.extend(blocks)
            else:
                texts.extend(content.splitlines())
    sample_chat = root / "data" / "text" / "sample_chat.txt"
    if sample_chat.exists():
        for line in sample_chat.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(line.strip())
    return texts


def finetune(config_path: str = "text_chat", checkpoint: str = None) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = load_chat_texts(config)
    if not texts:
        raise ValueError("No chat training data found.")
    random.shuffle(texts)
    split = max(1, int(len(texts) * (1 - config["data"]["val_split"])))
    train_texts = texts[:split]
    val_texts = texts[split:] if split < len(texts) else texts[:1]
    ckpt_dir = get_checkpoint_dir("text")
    tok_path = ckpt_dir / "tokenizer.json"
    ckpt_path = Path(checkpoint) if checkpoint else ckpt_dir / "latest.pt"
    if tok_path.exists() and ckpt_path.exists():
        tokenizer = NavineTokenizer.load(tok_path)
        model, _ = NavineTextModel.load_checkpoint(ckpt_path, device)
        model = model.to(device)
        print(f"Fine-tuning existing checkpoint: {ckpt_path}")
    else:
        tokenizer = NavineTokenizer(config["model"]["vocab_size"])
        tokenizer.build(train_texts + val_texts)
        model_cfg = config["model"].copy()
        model_cfg["vocab_size"] = len(tokenizer.token_to_id)
        model = NavineTextModel(**model_cfg).to(device)
        print("No checkpoint found; training from scratch on chat data.")
    print(f"Navine AI - Python Chat Fine-tune | Parameters: {model.count_parameters():,} | Device: {device}")
    train_ds = TextDataset(train_texts, tokenizer, config["model"]["max_seq_len"])
    val_ds = TextDataset(val_texts, tokenizer, config["model"]["max_seq_len"])
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    tokenizer.save(tok_path)
    step = 0
    max_steps = config["training"]["max_steps"]
    model.train()
    pbar = tqdm(total=max_steps, desc="Fine-tuning Navine AI - Python Chat")
    while step < max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_id
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step % config["training"]["eval_interval"] == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        v_logits = model(vx)
                        v_loss = torch.nn.functional.cross_entropy(
                            v_logits.reshape(-1, v_logits.size(-1)),
                            vy.reshape(-1),
                            ignore_index=tokenizer.pad_id,
                        )
                        val_losses.append(v_loss.item())
                print(f"Step {step} | Val Loss: {sum(val_losses)/len(val_losses):.4f}")
                model.train()
            if step % config["training"]["save_interval"] == 0 or step == max_steps:
                model.save_checkpoint(ckpt_dir / "latest.pt")
            if step >= max_steps:
                break
    pbar.close()
    model.save_checkpoint(ckpt_dir / "latest.pt")
    print(f"Chat fine-tune complete. Checkpoint: {ckpt_dir / 'latest.pt'}")


if __name__ == "__main__":
    finetune()
