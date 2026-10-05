from pathlib import Path
from typing import Any, Dict, List

import torch

from navine.text.model import NavineTextModel
from navine.text.tokenizer import NavineTokenizer
from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir, get_project_root


def _logs_dir() -> Path:
    root = get_project_root()
    try:
        import yaml

        navine_cfg = yaml.safe_load((root / "configs" / "navine.yaml").read_text(encoding="utf-8")) or {}
        rel = str(navine_cfg.get("logs_dir") or "logs")
    except Exception:
        rel = "logs"
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def inspect_text_model(write_report: bool = True) -> Dict[str, Any]:
    config = load_config("text")
    ckpt_dir = get_checkpoint_dir("text")
    ckpt_path = ckpt_dir / "latest.pt"
    tok_path = ckpt_dir / "tokenizer.json"
    lines: List[str] = ["Navine AI - Python Text Model Inspector", "=" * 40]
    report: Dict[str, Any] = {"module": "text", "checkpoint": str(ckpt_path), "ok": False}
    if not ckpt_path.exists():
        lines.append(f"Missing checkpoint: {ckpt_path}")
        report["error"] = "missing_checkpoint"
        if write_report:
            _write_lines(lines)
        return report
    device = torch.device("cpu")
    model, payload = NavineTextModel.load_checkpoint(ckpt_path, device)
    cfg = payload.get("config") or {}
    vocab_size = int(cfg.get("vocab_size") or model.token_emb.num_embeddings)
    d_model = int(cfg.get("d_model") or model.token_emb.embedding_dim)
    n_heads = int(cfg.get("n_heads") or model.blocks[0].attn.n_heads)
    n_layers = int(cfg.get("n_layers") or len(model.blocks))
    d_ff = int(cfg.get("d_ff") or model.blocks[0].ff.net[0].out_features)
    max_seq_len = int(cfg.get("max_seq_len") or model.max_seq_len)
    params = model.count_parameters()
    tokenizer_vocab = None
    if tok_path.exists():
        tok = NavineTokenizer.load(tok_path)
        tokenizer_vocab = tok.vocab_size
    dtype = next(model.parameters()).dtype
    report.update(
        {
            "ok": True,
            "vocab_size": vocab_size,
            "tokenizer_vocab": tokenizer_vocab,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "parameters": params,
            "dtype": str(dtype),
            "architecture": "transformer",
            "train_file": str(get_project_root() / config["data"]["train_file"]),
        }
    )
    lines.extend(
        [
            f"Checkpoint: {ckpt_path}",
            f"Architecture: GPT-style transformer ({n_layers} layers)",
            f"Parameters: {params:,}",
            f"d_model: {d_model} | heads: {n_heads} | d_ff: {d_ff}",
            f"max_seq_len: {max_seq_len}",
            f"vocab_size: {vocab_size} | tokenizer_vocab: {tokenizer_vocab}",
            f"dtype: {dtype}",
            f"Train file: {report['train_file']}",
        ]
    )
    if write_report:
        _write_lines(lines, append=True)
    return report


def _write_lines(lines: List[str], append: bool = False) -> Path:
    out = _logs_dir() / "model_report.txt"
    text = "\n".join(lines) + "\n"
    if append and out.exists():
        existing = out.read_text(encoding="utf-8")
        if "Text Model Inspector" in existing:
            parts = existing.split("Navine AI - Python Image Model Inspector")
            text = parts[0].rstrip() + "\n\n" + text
            if len(parts) > 1:
                text += "\nNavine AI - Python Image Model Inspector" + parts[1]
        else:
            text = existing.rstrip() + "\n\n" + text
    out.write_text(text, encoding="utf-8")
    return out
