from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from navine.text.tokenizer import NavineTokenizer
from navine.utils.paths import get_project_root

TINY_GPT_DIR = "data/tiny_gpt"
STORIES_FILE = "stories.txt"
SFT_RAW_FILE = "sft_raw.jsonl"
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
META_FILE = "pack_meta.json"

PIRATE_RULES: Tuple[Tuple[str, str], ...] = (
    (r"\byou\b", "ye"),
    (r"\byour\b", "yer"),
    (r"\bare\b", "be"),
    (r"\bis\b", "be"),
    (r"\bam\b", "be"),
    (r"\bhello\b", "ahoy"),
    (r"\bhi\b", "ahoy"),
    (r"\bfriend\b", "matey"),
    (r"\byes\b", "aye"),
    (r"\bmy\b", "me"),
    (r"\bthe\b", "th'"),
)


def _out_dir() -> Path:
    path = get_project_root() / TINY_GPT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_style(text: str, style: Optional[str] = None) -> str:
    if not style or str(style).lower() in ("none", "off", "default"):
        return text
    if str(style).lower() != "pirate":
        return text
    out = text
    for pattern, repl in PIRATE_RULES:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    if not out.strip().lower().startswith("ahoy"):
        out = f"Ahoy, {out.strip()}"
    return out


def _write_lines(path: Path, lines: Iterable[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            piece = line.strip()
            if not piece:
                continue
            handle.write(piece + "\n")
            count += 1
    return count


def fetch_tinystories(max_stories: int = 5000, style: Optional[str] = None) -> Path:
    out = _out_dir() / STORIES_FILE
    max_stories = max(100, int(max_stories))
    rows: List[str] = []
    try:
        from datasets import load_dataset

        stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for i, row in enumerate(stream):
            if i >= max_stories:
                break
            text = str(row.get("text") or "").strip()
            if len(text) >= 40:
                rows.append(apply_style(text, style))
        if rows:
            _write_lines(out, rows)
            print(f"Fetched {len(rows)} TinyStories via Hugging Face datasets -> {out}")
            return out
    except Exception as exc:
        print(f"HF datasets fetch skipped ({exc})")

    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        for block in raw.split("<|endoftext|>"):
            piece = block.strip()
            if len(piece) >= 40:
                rows.append(apply_style(piece, style))
            if len(rows) >= max_stories:
                break
    except Exception as exc:
        print(f"Direct TinyStories download failed ({exc})")

    if not rows:
        rows = [
            "Once upon a time, a little girl found a shiny red ball in the garden.",
            "The cat sat on the mat and watched the birds outside the window.",
            "Tom and Lily built a tall tower with colorful blocks.",
            "A friendly dog ran across the green field under the bright sun.",
            "The boy opened a book and started reading a story about the sea.",
        ]
        rows = [apply_style(t, style) for t in rows]
    _write_lines(out, rows[:max_stories])
    print(f"Wrote {min(len(rows), max_stories)} story lines -> {out}")
    return out


def fetch_instruction_sft(max_rows: int = 15000, style: Optional[str] = None) -> Path:
    out = _out_dir() / SFT_RAW_FILE
    max_rows = max(50, int(max_rows))
    pairs: List[dict] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        for row in ds:
            instruction = str(row.get("instruction") or "").strip()
            context = str(row.get("context") or "").strip()
            response = str(row.get("response") or "").strip()
            if not instruction or not response:
                continue
            question = instruction if not context else f"{instruction}\n{context}"
            pairs.append(
                {
                    "instruction": apply_style(question, style),
                    "response": apply_style(response, style),
                }
            )
            if len(pairs) >= max_rows:
                break
    except Exception as exc:
        print(f"Dolly fetch skipped ({exc})")

    if not pairs:
        pairs = [
            {"instruction": "Hello", "response": "Hello! I am Navine AI - Python. How can I help?"},
            {"instruction": "Tell me a short story", "response": "Once there was a ship that sailed the wide blue sea."},
            {"instruction": "What is Python?", "response": "Python is a programming language used for apps, scripts, and AI."},
        ]
        if style and str(style).lower() == "pirate":
            pairs = [
                {"instruction": "Ahoy", "response": "Ahoy matey! How can I help ye today?"},
                {"instruction": "Tell me a tale", "response": "There once was a ship that put to sea, and the name of the ship was the Billy of Tea."},
                {"instruction": "What be Python?", "response": "Python be a language for coding apps, scripts, and clever machines."},
            ]

    with out.open("w", encoding="utf-8") as handle:
        for row in pairs[:max_rows]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs[:max_rows])} instruction pairs -> {out}")
    return out


def instruction_pairs_to_sft_blocks(pairs: List[dict]) -> List[str]:
    blocks: List[str] = []
    for row in pairs:
        instruction = str(row.get("instruction") or row.get("question") or "").strip()
        response = str(row.get("response") or row.get("answer") or row.get("output") or "").strip()
        if not instruction or not response:
            continue
        blocks.append(f"### User: {instruction}\n### Assistant: {response}")
    return blocks


def load_sft_raw_blocks() -> List[str]:
    path = _out_dir() / SFT_RAW_FILE
    if not path.exists():
        return []
    pairs: List[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            pairs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return instruction_pairs_to_sft_blocks(pairs)


def merge_pretrain_corpus(include_stories: bool = True, include_corpus: bool = True) -> List[str]:
    root = get_project_root()
    texts: List[str] = []
    if include_corpus:
        corpus = root / "data" / "corpus"
        if corpus.exists():
            for path in sorted(corpus.rglob("*")):
                if path.suffix.lower() not in (".md", ".txt", ".markdown"):
                    continue
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for block in raw.split("\n\n"):
                    piece = block.strip()
                    if len(piece) >= 16 and "### User:" not in piece:
                        texts.append(piece)
    if include_stories:
        stories = _out_dir() / STORIES_FILE
        if stories.exists():
            for line in stories.read_text(encoding="utf-8", errors="ignore").splitlines():
                piece = line.strip()
                if len(piece) >= 20:
                    texts.append(piece)
    return texts


def pack_token_bins(
    tokenizer: NavineTokenizer,
    texts: List[str],
    val_ratio: float = 0.05,
    eot_id: Optional[int] = None,
) -> Dict[str, object]:
    out_dir = _out_dir()
    ids: List[int] = []
    eos = eot_id if eot_id is not None else tokenizer.eos_id
    pad = tokenizer.pad_id if tokenizer.pad_id is not None else 0
    for text in texts:
        if not text.strip():
            continue
        chunk_ids = tokenizer.encode(text.strip())
        if not chunk_ids:
            continue
        ids.extend(chunk_ids)
        if eos is not None and eos >= 0:
            ids.append(int(eos))
        else:
            ids.append(pad)
    if len(ids) < 64:
        raise ValueError("Not enough tokens to pack bins. Fetch more stories first.")
    arr = np.asarray(ids, dtype=np.uint16)
    split = max(64, int(len(arr) * (1.0 - val_ratio)))
    train_arr = arr[:split]
    val_arr = arr[split:] if split < len(arr) else arr[-64:]
    train_path = out_dir / TRAIN_BIN
    val_path = out_dir / VAL_BIN
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    meta = {
        "train_tokens": int(train_arr.size),
        "val_tokens": int(val_arr.size),
        "total_tokens": int(arr.size),
        "vocab_size": len(tokenizer.token_to_id),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Packed token bins | train={meta['train_tokens']:,} val={meta['val_tokens']:,} "
        f"vocab={meta['vocab_size']:,}"
    )
    return meta


def load_pack_meta() -> dict:
    path = _out_dir() / META_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


class TokenBinDataset:
    def __init__(self, bin_path: Path, block_size: int):
        self.block_size = max(32, int(block_size))
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if self.data.size <= self.block_size + 1:
            raise ValueError(f"Bin file too small: {bin_path}")

    def __len__(self) -> int:
        return max(1, (int(self.data.size) - self.block_size - 1) // self.block_size)

    def __getitem__(self, idx: int):
        import torch

        start = (idx * self.block_size) % max(1, int(self.data.size) - self.block_size - 1)
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def fetch_all(stories: int = 5000, sft_rows: int = 15000, style: Optional[str] = None) -> dict:
    story_path = fetch_tinystories(max_stories=stories, style=style)
    sft_path = fetch_instruction_sft(max_rows=sft_rows, style=style)
    return {"stories": str(story_path), "sft_raw": str(sft_path)}
