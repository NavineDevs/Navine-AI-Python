from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from navine.text.arch import apply_size_tier
from navine.text.story_pipeline import (
    fetch_all,
    load_sft_raw_blocks,
    merge_pretrain_corpus,
    pack_token_bins,
)
from navine.text.infer import clear_model_cache, generate
from navine.text.model import build_text_model
from navine.text.prompts import prompt_prefixes
from navine.text.tokenizer import NavineTokenizer
from navine.text.train import train
from navine.utils.config import load_config
from navine.utils.paths import get_project_root
from navine.utils.tier import resolve_checkpoint_dir

CONFIG_NAME = "tiny_gpt"
CORPUS_DIRS = ("data/corpus",)
OUT_DIR = "data/tiny_gpt"
PRETRAIN_FILE = "data/tiny_gpt/pretrain.txt"
SFT_FILE = "data/tiny_gpt/sft.txt"


def _root() -> Path:
    return get_project_root()


def _out_path(name: str) -> Path:
    path = _root() / OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ckpt_dir() -> Path:
    config = apply_size_tier(load_config(CONFIG_NAME))
    return resolve_checkpoint_dir("text", config)


def _read_corpus_blocks() -> List[str]:
    root = _root()
    blocks: List[str] = []
    for rel in CORPUS_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".md", ".txt", ".markdown"):
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for block in raw.split("\n\n"):
                piece = block.strip()
                if len(piece) >= 16:
                    blocks.append(piece)
    return blocks


def load_corpus_texts() -> List[str]:
    texts = merge_pretrain_corpus(include_stories=True, include_corpus=True)
    try:
        from navine.text.multilingual import load_multilingual_pretrain_lines

        texts.extend(load_multilingual_pretrain_lines())
    except Exception:
        pass
    return texts


def _extract_sft_blocks(texts: List[str]) -> List[str]:
    blocks: List[str] = []
    for text in texts:
        if "### User:" in text and "### Assistant:" in text:
            blocks.append(text.strip())
    for piece in _read_corpus_blocks():
        if "### User:" in piece and "### Assistant:" in piece:
            blocks.append(piece.strip())
    instruct = _root() / "data" / "text" / "instruction_chat.txt"
    if instruct.exists():
        content = instruct.read_text(encoding="utf-8", errors="ignore")
        for block in content.split("\n\n"):
            piece = block.strip()
            if "### User:" in piece and "### Assistant:" in piece:
                blocks.append(piece)
    try:
        from navine.text.multilingual import load_multilingual_sft_blocks

        blocks.extend(load_multilingual_sft_blocks())
    except Exception:
        pass
    dedup: List[str] = []
    seen = set()
    for block in blocks:
        key = re.sub(r"\s+", " ", block.lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(block)
    return dedup


def train_tokenizer(force: bool = False) -> NavineTokenizer:
    config = apply_size_tier(load_config(CONFIG_NAME))
    ckpt = _ckpt_dir()
    ckpt.mkdir(parents=True, exist_ok=True)
    tok_path = ckpt / "tokenizer.json"
    vocab_size = int(config["model"]["vocab_size"])
    use_bpe = bool((config.get("tokenizer") or {}).get("use_bpe", True))
    if tok_path.exists() and not force:
        tokenizer = NavineTokenizer.load(tok_path)
        print(f"Tokenizer loaded from {tok_path} (vocab={len(tokenizer.token_to_id)})")
        return tokenizer
    corpus = load_corpus_texts()
    if not corpus:
        raise FileNotFoundError(
            f"No corpus found. Add .md or .txt files under {_root() / 'data' / 'corpus'}"
        )
    tokenizer = NavineTokenizer(vocab_size, use_bpe=use_bpe)
    print(f"Training {'BPE' if use_bpe else 'word'} tokenizer on {len(corpus)} blocks (target vocab={vocab_size})...")
    tokenizer.build(corpus, use_bpe=use_bpe)
    tokenizer.save(tok_path)
    print(f"Tokenizer saved to {tok_path} (vocab={len(tokenizer.token_to_id)})")
    return tokenizer


def prepare_data() -> Tuple[Path, Path, dict]:
    corpus = load_corpus_texts()
    if not corpus:
        raise FileNotFoundError(
            f"No corpus found. Add markdown files to {_root() / 'data' / 'corpus'}"
        )
    ckpt = _ckpt_dir()
    tok_path = ckpt / "tokenizer.json"
    if not tok_path.exists():
        train_tokenizer(force=True)
    tokenizer = NavineTokenizer.load(tok_path)
    pretrain_path = _out_path("pretrain.txt")
    sft_path = _out_path("sft.txt")
    pretrain_path.write_text("\n\n".join(corpus) + "\n", encoding="utf-8")
    sft_blocks = _extract_sft_blocks(corpus)
    sft_blocks.extend(load_sft_raw_blocks())
    dedup_sft: List[str] = []
    seen_sft = set()
    for block in sft_blocks:
        key = re.sub(r"\s+", " ", block.lower())
        if key in seen_sft:
            continue
        seen_sft.add(key)
        dedup_sft.append(block)
    sft_blocks = dedup_sft
    if not sft_blocks:
        sft_blocks = [
            "### User: Hello\n### Assistant: Hello! I am Navine AI - Python. How can I help?",
            "### User: What are you?\n### Assistant: I am a small local GPT-style language model trained on your data.",
        ]
    sft_path.write_text("\n\n".join(sft_blocks) + "\n", encoding="utf-8")
    bin_meta = {}
    try:
        bin_meta = pack_token_bins(tokenizer, corpus)
    except Exception as exc:
        print(f"Token bin pack skipped ({exc})")
    train_tokens = int(bin_meta.get("train_tokens") or sum(len(tokenizer.encode(t)) for t in corpus))
    val_tokens = int(bin_meta.get("val_tokens") or max(1, int(train_tokens * 0.05)))
    stats = {
        "documents": len(corpus),
        "vocab_size": len(tokenizer.token_to_id),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "sft_blocks": len(sft_blocks),
        "pretrain_path": str(pretrain_path),
        "sft_path": str(sft_path),
    }
    print(f"Tokenizer loaded with vocab size of {stats['vocab_size']:,}")
    print(f"Loaded {stats['documents']} document(s) from corpus")
    print(f"Prepared pretrain -> {pretrain_path}")
    print(f"Prepared SFT ({stats['sft_blocks']} blocks) -> {sft_path}")
    print(f"Train tokens ~{stats['train_tokens']:,} | Val tokens ~{stats['val_tokens']:,}")
    return pretrain_path, sft_path, stats


def fetch_data(stories: int = 5000, sft_rows: int = 15000, style: Optional[str] = None) -> dict:
    print("Fetching TinyStories-style pretrain corpus...")
    result = fetch_all(stories=stories, sft_rows=sft_rows, style=style)
    print("Run prepare-data next to tokenize and pack bins.")
    return result


def sample_completions(
    prompts: Optional[List[str]] = None,
    count: int = 3,
    temperature: Optional[float] = None,
) -> None:
    clear_model_cache()
    config = apply_size_tier(load_config(CONFIG_NAME))
    infer = dict(config.get("inference") or {})
    temp = float(temperature if temperature is not None else infer.get("temperature", 0.8))
    tk = int(infer.get("top_k", 40))
    max_new = int(infer.get("max_new_tokens", 160))
    params, ckpt = _model_stats()
    default_prompts = list(config.get("sample_prompts") or [])
    items = prompts or default_prompts or ["Once upon a time"]
    print(f"Samples | params={params:,} | temp={temp} | top_k={tk}")
    for i, prompt in enumerate(items[: max(1, count)], start=1):
        text = generate(
            prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=tk,
            config_name=CONFIG_NAME,
        )
        print(f"\n--- sample {i} ---")
        print(f"Prompt: {prompt}")
        print(f"Output: {text.strip()}")


def pretrain(steps: int = 5000, fresh: bool = True, require_cuda: bool = False) -> None:
    prepare_data()
    overrides = {
        "data": {
            "train_file": PRETRAIN_FILE,
            "use_data_pipeline": False,
            "use_token_bins": True,
        },
        "training": {"sft_mask": False},
    }
    print(f"Pretraining Tiny GPT on story bins ({steps} steps)...")
    train(
        config_path=CONFIG_NAME,
        max_steps=steps,
        fresh=fresh,
        require_cuda=require_cuda,
        config_overrides=overrides,
    )


def sft(steps: int = 1500, fresh: bool = False, require_cuda: bool = False) -> None:
    prepare_data()
    ckpt = _ckpt_dir() / "latest.pt"
    if not ckpt.exists():
        print("No pretrain checkpoint found. Running pretrain first...")
        pretrain(steps=max(steps, 3000), fresh=True, require_cuda=require_cuda)
    config = apply_size_tier(load_config(CONFIG_NAME))
    sft_lr = float(config.get("training", {}).get("sft_learning_rate") or 0.00008)
    overrides = {
        "data": {
            "train_file": SFT_FILE,
            "use_data_pipeline": False,
            "use_token_bins": False,
        },
        "training": {
            "sft_mask": True,
            "learning_rate": sft_lr,
            "warmup_steps": 50,
        },
    }
    print(f"SFT fine-tuning Tiny GPT ({steps} steps, lr={sft_lr})...")
    train(
        config_path=CONFIG_NAME,
        max_steps=steps,
        fresh=fresh,
        require_cuda=require_cuda,
        config_overrides=overrides,
    )


def run_all(
    pretrain_steps: int = 5000,
    sft_steps: int = 1500,
    require_cuda: bool = False,
    stories: int = 5000,
    sft_rows: int = 15000,
    style: Optional[str] = None,
) -> None:
    print("Step 1/6: Fetch TinyStories + instruction SFT data")
    fetch_data(stories=stories, sft_rows=sft_rows, style=style)
    print("Step 2/6: Train tokenizer (fixed vocab — do not retrain after pretrain)")
    train_tokenizer(force=True)
    print("Step 3/6: Prepare data + pack token bins")
    prepare_data()
    print("Step 4/6: Pretrain")
    pretrain(steps=pretrain_steps, fresh=True, require_cuda=require_cuda)
    print("Step 5/6: SFT fine-tune (lower learning rate)")
    sft(steps=sft_steps, fresh=False, require_cuda=require_cuda)
    print("Step 6/6: Sample + chat")
    sample_completions()
    print("Run: python scripts/tiny_gpt.py chat")


def _model_stats() -> Tuple[int, str]:
    config = apply_size_tier(load_config(CONFIG_NAME))
    ckpt_dir = _ckpt_dir()
    ckpt = ckpt_dir / "latest.pt"
    tok = ckpt_dir / "tokenizer.json"
    if not ckpt.exists() or not tok.exists():
        raise FileNotFoundError(
            f"No trained Tiny GPT at {ckpt_dir}. Run: python scripts/tiny_gpt.py all"
        )
    tokenizer = NavineTokenizer.load(tok)
    model_cfg = dict(config.get("model") or {})
    model_cfg["vocab_size"] = len(tokenizer.token_to_id)
    model = build_text_model(model_cfg, len(tokenizer.token_to_id))
    try:
        import torch

        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = payload.get("model_state") or payload
        if isinstance(state, dict):
            params = sum(int(t.numel()) for t in state.values() if hasattr(t, "numel"))
            return params, str(ckpt)
    except Exception:
        pass
    return model.count_parameters(), str(ckpt)


def chat(
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> None:
    clear_model_cache()
    config = apply_size_tier(load_config(CONFIG_NAME))
    infer = dict(config.get("inference") or {})
    temp = float(temperature if temperature is not None else infer.get("temperature", 0.7))
    tk = int(top_k if top_k is not None else infer.get("top_k", 40))
    max_new = int(max_tokens if max_tokens is not None else infer.get("max_new_tokens", 128))
    params, ckpt = _model_stats()
    prefixes = prompt_prefixes()
    user_p = prefixes["user"]
    asst_p = prefixes["assistant"]
    try:
        from navine.text.multilingual import reply_language_instruction
    except Exception:
        reply_language_instruction = None
    print("Tiny GPT Chat")
    print(f"Model: {params:,} parameters | checkpoint: {ckpt}")
    print(f"Settings: temperature={temp} | top_k={tk} | max_tokens={max_new}")
    print("Multilingual: replies match the user's language")
    print("Type 'exit' or 'quit' to leave.")
    print()
    history: List[str] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break
        lang_hint = reply_language_instruction(user_input) if reply_language_instruction else ""
        parts = [f"{prefixes['system']} {lang_hint}"] if lang_hint else []
        parts.append(f"{prefixes['user']} {user_input.strip()}")
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                parts.append(history[i])
                parts.append(history[i + 1])
        parts.append(f"{prefixes['assistant']} ")
        prompt = "\n".join(parts)
        try:
            reply = generate(
                prompt,
                max_new_tokens=max_new,
                temperature=temp,
                top_k=tk,
                config_name=CONFIG_NAME,
            )
        except FileNotFoundError as exc:
            print(str(exc))
            break
        except Exception as exc:
            print(f"Error: {exc}")
            continue
        reply = reply.split(user_p)[0].split(asst_p)[0].strip()
        for marker in ("\n### User:", "\nUser:", "\n### Assistant:"):
            if marker in reply:
                reply = reply.split(marker)[0].strip()
        print(f"AI: {reply}")
        print()
        history.append(f"{user_p} {user_input.strip()}")
        history.append(f"{asst_p} {reply}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiny_gpt",
        description="Tiny GPT pipeline (tokenizer -> data -> pretrain -> SFT -> chat)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train-tokenizer", help="Train BPE tokenizer on data/corpus")

    sub.add_parser("prepare-data", help="Build pretrain.txt, sft.txt, and token bins")

    p_fetch = sub.add_parser("fetch-data", help="Download TinyStories + Dolly-style SFT pairs")
    p_fetch.add_argument("--stories", type=int, default=5000)
    p_fetch.add_argument("--sft-rows", type=int, default=15000)
    p_fetch.add_argument("--style", type=str, default=None, help="none or pirate")

    p_sample = sub.add_parser("sample", help="Generate autocomplete samples like nanoBeard")
    p_sample.add_argument("--count", type=int, default=3)
    p_sample.add_argument("--temperature", type=float, default=None)
    p_sample.add_argument("prompt", nargs="*", help="Optional prompt prefix")

    p_pre = sub.add_parser("pretrain", help="Pretrain on raw corpus text")
    p_pre.add_argument("--steps", type=int, default=5000)
    p_pre.add_argument("--fresh", action="store_true", default=True)
    p_pre.add_argument("--require-cuda", action="store_true")

    p_sft = sub.add_parser("sft", help="Supervised fine-tune on chat blocks")
    p_sft.add_argument("--steps", type=int, default=1500)
    p_sft.add_argument("--fresh", action="store_true")
    p_sft.add_argument("--require-cuda", action="store_true")

    p_all = sub.add_parser("all", help="Run full nanoBeard-style pipeline")
    p_all.add_argument("--pretrain-steps", type=int, default=5000)
    p_all.add_argument("--sft-steps", type=int, default=1500)
    p_all.add_argument("--stories", type=int, default=5000)
    p_all.add_argument("--sft-rows", type=int, default=15000)
    p_all.add_argument("--style", type=str, default=None)
    p_all.add_argument("--require-cuda", action="store_true")

    p_chat = sub.add_parser("chat", help="Interactive chat REPL")
    p_chat.add_argument("--temperature", type=float, default=None)
    p_chat.add_argument("--top-k", type=int, default=None)
    p_chat.add_argument("--max-tokens", type=int, default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train-tokenizer":
        train_tokenizer(force=True)
    elif args.command == "prepare-data":
        prepare_data()
    elif args.command == "fetch-data":
        fetch_data(stories=args.stories, sft_rows=args.sft_rows, style=args.style)
    elif args.command == "sample":
        prompts = [" ".join(args.prompt)] if args.prompt else None
        sample_completions(prompts=prompts, count=args.count, temperature=args.temperature)
    elif args.command == "pretrain":
        pretrain(
            steps=args.steps,
            fresh=bool(args.fresh),
            require_cuda=bool(args.require_cuda),
        )
    elif args.command == "sft":
        sft(
            steps=args.steps,
            fresh=bool(args.fresh),
            require_cuda=bool(args.require_cuda),
        )
    elif args.command == "all":
        run_all(
            pretrain_steps=args.pretrain_steps,
            sft_steps=args.sft_steps,
            require_cuda=bool(args.require_cuda),
            stories=args.stories,
            sft_rows=args.sft_rows,
            style=args.style,
        )
    elif args.command == "chat":
        chat(
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
