import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from navine.utils.paths import get_project_root

SKIP_NAME_PARTS = (
    "_quarantine",
    "autolearn_instructions",
    "autolearn_qa",
    "conversation_learned",
)

JUNK_MARKERS = (
    "<!-- sc_off -->",
    "<div class=",
    "<a href=",
    "&#x2f;",
    "from wikipedia, the free encyclopedia",
    "for other uses, see",
    "coordinates :",
    "disambiguation",
    "submitted by",
    "[link]",
    "[comments]",
    "tags:",
    "| [image",
    "| [human",
    "uncensored hentai",
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def _hash_line(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _should_skip_path(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if any(part == "_quarantine" for part in parts):
        return True
    return any(token in name for token in SKIP_NAME_PARTS)


def looks_like_contaminated(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    if any(marker in lower for marker in JUNK_MARKERS):
        return True
    if lower.count("<") >= 3 and lower.count(">") >= 3:
        return True
    if re.search(r"https?:&#x2f;&#x2f;", lower):
        return True
    if re.search(r"\|\s*\[(?:image|human|hentai|character|video)", lower):
        return True
    alpha = sum(1 for c in text if c.isalpha())
    if len(text) >= 40 and alpha / len(text) < 0.35:
        return True
    return False


def load_text_sources(folders: Optional[List[str]] = None, extensions: Tuple[str, ...] = (".txt", ".md")) -> List[str]:
    root = get_project_root()
    paths: List[Path] = []
    if folders:
        for rel in folders:
            base = root / rel
            if base.is_file():
                paths.append(base)
            elif base.is_dir():
                for ext in extensions:
                    paths.extend(base.rglob(f"*{ext}"))
    else:
        for rel in ("data/text", "data/train", "data/learn"):
            base = root / rel
            if base.exists():
                for ext in extensions:
                    paths.extend(base.rglob(f"*{ext}"))
    seen_hashes: set = set()
    texts: List[str] = []
    for path in sorted(set(paths)):
        if _should_skip_path(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for block in content.split("\n\n"):
            line = normalize_text(block)
            if len(line) < 8:
                continue
            if looks_like_contaminated(line):
                continue
            digest = _hash_line(line)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            texts.append(line)
    return texts


def pack_chunks(texts: Iterable[str], max_len: int = 512) -> List[str]:
    chunks: List[str] = []
    for text in texts:
        words = text.split()
        if len(words) <= max_len:
            chunks.append(text)
            continue
        for start in range(0, len(words), max_len):
            piece = " ".join(words[start : start + max_len])
            if piece.strip():
                chunks.append(piece)
    return chunks


def write_memmap_bins(
    texts: List[str],
    out_dir: Optional[Path] = None,
    val_ratio: float = 0.05,
) -> Dict[str, Path]:
    root = get_project_root()
    out = out_dir or (root / "data" / "text" / "packed")
    out.mkdir(parents=True, exist_ok=True)
    split = max(1, int(len(texts) * (1 - val_ratio)))
    train_text = "\n".join(texts[:split]) + "\n"
    val_text = "\n".join(texts[split:]) + "\n"
    train_path = out / "train.bin"
    val_path = out / "val.bin"
    train_path.write_bytes(train_text.encode("utf-8"))
    val_path.write_bytes(val_text.encode("utf-8"))
    return {"train": train_path, "val": val_path}


def build_training_corpus(folders: Optional[List[str]] = None) -> List[str]:
    raw = load_text_sources(folders=folders)
    return pack_chunks(raw)


def load_clean_instruct_blocks() -> List[str]:
    root = get_project_root()
    folders = [
        root / "data" / "train" / "chat",
        root / "data" / "train" / "thinking",
        root / "data" / "train" / "detective",
        root / "data" / "text",
    ]
    files: List[Path] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        files.extend(sorted(folder.glob("*.txt")))
        files.extend(sorted(folder.glob("*.md")))
    blocks: List[str] = []
    seen: set = set()
    for path in files:
        if _should_skip_path(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for block in content.split("\n\n"):
            text = block.strip()
            if "### User:" not in text or "### Assistant:" not in text:
                continue
            if looks_like_contaminated(text):
                continue
            digest = _hash_line(normalize_text(text))
            if digest in seen:
                continue
            seen.add(digest)
            blocks.append(text)
    return blocks
