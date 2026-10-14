import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class NavineTokenizer:
    SPECIAL_TOKENS = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<code>",
        "</code>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ]

    def __init__(self, vocab_size: int = 4096, use_bpe: bool = True):
        self.vocab_size = vocab_size
        self.use_bpe = bool(use_bpe)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}
        self._built = False

    def _normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text or "")

    def _word_pieces(self, text: str) -> List[str]:
        text = self._normalize(text)
        pieces = re.findall(r"\s+|[^\s]+", text, flags=re.UNICODE)
        return [p for p in pieces if p]

    def _is_cjk(self, ch: str) -> bool:
        if not ch:
            return False
        code = ord(ch[0])
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
            or 0x3400 <= code <= 0x4DBF
        )

    def _needs_space(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left[-1].isspace() or right[0].isspace():
            return False
        if self._is_cjk(left[-1]) or self._is_cjk(right[0]):
            return self._is_cjk(left[-1]) != self._is_cjk(right[0])
        if self.use_bpe and self.merges:
            return False
        if left[-1].isalnum() and right[0].isalnum() and left[-1].isascii() and right[0].isascii():
            return True
        if left[-1] in "([{\"'" or right[0] in ")]},\"'":
            return False
        return not right[0] in ".,!?;:)]}\"'"

    def _decode_tokens(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        out: List[str] = []
        for tok in tokens:
            if tok.isspace():
                if out and not out[-1].endswith((" ", "\n", "\t")):
                    out.append(tok if tok.strip() else " ")
                continue
            if not out:
                out.append(tok)
                continue
            prev = out[-1]
            if prev.endswith((" ", "\n", "\t")) or tok.startswith((" ", "\n", "\t")):
                out.append(tok)
            elif self._needs_space(prev, tok):
                out.append(" " + tok)
            else:
                out.append(tok)
        return "".join(out).strip()

    def _char_units(self, piece: str) -> List[str]:
        if piece.isspace():
            return [piece]
        return list(piece)

    def _tokenize_text(self, text: str) -> List[str]:
        return self._word_pieces(text)

    def _get_pairs(self, symbols: List[str]) -> Counter:
        pairs: Counter = Counter()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def _apply_merge(self, symbols: List[str], pair: Tuple[str, str]) -> List[str]:
        merged: List[str] = []
        i = 0
        a, b = pair
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def _build_word_vocab(self, texts: List[str]) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._tokenize_text(text))
        most_common = counter.most_common(self.vocab_size - len(self.SPECIAL_TOKENS))
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        for tok, _ in most_common:
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merges = []
        self._merge_ranks = {}
        self.use_bpe = False
        self._built = True

    def _build_bpe(self, texts: List[str]) -> None:
        max_chars = 2_500_000 if self.vocab_size >= 20000 else 800_000
        max_pieces = 25_000 if self.vocab_size >= 20000 else 6_000
        word_strings: Counter = Counter()
        budget = 0
        word_freq: Counter = Counter()
        for text in texts:
            if not text:
                continue
            for piece in self._word_pieces(text):
                word_strings[piece] += 1
                units = tuple(self._char_units(piece))
                if units and len(units) <= 48:
                    word_freq[units] += 1
            budget += len(text)
            if budget >= max_chars:
                break
        if len(word_freq) > max_pieces:
            word_freq = Counter(dict(word_freq.most_common(max_pieces)))

        alphabet = set()
        for units in word_freq:
            alphabet.update(units)
        for tok in self.SPECIAL_TOKENS:
            alphabet.add(tok)

        vocab = set(alphabet)
        merges: List[Tuple[str, str]] = []
        extra_merges = 8000 if self.vocab_size >= 20000 else 4000
        target = max(len(self.SPECIAL_TOKENS) + 256, min(self.vocab_size, len(alphabet) + extra_merges))

        while len(vocab) < target and len(merges) < self.vocab_size:
            pair_counts: Counter = Counter()
            for units, freq in word_freq.items():
                if len(units) < 2:
                    continue
                for i in range(len(units) - 1):
                    pair_counts[(units[i], units[i + 1])] += freq
            if not pair_counts:
                break
            best, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break
            merges.append(best)
            new_symbol = best[0] + best[1]
            vocab.add(new_symbol)
            new_freq: Counter = Counter()
            for units, freq in word_freq.items():
                merged = tuple(self._apply_merge(list(units), best))
                new_freq[merged] += freq
            word_freq = new_freq

        self.token_to_id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        for ch in sorted(alphabet):
            if ch not in self.token_to_id:
                self.token_to_id[ch] = len(self.token_to_id)
        for a, b in merges:
            piece = a + b
            if piece not in self.token_to_id:
                self.token_to_id[piece] = len(self.token_to_id)
            if len(self.token_to_id) >= self.vocab_size:
                break
        ranked = Counter()
        for units, freq in word_freq.items():
            for u in units:
                ranked[u] += freq
        for tok, _ in ranked.most_common(self.vocab_size):
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
            if len(self.token_to_id) >= self.vocab_size:
                break
        for token, _ in word_strings.most_common(self.vocab_size * 2):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
            if len(self.token_to_id) >= self.vocab_size:
                break
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.merges = merges[: max(0, self.vocab_size)]
        self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.use_bpe = True
        self._built = True

    def build(self, texts: List[str], use_bpe: Optional[bool] = None) -> None:
        flag = self.use_bpe if use_bpe is None else bool(use_bpe)
        if flag:
            self._build_bpe(texts)
        else:
            self._build_word_vocab(texts)

    def _bpe_encode_piece(self, piece: str) -> List[str]:
        symbols = self._char_units(piece)
        if not symbols:
            return []
        if len(symbols) == 1:
            return symbols
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self._merge_ranks[p], p) for p in pairs if p in self._merge_ranks]
            if not ranked:
                break
            ranked.sort(key=lambda x: x[0])
            symbols = self._apply_merge(symbols, ranked[0][1])
        return symbols

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        if not self._built:
            raise RuntimeError("Tokenizer not built. Train or load a checkpoint first.")
        text = self._normalize(text)
        tokens: List[str] = []
        if self.use_bpe and self.merges:
            for piece in self._word_pieces(text):
                tokens.extend(self._bpe_encode_piece(piece))
        else:
            tokens = self._tokenize_text(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = set(self.SPECIAL_TOKENS) if skip_special else set()
        parts = []
        for idx in ids:
            tok = self.id_to_token.get(idx, "<unk>")
            if tok in special:
                continue
            parts.append(tok)
        return self._decode_tokens(parts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "use_bpe": self.use_bpe,
            "merges": [[a, b] for a, b in self.merges],
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "NavineTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("vocab_size", 4096), use_bpe=bool(data.get("use_bpe", False)))
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        merges = data.get("merges") or []
        tok.merges = [(a, b) for a, b in merges]
        tok._merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok._built = True
        return tok

    def load_corpus_from_files(self, text_file: Path, code_file: Optional[Path] = None) -> List[str]:
        texts: List[str] = []
        if text_file.exists():
            texts.extend(text_file.read_text(encoding="utf-8").splitlines())
        if code_file and code_file.exists():
            for line in code_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    texts.append(entry.get("prompt", ""))
                    texts.append(entry.get("code", ""))
        return texts
