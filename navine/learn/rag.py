import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from navine.utils.paths import get_project_root

DB_NAME = "rag_index.db"
MIN_TOKEN_LEN = 2
MIN_RELEVANCE_SCORE = 0.001
MIN_CONTENT_TOKEN_LEN = 4

STOP_TOKENS = {
    "the", "and", "for", "are", "was", "were", "who", "what", "when", "where",
    "why", "how", "this", "that", "with", "you", "your", "from", "have", "has",
    "did", "does", "is", "of", "to", "in", "on", "or", "do", "it", "me", "my",
    "an", "as", "at", "be", "by", "we", "us", "am", "self", "not", "can", "if",
}

JUNK_MARKERS = [
    "login", "register", "sign up", "sign in", "subscribe", "recent upload",
    "click here", "read more", "view more", "load more", "next page",
    "page not found", "advertisement", "watch free", "related videos",
    "more videos", "menu", "navigation", "all rights reserved", "terms of service",
    "privacy policy", "cookie policy", "search results", "no results found",
]


def _db_path() -> Path:
    path = get_project_root() / "data" / "learn" / DB_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-z0-9]+", text) if len(t) >= MIN_TOKEN_LEN]


def _content_tokens(text: str) -> List[str]:
    return [
        t
        for t in _tokenize(text)
        if t not in STOP_TOKENS and len(t) >= MIN_CONTENT_TOKEN_LEN
    ]


def _looks_like_junk(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return True
    lower = text.lower()
    marker_hits = sum(1 for marker in JUNK_MARKERS if marker in lower)
    if marker_hits >= 2:
        return True
    words = re.findall(r"[a-zA-Z]+", text)
    if len(words) < 8:
        return True
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if letters / max(len(text), 1) < 0.6:
        return True
    return False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT,
            content TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_tokens (
            doc_id INTEGER,
            token TEXT,
            tf REAL,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_tokens ON doc_tokens(token)")
    conn.commit()
    return conn


def _compute_tf(tokens: List[str]) -> Counter:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return Counter({k: v / total for k, v in counts.items()})


def index_document(source: str, title: str, content: str) -> int:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (source, title, content) VALUES (?, ?, ?)",
        (source, title, content),
    )
    doc_id = cur.lastrowid
    tf = _compute_tf(_tokenize(content))
    for token, weight in tf.items():
        cur.execute(
            "INSERT INTO doc_tokens (doc_id, token, tf) VALUES (?, ?, ?)",
            (doc_id, token, weight),
        )
    conn.commit()
    conn.close()
    return doc_id


def index_file(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            last_id = None
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title") or entry.get("user") or path.name
                text = entry.get("text") or ""
                if not text and entry.get("user") and entry.get("assistant"):
                    text = f"Q: {entry['user']}\nA: {entry['assistant']}"
                if text:
                    last_id = index_document(str(path), title, text)
            return last_id
        if not isinstance(data, dict):
            return None
        return index_document(
            str(path),
            data.get("title", path.name),
            data.get("text", ""),
        )
    if path.suffix == ".jsonl":
        last_id = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            title = entry.get("title") or entry.get("user") or path.name
            text = entry.get("text") or ""
            if not text and entry.get("user") and entry.get("assistant"):
                text = f"Q: {entry['user']}\nA: {entry['assistant']}"
            last_id = index_document(str(path), title, text)
        return last_id
    if path.suffix == ".txt":
        return index_document(str(path), path.name, path.read_text(encoding="utf-8"))
    return None


def _index_roots() -> List[Path]:
    root = get_project_root()
    paths = [
        root / "data" / "learn",
        root / "data" / "conversations",
    ]
    return [path for path in paths if path.exists()]


def rebuild_index() -> int:
    conn = _connect()
    conn.execute("DELETE FROM doc_tokens")
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    count = 0
    for root in _index_roots():
        for path in root.rglob("*"):
            if path.suffix in (".json", ".jsonl", ".txt") and path.name != DB_NAME:
                if index_file(path) is not None:
                    count += 1
    return count


def _idf(token: str, conn: sqlite3.Connection) -> float:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM doc_tokens WHERE token = ?", (token,))
    df = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM documents")
    n = cur.fetchone()[0] or 1
    return math.log((n + 1) / (df + 1)) + 1.0


def retrieve_context(
    query: str,
    top_k: int = 3,
    max_chars: int = 600,
    source_filter: Optional[str] = None,
    exclude_source: Optional[str] = None,
) -> Optional[str]:
    tokens = _tokenize(query)
    if not tokens:
        return None
    query_content_tokens = set(_content_tokens(query))
    if not query_content_tokens:
        return None
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    if cur.fetchone()[0] == 0:
        conn.close()
        return None
    scores: Counter = Counter()
    for token in query_content_tokens:
        idf = _idf(token, conn)
        cur.execute(
            "SELECT doc_id, tf FROM doc_tokens WHERE token = ?",
            (token,),
        )
        for doc_id, tf in cur.fetchall():
            scores[doc_id] += (float(tf) + 0.2) * idf
    if not scores:
        conn.close()
        return None
    ranked: List[Tuple[int, float]] = scores.most_common(max(24, top_k * 8))
    parts: List[Tuple[int, str]] = []
    for doc_id, score in ranked:
        if score < MIN_RELEVANCE_SCORE:
            continue
        cur.execute("SELECT source, title, content FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        if not row:
            continue
        source, title, content = row
        source_text = str(source or "").lower()
        if source_filter and source_filter.lower() not in source_text:
            continue
        if exclude_source and exclude_source.lower() in source_text:
            continue
        doc_tokens = query_content_tokens & set(_content_tokens(f"{title} {content}"))
        min_overlap = 2 if len(query_content_tokens) >= 2 else 1
        if len(doc_tokens) < min_overlap:
            continue
        snippet = content[:max_chars]
        if _looks_like_junk(snippet):
            continue
        bonus = 4 if "youtube" in source_text or "youtube_" in str(title).lower() else 0
        parts.append((len(doc_tokens) + bonus, f"{title}: {snippet}"))
    conn.close()
    parts.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(text for _, text in parts[:top_k]) if parts else None
