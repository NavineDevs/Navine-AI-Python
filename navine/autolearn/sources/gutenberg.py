import re
from typing import Any, Dict, List, Optional

from navine.autolearn.http import learning_http_get

POPULAR_BOOK_IDS = [
    1342, 11, 1661, 84, 2701, 98, 74, 345, 2554, 5200,
    1232, 2600, 16328, 64317, 100, 120, 6130, 2542, 768, 1260,
]

CHAPTER_SPLIT = re.compile(r"\n\s*\n\s*\n")


def fetch_gutenberg_book(book_id: int, max_chars: int = 8000) -> Optional[Dict[str, Any]]:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        response = learning_http_get(url, timeout=60)
        if response.status_code != 200:
            alt_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            response = learning_http_get(alt_url, timeout=60)
            if response.status_code != 200:
                return None
            url = alt_url
        text = response.text
        start_markers = ("*** START OF", "***START OF")
        for marker in start_markers:
            idx = text.find(marker)
            if idx >= 0:
                text = text[idx + len(marker):]
                break
        end_markers = ("*** END OF", "***END OF")
        for marker in end_markers:
            idx = text.find(marker)
            if idx >= 0:
                text = text[:idx]
                break
        text = text.strip()
        if len(text) < 500:
            return None
        chunks = CHAPTER_SPLIT.split(text)
        excerpt = chunks[0] if chunks else text
        if len(excerpt) < 300 and len(chunks) > 1:
            excerpt = "\n\n".join(chunks[:2])
        title_match = re.search(r"Title:\s*(.+)", text[:2000], re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else f"Gutenberg Book {book_id}"
        return {
            "source": "gutenberg",
            "source_id": f"gutenberg:{book_id}",
            "title": title,
            "text": excerpt[:max_chars],
            "url": f"https://www.gutenberg.org/ebooks/{book_id}",
            "category": "creative",
        }
    except Exception:
        return None


def fetch_gutenberg(max_items: int = 5) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for book_id in POPULAR_BOOK_IDS:
        if len(items) >= max_items:
            break
        entry = fetch_gutenberg_book(book_id)
        if entry:
            items.append(entry)
    return items


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    limit = max_items or 5
    book_ids = config.get("gutenberg_book_ids") or POPULAR_BOOK_IDS
    items: List[Dict[str, Any]] = []
    for book_id in book_ids:
        if len(items) >= limit:
            break
        try:
            entry = fetch_gutenberg_book(int(book_id))
            if entry:
                items.append(entry)
        except Exception:
            continue
    return items[:limit]
