from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

HF_CURATED_QUERIES = [
    ("code", "code instruction python"),
    ("chat", "conversation chat"),
    ("blender", "blender 3d"),
    ("math", "math reasoning"),
    ("multimodal", "image text pairs"),
]


def search_datasets(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    limit = max(1, min(int(limit or 20), 50))
    url = f"https://huggingface.co/api/datasets?search={quote(q)}&limit={limit}"
    req = Request(url, headers={"User-Agent": "NavineAI/1.0"})
    with urlopen(req, timeout=25) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    rows: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        return rows
    for item in data:
        if not isinstance(item, dict):
            continue
        did = str(item.get("id") or item.get("_id") or "").strip()
        if not did:
            continue
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        rows.append(
            {
                "id": did,
                "author": str(item.get("author") or did.split("/")[0] if "/" in did else ""),
                "downloads": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "private": bool(item.get("private")),
                "tags": [str(t) for t in tags[:12]],
                "url": f"https://huggingface.co/datasets/{did}",
            }
        )
    return rows


def format_dataset_results(query: str, rows: List[Dict[str, Any]], limit: int = 12) -> str:
    if not rows:
        return f"No Hugging Face datasets found for: {query}"
    lines = [f'Hugging Face datasets for "{query}" ({min(len(rows), limit)} shown):', ""]
    for row in rows[:limit]:
        tags = ", ".join(row.get("tags") or [])[:80]
        lines.append(
            f"- {row['id']} · downloads {row.get('downloads', 0):,} · likes {row.get('likes', 0):,}"
        )
        lines.append(f"  {row.get('url')}")
        if tags:
            lines.append(f"  tags: {tags}")
    lines.append("")
    lines.append("Fetch for training: navine learn hf-code --list")
    return "\n".join(lines)


def curated_dataset_search() -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for key, query in HF_CURATED_QUERIES:
        try:
            rows[key] = {"query": query, "datasets": search_datasets(query, limit=5)}
        except Exception as exc:
            rows[key] = {"query": query, "error": str(exc), "datasets": []}
    return rows


def parse_hf_search_message(message: str) -> Optional[str]:
    text = (message or "").strip()
    patterns = [
        r"^(?:hf|huggingface)\s+(?:datasets?\s+)?search\s+(.+)$",
        r"^search\s+(?:hf|huggingface)\s+(?:for\s+)?(?:datasets?\s+)?(.+)$",
        r"^search\s+(?:for\s+)?datasets?\s+(?:on\s+)?(?:hf|huggingface)\s+(.+)$",
        r"^(?:find|list)\s+(?:hf|huggingface)\s+datasets?\s+(?:for\s+|about\s+)?(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, re.I)
        if match:
            return match.group(1).strip(" .?")
    return None
