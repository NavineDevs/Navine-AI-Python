import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from navine.utils.paths import get_project_root


def _search_learn_dir() -> Path:
    path = get_project_root() / "data" / "learn" / "search"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _train_path() -> Path:
    path = get_project_root() / "data" / "train" / "general" / "learned_search.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def learn_from_search_results(query: str, results: List[Dict[str, Any]]) -> int:
    if not query.strip() or not results:
        return 0
    saved = 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = abs(hash(query.strip().lower())) % 100000
    out_file = _search_learn_dir() / f"search_{stamp}_{digest}.json"
    payload = {
        "query": query.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results[:8],
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    blocks: List[str] = []
    for row in results[:5]:
        title = str(row.get("title") or "").strip()
        snippet = str(row.get("snippet") or row.get("body") or "").strip()
        url = str(row.get("url") or "").strip()
        if not snippet and not title:
            continue
        block = f"### User: {query.strip()}\n### Assistant: {title}\n{snippet}"
        if url:
            block += f"\nSource: {url}"
        blocks.append(block[:4000])
        saved += 1
    if blocks:
        with open(_train_path(), "a", encoding="utf-8") as handle:
            handle.write("\n\n".join(blocks) + "\n\n")
        try:
            from navine.learn.rag import index_document

            combined = "\n\n".join(blocks)
            index_document(f"search:{query[:80]}", query[:120], combined)
        except Exception:
            pass
    return saved
