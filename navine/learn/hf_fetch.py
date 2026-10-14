from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from navine.learn.hf_code import fetch_hf_code_datasets
from navine.learn.hf_search import search_datasets
from navine.utils.paths import get_project_root

HF_CHAT_SPECS: List[Dict[str, Any]] = [
    {
        "id": "databricks/databricks-dolly-15k",
        "alias": "dolly_15k",
        "split": "train",
        "max": 2500,
    },
    {
        "id": "HuggingFaceH4/no_robots",
        "alias": "no_robots",
        "split": "train",
        "max": 2000,
    },
    {
        "id": "Open-Orca/OpenOrca",
        "alias": "open_orca",
        "split": "train",
        "max": 1500,
    },
]

SEARCH_QUERIES = [
    "code instruction python",
    "conversation chat instruction",
    "alpaca instruct",
    "math reasoning",
]


def _chat_dir() -> Path:
    path = get_project_root() / "data" / "train" / "chat"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _learn_dir() -> Path:
    path = get_project_root() / "data" / "learn" / "huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_messages(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        user_parts: List[str] = []
        assistant_parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role in ("user", "human"):
                user_parts.append(content)
            elif role in ("assistant", "gpt", "model"):
                assistant_parts.append(content)
        if user_parts and assistant_parts:
            return "\n".join(user_parts), "\n".join(assistant_parts)

    instruction = str(
        row.get("instruction")
        or row.get("prompt")
        or row.get("question")
        or row.get("input")
        or row.get("query")
        or ""
    ).strip()
    response = str(
        row.get("response")
        or row.get("output")
        or row.get("completion")
        or row.get("answer")
        or row.get("chosen")
        or ""
    ).strip()
    if instruction and response and len(instruction) >= 4 and len(response) >= 8:
        return instruction, response

    system = str(row.get("system") or "").strip()
    if system and instruction and response:
        return f"{system}\n\n{instruction}", response
    return None


def _format_chat_block(user: str, assistant: str) -> str:
    return f"User: {user.strip()}\nAssistant: {assistant.strip()}"


def _iter_dataset_rows(dataset_id: str, split: str, max_rows: int) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    try:
        stream = load_dataset(dataset_id, split=split, streaming=True)
    except Exception:
        stream = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=True)
    count = 0
    for row in stream:
        yield dict(row)
        count += 1
        if count >= max_rows:
            break


def fetch_hf_chat_datasets(max_per_dataset: Optional[int] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {"datasets": {}, "total_pairs": 0}
    blocks: List[str] = []
    seen: set[str] = set()
    for spec in HF_CHAT_SPECS:
        name = str(spec["alias"])
        limit = int(max_per_dataset or spec.get("max") or 1000)
        entries: List[str] = []
        err: Optional[str] = None
        try:
            for row in _iter_dataset_rows(str(spec["id"]), str(spec.get("split") or "train"), limit * 2):
                pair = _extract_messages(row)
                if not pair:
                    continue
                user, assistant = pair
                if len(user) < 4 or len(assistant) < 8:
                    continue
                if len(user) > 2000:
                    user = user[:2000]
                if len(assistant) > 3000:
                    assistant = assistant[:3000]
                block = _format_chat_block(user, assistant)
                key = block[:400]
                if key in seen:
                    continue
                seen.add(key)
                entries.append(block)
                if len(entries) >= limit:
                    break
        except Exception as exc:
            err = str(exc)
        blocks.extend(entries)
        report["datasets"][name] = {
            "id": spec["id"],
            "pairs": len(entries),
            "error": err,
        }
        report["total_pairs"] += len(entries)

    out_txt = _chat_dir() / "hf_fetched.txt"
    if blocks:
        out_txt.write_text("\n\n".join(blocks), encoding="utf-8")
    dump = _learn_dir() / "hf_chat_fetched.jsonl"
    with dump.open("w", encoding="utf-8") as handle:
        for block in blocks:
            if "User:" not in block or "Assistant:" not in block:
                continue
            user, _, assistant = block.partition("\nAssistant:")
            handle.write(
                json.dumps(
                    {
                        "user": user.replace("User:", "", 1).strip(),
                        "assistant": assistant.strip(),
                        "source": "hf_fetch",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    report["chat_file"] = str(out_txt)
    report["jsonl_file"] = str(dump)
    return report


def search_hf_catalog(limit: int = 8) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {}
    for query in SEARCH_QUERIES:
        try:
            rows = search_datasets(query, limit=limit)
            catalog[query] = rows
        except Exception as exc:
            catalog[query] = {"error": str(exc)}
    path = _learn_dir() / "hf_search_catalog.json"
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"catalog_path": str(path), "queries": list(catalog.keys())}


def fetch_hf_training_pack(
    code_max: int = 800,
    chat_max: int = 1500,
) -> Dict[str, Any]:
    search = search_hf_catalog()
    code_report = fetch_hf_code_datasets(max_per_dataset=code_max, use_datasets_lib=True)
    chat_report = fetch_hf_chat_datasets(max_per_dataset=chat_max)
    summary = {
        "search": search,
        "code": code_report,
        "chat": chat_report,
        "code_added": int(code_report.get("total_added") or 0),
        "chat_pairs": int(chat_report.get("total_pairs") or 0),
    }
    path = _learn_dir() / "hf_fetch_report.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["report_path"] = str(path)
    return summary
