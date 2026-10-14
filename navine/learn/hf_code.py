import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from navine.utils.paths import get_project_root

HF_CODE_DATASETS: List[Dict[str, Any]] = [
    {
        "id": "sahil280114/codealpaca",
        "alias": "code_alpaca",
        "note": "Classic code instruction pairs (JSON mirror preferred)",
        "max": 20000,
        "kind": "codealpaca_json",
        "url": "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json",
    },
    {
        "id": "HuggingFaceH4/CodeAlpaca_20K",
        "alias": "hf_codealpaca_20k",
        "note": "HF hub CodeAlpaca 20k",
        "max": 20000,
        "kind": "datasets",
        "split": "train",
        "map": "codealpaca",
    },
    {
        "id": "iamtarun/python_code_instructions_18k_alpaca",
        "alias": "python_code_18k",
        "note": "Python code instruction alpaca-style",
        "max": 18000,
        "kind": "datasets",
        "split": "train",
        "map": "alpaca_python",
    },
    {
        "id": "flytech/python-codes-25k",
        "alias": "python_codes_25k",
        "note": "Python code tasks",
        "max": 25000,
        "kind": "datasets",
        "split": "train",
        "map": "python_codes",
    },
    {
        "id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "alias": "magicoder_oss",
        "note": "OSS-Instruct style coding pairs",
        "max": 40000,
        "kind": "datasets",
        "split": "train",
        "map": "magicoder",
    },
    {
        "id": "nickrosh/Evol-Instruct-Code-80k-v1",
        "alias": "evol_instruct_code",
        "note": "Evol-Instruct code instructions",
        "max": 40000,
        "kind": "datasets",
        "split": "train",
        "map": "evol_code",
    },
    {
        "id": "bigcode/the-stack-smol-xs",
        "alias": "stack_smol_xs",
        "note": "Real repo snippets",
        "max": 30000,
        "kind": "datasets",
        "split": "train",
        "map": "stack",
        "languages": ["Python", "python", "JavaScript", "javascript", "TypeScript", "typescript"],
    },
]


def _coding_dir() -> Path:
    path = get_project_root() / "data" / "train" / "coding"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _learn_dir() -> Path:
    path = get_project_root() / "data" / "learn" / "huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _guess_language(text: str, default: str = "python") -> str:
    lower = (text or "").lower()
    checks = (
        ("```python", "python"),
        ("```js", "javascript"),
        ("```javascript", "javascript"),
        ("```typescript", "typescript"),
        ("```ts", "typescript"),
        ("```rust", "rust"),
        ("```go", "go"),
        ("```java", "java"),
        ("```cpp", "cpp"),
        ("```c++", "cpp"),
        ("```csharp", "csharp"),
        ("```ruby", "ruby"),
        ("```php", "php"),
        ("```swift", "swift"),
        ("```kotlin", "kotlin"),
        ("```lua", "lua"),
        ("```bash", "shell"),
        ("```sh", "shell"),
    )
    for needle, lang in checks:
        if needle in lower:
            return lang
    if "def " in lower or "import " in lower or "print(" in lower:
        return "python"
    if "function " in lower or "const " in lower or "=>" in lower:
        return "javascript"
    if "fn " in lower and "let " in lower:
        return "rust"
    return default


def _strip_code_fences(code: str) -> str:
    text = (code or "").strip()
    m = re.search(r"```(?:[\w+-]+)?\n([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text


def _map_row(spec: Dict[str, Any], row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = spec.get("map") or "auto"
    if kind == "codealpaca":
        prompt = row.get("prompt") or row.get("instruction") or row.get("input") or ""
        code = row.get("completion") or row.get("output") or row.get("response") or ""
    elif kind == "alpaca_python":
        instruction = row.get("instruction") or row.get("prompt") or ""
        inp = row.get("input") or ""
        prompt = instruction if not inp else f"{instruction}\nInput: {inp}"
        code = row.get("output") or row.get("completion") or ""
    elif kind == "python_codes":
        prompt = row.get("instruction") or row.get("text") or row.get("prompt") or ""
        code = row.get("code") or row.get("output") or row.get("response") or ""
    elif kind == "magicoder":
        prompt = row.get("problem") or row.get("instruction") or row.get("query") or row.get("prompt") or ""
        code = row.get("solution") or row.get("response") or row.get("output") or ""
    elif kind == "evol_code":
        prompt = row.get("instruction") or row.get("prompt") or ""
        code = row.get("output") or row.get("response") or ""
    elif kind == "stack":
        content = row.get("content") or row.get("code") or ""
        lang = str(row.get("lang") or row.get("language") or "unknown")
        path = row.get("path") or row.get("max_stars_repo_path") or "snippet"
        if not content or len(content) < 40:
            return None
        allowed = [x.lower() for x in (spec.get("languages") or [])]
        if allowed and lang.lower() not in {a.lower() for a in allowed} and lang not in allowed:
            return None
        return {
            "language": lang.lower() if lang else "python",
            "prompt": f"Show a real {lang} code example from {path}",
            "code": content[:6000],
            "source": f"hf:{spec['id']}",
            "source_id": str(row.get("hexsha") or row.get("id") or path)[:120],
        }
    else:
        prompt = (
            row.get("instruction")
            or row.get("prompt")
            or row.get("input")
            or row.get("question")
            or row.get("problem")
            or ""
        )
        code = (
            row.get("output")
            or row.get("completion")
            or row.get("response")
            or row.get("solution")
            or row.get("code")
            or row.get("answer")
            or ""
        )
    prompt = str(prompt or "").strip()
    code = _strip_code_fences(str(code or ""))
    if len(prompt) < 8 or len(code) < 12:
        return None
    if len(code) > 8000:
        code = code[:8000]
    return {
        "language": _guess_language(code + "\n" + prompt),
        "prompt": prompt[:1500],
        "code": code,
        "source": f"hf:{spec['id']}",
    }


def _is_junk_code_entry(entry: Dict[str, Any]) -> bool:
    blob = f"{entry.get('prompt', '')}\n{entry.get('code', '')}".lower()
    if not blob.strip():
        return True
    junk_markers = (
        "<!doctype",
        "<html",
        "wikipedia",
        "wiktionary",
        "4chan.org",
        "reddit.com/r/",
        "from wikipedia",
        "free encyclopedia",
        "||",
        "tags:",
    )
    if any(m in blob for m in junk_markers):
        return True
    code = str(entry.get("code") or "")
    if code.count("<") > 12 and "def " not in code and "function " not in code and "class " not in code:
        return True
    return False


def _append_jsonl(path: Path, entries: List[Dict[str, Any]]) -> int:
    if not entries:
        return 0
    existing = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (obj.get("prompt", "")[:200], obj.get("code", "")[:200])
            existing.add(key)
    added = 0
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            if _is_junk_code_entry(entry):
                continue
            key = (entry.get("prompt", "")[:200], entry.get("code", "")[:200])
            if key in existing:
                continue
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing.add(key)
            added += 1
    return added


def _fetch_codealpaca_json(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    from navine.autolearn.http import learning_http_get

    url = spec["url"]
    resp = learning_http_get(url, timeout=120, sleep=0.2)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    limit = int(spec.get("max") or 2000)
    for row in data[:limit]:
        if not isinstance(row, dict):
            continue
        instruction = str(row.get("instruction") or "").strip()
        inp = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        prompt = instruction if not inp else f"{instruction}\nInput: {inp}"
        mapped = _map_row({"map": "auto", "id": "codealpaca"}, {"instruction": prompt, "output": output})
        if mapped:
            mapped["source"] = "hf-mirror:code_alpaca_20k"
            out.append(mapped)
    return out


def _iter_datasets_stream(dataset_id: str, split: str, max_rows: int) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            f"datasets package required for HF hub streaming ({dataset_id}). "
            f"Install: pip install datasets. ({exc})"
        ) from exc
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


def _fetch_datasets_spec(spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    limit = int(spec.get("max") or 1000)
    # oversample stream a bit for stacked filters
    raw_limit = limit * 3 if spec.get("map") == "stack" else limit
    try:
        rows = list(_iter_datasets_stream(spec["id"], str(spec.get("split") or "train"), raw_limit))
    except Exception as exc:
        return [], str(exc)
    out: List[Dict[str, Any]] = []
    for row in rows:
        mapped = _map_row(spec, row)
        if mapped:
            out.append(mapped)
        if len(out) >= limit:
            break
    return out, None


def fetch_hf_code_datasets(
    max_per_dataset: Optional[int] = None,
    only: Optional[List[str]] = None,
    use_datasets_lib: bool = True,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"datasets": {}, "total_added": 0, "files": []}
    aliases = {s.strip().lower() for s in (only or []) if s and s.strip()}
    for spec in HF_CODE_DATASETS:
        name = str(spec.get("alias") or spec["id"])
        if aliases and name.lower() not in aliases and str(spec["id"]).lower() not in aliases:
            continue
        local = dict(spec)
        if max_per_dataset is not None:
            local["max"] = int(max_per_dataset)
        entries: List[Dict[str, Any]] = []
        err: Optional[str] = None
        kind = local.get("kind")
        try:
            if kind == "codealpaca_json":
                entries = _fetch_codealpaca_json(local)
            elif kind == "datasets" and use_datasets_lib:
                entries, err = _fetch_datasets_spec(local)
            else:
                err = f"unsupported kind {kind}"
        except Exception as exc:
            err = str(exc)
            entries = []
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            lang = str(entry.get("language") or "python").lower()
            by_lang.setdefault(lang, []).append(entry)
        added_total = 0
        files = []
        for lang, items in by_lang.items():
            path = _coding_dir() / f"{lang}.jsonl"
            added = _append_jsonl(path, items)
            added_total += added
            if added:
                files.append(str(path))
        dump = _learn_dir() / f"{name.replace('/', '_')}.jsonl"
        if entries:
            with dump.open("w", encoding="utf-8") as handle:
                for entry in entries[:5000]:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            files.append(str(dump))
        report["datasets"][name] = {
            "id": local.get("id"),
            "added": added_total,
            "fetched": len(entries),
            "error": err,
            "note": local.get("note"),
        }
        report["total_added"] += added_total
        report["files"].extend(files)
    catalog = _learn_dir() / "HF_CODE_CATALOG.md"
    lines = [
        "# Hugging Face / open code datasets used for Navine",
        "",
        "Data only (instruction/code pairs). Custom text NN weights stay local.",
        "",
        "| Alias | Dataset | Fetched | Added | Error |",
        "|-------|---------|---------|-------|-------|",
    ]
    for name, info in report["datasets"].items():
        lines.append(
            f"| {name} | {info.get('id')} | {info.get('fetched')} | {info.get('added')} | {info.get('error') or ''} |"
        )
    lines.extend(
        [
            "",
            "## How to refresh",
            "",
            "```text",
            "python -m navine.cli learn hf-code --max 1500",
            "python -m navine.cli train coding --steps 200",
            "```",
            "",
            "Inference HF model APIs stay blocked by policy; hub dataset hosts are allowed for learning.",
        ]
    )
    catalog.write_text("\n".join(lines), encoding="utf-8")
    report["catalog"] = str(catalog)
    return report


def list_hf_code_datasets() -> List[Dict[str, Any]]:
    return [
        {
            "alias": s.get("alias"),
            "id": s.get("id"),
            "max": s.get("max"),
            "note": s.get("note"),
            "kind": s.get("kind"),
        }
        for s in HF_CODE_DATASETS
    ]
