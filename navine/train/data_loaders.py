import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.paths import get_project_root


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def read_txt_blocks(path: Path) -> List[str]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    if "### User:" in content or "### System:" in content or "### Assistant:" in content:
        return [b.strip() for b in content.split("\n\n") if b.strip()]
    if "Human:" in content and "Assistant:" in content:
        return [b.strip() for b in content.split("\n\n") if b.strip() and not b.strip().startswith("#")]
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("# ")]
    return lines


def resolve_data_path(config: Dict[str, Any], key: str = "train_file") -> Optional[Path]:
    root = get_project_root()
    rel = config.get("data", {}).get(key)
    if not rel:
        return None
    return root / rel
