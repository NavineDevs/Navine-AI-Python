import re
from typing import List

from navine.selfcode.io import get_project_root, iter_source_files

MAX_MATCHES = 12
MAX_LINE_SNIPPET = 180
CONTEXT_LINES = 1


def search_code(query: str, max_results: int = MAX_MATCHES) -> List[dict]:
    stripped = query.strip()
    if not stripped:
        return []
    from navine.utils.brand import brand_package

    root = get_project_root().resolve()
    pattern = re.compile(re.escape(stripped), re.I)
    results: List[dict] = []
    for top in (brand_package(), "web", "configs", "scripts", "app"):
        base = root / top
        if not base.is_dir():
            continue
        for file_path in iter_source_files(base):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(file_path.relative_to(root)).replace("\\", "/")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if not pattern.search(line):
                    continue
                start = max(0, idx - CONTEXT_LINES)
                end = min(len(lines), idx + CONTEXT_LINES + 1)
                snippet_lines = lines[start:end]
                snippet = "\n".join(snippet_lines)
                if len(snippet) > MAX_LINE_SNIPPET:
                    snippet = snippet[: MAX_LINE_SNIPPET - 3] + "..."
                results.append(
                    {
                        "path": rel,
                        "line": idx + 1,
                        "snippet": snippet,
                    }
                )
                if len(results) >= max_results:
                    return results
    return results
