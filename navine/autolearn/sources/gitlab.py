from typing import Any, Dict, List, Optional
from urllib.parse import quote

from navine.autolearn.http import learning_http_get

CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".lua": "lua",
    ".sh": "shell",
}
MAX_FILE_BYTES = 8000


def _search_projects(host: str, language: str, max_repos: int) -> List[Dict[str, Any]]:
    query = quote(f"language:{language} stars:>5")
    url = f"https://{host}/api/v4/projects?search={query}&order_by=stars&sort=desc&per_page={max_repos}"
    try:
        response = learning_http_get(url)
        if response.status_code != 200:
            url = f"https://{host}/api/v1/repos/search?q={query}&limit={max_repos}"
            response = learning_http_get(url)
            if response.status_code != 200:
                return []
            payload = response.json()
            if isinstance(payload, dict):
                return payload.get("data", [])[:max_repos]
            return payload[:max_repos] if isinstance(payload, list) else []
        return response.json()[:max_repos]
    except Exception:
        return []


def _project_path(project: Dict[str, Any], host: str) -> Optional[str]:
    if host == "gitlab.com":
        return project.get("path_with_namespace") or project.get("path")
    return project.get("full_name") or project.get("name")


def _default_branch(project: Dict[str, Any]) -> str:
    return project.get("default_branch") or project.get("default_branch_name") or "main"


def _list_tree(host: str, path: str, ref: str, tree_path: str = "") -> List[Dict[str, Any]]:
    if host == "gitlab.com":
        segment = quote(tree_path, safe="") if tree_path else ""
        url = f"https://gitlab.com/api/v4/projects/{quote(path, safe='')}/repository/tree?ref={ref}&path={segment}&per_page=50"
    else:
        url = f"https://codeberg.org/api/v1/repos/{path}/contents/{tree_path}?ref={ref}"
    try:
        response = learning_http_get(url)
        if response.status_code != 200:
            return []
        data = response.json()
        if isinstance(data, dict):
            return [data]
        return data
    except Exception:
        return []


def _fetch_raw(host: str, path: str, ref: str, file_path: str) -> Optional[str]:
    if host == "gitlab.com":
        url = f"https://gitlab.com/api/v4/projects/{quote(path, safe='')}/repository/files/{quote(file_path, safe='')}/raw?ref={ref}"
    else:
        url = f"https://codeberg.org/api/v1/repos/{path}/raw/{file_path}?ref={ref}"
    try:
        response = learning_http_get(url)
        if response.status_code != 200:
            raw_url = f"https://{host}/{path}/raw/branch/{ref}/{file_path}"
            response = learning_http_get(raw_url)
            if response.status_code != 200:
                return None
        return response.text[:MAX_FILE_BYTES]
    except Exception:
        return None


def _collect_files(host: str, path: str, ref: str, language: str, max_files: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    queue: List[str] = [""]
    seen = set()
    while queue and len(items) < max_files:
        tree_path = queue.pop(0)
        entries = _list_tree(host, path, ref, tree_path)
        for entry in entries:
            if host == "gitlab.com":
                entry_type = entry.get("type")
                name = entry.get("name", "")
                entry_path = entry.get("path", "")
            else:
                entry_type = "blob" if entry.get("type") != "dir" else "dir"
                name = entry.get("name", "")
                entry_path = entry.get("path", name)
            if not entry_path or entry_path in seen:
                continue
            seen.add(entry_path)
            if name.startswith(".") or name in ("node_modules", "vendor", "dist", "build"):
                continue
            is_dir = entry_type in ("tree", "dir")
            if is_dir:
                if entry_path.count("/") < 3:
                    queue.append(entry_path)
                continue
            lower = name.lower()
            ext = ""
            lang = language
            for candidate, mapped in CODE_EXTENSIONS.items():
                if lower.endswith(candidate):
                    ext = candidate
                    lang = mapped
                    break
            if not ext:
                continue
            code = _fetch_raw(host, path, ref, entry_path)
            if not code or len(code.strip()) < 30:
                continue
            source_name = "gitlab" if host == "gitlab.com" else "codeberg"
            if host == "gitlab.com":
                file_url = f"https://{host}/{path}/-/blob/{ref}/{entry_path}"
            else:
                file_url = f"https://{host}/{path}/src/branch/{ref}/{entry_path}"
            items.append(
                {
                    "source": source_name,
                    "source_id": f"{source_name}:{path}:{entry_path}",
                    "title": f"{path}/{entry_path}",
                    "text": code,
                    "url": file_url,
                    "language": lang,
                    "kind": "code",
                }
            )
    return items


def fetch_host(host: str, language: str = "python", max_repos: int = 3, max_files: int = 10) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    projects = _search_projects(host, language, max_repos)
    if not projects:
        return items
    files_per = max(1, max_files // max(1, len(projects)))
    for project in projects:
        path = _project_path(project, host)
        if not path:
            continue
        ref = _default_branch(project)
        try:
            batch = _collect_files(host, path, ref, language, files_per)
            items.extend(batch)
        except Exception:
            continue
        if len(items) >= max_files:
            break
    return items[:max_files]


def fetch_gitlab(language: str = "python", max_repos: int = 3, max_files: int = 10) -> List[Dict[str, Any]]:
    return fetch_host("gitlab.com", language, max_repos, max_files)


def fetch_codeberg(language: str = "python", max_repos: int = 3, max_files: int = 10) -> List[Dict[str, Any]]:
    return fetch_host("codeberg.org", language, max_repos, max_files)


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    languages = config.get("github_languages") or ["python"]
    max_repos = 2
    max_files = 8
    if max_items:
        max_repos = max(1, min(4, max_items // 15 or 1))
        max_files = max(4, max_items // max(1, len(languages)))
    items: List[Dict[str, Any]] = []
    hosts = config.get("code_hosts") or ["gitlab", "codeberg"]
    for host_key in hosts:
        host = "gitlab.com" if host_key == "gitlab" else "codeberg.org"
        for language in languages[:3]:
            try:
                batch = fetch_host(host, language, max_repos, max(2, max_files // 2))
                items.extend(batch)
            except Exception:
                continue
            if max_items and len(items) >= max_items:
                break
        if max_items and len(items) >= max_items:
            break
    if max_items:
        return items[:max_items]
    return items
