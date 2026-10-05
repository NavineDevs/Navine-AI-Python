import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from navine.autolearn.http import learning_http_get

CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".lua": "lua",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}
MAX_FILE_BYTES = 8000

AWESOME_LISTS = [
    "https://raw.githubusercontent.com/vinta/awesome-python/master/README.md",
    "https://raw.githubusercontent.com/sindresorhus/awesome/master/readme.md",
    "https://raw.githubusercontent.com/sorrycc/awesome-javascript/master/README.md",
    "https://raw.githubusercontent.com/rust-unofficial/awesome-rust/master/README.md",
    "https://raw.githubusercontent.com/avelino/awesome-go/master/README.md",
    "https://raw.githubusercontent.com/uhub/awesome-java/master/README.md",
    "https://raw.githubusercontent.com/fffaraz/awesome-cpp/master/README.md",
    "https://raw.githubusercontent.com/uhub/awesome-c-sharp/master/README.md",
    "https://raw.githubusercontent.com/markets/awesome-ruby/master/README.md",
    "https://raw.githubusercontent.com/ziadoz/awesome-php/master/README.md",
]

GITHUB_SEARCH_QUERIES = [
    "stars:>100 language:python",
    "stars:>100 language:javascript",
    "stars:>100 language:rust",
    "stars:>100 language:go",
    "topic:machine-learning stars:>50",
    "topic:web-development stars:>50",
    "topic:open-source stars:>100",
]


def _search_repos(query: str, max_repos: int) -> List[Dict[str, Any]]:
    encoded = quote(query)
    url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&order=desc&per_page={max_repos}"
    response = learning_http_get(
        url,
        headers={"Accept": "application/vnd.github+json"},
    )
    if response.status_code != 200:
        return []
    payload = response.json()
    return payload.get("items", [])


def _fetch_trending(max_repos: int = 5) -> List[Dict[str, Any]]:
    return _search_repos("stars:>50 pushed:>2024-01-01", max_repos)


def _fetch_gists(max_gists: int = 10) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/gists/public?per_page={max_gists}"
    try:
        response = learning_http_get(url, headers={"Accept": "application/vnd.github+json"})
        if response.status_code != 200:
            return []
        gists = response.json()
    except Exception:
        return []
    items: List[Dict[str, Any]] = []
    for gist in gists:
        gist_id = gist.get("id")
        files = gist.get("files") or {}
        for fname, finfo in files.items():
            raw_url = finfo.get("raw_url")
            language = finfo.get("language") or "text"
            if not raw_url:
                continue
            try:
                file_resp = learning_http_get(raw_url)
                if file_resp.status_code != 200:
                    continue
                code = file_resp.text[:MAX_FILE_BYTES]
                if len(code.strip()) < 20:
                    continue
                ext = Path(fname).suffix.lower()
                lang = CODE_EXTENSIONS.get(ext, language.lower() if language else "text")
                items.append(
                    {
                        "source": "github",
                        "source_id": f"github:gist:{gist_id}:{fname}",
                        "title": f"Gist {gist_id}/{fname}",
                        "text": code,
                        "url": raw_url,
                        "language": lang,
                        "kind": "code",
                    }
                )
            except Exception:
                continue
    return items


def _fetch_awesome_lists(max_items: int = 5) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for url in AWESOME_LISTS:
        if len(items) >= max_items:
            break
        try:
            response = learning_http_get(url)
            if response.status_code != 200:
                continue
            text = response.text[:12000]
            if len(text.strip()) < 200:
                continue
            name = url.rsplit("/", 2)[-2]
            items.append(
                {
                    "source": "github",
                    "source_id": f"github:awesome:{name}",
                    "title": f"Awesome List: {name}",
                    "text": text,
                    "url": url.replace("raw.githubusercontent.com", "github.com").replace("/master/", "/tree/master/"),
                    "category": "technical",
                }
            )
        except Exception:
            continue
    return items


def _fetch_readme(full_name: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{full_name}/readme"
    try:
        response = learning_http_get(
            url,
            headers={"Accept": "application/vnd.github.raw"},
        )
        if response.status_code != 200:
            return None
        text = response.text.strip()
        return text[:12000] if text else None
    except Exception:
        return None


def _list_contents(full_name: str, path: str = "") -> List[Dict[str, Any]]:
    segment = f"/{path}" if path else ""
    url = f"https://api.github.com/repos/{full_name}/contents{segment}"
    try:
        response = learning_http_get(
            url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status_code != 200:
            return []
        data = response.json()
        if isinstance(data, dict):
            return [data]
        return data
    except Exception:
        return []


def _collect_files(
    full_name: str,
    default_branch: str,
    max_files: int,
    language: str,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    queue: List[str] = [""]
    seen_paths = set()
    while queue and len(items) < max_files:
        current = queue.pop(0)
        entries = _list_contents(full_name, current)
        for entry in entries:
            entry_type = entry.get("type")
            path = entry.get("path", "")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            name = entry.get("name", "")
            if name.startswith(".") or name in ("node_modules", "vendor", "dist", "build", "__pycache__"):
                continue
            if entry_type == "dir":
                if path.count("/") < 4:
                    queue.append(path)
                continue
            if entry_type != "file":
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
            raw_url = entry.get("download_url")
            if not raw_url:
                raw_url = f"https://raw.githubusercontent.com/{full_name}/{default_branch}/{path}"
            try:
                file_resp = learning_http_get(raw_url)
                if file_resp.status_code != 200:
                    continue
                code = file_resp.text[:MAX_FILE_BYTES]
                if len(code.strip()) < 30:
                    continue
                items.append(
                    {
                        "source": "github",
                        "source_id": f"github:{full_name}:{path}",
                        "title": f"{full_name}/{path}",
                        "text": code,
                        "url": raw_url,
                        "language": lang,
                        "kind": "code",
                    }
                )
            except Exception:
                continue
    return items


def fetch_github(
    language: str = "python",
    max_repos: int = 5,
    max_files: int = 20,
    search_query: Optional[str] = None,
    include_gists: bool = True,
    include_awesome: bool = True,
    include_trending: bool = False,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if include_awesome:
        items.extend(_fetch_awesome_lists(max_items=min(3, max_files // 5 or 1)))
    if include_gists:
        items.extend(_fetch_gists(max_gists=min(5, max_files // 4 or 1)))
    query = search_query or f"language:{language} stars:>100"
    repos = _search_repos(query, max_repos)
    if not repos and include_trending:
        repos = _fetch_trending(max_repos)
    files_per_repo = max(1, max_files // max(1, len(repos)))
    for repo in repos:
        full_name = repo.get("full_name")
        if not full_name:
            continue
        default_branch = repo.get("default_branch") or "main"
        readme = _fetch_readme(full_name)
        if readme:
            items.append(
                {
                    "source": "github",
                    "source_id": f"github:{full_name}:readme",
                    "title": f"{full_name} README",
                    "text": readme,
                    "url": f"https://github.com/{full_name}",
                    "category": "technical",
                }
            )
        items.extend(_collect_files(full_name, default_branch, files_per_repo, language))
        if len(items) >= max_files + max_repos:
            break
    return items[: max_files + max_repos]


def fetch(config: Dict[str, Any], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    languages = config.get("github_languages") or ["python"]
    cycle_all = config.get("cycle_all_languages", True)
    max_repos = config.get("github_max_repos", 5)
    max_files = config.get("github_max_files", 25)
    if max_items:
        max_repos = max(2, min(10, max_items // 15 or 2))
        max_files = max(10, max_items // max(1, len(languages)))
    items: List[Dict[str, Any]] = []
    search_queries = config.get("github_search_queries") or GITHUB_SEARCH_QUERIES
    if config.get("github_include_awesome", True):
        items.extend(_fetch_awesome_lists(max_items=min(5, max_items // 10 if max_items else 3)))
    if config.get("github_include_gists", True):
        items.extend(_fetch_gists(max_gists=min(8, max_items // 10 if max_items else 5)))
    lang_list = languages if cycle_all else languages[:2]
    state_index = int(config.get("_github_lang_index", 0))
    if lang_list:
        start = state_index % len(lang_list)
        lang_list = lang_list[start:] + lang_list[:start]
    for idx, language in enumerate(lang_list):
        query = search_queries[idx % len(search_queries)] if search_queries else f"language:{language} stars:>100"
        if "language:" not in query:
            query = f"language:{language} {query}"
        try:
            batch = fetch_github(
                language=language,
                max_repos=max_repos,
                max_files=max_files,
                search_query=query,
                include_gists=False,
                include_awesome=False,
                include_trending=(idx == 0),
            )
            items.extend(batch)
        except Exception:
            continue
        if max_items and len(items) >= max_items:
            break
    if max_items:
        return items[:max_items]
    return items
