import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.policy import http_get
from navine.utils.paths import get_project_root

REMOTE_CODING_SOURCES = [
    "github",
    "gitlab",
    "codeberg",
    "stackoverflow",
    "hackernews",
    "docs",
]

LANG_ALIASES = {
    "shell": "bash",
    "sh": "bash",
    "zsh": "bash",
    "c#": "csharp",
    "c++": "cpp",
    "cxx": "cpp",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "kt": "kotlin",
    "rs": "rust",
    "golang": "go",
}

CODE_HOST_SOURCES = {    "python": [
        {
            "url": "https://raw.githubusercontent.com/python/cpython/main/Lib/textwrap.py",
            "prompt": "Show a Python textwrap module example",
        },
        {
            "url": "https://raw.githubusercontent.com/python/cpython/main/Lib/collections/__init__.py",
            "prompt": "Show Python collections module patterns",
        },
        {
            "url": "https://raw.githubusercontent.com/python/cpython/main/Lib/json/__init__.py",
            "prompt": "Show Python json module usage",
        },
        {
            "url": "https://raw.githubusercontent.com/python/cpython/main/Lib/pathlib.py",
            "prompt": "Show Python pathlib examples",
        },
        {
            "url": "https://raw.githubusercontent.com/python/cpython/main/Lib/datetime.py",
            "prompt": "Show Python datetime module patterns",
        },
    ],
    "javascript": [
        {
            "url": "https://raw.githubusercontent.com/nodejs/node/main/lib/util.js",
            "prompt": "Show Node.js util module patterns",
        },
        {
            "url": "https://raw.githubusercontent.com/lodash/lodash/master/compact.js",
            "prompt": "Show a lodash compact utility function",
        },
    ],
    "typescript": [
        {
            "url": "https://raw.githubusercontent.com/microsoft/TypeScript/main/src/lib/es5.d.ts",
            "prompt": "Show TypeScript ES5 type definitions",
        },
    ],
    "rust": [
        {
            "url": "https://raw.githubusercontent.com/rust-lang/rust/master/library/core/src/option.rs",
            "prompt": "Show Rust Option patterns",
        },
    ],
    "go": [
        {
            "url": "https://raw.githubusercontent.com/golang/go/master/src/strings/strings.go",
            "prompt": "Show Go strings package helpers",
        },
    ],
    "java": [
        {
            "url": "https://raw.githubusercontent.com/openjdk/jdk/master/src/java.base/share/classes/java/util/Objects.java",
            "prompt": "Show Java Objects utility patterns",
        },
    ],
    "cpp": [
        {
            "url": "https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json_fwd.hpp",
            "prompt": "Show a C++ JSON forward header example",
        },
    ],
    "csharp": [
        {
            "url": "https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Private.CoreLib/src/System/Math.cs",
            "prompt": "Show C# Math helper patterns",
        },
    ],
    "kotlin": [
        {
            "url": "https://raw.githubusercontent.com/JetBrains/kotlin/master/libraries/stdlib/src/kotlin/util/Standard.kt",
            "prompt": "Show Kotlin standard helpers",
        },
    ],
    "swift": [
        {
            "url": "https://raw.githubusercontent.com/apple/swift/main/stdlib/public/core/String.swift",
            "prompt": "Show Swift String core patterns",
        },
    ],
    "php": [
        {
            "url": "https://raw.githubusercontent.com/php/php-src/master/ext/standard/basic_functions.stub.php",
            "prompt": "Show PHP standard function stubs",
        },
    ],
    "ruby": [
        {
            "url": "https://raw.githubusercontent.com/ruby/ruby/master/lib/set.rb",
            "prompt": "Show Ruby Set library patterns",
        },
    ],
    "lua": [
        {
            "url": "https://raw.githubusercontent.com/lua/lua/master/lstrlib.c",
            "prompt": "Show Lua string library C source patterns",
        },
    ],
    "bash": [
        {
            "url": "https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/check_for_upgrade.sh",
            "prompt": "Show a bash upgrade check script",
        },
    ],
    "html": [
        {
            "url": "https://raw.githubusercontent.com/mdn/learning-area/main/html/introduction-to-html/getting-started/index.html",
            "prompt": "Show a simple HTML page example",
        },
    ],
    "css": [
        {
            "url": "https://raw.githubusercontent.com/mdn/learning-area/main/css/styling-text/styling-links/index.html",
            "prompt": "Show CSS styling for links",
        },
    ],
    "sql": [
        {
            "url": "https://raw.githubusercontent.com/sqlalchemy/sqlalchemy/main/lib/sqlalchemy/sql/selectable.py",
            "prompt": "Show SQLAlchemy selectable query patterns",
        },
    ],
    "dart": [
        {
            "url": "https://gitlab.com/dart-lang/sdk/-/raw/main/sdk/lib/core/string.dart",
            "prompt": "Show Dart string core patterns",
        },
    ],
    "scala": [
        {
            "url": "https://raw.githubusercontent.com/scala/scala/dev/src/library/scala/collection/immutable/List.scala",
            "prompt": "Show Scala List collection patterns",
        },
    ],
    "haskell": [
        {
            "url": "https://raw.githubusercontent.com/haskell/bytestring/master/Data/ByteString/Lazy/Char8.hs",
            "prompt": "Show Haskell ByteString patterns",
        },
    ],
    "elixir": [
        {
            "url": "https://raw.githubusercontent.com/elixir-lang/elixir/main/lib/elixir/lib/enum.ex",
            "prompt": "Show Elixir Enum module patterns",
        },
    ],
}

CODING_SOURCES = CODE_HOST_SOURCES
SEED_SAMPLES = [
    {
        "language": "python",
        "prompt": "create me a gui with tk that says UwU",
        "code": (
            "import tkinter as tk\n\n\ndef main():\n"
            "    root = tk.Tk()\n"
            "    root.title(\"UwU\")\n"
            "    root.geometry(\"360x180\")\n"
            "    label = tk.Label(root, text=\"UwU\", font=(\"Segoe UI\", 24))\n"
            "    label.pack(expand=True, padx=20, pady=20)\n"
            "    root.mainloop()\n\n\nif __name__ == \"__main__\":\n    main()"
        ),
    },
    {
        "language": "python",
        "prompt": "create me a gui with pyqt6 that says UwU",
        "code": (
            "import sys\n\ntry:\n"
            "    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel\n"
            "except ImportError:\n"
            "    from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel\n\n\n"
            "def main():\n"
            "    app = QApplication(sys.argv)\n"
            "    window = QMainWindow()\n"
            "    window.setWindowTitle(\"UwU\")\n"
            "    window.setGeometry(100, 100, 360, 180)\n"
            "    label = QLabel(\"UwU\", window)\n"
            "    label.setGeometry(40, 60, 280, 40)\n"
            "    window.show()\n"
            "    sys.exit(app.exec() if hasattr(app, \"exec\") else app.exec_())\n\n\n"
            "if __name__ == \"__main__\":\n    main()"
        ),
    },
    {
        "language": "python",
        "prompt": "create me a pygame window that says UwU",
        "code": (
            "import sys\n\nimport pygame\n\n\ndef main():\n"
            "    pygame.init()\n"
            "    screen = pygame.display.set_mode((480, 240))\n"
            "    pygame.display.set_caption(\"UwU\")\n"
            "    font = pygame.font.SysFont(\"segoeui\", 48)\n"
            "    text = font.render(\"UwU\", True, (255, 255, 255))\n"
            "    clock = pygame.time.Clock()\n"
            "    running = True\n"
            "    while running:\n"
            "        for event in pygame.event.get():\n"
            "            if event.type == pygame.QUIT:\n"
            "                running = False\n"
            "        screen.fill((30, 30, 40))\n"
            "        rect = text.get_rect(center=screen.get_rect().center)\n"
            "        screen.blit(text, rect)\n"
            "        pygame.display.flip()\n"
            "        clock.tick(60)\n"
            "    pygame.quit()\n"
            "    sys.exit(0)\n\n\nif __name__ == \"__main__\":\n    main()"
        ),
    },
    {
        "language": "java",
        "prompt": "write java hello world",
        "code": (
            "public class HelloWorld {\n"
            "    public static void main(String[] args) {\n"
            "        System.out.println(\"Hello, World!\");\n"
            "    }\n"
            "}"
        ),
    },
    {
        "language": "javascript",
        "prompt": "write a javascript function that reverses a string",
        "code": "function reverseString(s) {\n  return s.split(\"\").reverse().join(\"\");\n}",
    },
    {
        "language": "go",
        "prompt": "write golang hello world",
        "code": "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}",
    },
    {
        "language": "rust",
        "prompt": "write rust hello world",
        "code": "fn main() {\n    println!(\"Hello, World!\");\n}",
    },
    {
        "language": "cpp",
        "prompt": "write c++ hello world",
        "code": "#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}",
    },
]


def normalize_language(value: Optional[str]) -> str:
    raw = (value or "python").strip().lower()
    return LANG_ALIASES.get(raw, raw)


def _entry_key(entry: Dict[str, Any]) -> str:
    source_url = (entry.get("source_url") or "").strip().lower()
    source_id = (entry.get("source_id") or "").strip().lower()
    prompt = (entry.get("prompt") or "").strip().lower()
    if source_url:
        return f"url:{source_url}"
    if source_id:
        return f"id:{source_id}"
    code = (entry.get("code") or "").strip()
    return f"prompt:{prompt}|{hash(code) & 0xFFFFFFFF}"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _merge_entries(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = list(existing)
    seen = {_entry_key(row) for row in merged}
    for row in incoming:
        key = _entry_key(row)
        if key in seen:
            continue
        merged.append(row)
        seen.add(key)
    return merged


def _item_to_coding_entry(item: Dict[str, Any], fallback_language: str) -> Optional[Dict[str, Any]]:
    text = (item.get("text") or "").strip()
    if len(text) < 24:
        return None
    language = normalize_language(item.get("language") or fallback_language)
    kind = (item.get("kind") or "").strip().lower()
    title = (item.get("title") or item.get("prompt") or "Code sample").strip()
    source = (item.get("source") or "remote").strip()
    if kind == "qa" and source == "stackoverflow":
        language = normalize_language(fallback_language)
    if kind not in {"", "code", "qa", "technical", "docs"} and source not in {"github", "gitlab", "codeberg"}:
        if not item.get("language"):
            return None
    code = text[:8000]
    if source == "stackoverflow":
        code = re.sub(r"<[^>]+>", " ", code)
        code = re.sub(r"\s+", " ", code).strip()[:8000]
    return {
        "language": language,
        "prompt": title[:240] or f"Explain this {language} code",
        "code": code,
        "source_url": item.get("url") or "",
        "source_id": item.get("source_id") or "",
        "source": source,
    }


def fetch_remote_coding_samples(
    language: Optional[str] = None,
    max_per_source: int = 20,
    sources: Optional[List[str]] = None,
) -> Dict[str, int]:
    from navine.autolearn.config import load_config
    from navine.autolearn.sources import fetch_source
    from navine.autolearn.sources.github import fetch_github

    cfg = load_config()
    langs = [normalize_language(language)] if language else [normalize_language(x) for x in (cfg.get("github_languages") or list(CODING_SOURCES.keys()))]
    source_names = sources or REMOTE_CODING_SOURCES
    per_source = max(5, int(max_per_source or 20))
    stats: Dict[str, int] = {"items": 0, "entries": 0, "files": 0}
    by_lang: Dict[str, List[Dict[str, Any]]] = {lang: [] for lang in langs}

    for lang in langs:
        if "github" in source_names:
            try:
                batch = fetch_github(
                    language=lang,
                    max_repos=max(3, per_source // 4),
                    max_files=per_source,
                    include_gists=True,
                    include_awesome=lang in {"python", "javascript", "rust", "go", "java"},
                )
                stats["items"] += len(batch)
                for item in batch:
                    entry = _item_to_coding_entry(item, lang)
                    if entry:
                        by_lang.setdefault(normalize_language(entry["language"]), []).append(entry)
            except Exception:
                pass

        for name in source_names:
            if name == "github":
                continue
            try:
                local_cfg = dict(cfg)
                local_cfg["github_languages"] = [lang]
                batch = fetch_source(name, local_cfg, max_items=per_source)
                stats["items"] += len(batch)
                for item in batch:
                    entry = _item_to_coding_entry(item, lang)
                    if entry:
                        by_lang.setdefault(normalize_language(entry["language"]), []).append(entry)
            except Exception:
                continue

    root = get_project_root()
    out_dir = root / "data" / "train" / "coding"
    for lang, entries in by_lang.items():
        if not entries:
            continue
        out_path = out_dir / f"{lang}.jsonl"
        before_rows = _load_jsonl(out_path)
        merged = _merge_entries(before_rows, entries)
        if len(merged) > len(before_rows):
            _write_jsonl(out_path, merged)
            stats["files"] += 1
            stats["entries"] += len(merged) - len(before_rows)
    return stats


def seed_coding_samples() -> List[str]:
    root = get_project_root()
    out_dir = root / "data" / "train" / "coding"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    by_lang = {}
    for entry in SEED_SAMPLES:
        by_lang.setdefault(entry["language"], []).append(entry)
    for lang, entries in by_lang.items():
        out_path = out_dir / f"{lang}.jsonl"
        existing = []
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        seen = {(e.get("prompt") or "").strip().lower() for e in existing}
        changed = False
        for entry in entries:
            key = entry["prompt"].strip().lower()
            if key in seen:
                continue
            existing.append(entry)
            seen.add(key)
            changed = True
        if changed:
            with open(out_path, "w", encoding="utf-8") as handle:
                for row in existing:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            saved.append(str(out_path))
    return saved


def download_coding_samples(
    language: Optional[str] = None,
    max_count: Optional[int] = None,
    include_remote: bool = True,
    remote_max_per_source: int = 20,
) -> List[str]:
    root = get_project_root()
    out_dir = root / "data" / "train" / "coding"
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = [language] if language else list(CODING_SOURCES.keys())
    saved: List[str] = []
    seed_coding_samples()
    if include_remote:
        fetch_remote_coding_samples(
            language=language,
            max_per_source=remote_max_per_source,
        )
    for lang in langs:
        sources = CODING_SOURCES.get(lang, [])
        if not sources:
            continue
        if max_count is not None:
            sources = sources[:max_count]
        out_path = out_dir / f"{lang}.jsonl"
        existing = _load_jsonl(out_path)
        seen_urls = {e.get("source_url") for e in existing if e.get("source_url")}
        new_entries = list(existing)
        for spec in sources:
            if spec["url"] in seen_urls:
                continue
            try:
                resp = http_get(spec["url"], timeout=60)
                resp.raise_for_status()
                code = resp.text[:8000]
                new_entries.append(
                    {
                        "language": lang,
                        "prompt": spec["prompt"],
                        "code": code,
                        "source_url": spec["url"],
                        "source": "raw",
                    }
                )
                seen_urls.add(spec["url"])
            except Exception:
                continue
        if new_entries:
            _write_jsonl(out_path, new_entries)
            saved.append(str(out_path))
    if include_remote:
        for path in sorted(out_dir.glob("*.jsonl")):
            key = str(path)
            if key not in saved:
                saved.append(key)
    return saved