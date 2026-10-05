import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup

from navine.autolearn.http import learning_http_get
from navine.image.utils import validate_image_bytes
from navine.nsfw.config import load_nsfw_config
from navine.nsfw.media import enrich_media_item
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]

API_BASE = "https://a.4cdn.org"
CDN_BASE = "https://i.4cdn.org"

GROUP_ALIASES: Dict[str, str] = {
    "japanese": "japanese_culture",
    "culture": "japanese_culture",
    "games": "video_games",
    "game": "video_games",
    "interest": "interests",
    "misc": "miscellaneous",
    "nsfw": "adult",
}

DEFAULT_BOARD_CATEGORIES: Dict[str, str] = {
    "h": "hentai",
    "e": "hentai",
    "d": "hentai",
    "aco": "hentai",
    "u": "hentai",
    "y": "hentai",
    "t": "hentai",
    "gif": "animated",
    "camg": "animated",
    "s": "real",
    "hc": "real",
    "hr": "real",
    "hm": "human",
    "r": "mixed",
    "g": "mixed",
    "ic": "hentai",
    "wg": "hentai",
    "w": "hentai",
    "c": "hentai",
    "a": "hentai",
}

DEFAULT_ADULT_BOARDS = {
    "s", "hc", "hm", "h", "e", "u", "d", "gif", "aco", "r", "hr", "t", "y",
}

DEFAULT_TEXT_CATEGORIES: Dict[str, str] = {
    "g": "coding",
    "gd": "coding",
    "3": "coding",
    "diy": "coding",
    "pol": "unrestricted",
    "b": "unrestricted",
    "r9k": "unrestricted",
    "news": "news",
    "sci": "science",
    "his": "history",
    "lit": "creative",
    "mu": "creative",
    "fa": "creative",
    "po": "creative",
}

NSFW_MEDIA_CATEGORIES = {"hentai", "real", "human", "animated", "mixed", "porn"}

DEFAULT_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webm", "mp4"}


def _report(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def _sleep(cfg: Dict[str, Any]) -> None:
    fc = fourchan_config(cfg)
    delay = float(fc.get("sleep_between_requests") or cfg.get("sleep_between_requests", 1.5))
    time.sleep(delay)


def fourchan_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = cfg or load_nsfw_config()
    section = merged.get("fourchan") or {}
    if not isinstance(section, dict):
        return {"enabled": False}
    return section


def _normalize_board(board: str) -> str:
    return str(board).strip().lower().lstrip("/")


def _normalize_group(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return GROUP_ALIASES.get(key, key)


def board_groups(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    fc = fourchan_config(cfg)
    raw = fc.get("board_groups") or {}
    groups: Dict[str, List[str]] = {}
    if isinstance(raw, dict):
        for name, boards in raw.items():
            if not isinstance(boards, list):
                continue
            normalized = [_normalize_board(b) for b in boards if str(b).strip()]
            if normalized:
                groups[str(name).strip()] = normalized
    if groups:
        return groups
    legacy = fc.get("boards") or []
    if legacy:
        groups["default"] = [_normalize_board(b) for b in legacy if str(b).strip()]
    return groups


def all_configured_boards(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    fc = fourchan_config(cfg)
    explicit = fc.get("boards_all") or fc.get("boards") or []
    if explicit:
        seen: List[str] = []
        for board in explicit:
            name = _normalize_board(board)
            if name and name not in seen:
                seen.append(name)
        if seen:
            return seen
    seen: List[str] = []
    for boards in board_groups(cfg).values():
        for board in boards:
            if board not in seen:
                seen.append(board)
    return seen


def boards_in_group(group: str, cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    key = _normalize_group(group)
    groups = board_groups(cfg)
    if key in groups:
        return list(groups[key])
    for name, boards in groups.items():
        if _normalize_group(name) == key:
            return list(boards)
    return []


def list_fourchan_boards(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fc = fourchan_config(cfg)
    groups = board_groups(cfg)
    categories = board_media_categories(cfg)
    text_categories = board_text_categories(cfg)
    adult = adult_boards(cfg)
    rows: List[Dict[str, Any]] = []
    for group_name, boards in groups.items():
        for board in boards:
            media_cat = categories.get(board, "general")
            text_cat = text_categories.get(board)
            if board in adult and not text_cat:
                text_cat = "nsfw"
            rows.append(
                {
                    "group": group_name,
                    "board": board,
                    "media_category": media_cat,
                    "text_category": text_cat or "general",
                    "media_folder": media_folder_for_board(board, cfg),
                }
            )
    return {
        "enabled": bool(fc.get("enabled", False)),
        "groups": groups,
        "boards_all": all_configured_boards(cfg),
        "board_count": len(all_configured_boards(cfg)),
        "max_boards_per_run": int(fc.get("max_boards_per_run", 10)),
        "rotate_boards": bool(fc.get("rotate_boards", True)),
        "rows": rows,
    }


def _rotation_state_path() -> Path:
    path = get_project_root() / "data" / "nsfw" / "fourchan_rotation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_rotation_state() -> Dict[str, Any]:
    path = _rotation_state_path()
    if not path.exists():
        return {"offset": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"offset": 0}


def _save_rotation_state(state: Dict[str, Any]) -> None:
    _rotation_state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def _allowed_board(board: str, cfg: Dict[str, Any]) -> bool:
    return _normalize_board(board) in all_configured_boards(cfg)


def _allowed_extensions(cfg: Dict[str, Any]) -> Set[str]:
    fc = fourchan_config(cfg)
    raw = fc.get("allowed_extensions") or list(DEFAULT_ALLOWED_EXTENSIONS)
    return {str(ext).strip().lower().lstrip(".") for ext in raw if str(ext).strip()}


def board_media_categories(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    fc = fourchan_config(cfg)
    mapping = dict(DEFAULT_BOARD_CATEGORIES)
    custom = fc.get("board_categories") or {}
    if isinstance(custom, dict):
        for key, value in custom.items():
            mapping[_normalize_board(key)] = str(value).strip().lower()
    return mapping


def board_text_categories(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    fc = fourchan_config(cfg)
    mapping = dict(DEFAULT_TEXT_CATEGORIES)
    custom = fc.get("board_text_categories") or {}
    if isinstance(custom, dict):
        for key, value in custom.items():
            mapping[_normalize_board(key)] = str(value).strip().lower()
    return mapping


def adult_boards(cfg: Optional[Dict[str, Any]] = None) -> Set[str]:
    fc = fourchan_config(cfg)
    raw = fc.get("adult_boards")
    if isinstance(raw, list) and raw:
        return {_normalize_board(b) for b in raw if str(b).strip()}
    groups = board_groups(cfg)
    if "adult" in groups:
        return set(groups["adult"])
    return set(DEFAULT_ADULT_BOARDS)


def _board_media_category(board: str, cfg: Dict[str, Any]) -> str:
    normalized = _normalize_board(board)
    return board_media_categories(cfg).get(normalized, "general")


def _board_text_category(board: str, cfg: Dict[str, Any]) -> str:
    normalized = _normalize_board(board)
    if normalized in adult_boards(cfg):
        return "nsfw"
    return board_text_categories(cfg).get(normalized, "general")


def media_folder_for_board(board: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    normalized = _normalize_board(board)
    category = _board_media_category(normalized, cfg or load_nsfw_config())
    if category in NSFW_MEDIA_CATEGORIES:
        return f"data/nsfw/local/{category}"
    return f"data/learn/images/4chan/{normalized}"


def _local_media_dir(board: str, category: str) -> Path:
    root = get_project_root()
    if category in NSFW_MEDIA_CATEGORIES:
        return root / "data" / "nsfw" / "local" / category
    return root / "data" / "learn" / "images" / "4chan" / board


def _local_text_dir(board: str) -> Path:
    root = get_project_root()
    return root / "data" / "learn" / "4chan" / "text" / board


def _save_text_post(board: str, thread_id: int, post_no: Any, text: str) -> Optional[Path]:
    if not text.strip():
        return None
    dest_dir = _local_text_dir(board)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{board}_{thread_id}_{post_no}.txt"
    if dest.exists():
        return dest
    try:
        dest.write_text(text.strip() + "\n", encoding="utf-8")
        return dest
    except Exception:
        return None


def select_boards_for_run(
    cfg: Dict[str, Any],
    board: Optional[str] = None,
    group: Optional[str] = None,
) -> Tuple[List[str], str]:
    if board:
        target = _normalize_board(board)
        if not _allowed_board(target, cfg):
            return [], f"board /{target}/ not in configured allowlist"
        return [target], f"single board /{target}/"

    fc = fourchan_config(cfg)
    if group:
        selected = boards_in_group(group, cfg)
        if not selected:
            return [], f"group '{group}' not found or empty"
        pool = [_normalize_board(b) for b in selected if _allowed_board(b, cfg)]
        label = f"group {_normalize_group(group)}"
    else:
        pool = all_configured_boards(cfg)
        label = "all boards"

    if not pool:
        return [], "no boards configured"

    max_boards = int(fc.get("max_boards_per_run", 10))
    rotate = bool(fc.get("rotate_boards", True))
    if len(pool) <= max_boards:
        return pool, label

    if rotate:
        state = _load_rotation_state()
        offset = int(state.get("offset", 0)) % len(pool)
        selected_boards = [pool[(offset + idx) % len(pool)] for idx in range(max_boards)]
        state["offset"] = (offset + max_boards) % len(pool)
        state["last_selected"] = selected_boards
        _save_rotation_state(state)
        return selected_boards, f"{label} (rotated {max_boards}/{len(pool)})"

    return pool[:max_boards], f"{label} (first {max_boards}/{len(pool)})"


def _strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _media_url(board: str, tim: int, ext: str) -> str:
    ext_part = ext if ext.startswith(".") else f".{ext}"
    return f"{CDN_BASE}/{board}/{tim}{ext_part}"


def _media_kind(ext: str) -> Optional[str]:
    lower = ext.lower().lstrip(".")
    if lower in {"jpg", "jpeg", "png", "gif", "webp", "bmp"}:
        return "image"
    if lower in {"webm", "mp4", "mov", "m4v"}:
        return "video"
    return None


def _save_media_file(board: str, tim: int, ext: str, content: bytes, category: str) -> Optional[Path]:
    if _media_kind(ext) == "image" and not validate_image_bytes(content):
        return None
    dest_dir = _local_media_dir(board, category)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext_part = ext if ext.startswith(".") else f".{ext}"
    dest = dest_dir / f"{board}_{tim}{ext_part}"
    if dest.exists():
        return dest
    try:
        dest.write_bytes(content)
        return dest
    except Exception:
        return None


def _fetch_catalog(board: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/{board}/catalog.json"
    response = learning_http_get(url, headers={"Accept": "application/json"})
    if response.status_code != 200:
        return []
    payload = response.json()
    threads: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for page in payload:
            threads.extend(page.get("threads") or [])
    return threads


def _fetch_thread(board: str, thread_id: int, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/{board}/thread/{thread_id}.json"
    response = learning_http_get(url, headers={"Accept": "application/json"})
    if response.status_code != 200:
        return []
    payload = response.json()
    posts = payload.get("posts") or []
    return posts if isinstance(posts, list) else []


def _post_to_items(
    post: Dict[str, Any],
    board: str,
    thread_id: int,
    text_category: str,
    cfg: Dict[str, Any],
    allowed_ext: Set[str],
    want_text: bool,
    want_images: bool,
    want_videos: bool,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    post_no = post.get("no")
    subject = (post.get("sub") or "").strip()
    comment = _strip_html(post.get("com") or "")
    text_parts = [part for part in (subject, comment) if part]
    text = " ".join(text_parts).strip()
    thread_url = f"https://boards.4chan.org/{board}/thread/{thread_id}#p{post_no}"
    media_category = _board_media_category(board, cfg)
    if want_text and text:
        saved_text = _save_text_post(board, thread_id, post_no, text)
        text_item: Dict[str, Any] = {
            "source": "4chan",
            "source_id": f"4chan:{board}:{thread_id}:{post_no}:text",
            "title": subject or f"/{board}/ post {post_no}",
            "text": text,
            "url": thread_url,
            "category": text_category,
            "board": board,
            "thread_id": thread_id,
            "post_id": post_no,
            "media_category": media_category,
            "tags": ["4chan", board, media_category, text_category],
        }
        if saved_text:
            text_item["local_path"] = str(saved_text.relative_to(get_project_root()))
        items.append(text_item)
    tim = post.get("tim")
    ext = post.get("ext") or ""
    if not tim or not ext:
        return items
    ext_lower = str(ext).lower().lstrip(".")
    if ext_lower not in allowed_ext:
        return items
    kind = _media_kind(ext)
    if kind == "image" and not want_images:
        return items
    if kind == "video" and not want_videos:
        return items
    if not kind:
        return items
    media_url = _media_url(board, int(tim), str(ext))
    caption = subject or comment or f"/{board}/ {post_no}"
    payload: Dict[str, Any] = {
        "source": "4chan",
        "source_id": f"4chan:{board}:{thread_id}:{post_no}:{kind}",
        "title": caption[:120] or f"/{board}/ media",
        "text": text or caption,
        "url": thread_url,
        "category": text_category if text_category == "nsfw" else "nsfw" if media_category in NSFW_MEDIA_CATEGORIES else text_category,
        "kind": kind,
        "caption": caption[:200],
        "board": board,
        "thread_id": thread_id,
        "post_id": post_no,
        "media_category": media_category,
        "tags": ["4chan", board, media_category, kind],
    }
    if kind == "image":
        payload["image_url"] = media_url
    else:
        payload["video_url"] = media_url
    items.append(enrich_media_item(payload, cfg))
    return items


def fetch_4chan_board(
    board: str,
    max_threads: Optional[int] = None,
    max_images: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    progress: ProgressCallback = None,
    download_media: bool = True,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    fc = fourchan_config(cfg)
    if not fc.get("enabled", False):
        return []
    normalized = _normalize_board(board)
    if not _allowed_board(normalized, cfg):
        _report(progress, f"4chan board /{normalized}/ skipped (not in configs/nsfw.yaml)")
        return []
    thread_limit = max_threads if max_threads is not None else int(fc.get("max_threads_per_board", 5))
    image_limit = max_images if max_images is not None else int(fc.get("max_images_per_run", 30))
    want_text = cfg.get("include_text", True)
    want_images = cfg.get("include_images", True)
    want_videos = cfg.get("include_videos", True)
    allowed_ext = _allowed_extensions(cfg)
    text_category = _board_text_category(normalized, cfg)
    items: List[Dict[str, Any]] = []
    media_saved = 0
    _report(progress, f"4chan: fetching catalog for /{normalized}/...")
    catalog = _fetch_catalog(normalized, cfg)
    _sleep(cfg)
    if not catalog:
        _report(progress, f"4chan: /{normalized}/ catalog returned 0 threads")
        return items
    thread_ids: List[int] = []
    for thread in catalog:
        thread_no = thread.get("no")
        if thread_no is None:
            continue
        thread_ids.append(int(thread_no))
        if len(thread_ids) >= thread_limit:
            break
    for thread_id in thread_ids:
        if media_saved >= image_limit:
            break
        posts = _fetch_thread(normalized, thread_id, cfg)
        _sleep(cfg)
        if not posts:
            continue
        for post in posts:
            if media_saved >= image_limit:
                break
            batch = _post_to_items(
                post,
                normalized,
                thread_id,
                text_category,
                cfg,
                allowed_ext,
                want_text,
                want_images,
                want_videos,
            )
            for item in batch:
                if item.get("kind") in ("image", "video") and download_media:
                    url = item.get("image_url") or item.get("video_url")
                    tim = post.get("tim")
                    ext = post.get("ext") or ""
                    if url and tim and ext:
                        try:
                            response = learning_http_get(url)
                            if response.status_code == 200:
                                media_cat = item.get("media_category") or "general"
                                saved = _save_media_file(normalized, int(tim), str(ext), response.content, media_cat)
                                if saved:
                                    item["local_path"] = str(saved.relative_to(get_project_root()))
                                    media_saved += 1
                                    _report(
                                        progress,
                                        f"4chan: saved /{normalized}/ media ({media_saved}/{image_limit})",
                                    )
                            _sleep(cfg)
                        except Exception:
                            pass
                items.append(item)
    _report(progress, f"4chan: /{normalized}/ collected {len(items)} item(s), {media_saved} media file(s)")
    return items


def fetch_fourchan(
    config: Optional[Dict[str, Any]] = None,
    board: Optional[str] = None,
    group: Optional[str] = None,
    max_threads: Optional[int] = None,
    max_images: Optional[int] = None,
    progress: ProgressCallback = None,
    download_media: bool = True,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    fc = fourchan_config(cfg)
    if not fc.get("enabled", False):
        _report(progress, "4chan learning disabled in configs/nsfw.yaml")
        return []
    boards, selection_label = select_boards_for_run(cfg, board=board, group=group)
    if not boards:
        _report(progress, f"4chan: {selection_label}")
        return []
    _report(progress, f"4chan: selected {len(boards)} board(s) from {selection_label}")
    total_limit = int(fc.get("max_images_per_run", 30))
    per_board_images = max_images
    if per_board_images is None and len(boards) > 1:
        per_board_images = max(1, total_limit // len(boards))
    items: List[Dict[str, Any]] = []
    for board_name in boards:
        batch = fetch_4chan_board(
            board_name,
            max_threads=max_threads,
            max_images=per_board_images,
            config=cfg,
            progress=progress,
            download_media=download_media,
        )
        items.extend(batch)
        if len(items) >= total_limit * 3:
            break
    return items


def fetch(config: Optional[Dict[str, Any]] = None, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    items = fetch_fourchan(config=cfg, download_media=False)
    if max_items and len(items) > max_items:
        return items[:max_items]
    return items
