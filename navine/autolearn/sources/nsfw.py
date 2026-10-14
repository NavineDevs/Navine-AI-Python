import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from bs4 import BeautifulSoup

from navine.autolearn.sources.reddit import fetch_reddit
from navine.nsfw.config import (
    configured_internet_sources,
    configured_source_groups,
    configured_subreddits,
    ensure_local_dirs,
    load_nsfw_config,
    local_dir_status,
    media_mix_config,
    subreddit_media_map,
)
from navine.nsfw.internet import fetch_internet_sources
from navine.nsfw.media import (
    MEDIA_CATEGORIES,
    append_media_manifest,
    apply_mix_weights,
    enrich_items,
    enrich_media_item,
)
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
REDDIT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

IMAGE_HOSTS = (
    "i.redd.it",
    "i.imgur.com",
    "preview.redd.it",
    "external-preview.redd.it",
    "i.4cdn.org",
)

VIDEO_HOSTS = (
    "v.redd.it",
    "i.imgur.com",
    "redgifs.com",
    "thumbs2.redgifs.com",
    "thumbs.redgifs.com",
    "i.4cdn.org",
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VIDEO_EXT = {".mp4", ".webm", ".gif", ".mov"}


def _report(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def _tag_item_from_subreddit(item: Dict[str, Any], subreddit: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    tagged = dict(item)
    tagged["subreddit"] = subreddit
    mapping = subreddit_media_map(cfg)
    media_category, group_tags = mapping.get(subreddit.lower(), ("general", []))
    tagged.setdefault("media_category", media_category)
    existing = tagged.get("tags") or []
    if isinstance(existing, str):
        existing = [existing]
    tagged["tags"] = list(dict.fromkeys(list(existing) + list(group_tags)))
    return enrich_media_item(tagged, cfg)


def fetch_mixed_reddit(config: Optional[Dict[str, Any]] = None, max_posts: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    limit = max_posts or int(cfg.get("max_reddit_posts", 25))
    want_images = cfg.get("include_images", True)
    want_videos = cfg.get("include_videos", True)
    groups = configured_source_groups(cfg)
    items: List[Dict[str, Any]] = []
    for group_name, group in groups.items():
        media_category = group.get("media_category") or group_name
        media_type = group.get("media_type") or "any"
        group_tags = group.get("tags") or []
        want_img = want_images and media_type in ("any", "image")
        want_vid = want_videos and media_type in ("any", "video")
        for sub in group.get("subreddits") or []:
            batch = _reddit_media_items(sub, limit, "nsfw", want_img, want_vid)
            for entry in batch:
                tagged = _tag_item_from_subreddit(entry, sub, cfg)
                tagged["source_group"] = group_name
                tagged["media_category"] = media_category
                tagged["tags"] = list(dict.fromkeys(list(tagged.get("tags") or []) + list(group_tags)))
                items.append(enrich_media_item(tagged, cfg))
            time.sleep(float(cfg.get("sleep_between_requests", 1.5)))
    mix = media_mix_config(cfg)
    if mix.get("enabled", True):
        items = apply_mix_weights(enrich_items(items, cfg), mix)
    return items


def _local_media_category(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for cat in ("hentai", "real", "human", "mixed", "animated"):
        if cat in parts:
            return cat
    return "general"


def _subfolder_label(path: Path, base: Path) -> Optional[str]:
    try:
        rel = path.relative_to(base)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) > 1:
        return parts[0].strip().lower() or None
    return None


def _media_category_for(path: Path, label: Optional[str]) -> str:
    if label and label in MEDIA_CATEGORIES:
        return label
    return _local_media_category(path)


def _media_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "media_category": item.get("media_category"),
        "subject_type": item.get("subject_type"),
        "style_type": item.get("style_type"),
        "tags": item.get("tags") or [],
        "caption": item.get("caption"),
        "source": item.get("source"),
        "source_id": item.get("source_id"),
        "subreddit": item.get("subreddit"),
        "kind": item.get("kind"),
    }


def _subreddit_list(cfg: Dict[str, Any], key: str, fallback_key: str = "reddit_subreddits") -> List[str]:
    specific = cfg.get(key) or []
    if specific:
        return [str(s).strip() for s in specific if str(s).strip()]
    general = cfg.get(fallback_key) or []
    return [str(s).strip() for s in general if str(s).strip()]


def _collect_subreddits(cfg: Dict[str, Any]) -> List[str]:
    return list(
        dict.fromkeys(
            _subreddit_list(cfg, "reddit_text_subreddits")
            + _subreddit_list(cfg, "reddit_image_subreddits")
            + _subreddit_list(cfg, "reddit_video_subreddits")
            + _subreddit_list(cfg, "reddit_subreddits")
        )
    )


def _is_image_url(url: str) -> bool:
    if not url:
        return False
    lower = url.lower()
    if any(host in lower for host in IMAGE_HOSTS):
        return True
    path = lower.split("?")[0]
    return any(path.endswith(ext) for ext in IMAGE_EXT)


def _is_video_url(url: str) -> bool:
    if not url:
        return False
    lower = url.lower()
    if any(host in lower for host in VIDEO_HOSTS):
        return True
    path = lower.split("?")[0]
    return any(path.endswith(ext) for ext in VIDEO_EXT)


def _fetch_reddit_json(subreddit: str, max_posts: int) -> List[Dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "User-Agent": REDDIT_USER_AGENT,
    }
    json_urls = [
        f"https://old.reddit.com/r/{subreddit}/hot.json?limit={max_posts}&raw_json=1",
        f"https://www.reddit.com/r/{subreddit}/hot.json?limit={max_posts}&raw_json=1",
    ]
    try:
        from navine.autolearn.http import learning_http_get

        for json_url in json_urls:
            response = learning_http_get(json_url, headers=headers)
            if response.status_code != 200:
                continue
            return response.json().get("data", {}).get("children", [])
    except Exception:
        return []
    return []


def _fetch_reddit_rss_entries(subreddit: str, max_posts: int) -> List[Dict[str, Any]]:
    headers = {
        "Accept": "application/atom+xml",
        "User-Agent": REDDIT_USER_AGENT,
    }
    rss_urls = [
        f"https://old.reddit.com/r/{subreddit}/.rss",
        f"https://www.reddit.com/r/{subreddit}/.rss",
    ]
    try:
        from navine.autolearn.http import learning_http_get

        for rss_url in rss_urls:
            response = learning_http_get(rss_url, headers=headers)
            if response.status_code != 200:
                continue
            root = ET.fromstring(response.text)
            entries: List[Dict[str, Any]] = []
            for idx, entry in enumerate(root.findall("atom:entry", ATOM_NS)):
                title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
                link_el = entry.find("atom:link", ATOM_NS)
                link = link_el.get("href", "") if link_el is not None else ""
                entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS) or f"{subreddit}:{idx}"
                content_el = entry.find("atom:content", ATOM_NS)
                content_html = ""
                if content_el is not None and content_el.text:
                    content_html = content_el.text
                summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or ""
                text = summary.strip() if summary else title
                post_id = entry_id.rstrip("/").split("/")[-1] or str(idx)
                if not title and not text:
                    continue
                entries.append(
                    {
                        "post_id": post_id,
                        "title": title,
                        "text": text,
                        "url": link,
                        "content_html": content_html,
                    }
                )
            if entries:
                return entries[:max_posts]
    except Exception:
        return []
    return []


def _extract_media_from_rss_html(
    html: str,
    subreddit: str,
    post_id: str,
    post_url: str,
    title: str,
    category: str,
    want_images: bool,
    want_videos: bool,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not html:
        return items
    soup = BeautifulSoup(html, "html.parser")
    seen: set = set()
    for img in soup.find_all("img"):
        src = (img.get("src") or "").replace("&amp;", "&").strip()
        if not src or src.startswith("data:") or src in seen:
            continue
        if want_images and _is_image_url(src):
            seen.add(src)
            items.append(
                {
                    "source": "nsfw_reddit",
                    "source_id": f"nsfw_reddit:{subreddit}:{post_id}:rss:img:{len(seen)}",
                    "title": title or f"Image from r/{subreddit}",
                    "text": title or "",
                    "url": post_url,
                    "category": category,
                    "kind": "image",
                    "image_url": src,
                    "caption": title,
                }
            )
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip().replace("&amp;", "&")
        if not href or href.startswith("#") or href in seen:
            continue
        if want_videos and _is_video_url(href):
            seen.add(href)
            items.append(
                {
                    "source": "nsfw_reddit",
                    "source_id": f"nsfw_reddit:{subreddit}:{post_id}:rss:video:{len(seen)}",
                    "title": title or f"Video from r/{subreddit}",
                    "text": title or "",
                    "url": post_url,
                    "category": category,
                    "kind": "video",
                    "video_url": href,
                    "caption": title,
                }
            )
        elif want_images and _is_image_url(href):
            seen.add(href)
            items.append(
                {
                    "source": "nsfw_reddit",
                    "source_id": f"nsfw_reddit:{subreddit}:{post_id}:rss:link:{len(seen)}",
                    "title": title or f"Image from r/{subreddit}",
                    "text": title or "",
                    "url": post_url,
                    "category": category,
                    "kind": "image",
                    "image_url": href,
                    "caption": title,
                }
            )
    return items


def _reddit_json_post_to_items(
    data: Dict[str, Any],
    subreddit: str,
    category: str,
    want_images: bool,
    want_videos: bool,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    post_id = data.get("id")
    title = (data.get("title") or "").strip()
    url = (data.get("url") or "").strip()
    selftext = (data.get("selftext") or "").strip()
    permalink = data.get("permalink", "")
    post_url = f"https://www.reddit.com{permalink}" if permalink else url
    if selftext or title:
        items.append(
            {
                "source": "nsfw_reddit",
                "source_id": f"nsfw_reddit:{subreddit}:{post_id}:text",
                "title": title,
                "text": selftext or title,
                "url": post_url,
                "category": category,
            }
        )
    video_url = url
    media = data.get("media") or {}
    reddit_video = media.get("reddit_video") or {}
    if reddit_video.get("fallback_url"):
        video_url = reddit_video["fallback_url"]
    secure_media = data.get("secure_media") or {}
    secure_reddit_video = secure_media.get("reddit_video") or {}
    if secure_reddit_video.get("fallback_url"):
        video_url = secure_reddit_video["fallback_url"]
    if want_videos and _is_video_url(video_url):
        items.append(
            {
                "source": "nsfw_reddit",
                "source_id": f"nsfw_reddit:{subreddit}:{post_id}:video",
                "title": title or f"Video from r/{subreddit}",
                "text": title or "",
                "url": post_url,
                "category": category,
                "kind": "video",
                "video_url": video_url,
                "caption": title,
            }
        )
    elif want_images and _is_image_url(url):
        items.append(
            {
                "source": "nsfw_reddit",
                "source_id": f"nsfw_reddit:{subreddit}:{post_id}:img",
                "title": title or f"Image from r/{subreddit}",
                "text": title or "",
                "url": post_url,
                "category": category,
                "kind": "image",
                "image_url": url,
                "caption": title,
            }
        )
    preview = data.get("preview") or {}
    preview_images = preview.get("images") or []
    if want_images and preview_images:
        for pidx, preview_image in enumerate(preview_images[:5]):
            source = preview_image.get("source") or {}
            img_url = source.get("url") or ""
            if img_url:
                img_url = img_url.replace("&amp;", "&")
                if _is_image_url(img_url):
                    items.append(
                        {
                            "source": "nsfw_reddit",
                            "source_id": f"nsfw_reddit:{subreddit}:{post_id}:preview:{pidx}",
                            "title": title or f"Preview r/{subreddit}",
                            "text": title or "",
                            "url": post_url,
                            "category": category,
                            "kind": "image",
                            "image_url": img_url,
                            "caption": title,
                        }
                    )
    gallery = data.get("gallery_data") or {}
    media_metadata = data.get("media_metadata") or {}
    if want_images and gallery.get("items"):
        for gidx, gitem in enumerate(gallery.get("items", [])[:10]):
            media_id = gitem.get("media_id")
            meta = media_metadata.get(media_id) or {}
            src = meta.get("s") or {}
            img_url = src.get("u") or src.get("gif") or ""
            if img_url:
                img_url = img_url.replace("&amp;", "&")
                if _is_image_url(img_url):
                    items.append(
                        {
                            "source": "nsfw_reddit",
                            "source_id": f"nsfw_reddit:{subreddit}:{post_id}:gallery:{gidx}",
                            "title": title or f"Gallery r/{subreddit}",
                            "text": title or "",
                            "url": post_url,
                            "category": category,
                            "kind": "image",
                            "image_url": img_url,
                            "caption": title,
                        }
                    )
    return items


def _reddit_rss_entry_to_items(
    entry: Dict[str, Any],
    subreddit: str,
    category: str,
    want_images: bool,
    want_videos: bool,
) -> List[Dict[str, Any]]:
    post_id = entry.get("post_id") or "unknown"
    title = (entry.get("title") or "").strip()
    text = (entry.get("text") or title).strip()
    post_url = (entry.get("url") or "").strip()
    items: List[Dict[str, Any]] = []
    if title or text:
        items.append(
            {
                "source": "nsfw_reddit",
                "source_id": f"nsfw_reddit:{subreddit}:{post_id}:text",
                "title": title,
                "text": text or title,
                "url": post_url,
                "category": category,
            }
        )
    items.extend(
        _extract_media_from_rss_html(
            entry.get("content_html") or "",
            subreddit,
            post_id,
            post_url,
            title,
            category,
            want_images,
            want_videos,
        )
    )
    return items


def _reddit_media_items(subreddit: str, max_posts: int, category: str, want_images: bool, want_videos: bool) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    children = _fetch_reddit_json(subreddit, max_posts)
    if children:
        for child in children:
            data = child.get("data", {})
            items.extend(_reddit_json_post_to_items(data, subreddit, category, want_images, want_videos))
        return items
    rss_entries = _fetch_reddit_rss_entries(subreddit, max_posts)
    if rss_entries:
        for entry in rss_entries:
            items.extend(_reddit_rss_entry_to_items(entry, subreddit, category, want_images, want_videos))
        return items
    for item in fetch_reddit(subreddit, max_posts=max_posts):
        tagged = dict(item)
        tagged["category"] = category
        tagged["source"] = "nsfw_reddit"
        items.append(tagged)
    return items


def _scan_local_dir(directory: Path, cfg: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    text_ext = {ext.lower() for ext in (cfg.get("text_extensions") or [".txt", ".md", ".jsonl"])}
    image_ext = {ext.lower() for ext in (cfg.get("image_extensions") or list(IMAGE_EXT))}
    video_ext = {ext.lower() for ext in (cfg.get("video_extensions") or list(VIDEO_EXT))}
    if not directory.exists():
        return items
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        label = _subfolder_label(path, directory)
        try:
            rel = path.relative_to(get_project_root())
        except ValueError:
            rel = path
        if suffix in text_ext:
            text_tags = [t for t in (label, category, "local") if t]
            try:
                if suffix == ".jsonl":
                    with open(path, "r", encoding="utf-8") as handle:
                        for idx, line in enumerate(handle):
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            title = entry.get("title") or entry.get("prompt") or path.stem
                            text = entry.get("text") or entry.get("content") or entry.get("response") or ""
                            items.append(
                                {
                                    "source": "nsfw_local",
                                    "source_id": f"nsfw_local:{rel}:{idx}",
                                    "title": title,
                                    "text": text,
                                    "url": str(rel),
                                    "category": category,
                                    "topic": label,
                                    "tags": list(text_tags),
                                }
                            )
                else:
                    content = path.read_text(encoding="utf-8", errors="replace").strip()
                    if content:
                        items.append(
                            {
                                "source": "nsfw_local",
                                "source_id": f"nsfw_local:{rel}",
                                "title": path.stem,
                                "text": content,
                                "url": str(rel),
                                "category": category,
                                "topic": label,
                                "tags": list(text_tags),
                            }
                        )
            except Exception:
                continue
        elif suffix in image_ext and cfg.get("include_images", True):
            media_cat = _media_category_for(path, label)
            dest_root = get_project_root() / "data" / "nsfw" / "images" / "local" / media_cat
            dest_root.mkdir(parents=True, exist_ok=True)
            dest = dest_root / f"{path.stem}_{path.stat().st_mtime_ns}{suffix}"
            if not dest.exists():
                try:
                    shutil.copy2(path, dest)
                except Exception:
                    continue
            media_tags = [t for t in (label, media_cat, "local") if t]
            item = {
                "source": "nsfw_local",
                "source_id": f"nsfw_local:img:{rel}",
                "title": path.stem,
                "text": path.stem,
                "url": str(dest.relative_to(get_project_root())),
                "category": category,
                "kind": "image",
                "image_url": str(dest),
                "caption": path.stem,
                "media_category": media_cat,
                "topic": label,
                "tags": media_tags,
            }
            items.append(enrich_media_item(item, cfg))
        elif suffix in video_ext and cfg.get("include_videos", True):
            media_cat = _media_category_for(path, label)
            dest_root = get_project_root() / "data" / "nsfw" / "videos" / "local" / media_cat
            dest_root.mkdir(parents=True, exist_ok=True)
            dest = dest_root / f"{path.stem}_{path.stat().st_mtime_ns}{suffix}"
            if not dest.exists():
                try:
                    shutil.copy2(path, dest)
                except Exception:
                    continue
            media_tags = [t for t in (label, media_cat, "local", "video") if t]
            item = {
                "source": "nsfw_local",
                "source_id": f"nsfw_local:vid:{rel}",
                "title": path.stem,
                "text": path.stem,
                "url": str(dest.relative_to(get_project_root())),
                "category": category,
                "kind": "video",
                "video_url": str(dest),
                "caption": path.stem,
                "media_category": media_cat,
                "topic": label,
                "tags": media_tags,
            }
            items.append(enrich_media_item(item, cfg))
    return items


def fetch_local(local_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, category: str = "nsfw") -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    items: List[Dict[str, Any]] = []
    if local_path:
        path = Path(local_path)
        if not path.is_absolute():
            path = get_project_root() / path
        if path.is_file():
            items.extend(_scan_local_dir(path.parent, cfg, category))
        elif path.is_dir():
            items.extend(_scan_local_dir(path, cfg, category))
        return items
    ensure_local_dirs(cfg)
    scanned: List[Path] = []
    for rel in cfg.get("local_dirs") or []:
        cat = "unrestricted" if "unrestricted" in str(rel) else "nsfw"
        directory = (get_project_root() / rel).resolve()
        if any(directory == parent or parent in directory.parents for parent in scanned):
            continue
        scanned.append(directory)
        items.extend(_scan_local_dir(directory, cfg, cat))
    return items


def fetch_reddit_nsfw(subreddit: Optional[str] = None, max_posts: Optional[int] = None, config: Optional[Dict[str, Any]] = None, category: str = "nsfw") -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    limit = max_posts or int(cfg.get("max_reddit_posts", 25))
    want_images = cfg.get("include_images", True)
    want_videos = cfg.get("include_videos", True)
    if subreddit:
        subs = [subreddit]
    else:
        subs = _collect_subreddits(cfg)
    items: List[Dict[str, Any]] = []
    for sub in subs:
        if not sub:
            continue
        cat = category
        if sub in _subreddit_list(cfg, "reddit_unrestricted_subreddits"):
            cat = "unrestricted"
        batch = _reddit_media_items(sub, limit, cat, want_images, want_videos)
        for entry in batch:
            items.append(_tag_item_from_subreddit(entry, sub, cfg))
        time.sleep(float(cfg.get("sleep_between_requests", 1.5)))
    return items


def fetch_unrestricted_reddit(config: Optional[Dict[str, Any]] = None, max_posts: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    limit = max_posts or int(cfg.get("max_reddit_posts", 25))
    items: List[Dict[str, Any]] = []
    for sub in _subreddit_list(cfg, "reddit_unrestricted_subreddits"):
        items.extend(_reddit_media_items(sub, limit, "unrestricted", False, False))
        time.sleep(float(cfg.get("sleep_between_requests", 1.5)))
    return items


def download_nsfw_images(items: List[Dict[str, Any]], progress: ProgressCallback = None) -> int:
    count = 0
    failed = 0
    skipped = 0
    try:
        from navine.learn.image_learn import download_image_url, ingest_learned_images, _save_image_entry
    except ImportError as exc:
        _report(progress, f"Warning: image learning module unavailable: {exc}")
        return 0
    for item in items:
        if item.get("kind") != "image":
            continue
        url = item.get("image_url") or item.get("url")
        if not url:
            skipped += 1
            continue
        if not _is_image_url(url) and not Path(url).exists():
            skipped += 1
            continue
        caption = item.get("caption") or item.get("title") or "nsfw image"
        meta = _media_metadata(item)
        try:
            if url.startswith("http://") or url.startswith("https://"):
                download_image_url(url, caption=caption, metadata=meta)
                count += 1
                _report(progress, f"NSFW image downloaded ({count}) [{item.get('media_category', 'general')}]")
            else:
                path = Path(url)
                if path.exists():
                    suffix = path.suffix.lower().lstrip(".") or "jpeg"
                    mime = f"image/{suffix}"
                    if suffix == "jpg":
                        mime = "image/jpeg"
                    _save_image_entry(str(path), path.read_bytes(), mime, caption, metadata=meta)
                    count += 1
                    _report(progress, f"NSFW local image ingested ({count})")
                else:
                    skipped += 1
        except Exception as exc:
            failed += 1
            _report(progress, f"Warning: NSFW image download failed for {str(url)[:100]}: {exc}")
    local_root = get_project_root() / "data" / "nsfw" / "images" / "local"
    if local_root.exists():
        try:
            from navine.learn.image_learn import _save_image_entry

            for path in local_root.iterdir():
                if path.suffix.lower() not in IMAGE_EXT:
                    continue
                try:
                    _save_image_entry(str(path), path.read_bytes(), f"image/{path.suffix[1:]}", path.stem)
                    count += 1
                except Exception:
                    failed += 1
        except Exception as exc:
            _report(progress, f"Warning: local NSFW image ingest failed: {exc}")
    if skipped:
        _report(progress, f"Warning: skipped {skipped} image item(s) with missing or unsupported URLs.")
    if failed:
        _report(progress, f"Warning: {failed} NSFW image download(s) failed.")
    if count:
        try:
            append_media_manifest([item for item in items if item.get("kind") == "image"])
            ingest_learned_images()
        except Exception as exc:
            _report(progress, f"Warning: image ingest step failed: {exc}")
    return count


def download_nsfw_videos(items: List[Dict[str, Any]], progress: ProgressCallback = None) -> int:
    count = 0
    failed = 0
    skipped = 0
    try:
        from navine.learn.video_learn import download_gif_url, download_video_url, ingest_learned_video
    except ImportError as exc:
        _report(progress, f"Warning: video learning module unavailable: {exc}")
        return 0
    for item in items:
        if item.get("kind") != "video":
            continue
        url = item.get("video_url") or item.get("url")
        if not url:
            skipped += 1
            continue
        lower = url.lower()
        meta = _media_metadata(item)
        try:
            if url.startswith("http://") or url.startswith("https://"):
                if lower.endswith(".gif") or "gif" in lower:
                    download_gif_url(url, metadata=meta)
                else:
                    download_video_url(url, metadata=meta)
                count += 1
                _report(progress, f"NSFW video downloaded ({count}) [{item.get('media_category', 'general')}]")
            elif Path(url).exists():
                from navine.learn import video_learn

                path = Path(url)
                content = path.read_bytes()
                lower = str(path).lower()
                if lower.endswith(".gif"):
                    frames = video_learn._extract_gif_frames(content)
                    video_learn._write_sequence(str(path), frames, "gif")
                else:
                    frames = video_learn._extract_video_frames(content)
                    if frames:
                        video_learn._write_sequence(str(path), frames, "video")
                    else:
                        skipped += 1
                        continue
                count += 1
                _report(progress, f"NSFW local video ingested ({count})")
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            _report(progress, f"Warning: NSFW video download failed for {str(url)[:100]}: {exc}")
    local_root = get_project_root() / "data" / "nsfw" / "videos" / "local"
    if local_root.exists():
        try:
            from navine.learn import video_learn

            for path in local_root.iterdir():
                if not path.is_file() or path.suffix.lower() not in VIDEO_EXT:
                    continue
                try:
                    content = path.read_bytes()
                    lower = str(path).lower()
                    if lower.endswith(".gif"):
                        frames = video_learn._extract_gif_frames(content)
                        video_learn._write_sequence(str(path), frames, "gif")
                    else:
                        frames = video_learn._extract_video_frames(content)
                        if frames:
                            video_learn._write_sequence(str(path), frames, "video")
                        else:
                            failed += 1
                            continue
                    count += 1
                except Exception:
                    failed += 1
        except Exception as exc:
            _report(progress, f"Warning: local NSFW video ingest failed: {exc}")
    if skipped:
        _report(progress, f"Warning: skipped {skipped} video item(s) with missing or unsupported URLs.")
    if failed:
        _report(progress, f"Warning: {failed} NSFW video download(s) failed.")
    if count:
        try:
            append_media_manifest([item for item in items if item.get("kind") == "video"])
            ingest_learned_video()
        except Exception as exc:
            _report(progress, f"Warning: video ingest step failed: {exc}")
    return count


def fetch_mixed_cycle(progress: ProgressCallback = None, config: Optional[Dict[str, Any]] = None, max_items: Optional[int] = None) -> Dict[str, Any]:
    cfg = dict(config or load_nsfw_config())
    cfg["mixed_mode"] = True
    return fetch_full_cycle(progress=progress, config=cfg, max_items=max_items)


def fetch_full_cycle(progress: ProgressCallback = None, config: Optional[Dict[str, Any]] = None, max_items: Optional[int] = None) -> Dict[str, Any]:
    cfg = dict(config or load_nsfw_config())
    limit = max_items or int(cfg.get("max_items_per_run", 80))
    warnings: List[str] = []
    text_items: List[Dict[str, Any]] = []
    subs = configured_subreddits(cfg)
    created_dirs = ensure_local_dirs(cfg)
    local_status = local_dir_status(cfg)
    if not subs:
        warnings.append(
            "No Reddit subreddits configured in configs/nsfw.yaml. "
            "Add reddit_subreddits, reddit_text_subreddits, reddit_image_subreddits, "
            "reddit_video_subreddits, or reddit_unrestricted_subreddits."
        )
    elif cfg.get("include_text", True) or cfg.get("include_images", True) or cfg.get("include_videos", True):
        _report(progress, f"Reddit: scanning {len(subs)} configured subreddit(s)...")
    _report(progress, "Fetching NSFW and unrestricted text from Reddit...")
    reddit_count = 0
    if cfg.get("include_text", True) or cfg.get("include_images", True) or cfg.get("include_videos", True):
        if subs:
            if cfg.get("mixed_mode", True):
                _report(progress, "Fetching mixed NSFW Reddit groups (hentai, real, human, video)...")
                reddit_items = fetch_mixed_reddit(config=cfg)
            else:
                reddit_items = fetch_reddit_nsfw(config=cfg)
            reddit_count = len(reddit_items)
            text_items.extend(reddit_items)
        else:
            _report(progress, "Reddit fetch skipped: no subreddits configured.")
    if cfg.get("include_unrestricted", True):
        unrestricted_subs = _subreddit_list(cfg, "reddit_unrestricted_subreddits")
        if unrestricted_subs:
            unrestricted_items = fetch_unrestricted_reddit(config=cfg)
            reddit_count += len(unrestricted_items)
            text_items.extend(unrestricted_items)
        try:
            from navine.autolearn.sources.unrestricted import fetch_unrestricted_topics

            topic_items = fetch_unrestricted_topics(config=cfg)
            text_items.extend(topic_items)
        except Exception:
            pass
    _report(progress, "Scanning local NSFW and unrestricted folders...")
    if created_dirs:
        for directory in created_dirs:
            _report(progress, f"Created local folder: {directory.relative_to(get_project_root())}")
    local_before = len(text_items)
    text_items.extend(fetch_local(config=cfg))
    local_count = len(text_items) - local_before
    if local_status["file_count"] == 0:
        warnings.append(
            "Local folders are empty. Add text, images, or videos to "
            "data/nsfw/local/ and/or data/unrestricted/local/."
        )
    elif local_count == 0 and local_status["file_count"] > 0:
        warnings.append(
            "Local files were found but none were ingested. Check file extensions in configs/nsfw.yaml."
        )
    else:
        _report(progress, f"Local scan: {local_count} item(s) from {local_status['file_count']} file(s).")
    fourchan_count = 0
    try:
        from navine.autolearn.sources.fourchan import fetch_fourchan, fourchan_config

        if fourchan_config(cfg).get("enabled", False):
            _report(progress, "Fetching 4chan boards (official JSON API)...")
            fourchan_items = fetch_fourchan(config=cfg, progress=progress, download_media=True)
            fourchan_count = len(fourchan_items)
            text_items.extend(enrich_items(fourchan_items, cfg))
            _report(progress, f"4chan: {fourchan_count} item(s) from configured boards.")
    except Exception as exc:
        warnings.append(f"4chan fetch failed: {exc}")
    waifu_count = 0
    waifu_downloaded = 0
    try:
        from navine.reference.sources.waifu import fetch_and_download_waifu, waifu_im_config

        if waifu_im_config(cfg).get("enabled", False):
            _report(progress, "Fetching waifu.im hentai references (REST API)...")
            waifu_result = fetch_and_download_waifu(config=cfg, progress=progress)
            waifu_items = waifu_result.get("items") or []
            waifu_count = len(waifu_items)
            waifu_downloaded = int(waifu_result.get("downloaded") or 0)
            text_items.extend(enrich_items(waifu_items, cfg))
            _report(progress, f"Waifu.im: {waifu_downloaded} reference image(s) saved to hentai folder.")
    except Exception as exc:
        warnings.append(f"waifu.im fetch failed: {exc}")
    infini_count = 0
    infini_downloaded = 0
    try:
        from navine.reference.sources.infini_atomic import (
            fetch_and_download_infini_atomic,
            infini_atomic_config,
        )

        if infini_atomic_config(cfg).get("enabled", False):
            _report(progress, "Fetching infini-atomic.w3spaces.com hentai references...")
            infini_result = fetch_and_download_infini_atomic(config=cfg, progress=progress)
            infini_items = infini_result.get("items") or []
            infini_count = len(infini_items)
            infini_downloaded = int(infini_result.get("downloaded") or 0)
            text_items.extend(enrich_items(infini_items, cfg))
            _report(
                progress,
                f"Infini-atomic: {infini_downloaded} reference image(s) saved to hentai folder.",
            )
    except Exception as exc:
        warnings.append(f"infini-atomic fetch failed: {exc}")
    internet_counts = {}
    internet_total = 0
    source_summary = configured_internet_sources(cfg)
    if any(source_summary.values()):
        _report(progress, "Fetching NSFW internet sources (web, crawl, feeds, search)...")
        internet_result = fetch_internet_sources(config=cfg, progress=progress)
        internet_items = internet_result.get("items") or []
        internet_counts = internet_result.get("counts") or {}
        internet_total = len(internet_items)
        text_items.extend(enrich_items(internet_items, cfg))
        _report(
            progress,
            "Internet sources: "
            f"{internet_counts.get('reddit_posts', 0)} post URLs, "
            f"{internet_counts.get('web_urls', 0)} web, "
            f"{internet_counts.get('crawl', 0)} crawl, "
            f"{internet_counts.get('direct_media', 0)} direct, "
            f"{internet_counts.get('feeds', 0)} feeds, "
            f"{internet_counts.get('search', 0)} search, "
            f"waifu {waifu_downloaded}, infini {infini_downloaded}.",
        )
    if reddit_count == 0 and subs:
        warnings.append(
            "Reddit returned 0 items. Subreddits may be private, banned, or rate-limited. "
            "Check subreddit names and network access."
        )
    image_candidates = sum(1 for item in text_items if item.get("kind") == "image")
    video_candidates = sum(1 for item in text_items if item.get("kind") == "video")
    if cfg.get("include_images", True) and image_candidates == 0:
        warnings.append(
            "No image items collected from Reddit RSS/JSON, internet sources, or local folders. "
            "Reddit JSON may be blocked (403); verify RSS media extraction and booru search sources."
        )
    if cfg.get("include_videos", True) and video_candidates == 0:
        warnings.append(
            "No video items collected from Reddit RSS/JSON, internet sources, or local folders."
        )
    if image_candidates or video_candidates:
        _report(
            progress,
            f"Media candidates: {image_candidates} image(s), {video_candidates} video(s).",
        )
    images_downloaded = 0
    videos_downloaded = 0
    if cfg.get("include_images", True):
        _report(progress, f"Downloading NSFW images ({image_candidates} candidate(s))...")
        images_downloaded = download_nsfw_images(text_items, progress=progress)
        if image_candidates > 0 and images_downloaded == 0:
            warnings.append(
                f"Found {image_candidates} image item(s) but downloaded 0. Check network access and URL policy."
            )
    if cfg.get("include_videos", True):
        _report(progress, f"Downloading NSFW videos/GIFs ({video_candidates} candidate(s))...")
        videos_downloaded = download_nsfw_videos(text_items, progress=progress)
        if video_candidates > 0 and videos_downloaded == 0:
            warnings.append(
                f"Found {video_candidates} video item(s) but downloaded 0. Check network access and URL policy."
            )
    for warning in warnings:
        _report(progress, f"Warning: {warning}")
    text_items = enrich_items(text_items, cfg)
    mix = media_mix_config(cfg)
    if mix.get("enabled", True) and cfg.get("mixed_mode", True):
        text_items = apply_mix_weights(text_items, mix)
    if len(text_items) > limit:
        text_items = text_items[:limit]
    category_counts: Dict[str, int] = {}
    for item in text_items:
        cat = item.get("media_category") or "general"
        category_counts[cat] = category_counts.get(cat, 0) + 1
    return {
        "text_items": text_items,
        "images_downloaded": images_downloaded,
        "videos_downloaded": videos_downloaded,
        "items_fetched": len(text_items),
        "reddit_items": reddit_count,
        "fourchan_items": fourchan_count,
        "waifu_items": waifu_count,
        "waifu_downloaded": waifu_downloaded,
        "infini_items": infini_count,
        "infini_downloaded": infini_downloaded,
        "local_items": local_count,
        "internet_items": internet_total,
        "internet_counts": internet_counts,
        "image_candidates": image_candidates,
        "video_candidates": video_candidates,
        "media_category_counts": category_counts,
        "warnings": warnings,
    }


def fetch(config: Optional[Dict[str, Any]] = None, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    result = fetch_full_cycle(config=config, max_items=max_items)
    return result.get("text_items") or []
