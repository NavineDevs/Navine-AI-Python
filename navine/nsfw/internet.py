import json
import re
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from navine.autolearn.http import learning_http_get
from navine.autolearn.sources.feeds import fetch_feed
from navine.learn.web import USER_AGENT, TIMEOUT
from navine.nsfw.config import load_nsfw_config
from navine.nsfw.media import enrich_media_item
from navine.policy import assert_learning_url_allowed, is_learning_url_allowed
from navine.utils.paths import get_project_root

REDDIT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

IMAGE_HOSTS = (
    "i.redd.it",
    "i.imgur.com",
    "preview.redd.it",
    "external-preview.redd.it",
    "imgur.com",
    "thumbs.redgifs.com",
    "thumbs2.redgifs.com",
    "rule34.xxx",
    "danbooru.donmai.us",
    "gelbooru.com",
    "e621.net",
    "sankakucomplex.com",
    "i.4cdn.org",
    "cdn.waifu.im",
)

VIDEO_HOSTS = (
    "v.redd.it",
    "i.imgur.com",
    "redgifs.com",
    "www.redgifs.com",
    "gfycat.com",
    "thumbs.redgifs.com",
    "thumbs2.redgifs.com",
    "i.4cdn.org",
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".webm", ".gif", ".mov", ".m4v"}

ProgressCallback = Optional[Callable[[str], None]]


def _report(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def _sleep(cfg: Dict[str, Any]) -> None:
    time.sleep(float(cfg.get("sleep_between_requests", 1.5)))


def _nsfw_url_allowed(url: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    if not url:
        return False
    if is_learning_url_allowed(url):
        return True
    merged = cfg or load_nsfw_config()
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().split(":")[0]
    if not host:
        return False
    for domain in merged.get("allowed_learning_domains") or []:
        domain = str(domain).strip().lower()
        if not domain:
            continue
        if host == domain or host.endswith("." + domain):
            return True
    for pattern in merged.get("allowed_url_patterns") or []:
        pattern = str(pattern).strip()
        if pattern and re.search(pattern, url, re.I):
            return True
    return False


def _assert_nsfw_url(url: str, cfg: Optional[Dict[str, Any]] = None) -> None:
    if _nsfw_url_allowed(url, cfg):
        try:
            assert_learning_url_allowed(url)
        except Exception:
            pass
        return
    raise RuntimeError(f"Navine AI - Python NSFW policy blocked URL: {url}")


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


def _reddit_post_json_url(url: str) -> Optional[str]:
    match = re.search(r"reddit\.com/(?:r/[\w]+/)?comments/([a-z0-9]+)", url, re.I)
    if not match:
        return None
    post_id = match.group(1)
    return post_id


def _reddit_json_endpoints(post_id: str) -> List[str]:
    return [
        f"https://www.reddit.com/comments/{post_id}.json?raw_json=1",
        f"https://old.reddit.com/comments/{post_id}.json?raw_json=1",
        f"https://www.reddit.com/comments/{post_id}/.json?raw_json=1",
    ]


def _reddit_html_endpoints(url: str, post_id: str) -> List[str]:
    endpoints = [url]
    if "old.reddit.com" not in url:
        endpoints.append(re.sub(r"https?://(?:www\.)?reddit\.com", "https://old.reddit.com", url, flags=re.I))
    endpoints.append(f"https://old.reddit.com/comments/{post_id}/")
    return list(dict.fromkeys(endpoints))


def _fetch_reddit_post_json(post_id: str, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headers = {"Accept": "application/json", "User-Agent": REDDIT_USER_AGENT}
    for endpoint in _reddit_json_endpoints(post_id):
        try:
            response = learning_http_get(endpoint, headers=headers, timeout=TIMEOUT)
            if response.status_code != 200:
                continue
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                continue
            children = payload[0].get("data", {}).get("children", [])
            if children:
                return children[0].get("data", {})
        except Exception:
            continue
        _sleep(cfg)
    return None


def _fetch_reddit_post_html(url: str, post_id: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    subreddit = _subreddit_from_url(url)
    headers = {"User-Agent": REDDIT_USER_AGENT, "Accept": "text/html"}
    for page_url in _reddit_html_endpoints(url, post_id):
        try:
            response = learning_http_get(page_url, headers=headers, timeout=TIMEOUT)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            title = ""
            title_el = soup.find("a", class_="title") or soup.find("h1")
            if title_el:
                title = title_el.get_text(strip=True)
            if not title and soup.title and soup.title.string:
                title = soup.title.string.strip()
            post_url = page_url
            if title:
                items.append(
                    {
                        "source": "nsfw_reddit_post",
                        "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:html:text",
                        "title": title,
                        "text": title,
                        "url": post_url,
                        "category": "nsfw",
                    }
                )
            for meta_name, attr in (("og:video", "content"), ("og:video:url", "content"), ("og:video:secure_url", "content")):
                tag = soup.find("meta", property=meta_name) or soup.find("meta", attrs={"name": meta_name})
                if tag and tag.get(attr):
                    video_url = tag[attr]
                    if _is_video_url(video_url):
                        items.append(
                            {
                                "source": "nsfw_reddit_post",
                                "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:html:video",
                                "title": title or f"Video from r/{subreddit}",
                                "text": title or "",
                                "url": post_url,
                                "category": "nsfw",
                                "kind": "video",
                                "video_url": video_url,
                                "caption": title,
                            }
                        )
            for meta_name in ("og:image", "twitter:image"):
                tag = soup.find("meta", property=meta_name) or soup.find("meta", attrs={"name": meta_name})
                if tag and tag.get("content"):
                    image_url = tag["content"]
                    if _is_image_url(image_url):
                        items.append(
                            {
                                "source": "nsfw_reddit_post",
                                "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:html:img",
                                "title": title or f"Image from r/{subreddit}",
                                "text": title or "",
                                "url": post_url,
                                "category": "nsfw",
                                "kind": "image",
                                "image_url": image_url,
                                "caption": title,
                            }
                        )
            for video in soup.find_all("video"):
                src = video.get("src")
                if src:
                    video_url = urljoin(page_url, src)
                    if _is_video_url(video_url):
                        items.append(
                            {
                                "source": "nsfw_reddit_post",
                                "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:html:vidtag",
                                "title": title or f"Video from r/{subreddit}",
                                "text": title or "",
                                "url": post_url,
                                "category": "nsfw",
                                "kind": "video",
                                "video_url": video_url,
                                "caption": title,
                            }
                        )
                for source_tag in video.find_all("source"):
                    src = source_tag.get("src")
                    if src:
                        video_url = urljoin(page_url, src)
                        if _is_video_url(video_url):
                            items.append(
                                {
                                    "source": "nsfw_reddit_post",
                                    "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:html:source",
                                    "title": title or f"Video from r/{subreddit}",
                                    "text": title or "",
                                    "url": post_url,
                                    "category": "nsfw",
                                    "kind": "video",
                                    "video_url": video_url,
                                    "caption": title,
                                }
                            )
            expando = soup.find("div", class_="expando")
            if expando:
                items.extend(_extract_media_from_html(str(expando), page_url, title, "nsfw"))
            if items:
                return items
        except Exception:
            continue
        _sleep(cfg)
    return items


def _subreddit_from_url(url: str) -> str:
    match = re.search(r"reddit\.com/r/([\w]+)/", url, re.I)
    return match.group(1) if match else "reddit"


def _reddit_post_data_to_items(
    data: Dict[str, Any],
    subreddit: str,
    category: str,
    want_images: bool,
    want_videos: bool,
    want_text: bool,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    post_id = data.get("id")
    title = (data.get("title") or "").strip()
    url = (data.get("url") or "").strip()
    selftext = (data.get("selftext") or "").strip()
    permalink = data.get("permalink", "")
    post_url = f"https://www.reddit.com{permalink}" if permalink else url
    if want_text and (selftext or title):
        items.append(
            {
                "source": "nsfw_reddit_post",
                "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:text",
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
                "source": "nsfw_reddit_post",
                "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:video",
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
                "source": "nsfw_reddit_post",
                "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:img",
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
                            "source": "nsfw_reddit_post",
                            "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:preview:{pidx}",
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
                            "source": "nsfw_reddit_post",
                            "source_id": f"nsfw_reddit_post:{subreddit}:{post_id}:gallery:{gidx}",
                            "title": title or f"Gallery r/{subreddit}",
                            "text": title or "",
                            "url": post_url,
                            "category": category,
                            "kind": "image",
                            "image_url": img_url,
                            "caption": title,
                        }
                    )
    crossposts = data.get("crosspost_parent_list") or []
    for cidx, crosspost in enumerate(crossposts[:3]):
        items.extend(
            _reddit_post_data_to_items(
                crosspost,
                subreddit,
                category,
                want_images,
                want_videos,
                want_text and cidx == 0,
            )
        )
    return items


def fetch_reddit_post(
    url: str,
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    post_id = _reddit_post_json_url(url)
    if not post_id:
        return []
    _assert_nsfw_url(url, cfg)
    want_images = cfg.get("include_images", True)
    want_videos = cfg.get("include_videos", True)
    want_text = cfg.get("include_text", True)
    subreddit = _subreddit_from_url(url)
    data = _fetch_reddit_post_json(post_id, cfg)
    if data:
        return _reddit_post_data_to_items(data, subreddit, category, want_images, want_videos, want_text)
    return _fetch_reddit_post_html(url, post_id, cfg)


def _extract_media_from_html(html: str, base_url: str, title: str, category: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "html.parser")
    seen: Set[str] = set()
    page_title = title
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    for tag in soup.find_all("img"):
        src = tag.get("src") or tag.get("data-src") or tag.get("data-lazy-src") or tag.get("data-original")
        if not src or src.startswith("data:"):
            continue
        full = urljoin(base_url, src)
        if full in seen:
            continue
        seen.add(full)
        if _is_image_url(full):
            alt = (tag.get("alt") or page_title or "").strip()
            items.append(
                {
                    "source": "nsfw_web",
                    "source_id": f"nsfw_web:img:{hash(full) & 0xFFFFFFFF:08x}",
                    "title": alt or page_title,
                    "text": alt or page_title,
                    "url": base_url,
                    "category": category,
                    "kind": "image",
                    "image_url": full,
                    "caption": alt or page_title,
                }
            )
    for tag in soup.find_all("video"):
        src = tag.get("src")
        if src:
            full = urljoin(base_url, src)
            if full not in seen and _is_video_url(full):
                seen.add(full)
                items.append(
                    {
                        "source": "nsfw_web",
                        "source_id": f"nsfw_web:vid:{hash(full) & 0xFFFFFFFF:08x}",
                        "title": page_title,
                        "text": page_title,
                        "url": base_url,
                        "category": category,
                        "kind": "video",
                        "video_url": full,
                        "caption": page_title,
                    }
                )
        for source_tag in tag.find_all("source"):
            src = source_tag.get("src")
            if not src:
                continue
            full = urljoin(base_url, src)
            if full in seen:
                continue
            seen.add(full)
            if _is_video_url(full):
                items.append(
                    {
                        "source": "nsfw_web",
                        "source_id": f"nsfw_web:vid:{hash(full) & 0xFFFFFFFF:08x}",
                        "title": page_title,
                        "text": page_title,
                        "url": base_url,
                        "category": category,
                        "kind": "video",
                        "video_url": full,
                        "caption": page_title,
                    }
                )
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        full = urljoin(base_url, href.split("#")[0])
        if full in seen:
            continue
        if _is_image_url(full):
            seen.add(full)
            label = (tag.get_text() or page_title or "").strip()
            items.append(
                {
                    "source": "nsfw_web",
                    "source_id": f"nsfw_web:img:{hash(full) & 0xFFFFFFFF:08x}",
                    "title": label or page_title,
                    "text": label or page_title,
                    "url": base_url,
                    "category": category,
                    "kind": "image",
                    "image_url": full,
                    "caption": label or page_title,
                }
            )
        elif _is_video_url(full):
            seen.add(full)
            label = (tag.get_text() or page_title or "").strip()
            items.append(
                {
                    "source": "nsfw_web",
                    "source_id": f"nsfw_web:vid:{hash(full) & 0xFFFFFFFF:08x}",
                    "title": label or page_title,
                    "text": label or page_title,
                    "url": base_url,
                    "category": category,
                    "kind": "video",
                    "video_url": full,
                    "caption": label or page_title,
                }
            )
    return items


def _extract_page_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    main = soup.find("main") or soup.find("article") or soup.find("body")
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def fetch_web_url(
    url: str,
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    _assert_nsfw_url(url, cfg)
    if _reddit_post_json_url(url):
        return fetch_reddit_post(url, config=cfg, category=category)
    if _is_image_url(url):
        return [
            {
                "source": "nsfw_direct",
                "source_id": f"nsfw_direct:img:{hash(url) & 0xFFFFFFFF:08x}",
                "title": url,
                "text": url,
                "url": url,
                "category": category,
                "kind": "image",
                "image_url": url,
                "caption": urlparse(url).path.split("/")[-1],
            }
        ]
    if _is_video_url(url):
        return [
            {
                "source": "nsfw_direct",
                "source_id": f"nsfw_direct:vid:{hash(url) & 0xFFFFFFFF:08x}",
                "title": url,
                "text": url,
                "url": url,
                "category": category,
                "kind": "video",
                "video_url": url,
                "caption": urlparse(url).path.split("/")[-1],
            }
        ]
    items: List[Dict[str, Any]] = []
    try:
        response = learning_http_get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if response.status_code != 200:
            return items
        content_type = response.headers.get("Content-Type", "")
        if content_type.lower().startswith("image/"):
            return [
                {
                    "source": "nsfw_direct",
                    "source_id": f"nsfw_direct:img:{hash(url) & 0xFFFFFFFF:08x}",
                    "title": url,
                    "text": url,
                    "url": url,
                    "category": category,
                    "kind": "image",
                    "image_url": url,
                    "caption": urlparse(url).path.split("/")[-1],
                }
            ]
        if content_type.lower().startswith("video/"):
            return [
                {
                    "source": "nsfw_direct",
                    "source_id": f"nsfw_direct:vid:{hash(url) & 0xFFFFFFFF:08x}",
                    "title": url,
                    "text": url,
                    "url": url,
                    "category": category,
                    "kind": "video",
                    "video_url": url,
                    "caption": urlparse(url).path.split("/")[-1],
                }
            ]
        title, text = _extract_page_text(response.text)
        if cfg.get("include_text", True) and text:
            items.append(
                {
                    "source": "nsfw_web",
                    "source_id": f"nsfw_web:text:{hash(url) & 0xFFFFFFFF:08x}",
                    "title": title or url,
                    "text": text[:8000],
                    "url": url,
                    "category": category,
                }
            )
        if cfg.get("include_images", True) or cfg.get("include_videos", True):
            items.extend(_extract_media_from_html(response.text, url, title, category))
    except Exception:
        pass
    return items


def _same_domain(base: str, url: str) -> bool:
    return urlparse(base).netloc == urlparse(url).netloc


def _extract_links(page_url: str, html: str) -> Set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        full = urljoin(page_url, href.split("#")[0])
        parsed = urlparse(full)
        if parsed.scheme in ("http", "https"):
            links.add(full)
    return links


def fetch_crawl_url(
    start_url: str,
    max_pages: int = 15,
    same_domain_only: bool = True,
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    _assert_nsfw_url(start_url, cfg)
    items: List[Dict[str, Any]] = []
    visited: Set[str] = set()
    queue: deque = deque([start_url])
    count = 0
    while queue and count < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if not _nsfw_url_allowed(url, cfg):
            continue
        try:
            response = learning_http_get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            if response.status_code != 200:
                continue
            title, text = _extract_page_text(response.text)
            if cfg.get("include_text", True) and text:
                items.append(
                    {
                        "source": "nsfw_crawl",
                        "source_id": f"nsfw_crawl:text:{hash(url) & 0xFFFFFFFF:08x}",
                        "title": title or url,
                        "text": text[:8000],
                        "url": url,
                        "category": category,
                    }
                )
            if cfg.get("include_images", True) or cfg.get("include_videos", True):
                items.extend(_extract_media_from_html(response.text, url, title, category))
            count += 1
            _report(progress, f"Crawled page {count}/{max_pages}: {url}")
            for link in _extract_links(url, response.text):
                if link in visited:
                    continue
                if same_domain_only and not _same_domain(start_url, link):
                    continue
                if _nsfw_url_allowed(link, cfg):
                    queue.append(link)
            _sleep(cfg)
        except Exception:
            continue
    return items


def fetch_direct_media(
    urls: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    source_urls = urls or cfg.get("direct_media_urls") or []
    items: List[Dict[str, Any]] = []
    for url in source_urls:
        url = str(url).strip()
        if not url:
            continue
        try:
            items.extend(fetch_web_url(url, config=cfg, category=category))
            _sleep(cfg)
        except Exception:
            continue
    return items


def fetch_configured_feeds(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    feed_urls = cfg.get("feed_urls") or []
    max_items = int(cfg.get("max_feed_items", 20))
    items: List[Dict[str, Any]] = []
    per_feed = max(3, max_items // max(1, len(feed_urls)))
    for feed_url in feed_urls:
        feed_url = str(feed_url).strip()
        if not feed_url or not _nsfw_url_allowed(feed_url, cfg):
            continue
        try:
            batch = fetch_feed(feed_url, max_items=per_feed)
            for entry in batch:
                tagged = dict(entry)
                tagged["source"] = "nsfw_feed"
                tagged["category"] = category
                items.append(tagged)
            _sleep(cfg)
        except Exception:
            continue
    return items[:max_items]


def _search_item(
    source_type: str,
    post_id: Any,
    idx: int,
    file_url: str,
    tags_text: str,
    category: str,
    source_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    media_category = source_cfg.get("media_category") or category
    extra_tags = source_cfg.get("tags") or []
    base_tags = tags_text.replace("_", " ")
    kind = None
    payload: Dict[str, Any] = {
        "source": "nsfw_search",
        "title": base_tags[:120],
        "text": base_tags,
        "url": file_url,
        "category": category,
        "search_tags": base_tags,
        "media_category": media_category,
        "tags": list(dict.fromkeys(_normalize_tag_list(extra_tags) + _normalize_tag_list(base_tags.split()))),
    }
    if _is_video_url(file_url):
        kind = "video"
        payload.update(
            {
                "source_id": f"nsfw_search:{source_type}:{post_id}:video",
                "kind": "video",
                "video_url": file_url,
                "caption": base_tags[:120],
            }
        )
    elif _is_image_url(file_url):
        kind = "image"
        payload.update(
            {
                "source_id": f"nsfw_search:{source_type}:{post_id}:img",
                "kind": "image",
                "image_url": file_url,
                "caption": base_tags[:120],
            }
        )
    if not kind:
        return None
    return enrich_media_item(payload)


def _normalize_tag_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [t.strip().lower() for t in raw.replace("_", " ").split() if t.strip()]
    if isinstance(raw, list):
        return [str(t).strip().lower() for t in raw if str(t).strip()]
    return []


def _fetch_rule34(source_cfg: Dict[str, Any], cfg: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    base_url = str(source_cfg.get("base_url") or "https://rule34.xxx").rstrip("/")
    tags = str(source_cfg.get("tags") or source_cfg.get("query") or "hentai")
    limit = int(source_cfg.get("limit") or cfg.get("max_search_images", 15))
    api_url = f"{base_url}/index.php?page=dapi&s=post&q=index&json=1&tags={tags}&limit={limit}"
    if not _nsfw_url_allowed(api_url, cfg):
        return items
    try:
        response = learning_http_get(
            api_url,
            headers={"Accept": "application/json", "User-Agent": REDDIT_USER_AGENT},
        )
        if response.status_code != 200:
            return items
        payload = response.json()
        posts = payload if isinstance(payload, list) else payload.get("post", [])
        if isinstance(posts, dict):
            posts = [posts]
        for idx, post in enumerate(posts[:limit]):
            file_url = post.get("file_url") or post.get("sample_url") or ""
            if not file_url:
                continue
            tags_text = (post.get("tags") or tags).replace("_", " ")
            item = _search_item("rule34", post.get("id", idx), idx, file_url, tags_text, category, source_cfg)
            if item:
                items.append(item)
    except Exception:
        pass
    return items


def _fetch_danbooru(source_cfg: Dict[str, Any], cfg: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    base_url = str(source_cfg.get("base_url") or "https://danbooru.donmai.us").rstrip("/")
    tags = str(source_cfg.get("tags") or "rating:explicit")
    limit = int(source_cfg.get("limit") or cfg.get("max_search_images", 15))
    api_url = f"{base_url}/posts.json?tags={tags}&limit={limit}"
    if not _nsfw_url_allowed(api_url, cfg):
        return items
    try:
        response = learning_http_get(api_url, headers={"Accept": "application/json"})
        if response.status_code != 200:
            return items
        posts = response.json()
        if not isinstance(posts, list):
            return items
        for post in posts[:limit]:
            file_url = post.get("large_file_url") or post.get("file_url") or ""
            if not file_url:
                continue
            tag_string = (post.get("tag_string") or tags).replace("_", " ")
            item = _search_item("danbooru", post.get("id"), 0, file_url, tag_string, category, source_cfg)
            if item:
                items.append(item)
    except Exception:
        pass
    return items


def _fetch_gelbooru(source_cfg: Dict[str, Any], cfg: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    base_url = str(source_cfg.get("base_url") or "https://gelbooru.com").rstrip("/")
    tags = str(source_cfg.get("tags") or "rating:explicit")
    limit = int(source_cfg.get("limit") or cfg.get("max_search_images", 15))
    api_url = f"{base_url}/index.php?page=dapi&s=post&q=index&json=1&tags={tags}&limit={limit}"
    if not _nsfw_url_allowed(api_url, cfg):
        return items
    try:
        response = learning_http_get(api_url, headers={"Accept": "application/json"})
        if response.status_code != 200:
            return items
        payload = response.json()
        posts = payload.get("post", []) if isinstance(payload, dict) else payload
        if isinstance(posts, dict):
            posts = [posts]
        for idx, post in enumerate((posts or [])[:limit]):
            file_url = post.get("file_url") or post.get("sample_url") or ""
            if not file_url:
                continue
            tags_text = (post.get("tags") or tags).replace("_", " ")
            item = _search_item("gelbooru", post.get("id", idx), idx, file_url, tags_text, category, source_cfg)
            if item:
                items.append(item)
    except Exception:
        pass
    return items


def _fetch_e621(source_cfg: Dict[str, Any], cfg: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    base_url = str(source_cfg.get("base_url") or "https://e621.net").rstrip("/")
    tags = str(source_cfg.get("tags") or "rating:e")
    limit = int(source_cfg.get("limit") or cfg.get("max_search_images", 15))
    api_url = f"{base_url}/posts.json?tags={tags}&limit={limit}"
    if not _nsfw_url_allowed(api_url, cfg):
        return items
    try:
        response = learning_http_get(
            api_url,
            headers={"Accept": "application/json", "User-Agent": REDDIT_USER_AGENT},
        )
        if response.status_code != 200:
            return items
        payload = response.json()
        posts = payload.get("posts") or []
        for post in posts[:limit]:
            file_obj = post.get("file") or {}
            file_url = file_obj.get("url") or ""
            if not file_url:
                continue
            tag_string = " ".join(post.get("tags", {}).get("general", [])[:20])
            item = _search_item("e621", post.get("id"), 0, file_url, tag_string or tags, category, source_cfg)
            if item:
                items.append(item)
    except Exception:
        pass
    return items


def _fetch_waifu_im_source(source_cfg: Dict[str, Any], cfg: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    try:
        from navine.reference.sources.waifu import fetch_waifu_im

        merged = dict(cfg)
        section = dict(merged.get("waifu_im") or {})
        if source_cfg.get("tags"):
            section["tags"] = source_cfg.get("tags")
        if source_cfg.get("limit"):
            section["max_per_run"] = int(source_cfg.get("limit"))
        if source_cfg.get("media_category"):
            section["media_category"] = source_cfg.get("media_category")
        section["enabled"] = True
        merged["waifu_im"] = section
        return fetch_waifu_im(config=merged, category=category)
    except Exception:
        return []


def fetch_waifu_im_sources(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    section = cfg.get("waifu_im") or {}
    if not isinstance(section, dict) or not section.get("enabled", False):
        return []
    _report(progress, "Fetching waifu.im hentai references...")
    try:
        from navine.reference.sources.waifu import fetch_waifu_im

        return fetch_waifu_im(config=cfg, category=category, progress=progress)
    except Exception as exc:
        _report(progress, f"Warning: waifu.im fetch failed: {exc}")
        return []


def fetch_infini_atomic_sources(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    section = cfg.get("infini_atomic") or {}
    if not isinstance(section, dict) or not section.get("enabled", False):
        return []
    _report(progress, "Fetching infini-atomic.w3spaces.com hentai references...")
    try:
        from navine.reference.sources.infini_atomic import fetch_infini_atomic

        return fetch_infini_atomic(config=cfg, category=category, progress=progress)
    except Exception as exc:
        _report(progress, f"Warning: infini-atomic fetch failed: {exc}")
        return []


def fetch_image_searches(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    sources = cfg.get("image_search_sources") or []
    queries = cfg.get("image_search_queries") or []
    if not sources and queries:
        sources = [{"type": "rule34", "tags": query, "limit": 10} for query in queries]
    max_total = int(cfg.get("max_search_images", 25))
    items: List[Dict[str, Any]] = []
    for source_cfg in sources:
        if len(items) >= max_total:
            break
        source_type = str(source_cfg.get("type") or "rule34").lower()
        if source_type == "rule34":
            batch = _fetch_rule34(source_cfg, cfg, category)
        elif source_type == "danbooru":
            batch = _fetch_danbooru(source_cfg, cfg, category)
        elif source_type == "gelbooru":
            batch = _fetch_gelbooru(source_cfg, cfg, category)
        elif source_type == "e621":
            batch = _fetch_e621(source_cfg, cfg, category)
        elif source_type == "waifu_im":
            batch = _fetch_waifu_im_source(source_cfg, cfg, category)
        else:
            _report(progress, f"Warning: unknown image search source type: {source_type}")
            continue
        if not batch:
            _report(progress, f"Warning: image search {source_type} returned 0 items (blocked, empty, or API error).")
        items.extend(batch)
        _sleep(cfg)
    return items[:max_total]


def fetch_video_searches(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    sources = cfg.get("video_search_sources") or []
    max_total = int(cfg.get("max_search_videos", 20))
    items: List[Dict[str, Any]] = []
    for source_cfg in sources:
        if len(items) >= max_total:
            break
        source_type = str(source_cfg.get("type") or "rule34").lower()
        if source_type == "rule34":
            batch = _fetch_rule34(source_cfg, cfg, category)
        elif source_type == "e621":
            batch = _fetch_e621(source_cfg, cfg, category)
        elif source_type == "danbooru":
            batch = _fetch_danbooru(source_cfg, cfg, category)
        else:
            _report(progress, f"Warning: unknown video search source type: {source_type}")
            continue
        video_batch = [item for item in batch if item.get("kind") == "video"]
        if not video_batch:
            video_batch = batch
        items.extend(video_batch)
        _sleep(cfg)
    return items[:max_total]


def fetch_configured_web_urls(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    urls = cfg.get("web_urls") or []
    max_pages = int(cfg.get("max_web_pages_per_run", 10))
    items: List[Dict[str, Any]] = []
    for url in urls[:max_pages]:
        url = str(url).strip()
        if not url:
            continue
        try:
            items.extend(fetch_web_url(url, config=cfg, category=category))
            _sleep(cfg)
        except Exception:
            continue
    return items


def fetch_configured_crawls(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
    progress: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    crawl_sources = cfg.get("crawl_start_urls") or []
    default_max = int(cfg.get("max_crawl_pages", 15))
    items: List[Dict[str, Any]] = []
    for entry in crawl_sources:
        if isinstance(entry, str):
            start_url = entry.strip()
            max_pages = default_max
            same_domain = True
        else:
            start_url = str(entry.get("url") or "").strip()
            max_pages = int(entry.get("max_pages") or default_max)
            same_domain = bool(entry.get("same_domain", True))
        if not start_url:
            continue
        try:
            items.extend(
                fetch_crawl_url(
                    start_url,
                    max_pages=max_pages,
                    same_domain_only=same_domain,
                    config=cfg,
                    category=category,
                    progress=progress,
                )
            )
        except Exception:
            continue
    return items


def fetch_reddit_post_urls(
    config: Optional[Dict[str, Any]] = None,
    category: str = "nsfw",
) -> List[Dict[str, Any]]:
    cfg = config or load_nsfw_config()
    post_urls = cfg.get("reddit_post_urls") or []
    items: List[Dict[str, Any]] = []
    for url in post_urls:
        url = str(url).strip()
        if not url:
            continue
        try:
            items.extend(fetch_reddit_post(url, config=cfg, category=category))
            _sleep(cfg)
        except Exception:
            continue
    return items


def fetch_internet_sources(
    config: Optional[Dict[str, Any]] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = config or load_nsfw_config()
    items: List[Dict[str, Any]] = []
    counts = {
        "reddit_posts": 0,
        "web_urls": 0,
        "crawl": 0,
        "direct_media": 0,
        "feeds": 0,
        "search": 0,
        "video_search": 0,
        "waifu_im": 0,
    }
    _report(progress, "Fetching configured Reddit post URLs...")
    post_items = fetch_reddit_post_urls(config=cfg)
    counts["reddit_posts"] = len(post_items)
    items.extend(post_items)
    _report(progress, "Fetching configured web URLs...")
    web_items = fetch_configured_web_urls(config=cfg)
    counts["web_urls"] = len(web_items)
    items.extend(web_items)
    _report(progress, "Crawling configured gallery/page sources...")
    crawl_items = fetch_configured_crawls(config=cfg, progress=progress)
    counts["crawl"] = len(crawl_items)
    items.extend(crawl_items)
    _report(progress, "Fetching direct media URLs...")
    direct_items = fetch_direct_media(config=cfg)
    counts["direct_media"] = len(direct_items)
    items.extend(direct_items)
    _report(progress, "Fetching RSS/feed sources...")
    feed_items = fetch_configured_feeds(config=cfg)
    counts["feeds"] = len(feed_items)
    items.extend(feed_items)
    if cfg.get("include_images", True):
        _report(progress, "Running image search sources...")
        search_items = fetch_image_searches(config=cfg, progress=progress)
        counts["search"] = len(search_items)
        items.extend(search_items)
        if not search_items and (cfg.get("image_search_sources") or cfg.get("image_search_queries")):
            _report(progress, "Warning: all configured image search sources returned 0 items.")
    if cfg.get("include_videos", True):
        _report(progress, "Running video search sources...")
        video_search_items = fetch_video_searches(config=cfg, progress=progress)
        counts["video_search"] = len(video_search_items)
        items.extend(video_search_items)
    enriched = [enrich_media_item(item, cfg) for item in items]
    return {"items": enriched, "counts": counts}


def save_crawl_snapshot(items: List[Dict[str, Any]], label: str) -> str:
    root = get_project_root()
    out_dir = root / "data" / "nsfw" / "crawl"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w\-]", "_", label)[:60]
    out_path = out_dir / f"{slug}.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(out_path)
