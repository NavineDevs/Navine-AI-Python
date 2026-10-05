import json
import re
from collections import deque
from typing import Optional, Set
from urllib.parse import urljoin, urlparse

from navine.learn.web import fetch_url
from navine.utils.paths import get_project_root


def _same_domain(base: str, url: str) -> bool:
    return urlparse(base).netloc == urlparse(url).netloc


def _extract_links(html_url: str, html_text: str) -> Set[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        full = urljoin(html_url, href)
        parsed = urlparse(full)
        if parsed.scheme in ("http", "https"):
            links.add(full.split("#")[0])
    return links


def crawl(
    start_url: str,
    max_pages: int = 20,
    same_domain_only: bool = True,
) -> str:
    root = get_project_root()
    out_dir = root / "data" / "learn" / "crawl"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w\-]", "_", urlparse(start_url).netloc)[:60]
    out_path = out_dir / f"{slug}.jsonl"
    visited: Set[str] = set()
    queue: deque = deque([start_url])
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while queue and count < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            try:
                doc = fetch_url(url)
            except Exception:
                continue
            entry = {
                "url": doc["url"],
                "title": doc["title"],
                "text": doc["text"][:8000],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if same_domain_only:
                try:
                    from navine.learn.web import USER_AGENT, TIMEOUT
                    from navine.policy import http_get

                    resp = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
                    for link in _extract_links(url, resp.text):
                        if link not in visited and _same_domain(start_url, link):
                            queue.append(link)
                except Exception:
                    pass
    return str(out_path)
