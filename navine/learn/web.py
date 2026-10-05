import re
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from navine.policy import http_get

from navine.utils.paths import get_project_root

USER_AGENT = "NavineAI/1.0 (local learning bot)"
TIMEOUT = 30


def fetch_url(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    response = http_get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    title = url
    text = response.text
    if "html" in content_type.lower():
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
    else:
        text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {
        "url": url,
        "title": title,
        "text": text,
        "content_type": content_type,
    }


def save_fetched(doc: dict, filename: Optional[str] = None) -> str:
    import json
    from datetime import datetime

    root = get_project_root()
    out_dir = root / "data" / "learn" / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not filename:
        slug = re.sub(r"[^\w\-]", "_", urlparse(doc["url"]).netloc + urlparse(doc["url"]).path)[:80]
        filename = f"{slug}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return str(path)
