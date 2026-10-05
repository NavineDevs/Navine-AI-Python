import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from navine.policy import http_get
from PIL import Image

from navine.learn.web import USER_AGENT, TIMEOUT
from navine.image.utils import quarantine_file, validate_image, validate_image_bytes
from navine.utils.paths import get_project_root

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MIN_IMAGE_BYTES = 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024


def _learned_dir() -> Path:
    path = get_project_root() / "data" / "image" / "learned"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ingest_dir() -> Path:
    path = get_project_root() / "data" / "image" / "ingested"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path() -> Path:
    return _learned_dir() / "manifest.json"


def _load_manifest() -> List[dict]:
    path = _manifest_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(entries: List[dict]) -> None:
    _manifest_path().write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _image_id(url: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()[:16]
    return digest


def _guess_extension(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    ct = content_type.lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    return ".jpg"


def _caption_from_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem if stem else "image"


def _download_bytes(url: str) -> tuple:
    response = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, stream=True)
    response.raise_for_status()
    content = response.content
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large: {len(content)} bytes")
    if len(content) < MIN_IMAGE_BYTES:
        raise ValueError("Downloaded file too small to be a valid image")
    content_type = response.headers.get("Content-Type", "")
    return content, content_type


def _is_direct_image_url(url: str, content_type: str) -> bool:
    if content_type.lower().startswith("image/"):
        return True
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix in IMAGE_EXTENSIONS


def _extract_image_urls_from_html(html: str, base_url: str) -> List[tuple]:
    soup = BeautifulSoup(html, "html.parser")
    found = []
    seen = set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src or src.startswith("data:"):
            continue
        full = urljoin(base_url, src)
        if full in seen:
            continue
        seen.add(full)
        alt = (img.get("alt") or "").strip()
        found.append((full, alt))
    return found


def _page_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def _save_image_entry(
    url: str,
    content: bytes,
    content_type: str,
    caption: Optional[str],
    metadata: Optional[dict] = None,
) -> str:
    learned = _learned_dir()
    ext = _guess_extension(url, content_type)
    image_id = _image_id(url, content)
    image_path = learned / f"{image_id}{ext}"
    meta_path = learned / f"{image_id}.json"
    if image_path.exists():
        return str(image_path)
    if not validate_image_bytes(content):
        raise ValueError(f"Invalid image data from {url}")
    image_path.write_bytes(content)
    if not caption:
        caption = _caption_from_filename(image_path.name)
    entry = {
        "id": image_id,
        "url": url,
        "caption": caption,
        "filename": image_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content_type": content_type,
    }
    if metadata:
        for key in (
            "media_category",
            "subject_type",
            "style_type",
            "tags",
            "source",
            "source_id",
            "subreddit",
            "kind",
        ):
            if key in metadata and metadata[key] is not None:
                entry[key] = metadata[key]
    meta_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = _load_manifest()
    if not any(e.get("id") == image_id for e in manifest):
        manifest.append(entry)
        _save_manifest(manifest)
    return str(image_path)


def download_image_url(url: str, caption: Optional[str] = None, metadata: Optional[dict] = None) -> List[str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    response = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    saved = []
    if _is_direct_image_url(url, content_type):
        path = _save_image_entry(url, response.content, content_type, caption, metadata=metadata)
        saved.append(path)
        return saved
    title = _page_title(response.text)
    candidates = _extract_image_urls_from_html(response.text, url)
    if not candidates:
        raise ValueError(f"No images found at {url}")
    for img_url, alt in candidates[:20]:
        try:
            content, ct = _download_bytes(img_url)
            cap = caption or alt or title or _caption_from_filename(img_url)
            saved.append(_save_image_entry(img_url, content, ct, cap, metadata=metadata))
        except Exception:
            continue
    if not saved:
        raise ValueError(f"Could not download any images from {url}")
    return saved


def crawl_images(page_url: str, max_images: int = 20) -> List[str]:
    response = http_get(page_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    title = _page_title(response.text)
    candidates = _extract_image_urls_from_html(response.text, page_url)
    saved = []
    for img_url, alt in candidates:
        if len(saved) >= max_images:
            break
        try:
            content, ct = _download_bytes(img_url)
            cap = alt or title or _caption_from_filename(img_url)
            saved.append(_save_image_entry(img_url, content, ct, cap))
        except Exception:
            continue
    if not saved:
        raise ValueError(f"No images downloaded from {page_url}")
    return saved


def search_images(query: str, max_count: int = 10) -> List[str]:
    saved = []
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": max_count,
        "prop": "imageinfo",
        "iiprop": "url",
    }
    try:
        resp = http_get(api_url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            img_url = infos[0].get("url")
            if not img_url:
                continue
            try:
                content, ct = _download_bytes(img_url)
                cap = page.get("title", query).replace("File:", "").rsplit(".", 1)[0]
                cap = re.sub(r"[_\-]+", " ", cap)
                saved.append(_save_image_entry(img_url, content, ct, cap))
            except Exception:
                continue
    except Exception:
        pass
    if len(saved) < max_count:
        for pid in range(1, max_count * 3):
            if len(saved) >= max_count:
                break
            picsum_url = f"https://picsum.photos/id/{pid}/256/256"
            try:
                content, ct = _download_bytes(picsum_url)
                cap = f"{query} sample {pid}"
                saved.append(_save_image_entry(picsum_url, content, ct, cap))
            except Exception:
                continue
    if not saved:
        raise ValueError(f"No images found for query: {query}")
    return saved[:max_count]


def ingest_learned_images(image_size: Optional[int] = None) -> str:
    if image_size is None:
        try:
            from navine.utils.config import load_config

            image_size = int(load_config("image")["model"]["image_size"])
        except Exception:
            image_size = 128
    learned = _learned_dir()
    out_dir = _ingest_dir()
    count = 0
    for meta_path in sorted(learned.glob("*.json")):
        if meta_path.name == "manifest.json":
            continue
        entry = json.loads(meta_path.read_text(encoding="utf-8"))
        filename = entry.get("filename")
        if not filename:
            continue
        src = learned / filename
        if not src.exists():
            continue
        if not validate_image(src):
            quarantine_file(src, reason="invalid learned image")
            continue
        try:
            img = Image.open(src).convert("RGB")
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            out_name = f"{entry['id']}.png"
            out_path = out_dir / out_name
            img.save(out_path, format="PNG")
            sidecar = out_dir / f"{entry['id']}.json"
            sidecar.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
            count += 1
        except Exception:
            continue
    samples_dir = get_project_root() / "data" / "image" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for png in out_dir.glob("*.png"):
        dest = samples_dir / png.name
        if not dest.exists():
            shutil.copy2(png, dest)
        sidecar_src = out_dir / f"{png.stem}.json"
        if sidecar_src.exists():
            sidecar_dest = samples_dir / f"{png.stem}.json"
            if not sidecar_dest.exists():
                shutil.copy2(sidecar_src, sidecar_dest)
    return str(out_dir) if count else str(learned)


def train_learned_image(config_path: str = "image", finetune_steps: Optional[int] = None) -> None:
    from navine.image.train import train

    ingest_learned_images()
    ingest_nsfw_images()
    train(config_path=config_path, finetune=True, finetune_steps=finetune_steps)


def ingest_nsfw_images(image_size: Optional[int] = None) -> int:
    if image_size is None:
        try:
            from navine.utils.config import load_config

            image_size = int(load_config("image")["model"]["image_size"])
        except Exception:
            image_size = 128
    root = get_project_root()
    nsfw_dirs = [
        root / "data" / "nsfw" / "images" / "local",
        root / "data" / "nsfw" / "images",
    ]
    manifest_path = root / "data" / "nsfw" / "media_manifest.jsonl"
    out_dir = _ingest_dir()
    samples_dir = root / "data" / "image" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set = set()
    for nsfw_dir in nsfw_dirs:
        if not nsfw_dir.exists():
            continue
        for path in nsfw_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            if not validate_image(path):
                quarantine_file(path, reason="invalid nsfw image")
                continue
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                out_name = f"nsfw_{digest}.png"
                out_path = out_dir / out_name
                img.save(out_path, format="PNG")
                caption = _caption_from_filename(path.name)
                sidecar = out_dir / f"nsfw_{digest}.json"
                sidecar.write_text(
                    json.dumps({"id": digest, "caption": caption, "source": str(path)}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                dest = samples_dir / out_name
                if not dest.exists():
                    shutil.copy2(out_path, dest)
                sidecar_dest = samples_dir / f"nsfw_{digest}.json"
                if not sidecar_dest.exists():
                    shutil.copy2(sidecar, sidecar_dest)
                count += 1
            except Exception:
                continue
    if manifest_path.exists():
        try:
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("kind") != "image":
                    continue
                url = entry.get("path") or entry.get("local_path") or entry.get("url")
                if not url:
                    continue
                path = Path(url)
                if not path.is_absolute():
                    path = root / path
                if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                if not validate_image(path):
                    quarantine_file(path, reason="invalid nsfw manifest image")
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                    out_name = f"nsfw_{digest}.png"
                    out_path = out_dir / out_name
                    img.save(out_path, format="PNG")
                    caption = entry.get("caption") or entry.get("title") or _caption_from_filename(path.name)
                    sidecar = out_dir / f"nsfw_{digest}.json"
                    sidecar.write_text(
                        json.dumps({"id": digest, "caption": caption, "source": str(path)}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    dest = samples_dir / out_name
                    if not dest.exists():
                        shutil.copy2(out_path, dest)
                    count += 1
                except Exception:
                    continue
        except Exception:
            pass
    return count
