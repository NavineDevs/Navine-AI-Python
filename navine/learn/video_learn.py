import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from PIL import Image

from navine.learn.web import USER_AGENT, TIMEOUT
from navine.policy import http_get
from navine.utils.paths import get_project_root

MAX_GIF_FRAMES = 64
MAX_VIDEO_FRAMES = 64
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}


def _is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host in YOUTUBE_HOSTS:
        return True
    if host.endswith(".youtube.com"):
        return True
    return "youtube.com/watch" in url or "youtu.be/" in url


def _parse_vtt_text(raw: str) -> str:
    lines: List[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("WEBVTT") or "-->" in text:
            continue
        if text.isdigit():
            continue
        if text.startswith("NOTE"):
            continue
        cleaned = re.sub(r"<[^>]+>", "", text).strip()
        if cleaned:
            lines.append(cleaned)
    deduped: List[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return "\n".join(deduped)


def _collect_youtube_transcript(tmpdir: Path, info: dict) -> str:
    for path in sorted(tmpdir.glob("*.vtt")) + sorted(tmpdir.glob("*.srt")):
        try:
            text = _parse_vtt_text(path.read_text(encoding="utf-8", errors="ignore"))
            if len(text.split()) >= 12:
                return text
        except Exception:
            continue
    return _collect_youtube_transcript_from_info(info)


def _collect_youtube_transcript_from_info(info: dict) -> str:
    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}
    for bucket in (subtitles, automatic):
        for lang in ("en", "en-US", "en-GB", "en-us", "en-gb"):
            entries = bucket.get(lang) or bucket.get(lang.lower())
            if not entries:
                continue
            for entry in entries:
                sub_url = entry.get("url")
                if not sub_url:
                    continue
                try:
                    content, _ = _download(sub_url)
                    text = _parse_vtt_text(content.decode("utf-8", errors="ignore"))
                    if len(text.split()) >= 12:
                        return text
                except Exception:
                    continue
    return ""


def _try_fetch_youtube_subtitles(url: str, tmpdir: Path) -> str:
    try:
        import yt_dlp
    except ImportError:
        return ""
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt/best",
        "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 60,
        "ignoreerrors": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if isinstance(info, dict):
            text = _collect_youtube_transcript(tmpdir, info)
            if text:
                return text
    except Exception:
        pass
    return ""


def _parse_youtube_spoken_caption(raw_text: str) -> str:
    idx = raw_text.find('"wireMagic"')
    if idx < 0:
        return ""
    blob = raw_text[idx:]
    parts = re.findall(r'"utf8":\s*"((?:\\.|[^"])*)"', blob)
    out: List[str] = []
    for part in parts:
        try:
            decoded = bytes(part, "utf-8").decode("unicode_escape")
        except Exception:
            decoded = part
        decoded = decoded.replace("\n", " ").strip()
        if decoded and (not out or decoded != out[-1]):
            out.append(decoded)
    return " ".join(out)


def _youtube_study_profile(title: str, description: str, spoken: str) -> dict:
    blob = f"{title} {description} {spoken}".lower()
    if any(
        k in blob
        for k in (
            "neural network",
            "from scratch",
            "genetic algorithm",
            "self-driving",
            "pygame",
            "hidden layer",
            "weights and biases",
            "mutation",
            "breeding pool",
            "tanh",
        )
    ):
        modality = "code"
        relevance = "high"
        notes = (
            "From-scratch neural net + genetic training (sensors, neurons, hidden layers, "
            "population, crossover, mutation). Strong coding/AI signal."
        )
    elif any(
        k in blob
        for k in (
            "jax",
            "autodiff",
            "xla",
            "jit compilation",
            "vmap",
        )
    ):
        modality = "code"
        relevance = "high"
        notes = "JAX autodiff/JIT is supported in Navine learn jax and racing AI math path."
    elif any(k in blob for k in ("llm", "large language", "transformer", "giant", "decoder", "token")):
        modality = "text"
        relevance = "high"
        notes = "Transformer/LLM theory aligns with Navine text stack (RoPE, SwiGLU, RMSNorm)."
    elif any(k in blob for k in ("video gen", "text-to-video", "narrated video", "video generation model")):
        modality = "video"
        relevance = "medium"
        notes = "Confirms 8–12GB GPUs need small custom video models; matches Navine hybrid video path."
    elif any(k in blob for k in ("ddpm", "diffusion", "denois", "unet", "image gen")):
        modality = "image"
        relevance = "medium"
        notes = "DDPM/denoising matches Navine image model; video demo stays at MNIST/tiny scale."
    elif "chatbot" in blob:
        modality = "chat"
        relevance = "low"
        notes = "Mostly UI/API chatbot wiring, not weight training — limited value for model quality."
    else:
        modality = "general"
        relevance = "low"
        notes = "General AI content with limited training signal."
    return {
        "navine_modality": modality,
        "helpful_for_navine": relevance,
        "study_notes": notes,
    }


def _youtube_training_caption(title: str, description: str, transcript: str, spoken: str, profile: dict) -> str:
    core = spoken.strip() or transcript.strip() or description.strip()
    core = re.sub(r"\s+", " ", core)
    if len(core.split()) > 48:
        core = " ".join(core.split()[:48])
    prefix = {
        "text": "transformer language model training",
        "image": "ddpm diffusion image generation training",
        "video": "temporal video motion generation training",
        "chat": "ai chatbot interface",
    }.get(str(profile.get("navine_modality") or ""), "ai model training")
    if core:
        return f"{prefix}, {title}, {core}"
    return f"{prefix}, {title}"


def study_youtube_learned(reindex_rag: bool = False) -> dict:
    root = get_project_root()
    pages_dir = root / "data" / "learn" / "pages"
    learned = _learned_root()
    rows: List[dict] = []
    updated = 0

    for page_path in sorted(pages_dir.glob("youtube_*.json")):
        doc = json.loads(page_path.read_text(encoding="utf-8"))
        video_id = str(doc.get("video_id") or page_path.stem.replace("youtube_", ""))
        title = str(doc.get("title") or "")
        body = str(doc.get("text") or "")
        spoken = _parse_youtube_spoken_caption(body)
        description = body.split("\n\n")[1] if "\n\n" in body else ""
        if '"wireMagic"' in description:
            description = body.split('"wireMagic"')[0].strip().split("\n\n", 1)[0]
        profile = _youtube_study_profile(title, description, spoken)
        caption = _youtube_training_caption(title, description, "", spoken, profile)
        doc.update(
            {
                **profile,
                "spoken_words": len(spoken.split()) if spoken else 0,
                "training_caption": caption,
            }
        )
        page_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

        seq_dir = None
        for candidate in learned.iterdir():
            meta_path = candidate / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("source_id") or "") == video_id:
                seq_dir = candidate
                meta["caption"] = caption
                meta["navine_modality"] = profile["navine_modality"]
                meta["helpful_for_navine"] = profile["helpful_for_navine"]
                meta["study_notes"] = profile["study_notes"]
                meta["spoken_words"] = doc["spoken_words"]
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                updated += 1
                break

        rows.append(
            {
                "video_id": video_id,
                "title": title,
                "spoken_words": doc["spoken_words"],
                "sequence": str(seq_dir) if seq_dir else None,
                **profile,
            }
        )

    rag_docs = None
    if reindex_rag:
        from navine.learn.rag import rebuild_index

        rag_docs = rebuild_index()

    helpful = [r for r in rows if r.get("helpful_for_navine") in ("high", "medium")]
    return {
        "videos_studied": len(rows),
        "sequences_updated": updated,
        "helpful_count": len(helpful),
        "rag_docs": rag_docs,
        "videos": rows,
    }


def _save_youtube_text(
    url: str,
    title: str,
    description: str,
    transcript: str,
    video_id: str,
    spoken: str = "",
    profile: Optional[dict] = None,
) -> str:
    from navine.learn.rag import index_document

    if profile is None:
        profile = _youtube_study_profile(title, description, spoken or transcript)
    parts = [title.strip()]
    if description.strip():
        parts.append(description.strip())
    if spoken.strip():
        parts.append(spoken.strip())
    elif transcript.strip():
        parts.append(transcript.strip())
    body = "\n\n".join(parts)
    doc = {
        "url": url,
        "title": title,
        "text": body,
        "content_type": "youtube/transcript",
        "video_id": video_id,
        "source": "youtube",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spoken_words": len(spoken.split()) if spoken else 0,
        "training_caption": _youtube_training_caption(title, description, transcript, spoken, profile),
        **profile,
    }
    root = get_project_root()
    out_dir = root / "data" / "learn" / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w\-]", "_", video_id or title)[:80]
    path = out_dir / f"youtube_{slug}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if body.strip():
        index_document(url, title, body)
    return str(path)


def download_youtube_url(url: str, metadata: Optional[dict] = None) -> str:
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError("Install yt-dlp to learn from YouTube: pip install yt-dlp") from exc

    with tempfile.TemporaryDirectory(prefix="navine_yt_") as tmp:
        tmpdir = Path(tmp)
        outtmpl = str(tmpdir / "%(id)s.%(ext)s")
        meta_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 60,
        }
        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        title = str(info.get("title") or url).strip()
        description = str(info.get("description") or "").strip()
        video_id = str(info.get("id") or "").strip()
        transcript = _collect_youtube_transcript_from_info(info)
        if not transcript:
            transcript = _try_fetch_youtube_subtitles(url, tmpdir)

        dl_opts = {
            "format": "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "socket_timeout": 60,
            "retries": 5,
            "fragment_retries": 5,
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])

        spoken = _parse_youtube_spoken_caption(f"{description}\n{transcript}")
        profile = _youtube_study_profile(title, description, spoken or transcript)
        text_path = _save_youtube_text(
            url, title, description, transcript, video_id, spoken=spoken, profile=profile
        )
        training_caption = _youtube_training_caption(title, description, transcript, spoken, profile)

        video_path = None
        for ext in (".mp4", ".webm", ".mkv", ".mov"):
            matches = sorted(tmpdir.glob(f"*{ext}"))
            if matches:
                video_path = matches[0]
                break
        if video_path is None:
            raise ValueError(f"Could not download video from {url}")

        content = video_path.read_bytes()
        frames = _extract_video_frames(content)
        if not frames:
            raise ValueError(f"Could not extract frames from YouTube video {url}")

        meta = dict(metadata or {})
        meta.update(
            {
                "caption": training_caption,
                "source": "youtube",
                "source_id": video_id,
                "text_path": text_path,
                "transcript_words": len(transcript.split()) if transcript else 0,
                "spoken_words": len(spoken.split()) if spoken else 0,
                "navine_modality": profile["navine_modality"],
                "helpful_for_navine": profile["helpful_for_navine"],
                "study_notes": profile["study_notes"],
            }
        )
        frame_dir = _write_sequence(url, frames, "youtube", metadata=meta)
        print(f"Navine AI - Python: learned YouTube video '{title}' | frames={len(frames)} | transcript_words={meta['transcript_words']}")
        return frame_dir


def _learned_root() -> Path:
    path = get_project_root() / "data" / "video" / "learned"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ingest_root() -> Path:
    path = get_project_root() / "data" / "video" / "ingested"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sequence_id(url: str, content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _download(url: str) -> tuple:
    response = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, stream=True)
    response.raise_for_status()
    content = response.content
    content_type = response.headers.get("Content-Type", "")
    return content, content_type


def _save_frame(frame_dir: Path, index: int, img: Image.Image) -> str:
    path = frame_dir / f"frame_{index:04d}.png"
    img.convert("RGB").save(path, format="PNG")
    return str(path)


def _extract_gif_frames(content: bytes, max_frames: int = MAX_GIF_FRAMES) -> List[Image.Image]:
    img = Image.open(BytesIO(content))
    frames = []
    try:
        while True:
            frame = img.copy().convert("RGB")
            frames.append(frame)
            if len(frames) >= max_frames:
                break
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames if frames else [Image.open(BytesIO(content)).convert("RGB")]


def _extract_video_frames(content: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> List[Image.Image]:
    frames = []
    try:
        import cv2
        import numpy as np
        import tempfile
        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            total = max_frames
        pick = min(max_frames, max(1, total))
        indices = [int(round(i * max(total - 1, 0) / max(pick - 1, 1))) for i in range(pick)] if pick > 1 else [0]
        seen = set()
        for target in indices:
            if target in seen:
                continue
            seen.add(target)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if len(frames) >= max_frames:
                break
        cap.release()
        Path(tmp_path).unlink(missing_ok=True)
    except ImportError:
        pass
    except Exception:
        pass
    if not frames:
        try:
            import imageio.v3 as iio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            for i, frame in enumerate(iio.imiter(tmp_path)):
                if i >= max_frames:
                    break
                frames.append(Image.fromarray(frame).convert("RGB"))
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
    return frames


def _write_sequence(url: str, frames: List[Image.Image], source_type: str, metadata: Optional[dict] = None) -> str:
    if not frames:
        raise ValueError("No frames extracted")
    seq_id = hashlib.sha256(url.encode()).hexdigest()[:16]
    frame_dir = _learned_root() / seq_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        paths.append(_save_frame(frame_dir, i, frame))
    meta = {
        "id": seq_id,
        "url": url,
        "source_type": source_type,
        "frame_count": len(frames),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        for key in (
            "media_category",
            "subject_type",
            "style_type",
            "tags",
            "caption",
            "source",
            "source_id",
            "subreddit",
            "kind",
        ):
            if key in metadata and metadata[key] is not None:
                meta[key] = metadata[key]
    (frame_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(frame_dir)


def download_video_url(url: str, metadata: Optional[dict] = None) -> str:
    if _is_youtube_url(url):
        return download_youtube_url(url, metadata=metadata)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    content, content_type = _download(url)
    lower_ct = content_type.lower()
    lower_path = parsed.path.lower()
    if "gif" in lower_ct or lower_path.endswith(".gif"):
        frames = _extract_gif_frames(content)
        return _write_sequence(url, frames, "gif", metadata=metadata)
    if "video" in lower_ct or lower_path.endswith((".mp4", ".webm", ".mov", ".avi")):
        frames = _extract_video_frames(content)
        if frames:
            return _write_sequence(url, frames, "video", metadata=metadata)
    try:
        frames = _extract_gif_frames(content)
        if len(frames) > 1:
            return _write_sequence(url, frames, "gif", metadata=metadata)
    except Exception:
        pass
    frames = _extract_video_frames(content)
    if frames:
        return _write_sequence(url, frames, "video", metadata=metadata)
    raise ValueError(f"Could not extract frames from {url}")


def download_gif_url(url: str, metadata: Optional[dict] = None) -> str:
    content, _ = _download(url)
    frames = _extract_gif_frames(content)
    return _write_sequence(url, frames, "gif", metadata=metadata)


def ingest_learned_video(frame_size: Optional[int] = None) -> str:
    if frame_size is None:
        try:
            from navine.utils.config import load_config

            frame_size = int(load_config("video")["model"]["frame_size"])
        except Exception:
            frame_size = 96
    learned = _learned_root()
    out_root = _ingest_root()
    samples_dir = get_project_root() / "data" / "video" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    seq_count = 0
    for seq_dir in sorted(learned.iterdir()):
        if not seq_dir.is_dir():
            continue
        frames = sorted(seq_dir.glob("frame_*.png"))
        if not frames:
            continue
        out_seq = out_root / seq_dir.name
        out_seq.mkdir(parents=True, exist_ok=True)
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            img = img.resize((frame_size, frame_size), Image.Resampling.LANCZOS)
            out_path = out_seq / fp.name
            img.save(out_path, format="PNG")
            dest = samples_dir / f"{seq_dir.name}_{fp.name}"
            if not dest.exists():
                shutil.copy2(out_path, dest)
        meta_src = seq_dir / "meta.json"
        if meta_src.exists():
            shutil.copy2(meta_src, out_seq / "meta.json")
        seq_count += 1
    return str(out_root) if seq_count else str(learned)


def train_learned_video(config_path: str = "video", finetune_steps: Optional[int] = None) -> None:
    ingest_learned_video()
    from navine.video.train import train
    train(config_path=config_path, finetune=True, finetune_steps=finetune_steps)
