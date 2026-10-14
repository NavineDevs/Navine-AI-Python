import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.paths import get_project_root


def _jobs_dir() -> Path:
    path = get_project_root() / "logs" / "train_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _python_exe() -> str:
    root = get_project_root()
    venv_py = root / "venv" / "Scripts" / "python.exe"
    return str(venv_py) if venv_py.exists() else sys.executable


def list_recent_jobs(limit: int = 12) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for meta_path in sorted(_jobs_dir().glob("*.json"), reverse=True):
        if len(rows) >= limit:
            break
        try:
            rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _write_job_meta(job_id: str, meta: Dict[str, Any]) -> Path:
    path = _jobs_dir() / f"{job_id}.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def ingest_custom_source(
    *,
    source: str,
    url: Optional[str] = None,
    text: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    root = get_project_root()
    source_key = str(source or "text").strip().lower()
    result: Dict[str, Any] = {"source": source_key, "paths": []}

    if source_key in ("url", "website", "web"):
        if not url:
            raise ValueError("URL is required for website learning")
        from navine.learn.rag import index_document
        from navine.learn.web import fetch_url, save_fetched

        doc = fetch_url(str(url).strip())
        saved = save_fetched(doc)
        index_document(doc["url"], doc["title"], doc["text"])
        train_path = root / "data" / "train" / "general" / f"web_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        block = f"### Title: {doc.get('title') or 'Web page'}\n### Source: {doc.get('url')}\n\n{doc.get('text') or ''}"
        train_path.write_text(block.strip() + "\n", encoding="utf-8")
        result["paths"] = [saved, str(train_path)]
        result["title"] = doc.get("title")
        return result

    if source_key in ("text", "paste", "custom"):
        body = str(text or "").strip()
        if len(body) < 8:
            raise ValueError("Text must be at least 8 characters")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        learn_path = root / "data" / "learn" / "custom" / f"custom_{stamp}.txt"
        train_path = root / "data" / "train" / "general" / f"custom_{stamp}.txt"
        learn_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        heading = str(title or "Custom training text").strip()
        payload = f"### Title: {heading}\n\n{body}\n"
        learn_path.write_text(payload, encoding="utf-8")
        train_path.write_text(payload, encoding="utf-8")
        try:
            from navine.learn.rag import index_document

            index_document(f"custom:{stamp}", heading, body)
        except Exception:
            pass
        result["paths"] = [str(learn_path), str(train_path)]
        result["title"] = heading
        return result

    raise ValueError(f"Unsupported custom source: {source}")


def spawn_learn_job(
    *,
    action: str,
    url: Optional[str] = None,
    text: Optional[str] = None,
    title: Optional[str] = None,
    max_pages: Optional[int] = None,
    started_by: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    log_path = _jobs_dir() / f"{job_id}.log"
    meta = {
        "id": job_id,
        "target": f"learn-{action}",
        "action": action,
        "url": url,
        "started_by": started_by,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued",
        "log": str(log_path),
        "custom": {"source": action, "url": url, "text": text, "title": title, "max_pages": max_pages},
    }
    _write_job_meta(job_id, meta)

    root = get_project_root()
    worker = root / "scripts" / "run_train_job.py"
    args = [_python_exe(), str(worker), "--job-id", job_id, "--learn-only"]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(args, cwd=str(root), creationflags=creationflags)
    meta["status"] = "started"
    _write_job_meta(job_id, meta)
    return meta


def spawn_train_job(
    *,
    target: str,
    steps: Optional[int] = None,
    config: Optional[str] = None,
    language: Optional[str] = None,
    require_cuda: bool = True,
    custom: Optional[Dict[str, Any]] = None,
    started_by: Optional[str] = None,
    voice_name: Optional[str] = None,
    voice_sample: Optional[str] = None,
    voice_set_default: bool = True,
    deepfake_strength: Optional[float] = None,
    deepfake_source: Optional[str] = None,
    deepfake_target: Optional[str] = None,
) -> Dict[str, Any]:
    from navine.utils.training_lock import training_lock_active

    target_key = str(target or "text").strip().lower().replace("_", "-")
    lock_name = target_key
    if lock_name in ("text-code", "textcode", "code"):
        lock_name = "text"
    elif lock_name in ("image-lora", "video-lora", "full", "all"):
        lock_name = "image" if "image" in lock_name else ("video" if "video" in lock_name else "text")
    elif lock_name not in ("text", "image", "video", "voice", "deepfake", "deep-fake", "face-swap"):
        lock_name = "text"

    if training_lock_active(lock_name) and lock_name in ("text", "image", "video", "voice", "deepfake"):
        raise RuntimeError(f"{lock_name} training is already running. Wait for it to finish.")

    job_id = uuid.uuid4().hex[:12]
    log_path = _jobs_dir() / f"{job_id}.log"
    meta = {
        "id": job_id,
        "target": target_key,
        "steps": steps,
        "config": config,
        "language": language,
        "require_cuda": bool(require_cuda),
        "custom": custom or None,
        "voice_name": voice_name,
        "voice_sample": voice_sample,
        "voice_set_default": bool(voice_set_default),
        "deepfake_strength": deepfake_strength,
        "deepfake_source": deepfake_source,
        "deepfake_target": deepfake_target,
        "started_by": started_by,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued",
        "log": str(log_path),
    }
    _write_job_meta(job_id, meta)

    root = get_project_root()
    worker = root / "scripts" / "run_train_job.py"
    if not worker.exists():
        raise FileNotFoundError("scripts/run_train_job.py not found")

    args = [_python_exe(), str(worker), "--job-id", job_id]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(args, cwd=str(root), creationflags=creationflags)
    meta["status"] = "started"
    _write_job_meta(job_id, meta)
    try:
        from navine.api.train_progress_helper import write_admin_live_progress

        write_admin_live_progress(
            {
                "running": True,
                "stage": target_key,
                "label": target_key,
                "steps": 0,
                "steps_total": int(steps or 0),
                "percent": 3.0,
                "job_id": job_id,
            }
        )
    except Exception:
        pass
    return meta
