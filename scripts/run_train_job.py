import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _jobs_dir() -> Path:
    return ROOT / "logs" / "train_jobs"


def _load_job(job_id: str) -> dict:
    path = _jobs_dir() / f"{job_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save_job(job_id: str, meta: dict) -> None:
    (_jobs_dir() / f"{job_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _log(job_id: str, line: str) -> None:
    path = _jobs_dir() / f"{job_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {line}\n")
    print(line, flush=True, file=sys.__stdout__)
    try:
        from navine.api.train_progress_helper import _parse_steps_from_log, write_admin_live_progress

        meta = _load_job(job_id)
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        steps, steps_total, pct = _parse_steps_from_log(text)
        if not steps_total:
            steps_total = int(meta.get("steps") or 0)
        write_admin_live_progress(
            {
                "running": str(meta.get("status") or "") == "running",
                "stage": str(meta.get("target") or "training"),
                "label": str(meta.get("target") or "training"),
                "steps": steps,
                "steps_total": steps_total,
                "percent": pct if pct else (min(99.0, 100.0 * steps / steps_total) if steps_total and steps else 8.0),
                "job_id": job_id,
            }
        )
    except Exception:
        pass


class _JobLogTee:
    def __init__(self, job_id: str, stream):
        self.job_id = job_id
        self.stream = stream

    def write(self, data: str) -> None:
        self.stream.write(data)
        if data and data.strip():
            for line in data.splitlines():
                if line.strip():
                    _log(self.job_id, line.strip())

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def _run_custom_ingest(custom: dict) -> None:
    from navine.api.train_jobs import ingest_custom_source

    ingest_custom_source(
        source=str(custom.get("source") or "text"),
        url=custom.get("url"),
        text=custom.get("text"),
        title=custom.get("title"),
    )


def _dispatch_target(target: str, steps, config, language, require_cuda: bool, meta: Optional[dict] = None) -> None:
    target_key = str(target or "text").strip().lower().replace("_", "-")
    meta = meta or {}

    if target_key == "voice":
        from navine.voice.train import train_voice

        train_voice(
            steps=steps,
            language=language,
            voice_name=meta.get("voice_name"),
            sample_path=meta.get("voice_sample"),
            set_default=bool(meta.get("voice_set_default", True)),
        )
        return

    if target_key in ("deepfake", "deep-fake", "face-swap"):
        from navine.deepfake.train import train_deepfake

        train_deepfake(steps=steps, language=language, strength=meta.get("deepfake_strength"), source_path=meta.get("deepfake_source"), target_path=meta.get("deepfake_target"))
        return

    if target_key in ("text", "text-enterprise"):
        from navine.text.train import train
        from navine.utils.tier import modality_config_name

        train(
            config_path=config or modality_config_name("text"),
            max_steps=steps,
            require_cuda=require_cuda,
        )
        return

    if target_key in ("text-code", "textcode", "code"):
        from navine.text.train import train

        train(config_path="text_code", max_steps=steps, require_cuda=require_cuda)
        return

    if target_key == "image":
        try:
            from navine.learn.image_learn import ingest_learned_images, ingest_nsfw_images

            ingest_learned_images()
            ingest_nsfw_images()
        except Exception:
            pass
        from navine.image.train import train
        from navine.utils.tier import modality_config_name

        train(
            config_path=config or modality_config_name("image"),
            finetune=True,
            finetune_steps=steps,
            require_cuda=require_cuda,
        )
        return

    if target_key == "video":
        try:
            from navine.learn.video_learn import ingest_learned_video

            ingest_learned_video()
        except Exception:
            pass
        from navine.video.train import train
        from navine.utils.tier import modality_config_name

        train(
            config_path=config or modality_config_name("video"),
            finetune=True,
            finetune_steps=steps,
            require_cuda=require_cuda,
        )
        return

    if target_key in ("image-lora", "image_lora"):
        from navine.image.lora_custom import train_custom_lora

        train_custom_lora(steps=steps)
        return

    if target_key in ("video-lora", "video_lora"):
        from navine.video.lora_custom import train_video_lora

        train_video_lora(steps=steps)
        return

    if target_key == "all":
        from navine.train.registry import train_all

        train_all(language=language, steps=steps)
        return

    if target_key == "full":
        from navine.train.full import train_full

        train_full(steps=steps, language=language)
        return

    from navine.train.registry import TRAINING_TYPES, train_type

    if target_key not in TRAINING_TYPES:
        raise ValueError(f"Unknown training target: {target_key}")
    train_type(target_key, language=language, steps=steps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--learn-only", action="store_true")
    args = parser.parse_args()
    job_id = str(args.job_id).strip()
    meta = _load_job(job_id)
    target = meta.get("target") or "text"
    steps = meta.get("steps")
    config = meta.get("config")
    language = meta.get("language")
    require_cuda = bool(meta.get("require_cuda", True))
    custom = meta.get("custom") or {}

    meta["status"] = "running"
    _save_job(job_id, meta)
    _log(job_id, f"JOB START target={target} steps={steps} learn_only={args.learn_only}")
    try:
        from navine.api.train_progress_helper import write_admin_live_progress

        write_admin_live_progress(
            {
                "running": True,
                "stage": str(target),
                "label": str(target),
                "steps": 0,
                "steps_total": int(steps or 0),
                "percent": 2.0,
                "job_id": job_id,
            }
        )
    except Exception:
        pass

    stdout_prev = sys.stdout
    stderr_prev = sys.stderr
    sys.stdout = _JobLogTee(job_id, stdout_prev)
    sys.stderr = _JobLogTee(job_id, stderr_prev)

    try:
        if args.learn_only:
            action = str(custom.get("source") or meta.get("action") or "url").lower()
            if action in ("url", "website", "web"):
                from navine.api.train_jobs import ingest_custom_source
                ingest_custom_source(source="url", url=custom.get("url") or meta.get("url"))
            elif action == "crawl":
                from navine.learn.crawl import crawl
                from navine.learn.rag import rebuild_index
                url = str(custom.get("url") or meta.get("url") or "").strip()
                pages = int(custom.get("max_pages") or 8)
                crawl(url, max_pages=pages)
                rebuild_index()
            elif action == "ingest":
                from navine.learn.datasets import ingest_learned_data
                from navine.learn.rag import rebuild_index
                ingest_learned_data()
                rebuild_index()
            elif action == "index":
                from navine.learn.rag import rebuild_index
                rebuild_index()
            elif action in ("text", "paste", "custom"):
                from navine.api.train_jobs import ingest_custom_source
                ingest_custom_source(
                    source="text",
                    text=custom.get("text"),
                    title=custom.get("title"),
                )
            else:
                raise ValueError(f"Unknown learn action: {action}")
            meta["status"] = "completed"
            meta["finished_at"] = datetime.now(timezone.utc).isoformat()
            _save_job(job_id, meta)
            _log(job_id, "LEARN COMPLETE")
            return 0
        if custom:
            _log(job_id, f"CUSTOM INGEST source={custom.get('source')}")
            _run_custom_ingest(custom)
        _dispatch_target(target, steps, config, language, require_cuda, meta)
        meta["status"] = "completed"
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_job(job_id, meta)
        _log(job_id, "JOB COMPLETE")
        return 0
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = str(exc)
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_job(job_id, meta)
        _log(job_id, f"JOB FAILED: {exc}")
        _log(job_id, traceback.format_exc())
        return 1
    finally:
        sys.stdout = stdout_prev
        sys.stderr = stderr_prev
        try:
            from navine.api.train_progress_helper import write_admin_live_progress

            write_admin_live_progress({"running": False, "percent": 100.0, "stage": "idle", "label": "idle"})
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
