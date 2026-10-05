import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from navine.autolearn.engine import AutolearnEngine
from navine.autolearn.sources.nsfw import fetch_full_cycle
from navine.nsfw.config import is_nsfw_enabled, load_nsfw_config, reload_nsfw_config
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]


def _state_path() -> Path:
    path = get_project_root() / "data" / "nsfw" / "autolearn_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"last_run": None, "runs": 0, "images_downloaded": 0, "videos_downloaded": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run": None, "runs": 0, "images_downloaded": 0, "videos_downloaded": 0}


def _save_state(state: Dict[str, Any]) -> None:
    _state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_nsfw_cycle(progress: ProgressCallback = None) -> Dict[str, Any]:
    cfg = reload_nsfw_config()

    def report(msg: str) -> None:
        if progress:
            progress(msg)

    report("Navine AI - Python NSFW autolearn cycle starting...")
    result = fetch_full_cycle(progress=report, config=cfg)
    fetched = int(result.get("items_fetched", 0))
    if fetched:
        report(f"Fetched {fetched} item(s) ({result.get('reddit_items', 0)} Reddit, {result.get('fourchan_items', 0)} 4chan, {result.get('local_items', 0)} local, {result.get('internet_items', 0)} internet).")
    engine = AutolearnEngine()
    ingest = {"new_samples": 0, "by_category": {}}
    if result.get("text_items"):
        ingest = engine.process_items(result["text_items"])
    unrestricted_new = int((ingest.get("by_category") or {}).get("unrestricted", 0))
    if cfg.get("auto_train_text", True) and ingest.get("new_samples", 0) > 0:
        report("Training NSFW text model...")
        try:
            from navine.train.registry import train_type

            train_type("nsfw", steps=cfg.get("text_train_steps"))
        except Exception as exc:
            report(f"NSFW text train skipped: {exc}")
    if cfg.get("include_unrestricted", True) and cfg.get("auto_train_unrestricted", True):
        if unrestricted_new > 0 or ingest.get("new_samples", 0) > 0:
            report("Training unrestricted text model...")
        else:
            report("Training unrestricted text model (existing corpus only)...")
        try:
            from navine.train.registry import train_type

            train_type("unrestricted", steps=cfg.get("text_train_steps"))
        except Exception as exc:
            report(f"Unrestricted train skipped: {exc}")
    if result.get("images_downloaded", 0) > 0 and cfg.get("auto_train_image", True):
        report("Training image model on NSFW images...")
        try:
            from navine.learn.image_learn import train_learned_image

            train_learned_image(finetune_steps=cfg.get("image_train_steps"))
        except Exception as exc:
            report(f"Image train skipped: {exc}")
    if result.get("videos_downloaded", 0) > 0 and cfg.get("auto_train_video", True):
        report("Training video model on NSFW videos...")
        try:
            from navine.learn.video_learn import train_learned_video

            train_learned_video(finetune_steps=cfg.get("video_train_steps"))
        except Exception as exc:
            report(f"Video train skipped: {exc}")
    state = _load_state()
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["runs"] = int(state.get("runs", 0)) + 1
    state["images_downloaded"] = int(state.get("images_downloaded", 0)) + int(result.get("images_downloaded", 0))
    state["videos_downloaded"] = int(state.get("videos_downloaded", 0)) + int(result.get("videos_downloaded", 0))
    state["last_new_samples"] = ingest.get("new_samples", 0)
    _save_state(state)
    report("Navine AI - Python NSFW autolearn cycle complete.")
    return {
        "status": "ok",
        "new_samples": ingest.get("new_samples", 0),
        "items_fetched": result.get("items_fetched", 0),
        "images_downloaded": result.get("images_downloaded", 0),
        "videos_downloaded": result.get("videos_downloaded", 0),
        "by_category": ingest.get("by_category", {}),
        "media_category_counts": result.get("media_category_counts", {}),
        "warnings": result.get("warnings", []),
    }


def run_nsfw_loop(hours: float = 8.0, progress: ProgressCallback = None) -> None:
    cfg = load_nsfw_config()
    interval = float(cfg.get("auto_learn_interval_minutes", 30)) * 60
    started = time.time()
    end = started + hours * 3600
    while time.time() < end:
        if not is_nsfw_enabled():
            break
        run_nsfw_cycle(progress=progress)
        remaining = end - time.time()
        if remaining <= 0:
            break
        sleep_for = min(interval, remaining)
        if progress:
            progress(f"NSFW autolearn sleeping {sleep_for / 60:.1f} minutes...")
        time.sleep(sleep_for)


def get_nsfw_autolearn_status() -> Dict[str, Any]:
    state = _load_state()
    cfg = load_nsfw_config()
    from navine.nsfw.config import configured_source_groups
    from navine.reference.config import list_reference_entries
    from navine.reference.media import count_folder_media, get_reference_usage_stats
    from navine.utils.paths import get_project_root

    reference_map: Dict[str, Dict[str, Any]] = {}
    root = get_project_root()
    for category, entry in list_reference_entries().items():
        folder = root / str(entry.get("dir") or "")
        image_count, video_count = count_folder_media(folder)
        reference_map[category] = {
            "dir": entry.get("dir"),
            "keywords": entry.get("keywords") or [],
            "images": image_count,
            "videos": video_count,
            "uses": int((get_reference_usage_stats().get("categories") or {}).get(category, 0)),
        }

    fourchan_info: Dict[str, Any] = {}
    try:
        from navine.autolearn.sources.fourchan import list_fourchan_boards

        fourchan_info = list_fourchan_boards(cfg)
    except Exception:
        fourchan_info = {}

    return {
        "enabled": True,
        "unrestricted_mode": True,
        "mixed_mode": cfg.get("mixed_mode", True),
        "auto_learn": cfg.get("auto_learn", True),
        "source_groups": list(configured_source_groups(cfg).keys()),
        "fourchan_enabled": bool((cfg.get("fourchan") or {}).get("enabled", False)),
        "fourchan_board_groups": {
            name: len(boards) for name, boards in (fourchan_info.get("groups") or {}).items()
        },
        "fourchan_board_count": fourchan_info.get("board_count", 0),
        "fourchan_max_boards_per_run": fourchan_info.get("max_boards_per_run", 0),
        "reference_keywords": reference_map,
        "last_run": state.get("last_run"),
        "runs": state.get("runs", 0),
        "images_downloaded": state.get("images_downloaded", 0),
        "videos_downloaded": state.get("videos_downloaded", 0),
        "last_new_samples": state.get("last_new_samples", 0),
        "product": "Navine AI - Python",
    }
