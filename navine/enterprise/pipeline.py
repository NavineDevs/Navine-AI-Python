import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from navine.utils.config import load_config
from navine.utils.paths import get_project_root

ProgressCallback = Optional[Callable[[str], None]]


def _enterprise_log_dir() -> Path:
    path = get_project_root() / "logs" / "enterprise"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path() -> Path:
    return _enterprise_log_dir() / "pipeline_state.json"


def _report_path() -> Path:
    return _enterprise_log_dir() / "pipeline_report.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_enterprise_config() -> Dict[str, Any]:
    return load_config("enterprise")


def load_pipeline_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"started_at": None, "phase_index": 0, "phases_done": [], "errors": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pipeline_state(state: Dict[str, Any]) -> None:
    with _state_path().open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def get_enterprise_status() -> Dict[str, Any]:
    state = load_pipeline_state()
    cfg = load_enterprise_config()
    phases = cfg.get("phases") or []
    current = None
    idx = int(state.get("phase_index", 0))
    if 0 <= idx < len(phases):
        current = phases[idx].get("id")
    return {
        "phase_index": idx,
        "current_phase": current,
        "phases_total": len(phases),
        "phases_done": state.get("phases_done") or [],
        "started_at": state.get("started_at"),
        "last_update": state.get("last_update"),
        "errors": state.get("errors") or [],
        "active_tier": _read_active_tier(),
    }


def _read_active_tier() -> str:
    try:
        navine = load_config("navine")
        return str((navine.get("enterprise") or {}).get("active_tier", "standard"))
    except Exception:
        return "standard"


def _set_active_tier(tier: str) -> None:
    root = get_project_root()
    navine_path = root / "configs" / "navine.yaml"
    with navine_path.open("r", encoding="utf-8") as handle:
        navine = yaml.safe_load(handle) or {}
    ent = dict(navine.get("enterprise") or {})
    ent["active_tier"] = tier
    navine["enterprise"] = ent
    with navine_path.open("w", encoding="utf-8") as handle:
        yaml.dump(navine, handle, default_flow_style=False, sort_keys=False)


def _log(msg: str, progress: ProgressCallback = None) -> None:
    line = f"[{_now_iso()}] {msg}"
    log_file = _enterprise_log_dir() / "pipeline.log"
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if progress:
        progress(line)


def _phase_ingest(phase: Dict[str, Any], progress: ProgressCallback = None) -> None:
    _log("Enterprise ingest: collecting training data", progress)
    try:
        from navine.autolearn.engine import AutolearnEngine

        engine = AutolearnEngine()
        max_items = int(phase.get("autolearn_max_items", 300))
        items = engine.fetch_everything(max_items=max_items)
        _log(f"Autolearn fetched {len(items)} items", progress)
        engine.run_cycle(aggressive=True)
        _log("Autolearn cycle complete", progress)
        if phase.get("nsfw_ingest", True):
            try:
                from navine.nsfw.autolearn import run_nsfw_cycle

                run_nsfw_cycle()
                _log("NSFW autolearn cycle complete", progress)
            except Exception as exc:
                _log(f"NSFW ingest skipped: {exc}", progress)
    except Exception as exc:
        _log(f"Ingest phase partial failure: {exc}", progress)


def _phase_text(phase: Dict[str, Any], progress: ProgressCallback = None) -> None:
    config_name = str(phase.get("config", "text_enterprise"))
    _log(f"Enterprise text training ({config_name})", progress)
    from navine.text.train import train as train_text

    train_text(config_name)
    from navine.enterprise.eval import eval_text_enterprise

    result = eval_text_enterprise()
    min_score = float(phase.get("min_pass_score", 0.5))
    _log(f"Text eval: {result}", progress)
    if float(result.get("score", 0)) < min_score:
        _log(f"Text score {result.get('score')} below {min_score}; continuing anyway", progress)


def _phase_image(phase: Dict[str, Any], progress: ProgressCallback = None) -> None:
    config_name = str(phase.get("config", "image_enterprise"))
    _log(f"Enterprise image training ({config_name})", progress)
    from navine.image.train import train as train_image

    train_image(config_name, finetune=False)
    cycles = int(phase.get("quality_cycles", 2))
    min_score = float(phase.get("min_pass_score", 0.7))
    from navine.enterprise.eval import eval_image_enterprise

    best = None
    for cycle in range(1, cycles + 1):
        _log(f"Image quality cycle {cycle}/{cycles}", progress)
        result = eval_image_enterprise()
        _log(f"Image eval cycle {cycle}: {result}", progress)
        if best is None or float(result.get("average_score", 0)) > float(best.get("average_score", 0)):
            best = result
        if result.get("all_pass") and float(result.get("average_score", 0)) >= min_score:
            break
        if cycle < cycles:
            from navine.image.train import train as train_image_ft

            train_image_ft(config_name, finetune=True, finetune_steps=250)
    if best:
        _log(f"Image enterprise best eval: {best}", progress)


def _phase_video(phase: Dict[str, Any], progress: ProgressCallback = None) -> None:
    config_name = str(phase.get("config", "video_enterprise"))
    _log(f"Enterprise video training ({config_name})", progress)
    from navine.video.train import train as train_video

    train_video(config_name, finetune=False)
    from navine.enterprise.eval import eval_video_enterprise

    result = eval_video_enterprise()
    _log(f"Video eval: {result}", progress)


def _phase_eval(phase: Dict[str, Any], progress: ProgressCallback = None) -> Dict[str, Any]:
    _log("Enterprise multimodal evaluation", progress)
    from navine.enterprise.eval import run_enterprise_eval

    report = run_enterprise_eval(phase.get("sample_prompts") or {})
    if phase.get("run_benchmark", True):
        try:
            from navine.benchmark import run_benchmark

            report["benchmark"] = run_benchmark()
        except Exception as exc:
            report["benchmark_error"] = str(exc)
    with _report_path().open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _log(f"Eval report written to {_report_path()}", progress)
    return report


def _phase_promote(phase: Dict[str, Any], progress: ProgressCallback = None) -> None:
    tier = str(phase.get("set_active_tier", "enterprise"))
    _set_active_tier(tier)
    _log(f"Active tier set to {tier}", progress)


_PHASE_RUNNERS = {
    "ingest": _phase_ingest,
    "text": _phase_text,
    "image": _phase_image,
    "video": _phase_video,
    "eval": _phase_eval,
    "promote": _phase_promote,
}


def run_enterprise_pipeline(
    resume: bool = False,
    start_phase: Optional[str] = None,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = load_enterprise_config()
    phases: List[Dict[str, Any]] = list(cfg.get("phases") or [])
    state = load_pipeline_state() if resume else {
        "started_at": _now_iso(),
        "phase_index": 0,
        "phases_done": [],
        "errors": [],
    }
    if not resume:
        save_pipeline_state(state)

    start_idx = int(state.get("phase_index", 0))
    if start_phase:
        for i, ph in enumerate(phases):
            if ph.get("id") == start_phase:
                start_idx = i
                break

    for idx in range(start_idx, len(phases)):
        phase = phases[idx]
        phase_id = str(phase.get("id", f"phase_{idx}"))
        state["phase_index"] = idx
        state["last_update"] = _now_iso()
        save_pipeline_state(state)
        _log(f"Starting phase {phase_id} ({idx + 1}/{len(phases)})", progress)
        runner = _PHASE_RUNNERS.get(phase_id)
        if not runner:
            _log(f"Unknown phase {phase_id}, skipping", progress)
            continue
        try:
            runner(phase, progress)
            done = list(state.get("phases_done") or [])
            if phase_id not in done:
                done.append(phase_id)
            state["phases_done"] = done
            state["phase_index"] = idx + 1
            state["last_update"] = _now_iso()
            save_pipeline_state(state)
            _log(f"Completed phase {phase_id}", progress)
        except Exception as exc:
            err = {"phase": phase_id, "error": str(exc), "trace": traceback.format_exc()}
            errors = list(state.get("errors") or [])
            errors.append(err)
            state["errors"] = errors
            state["last_update"] = _now_iso()
            save_pipeline_state(state)
            _log(f"Phase {phase_id} failed: {exc}", progress)
            raise
    state["completed_at"] = _now_iso()
    state["phase_index"] = len(phases)
    save_pipeline_state(state)
    _log("Enterprise pipeline complete", progress)
    return get_enterprise_status()
