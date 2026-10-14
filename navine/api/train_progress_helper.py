import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from navine.utils.paths import get_project_root

_LAST_REPORT: Dict[str, Any] = {"t": 0.0, "steps": -1, "stage": ""}


def _jobs_dir() -> Path:
    return get_project_root() / "logs" / "train_jobs"


def _parse_steps_from_log(text: str) -> tuple[int, int, float]:
    steps = 0
    steps_total = 0
    percent = 0.0
    progress_matches = re.findall(
        r"PROGRESS\s+steps?\s*=\s*(\d+)\s*/\s*(\d+)",
        text,
        re.I,
    )
    if progress_matches:
        steps = int(progress_matches[-1][0])
        steps_total = int(progress_matches[-1][1])
        if steps_total > 0:
            percent = min(100.0, 100.0 * steps / steps_total)
    step_matches = re.findall(r"(?:step|iter(?:ation)?)\s*[:=\s]\s*(\d+)", text, re.I)
    if not step_matches:
        step_matches = re.findall(r"Step\s+(\d+)", text, re.I)
    if step_matches:
        steps = max(steps, int(step_matches[-1]))
    tqdm_matches = re.findall(r"(\d+)%\|", text)
    if tqdm_matches:
        percent = float(tqdm_matches[-1])
    frac_matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
    for a, b in frac_matches:
        ai, bi = int(a), int(b)
        if bi >= 1 and ai <= bi and bi <= 200000:
            steps = max(steps, ai)
            if bi > steps_total:
                steps_total = bi
    if steps_total > 0 and steps > 0 and percent <= 0:
        percent = min(100.0, 100.0 * steps / steps_total)
    return steps, steps_total, percent


def _parse_stack_log(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    current = None
    for line in reversed(lines[-40:]):
        if "] DONE " in line:
            break
        if "] START " in line:
            current = line.split("] START ", 1)[-1].strip()
            break
    if not current:
        return None
    return {"running": True, "stage": current, "label": current, "percent": 12.0}


def _parse_comprehensive_log() -> Optional[Dict[str, Any]]:
    path = get_project_root() / "logs" / "comprehensive_train" / "run.log"
    if not path.exists():
        return None
    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    except Exception:
        return None
    if not lines:
        return None
    last = lines[-1]
    if "Wrote " in last and "summary.json" in last:
        return None
    if age > 600:
        return None
    if "comprehensive train start" in last.lower():
        return {
            "running": True,
            "stage": "comprehensive",
            "label": "starting",
            "percent": 2.0,
            "last_line": last,
            "recent_lines": lines[-6:],
        }
    match = re.search(r"\btrain\s+([A-Za-z0-9_\-]+)", last, re.I)
    if "] RUN " in last and match:
        stage = match.group(1)
        return {
            "running": True,
            "stage": stage,
            "label": stage,
            "percent": 8.0,
            "last_line": last,
            "recent_lines": lines[-6:],
        }
    if "] RUN " in last:
        return {
            "running": True,
            "stage": "comprehensive",
            "label": last.split("] RUN ", 1)[-1][:80],
            "percent": 5.0,
            "last_line": last,
            "recent_lines": lines[-6:],
        }
    return None


def _iso_age_seconds(value: Any) -> Optional[float]:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())
    except Exception:
        return None


def write_admin_live_progress(payload: Dict[str, Any]) -> None:
    path = _jobs_dir() / "progress.live.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def report_train_progress(
    *,
    stage: str,
    steps: int = 0,
    steps_total: int = 0,
    label: Optional[str] = None,
    running: bool = True,
    percent: Optional[float] = None,
    last_line: Optional[str] = None,
    force: bool = False,
    cycle: Optional[int] = None,
    max_cycles: Optional[int] = None,
) -> None:
    now = time.monotonic()
    stage_key = str(stage or "training")
    steps_i = int(max(0, steps))
    total_i = int(max(0, steps_total))
    if (
        not force
        and stage_key == _LAST_REPORT.get("stage")
        and steps_i == int(_LAST_REPORT.get("steps") or -1)
        and (now - float(_LAST_REPORT.get("t") or 0.0)) < 0.75
    ):
        return
    if (
        not force
        and stage_key == _LAST_REPORT.get("stage")
        and (now - float(_LAST_REPORT.get("t") or 0.0)) < 0.75
        and steps_i > 0
        and steps_i % 5 != 0
        and steps_i != total_i
    ):
        return
    if percent is None:
        if total_i > 0:
            percent = min(99.5 if running else 100.0, 100.0 * steps_i / max(total_i, 1))
        else:
            percent = 5.0 if running else 0.0
    payload: Dict[str, Any] = {
        "running": bool(running),
        "stage": stage_key,
        "label": str(label or stage_key),
        "steps": steps_i,
        "steps_total": total_i,
        "percent": float(percent),
    }
    if cycle is not None:
        payload["cycle"] = int(cycle)
    if max_cycles is not None:
        payload["max_cycles"] = int(max_cycles)
    if last_line:
        payload["last_line"] = str(last_line)
        payload["recent_lines"] = [str(last_line)]
    try:
        existing = {}
        live_path = _jobs_dir() / "progress.live.json"
        if live_path.exists():
            try:
                existing = json.loads(live_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        if payload.get("cycle") is None and existing.get("cycle") is not None:
            payload["cycle"] = existing.get("cycle")
        if payload.get("max_cycles") is None and existing.get("max_cycles") is not None:
            payload["max_cycles"] = existing.get("max_cycles")
        write_admin_live_progress(payload)
        _LAST_REPORT["t"] = now
        _LAST_REPORT["steps"] = steps_i
        _LAST_REPORT["stage"] = stage_key
    except Exception:
        pass


def _parse_rebuild_150m_log() -> Optional[Dict[str, Any]]:
    path = get_project_root() / "logs" / "rebuild_150m.log"
    if not path.exists():
        return None
    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
        lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    except Exception:
        return None
    if not lines:
        return None
    joined = "\n".join(lines)
    if "REBUILD_150M_COMPLETE" in joined:
        return None

    stages = ["text_enterprise", "text_code", "image_v2", "video", "voice"]
    begun = [(i, ln.split("STAGE_BEGIN", 1)[-1].strip()) for i, ln in enumerate(lines) if "STAGE_BEGIN" in ln]
    if not begun:
        return None
    begin_i, current = begun[-1]
    ended_after = [
        ln for ln in lines[begin_i + 1 :]
        if f"STAGE_END {current}" in ln or f"STAGE_FAIL {current}" in ln
    ]
    stage_done = bool(ended_after)
    cycle = sum(1 for ln in lines if "STAGE_END" in ln)
    max_cycles = len(stages)
    running = not stage_done
    if stage_done and age > 120:
        running = False
    if age > 900 and stage_done:
        return None

    progress_matches = re.findall(r"PROGRESS\s+steps?\s*=\s*(\d+)\s*/\s*(\d+)", "\n".join(lines[-40:]), re.I)
    steps = 0
    steps_total = 0
    percent = max(5.0, 100.0 * cycle / max(max_cycles, 1))
    if progress_matches and not stage_done:
        steps = int(progress_matches[-1][0])
        steps_total = int(progress_matches[-1][1])
        if steps_total > 0:
            stage_pct = min(99.0, 100.0 * steps / steps_total)
            percent = min(99.5, (cycle / max(max_cycles, 1)) * 100.0 + stage_pct / max(max_cycles, 1))
    if not running and cycle >= max_cycles:
        return None
    return {
        "running": running,
        "stage": current or "rebuild_150m",
        "label": f"rebuild 150m · {current or 'idle'}" + (" (done)" if stage_done else ""),
        "cycle": min(max_cycles, max(cycle, 1 if current else 0)),
        "max_cycles": max_cycles,
        "steps": steps,
        "steps_total": steps_total,
        "percent": round(percent, 1) if running else min(100.0, round(100.0 * cycle / max(max_cycles, 1), 1)),
        "last_line": lines[-1],
        "recent_lines": lines[-8:],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _limit_train_log_dir() -> Path:
    root = get_project_root()
    local = root / "logs" / "limit_train"
    markers = ("marathon.stdout.log", "queue.log", "marathon.lock")
    if any((local / name).exists() for name in markers):
        return local
    try:
        for candidate in sorted(root.parent.iterdir()):
            if not candidate.is_dir() or candidate.resolve() == root.resolve():
                continue
            log_dir = candidate / "logs" / "limit_train"
            if any((log_dir / name).exists() for name in markers):
                return log_dir
    except Exception:
        pass
    return local


def _limit_train_running(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip().splitlines()[0])
    except Exception:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return True


def _parse_marathon_round(lines: list[str]) -> int:
    for line in reversed(lines[-200:]):
        match = re.search(r"ROUND\s+(\d+)\s+(?:begin|done)", line, re.I)
        if match:
            return int(match.group(1))
    return 0


def _parse_limit_train_log() -> Optional[Dict[str, Any]]:
    log_dir = _limit_train_log_dir()
    if not log_dir.exists():
        return None
    marathon_log = log_dir / "marathon.stdout.log"
    queue_log = log_dir / "queue.log"
    lock_path = log_dir / "marathon.lock"
    running = _limit_train_running(lock_path)
    queue_lines: list[str] = []
    marathon_lines: list[str] = []
    if queue_log.exists():
        try:
            queue_lines = [
                ln for ln in queue_log.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()
            ]
        except Exception:
            queue_lines = []
    if marathon_log.exists():
        try:
            marathon_lines = [
                ln for ln in marathon_log.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()
            ]
        except Exception:
            marathon_lines = []
    if not queue_lines and not marathon_lines:
        return None

    cycle = _parse_marathon_round(queue_lines) or _parse_marathon_round(marathon_lines)
    last_queue_line = queue_lines[-1] if queue_lines else ""
    current_job = None
    vram_wait = False
    scan_lines = queue_lines[-120:] if queue_lines else marathon_lines[-120:]
    for line in reversed(scan_lines):
        if " SKIP " in line and "only" in line and "GB free" in line:
            vram_wait = True
        if " START " in line:
            match = re.search(r"START\s+(\w+)", line)
            if match:
                current_job = match.group(1)
                break
        if " ROUND " in line and " begin" in line.lower():
            break
    if queue_log.exists():
        try:
            age = max(0.0, time.time() - queue_log.stat().st_mtime)
            if age < 300.0 and (
                " START " in last_queue_line
                or " ROUND " in last_queue_line
                or "LIMIT TRAIN BEGIN" in last_queue_line
            ):
                running = True
        except Exception:
            pass
    if running and not current_job and vram_wait:
        current_job = "vram_wait"

    steps = 0
    steps_total = 0
    percent = 0.0
    recent_lines: list[str] = []
    last_line = last_queue_line or (marathon_lines[-1] if marathon_lines else None)
    if current_job and current_job != "vram_wait":
        out_log = log_dir / f"{current_job}.out.log"
        err_log = log_dir / f"{current_job}.err.log"
        for log_path in (out_log, err_log):
            if not log_path.exists():
                continue
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
                parsed_steps, parsed_total, parsed_pct = _parse_steps_from_log(text[-20000:])
                if parsed_steps > steps:
                    steps = parsed_steps
                if parsed_total > steps_total:
                    steps_total = parsed_total
                if parsed_pct > percent:
                    percent = parsed_pct
                tail = [ln for ln in text.splitlines() if ln.strip()][-8:]
                if tail:
                    recent_lines = tail
                    last_line = tail[-1]
            except Exception:
                pass
    if steps_total > 0 and steps > 0 and percent <= 0:
        percent = min(99.5, 100.0 * steps / steps_total)
    elif vram_wait and running:
        percent = max(percent, 3.0)

    if not running and cycle <= 0 and not current_job:
        return None

    if current_job == "vram_wait":
        stage = "vram_wait"
        label = f"marathon round {cycle} · waiting for GPU VRAM"
    elif current_job:
        stage = current_job
        label = f"marathon round {cycle} · {current_job}"
        if steps_total > 0:
            label = f"{label} · {steps}/{steps_total}"
    else:
        stage = "marathon"
        label = f"marathon round {cycle} · between jobs"

    tail_recent = scan_lines[-6:] if scan_lines else []
    return {
        "running": running,
        "stage": stage,
        "label": label,
        "cycle": cycle,
        "max_cycles": 0,
        "marathon_round": cycle,
        "marathon_loop": True,
        "steps": steps,
        "steps_total": steps_total,
        "percent": round(percent, 1) if percent else (5.0 if running else 0.0),
        "last_line": last_line,
        "recent_lines": recent_lines or tail_recent or ([last_line] if last_line else []),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def enrich_progress_from_admin_jobs(live: Dict[str, Any], locks: Dict[str, bool]) -> Dict[str, Any]:
    live = dict(live or {})
    jobs_dir = _jobs_dir()
    live_path = jobs_dir / "progress.live.json"
    if live_path.exists():
        try:
            admin_live = json.loads(live_path.read_text(encoding="utf-8"))
            age = _iso_age_seconds(admin_live.get("updated_at"))
            fresh = age is None or age < 180.0
            if admin_live.get("running") and fresh:
                live.update(admin_live)
            elif admin_live.get("running") and not fresh:
                admin_live = dict(admin_live)
                admin_live["running"] = False
                admin_live["label"] = f"{admin_live.get('label') or admin_live.get('stage') or 'train'} (stale)"
                write_admin_live_progress(admin_live)
            elif fresh and float(admin_live.get("percent") or 0) > 0 and not live.get("running"):
                live.setdefault("percent", admin_live.get("percent"))
                live.setdefault("steps", admin_live.get("steps"))
                live.setdefault("steps_total", admin_live.get("steps_total"))
                live.setdefault("stage", admin_live.get("stage"))
                live.setdefault("label", admin_live.get("label"))
                live.setdefault("last_line", admin_live.get("last_line"))
                live.setdefault("recent_lines", admin_live.get("recent_lines"))
                if admin_live.get("cycle") is not None:
                    live.setdefault("cycle", admin_live.get("cycle"))
                if admin_live.get("max_cycles") is not None:
                    live.setdefault("max_cycles", admin_live.get("max_cycles"))
        except Exception:
            pass

    running_job = None
    for meta_path in sorted(jobs_dir.glob("*.json"), reverse=True):
        if meta_path.name == "progress.live.json":
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(meta.get("status") or "").lower()
        if status in ("running", "started", "queued"):
            running_job = meta
            break

    if running_job:
        live["running"] = True
        target = str(running_job.get("target") or "training")
        live["stage"] = target
        live["label"] = target
        steps_total = int(running_job.get("steps") or 0)
        if steps_total > 0:
            live["steps_total"] = steps_total
        job_id = str(running_job.get("id") or "")
        log_path = jobs_dir / f"{job_id}.log"
        if log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
                steps, parsed_total, pct = _parse_steps_from_log(text)
                if steps:
                    live["steps"] = steps
                if parsed_total:
                    live["steps_total"] = parsed_total
                if pct:
                    live["percent"] = pct
                recent = [ln for ln in text.splitlines() if ln.strip()][-8:]
                if recent:
                    live["recent_lines"] = recent
                    live["last_line"] = recent[-1]
            except Exception:
                pass
        if live.get("steps_total") and live.get("steps") and not live.get("percent"):
            live["percent"] = min(100.0, 100.0 * int(live["steps"]) / max(int(live["steps_total"]), 1))
        if live.get("running") and not live.get("percent"):
            live["percent"] = 8.0

    stack = _parse_stack_log(get_project_root() / "logs" / "modality_training_stack.log")
    if stack and not live.get("running"):
        live.update(stack)

    comprehensive = _parse_comprehensive_log()
    if comprehensive and not live.get("running"):
        live.update(comprehensive)

    limit_train = _parse_limit_train_log()
    if limit_train:
        if limit_train.get("running") or int(limit_train.get("cycle") or 0) > 0:
            for key, value in limit_train.items():
                if value is None:
                    continue
                if key == "max_cycles" and int(value or 0) <= 0:
                    continue
                if key in ("cycle", "marathon_round") and int(value or 0) > int(live.get("cycle") or 0):
                    live["cycle"] = int(value)
                    live["marathon_round"] = int(value)
                elif key not in live or live.get(key) in (None, 0, "", []):
                    live[key] = value
                elif key in ("label", "stage", "last_line", "recent_lines", "percent", "steps", "steps_total"):
                    live[key] = value
            if limit_train.get("running"):
                live["running"] = True
            if limit_train.get("marathon_loop"):
                live["marathon_loop"] = True

    rebuild = _parse_rebuild_150m_log()
    if rebuild:
        if rebuild.get("running") and not live.get("running"):
            live.update({k: rebuild[k] for k in rebuild if rebuild[k] is not None})
        elif rebuild.get("running") or live.get("running"):
            for key in ("cycle", "max_cycles"):
                if rebuild.get(key) is not None and live.get(key) in (None, 0):
                    live[key] = rebuild[key]
            if rebuild.get("label") and (
                not live.get("label") or str(live.get("label")).startswith("rebuild")
            ):
                if "rebuild" in str(rebuild.get("label") or ""):
                    live["label"] = rebuild["label"]
            if not live.get("stage") and rebuild.get("stage"):
                live["stage"] = rebuild["stage"]
            if rebuild.get("running"):
                live["running"] = True
        elif not live.get("running") and rebuild.get("cycle"):
            live.setdefault("cycle", rebuild.get("cycle"))
            live.setdefault("max_cycles", rebuild.get("max_cycles"))
            live.setdefault("percent", rebuild.get("percent"))
            live.setdefault("label", rebuild.get("label"))
            live.setdefault("stage", rebuild.get("stage"))
            live.setdefault("last_line", rebuild.get("last_line"))
            live.setdefault("recent_lines", rebuild.get("recent_lines"))

    active_locks = [name for name, active in (locks or {}).items() if active]
    if active_locks and not live.get("running"):
        live["running"] = True
        live["stage"] = active_locks[0]
        live["label"] = active_locks[0]
        live["percent"] = max(float(live.get("percent") or 0), 15.0)

    if live.get("running") and float(live.get("percent") or 0) <= 0:
        live["percent"] = 5.0

    return live
