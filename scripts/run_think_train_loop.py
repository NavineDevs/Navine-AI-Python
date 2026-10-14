"""Continuous dual training for Navine AI and Navuryx AI with live progress."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DOWNLOADS = Path(__file__).resolve().parents[1]
if DOWNLOADS.name.endswith(" AI") or "Navine" in DOWNLOADS.name or "Navuryx" in DOWNLOADS.name:
    PARENT = DOWNLOADS.parent
else:
    PARENT = DOWNLOADS

PRODUCTS: List[Dict[str, Any]] = [
    {"name": "navuryx", "root": PARENT / "Navuryx AI", "pkg": "navuryx"},
    {"name": "navine", "root": PARENT / "Navine AI", "pkg": "navine"},
]

LOG_ROOT = PARENT / "Navine AI" / "logs" / "think_train"
STATE_PATH = LOG_ROOT / "state.json"
MASTER_LOG = LOG_ROOT / "master.log"
LIVE_PATH = LOG_ROOT / "progress.live.json"
LOCK_PATH = LOG_ROOT / "loop.lock"

STAGE_CATALOG = [
    "multimodal-text",
    "nsfw",
    "unrestricted",
    "general",
    "creative",
    "coding",
    "math",
    "chat",
    "image",
    "video",
    "voice",
    "eval",
]

_LIVE: Dict[str, Any] = {
    "running": False,
    "updated_at": None,
    "cycle": 0,
    "max_cycles": 48,
    "product": None,
    "stage": None,
    "stage_index": 0,
    "stage_total": 1,
    "steps": 0,
    "steps_total": 0,
    "label": "",
    "elapsed_s": 0,
    "eta_s": None,
    "percent": 0.0,
    "products": {},
    "stage_started_at": None,
    "cycle_started_at": None,
}
_STAGE_DURATIONS: List[float] = []
_HEARTBEAT_STOP = threading.Event()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with MASTER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _write_live(**updates: Any) -> None:
    _LIVE.update(updates)
    _LIVE["updated_at"] = _now()
    stage_i = int(_LIVE.get("stage_index") or 0)
    stage_t = max(1, int(_LIVE.get("stage_total") or 1))
    frac = 0.0
    if _LIVE.get("stage_started_at") and _STAGE_DURATIONS:
        avg = sum(_STAGE_DURATIONS[-8:]) / max(1, len(_STAGE_DURATIONS[-8:]))
        if avg > 0:
            elapsed = time.time() - float(_LIVE["stage_started_at"])
            frac = min(0.95, elapsed / avg)
    steps = int(_LIVE.get("steps") or 0)
    steps_total = int(_LIVE.get("steps_total") or 0)
    if steps_total > 0 and steps > 0:
        frac = max(frac, min(0.95, steps / steps_total))
    percent = min(100.0, max(0.0, ((stage_i + frac) / stage_t) * 100.0))
    _LIVE["percent"] = round(percent, 1)
    remaining = max(0, stage_t - stage_i - frac)
    if _STAGE_DURATIONS:
        avg = sum(_STAGE_DURATIONS[-8:]) / max(1, len(_STAGE_DURATIONS[-8:]))
        _LIVE["eta_s"] = int(remaining * avg)
    if _LIVE.get("cycle_started_at"):
        _LIVE["elapsed_s"] = int(time.time() - float(_LIVE["cycle_started_at"]))
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        LIVE_PATH.write_text(json.dumps(_LIVE, indent=2), encoding="utf-8")
    except Exception:
        pass


def _heartbeat_loop() -> None:
    while not _HEARTBEAT_STOP.wait(5.0):
        try:
            _write_live()
        except Exception:
            pass


def _stop_requested() -> bool:
    for product in PRODUCTS:
        if (product["root"] / "logs" / "think_train" / "STOP").exists():
            return True
    if (LOG_ROOT / "STOP").exists():
        return True
    return False


def _save_state(state: Dict[str, Any]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cycle": 0, "best_scores": {}, "consecutive_pass": {}, "history": []}


def _python(root: Path) -> Path:
    py = root / "venv" / "Scripts" / "python.exe"
    if py.exists():
        return py
    return PARENT / "Navine AI" / "venv" / "Scripts" / "python.exe"


def _stages_per_product(skip_image_video: bool) -> int:
    text = 8
    media = 0 if skip_image_video else 3
    eval_n = 1
    return text + media + eval_n


def _acquire_loop_lock() -> bool:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text(encoding="ascii").strip())
            if os.name == "nt":
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return False
            else:
                os.kill(pid, 0)
                return False
        except Exception:
            pass
    LOCK_PATH.write_text(str(os.getpid()), encoding="ascii")
    return True


def _release_loop_lock() -> None:
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception:
        pass


def _run(
    root: Path,
    args: List[str],
    label: str,
    *,
    product: str,
    stage: str,
    steps: int = 0,
    stage_index: int = 0,
    stage_total: int = 1,
) -> int:
    py = _python(root)
    cmd = [str(py)] + args
    _write_live(
        running=True,
        product=product,
        stage=stage,
        stage_index=stage_index,
        stage_total=stage_total,
        steps=0,
        steps_total=steps,
        label=label,
        stage_started_at=time.time(),
        products={
            **(_LIVE.get("products") or {}),
            product: {
                "phase": stage,
                "label": label,
                "score": (_LIVE.get("products") or {}).get(product, {}).get("score"),
            },
        },
    )
    _log(f"{root.name} | START {label}")
    start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if "Step " in line and "Val Loss" in line:
                try:
                    part = line.split("Step ", 1)[1]
                    n = int(part.split("|", 1)[0].strip())
                    _write_live(steps=n)
                except Exception:
                    pass
            elif "it/s" in line and "/" in line:
                try:
                    for tok in line.replace("\r", " ").split():
                        if "/" in tok and tok[0].isdigit():
                            cur, tot = tok.split("/", 1)
                            cur_i = int(cur)
                            tot_i = int("".join(ch for ch in tot if ch.isdigit()) or "0")
                            if tot_i > 0:
                                _write_live(steps=cur_i, steps_total=max(steps, tot_i))
                            break
                except Exception:
                    pass
        code = int(proc.wait())
    except Exception as exc:
        _log(f"{root.name} | ERROR {label}: {exc}")
        code = 1
    elapsed = time.time() - start
    _STAGE_DURATIONS.append(elapsed)
    if len(_STAGE_DURATIONS) > 40:
        del _STAGE_DURATIONS[:-40]
    _log(f"{root.name} | EXIT {label} code={code} {int(elapsed) // 60}m{int(elapsed) % 60}s")
    _write_live(
        steps=steps if steps else int(_LIVE.get("steps") or 0),
        steps_total=steps or int(_LIVE.get("steps_total") or 0),
        label=f"{label} done",
        stage_started_at=None,
    )
    return code


def _stage_steps(base: int, cycle: int, growth: int, cap: int) -> int:
    return min(cap, base + max(0, cycle - 1) * growth)


def _train_text_cycle(
    root: Path,
    pkg: str,
    product: str,
    cycle: int,
    stage_offset: int,
    stage_total: int,
) -> int:
    multi_s = _stage_steps(300, cycle, 40, 900)
    general_s = _stage_steps(500, cycle, 80, 1800)
    creative_s = _stage_steps(400, cycle, 50, 1000)
    code_s = _stage_steps(1000, cycle, 150, 3200)
    math_s = _stage_steps(800, cycle, 120, 2400)
    chat_s = _stage_steps(2000, cycle, 250, 5000)
    nsfw_s = min(600, 250 + cycle * 30)
    unres_s = min(600, 250 + cycle * 30)
    stages = [
        ("multimodal-text", multi_s),
        ("nsfw", nsfw_s),
        ("unrestricted", unres_s),
        ("general", general_s),
        ("creative", creative_s),
        ("coding", code_s),
        ("math", math_s),
        ("chat", chat_s),
    ]
    idx = stage_offset
    for name, steps in stages:
        if _stop_requested():
            return idx
        script = (
            f"import sys; sys.path.insert(0, r'{root}'); "
            f"from {pkg}.device_manager import configure_compute_environment; "
            f"from {pkg}.text.infer import clear_model_cache; "
            f"from {pkg}.train.registry import train_type; "
            f"from {pkg}.utils.training_lock import clear_stale_lock; "
            f"[clear_stale_lock(n) for n in ('text','image','video','voice')]; "
            f"configure_compute_environment(); "
            f"train_type('{name}', steps={steps}); "
            f"clear_model_cache()"
        )
        _run(
            root,
            ["-u", "-c", script],
            f"text:{name}:{steps}",
            product=product,
            stage=name,
            steps=steps,
            stage_index=idx,
            stage_total=stage_total,
        )
        idx += 1
    return idx


def _train_image_video(
    root: Path,
    pkg: str,
    product: str,
    cycle: int,
    stage_offset: int,
    stage_total: int,
) -> int:
    if _stop_requested():
        return stage_offset
    img_steps = min(600, 300 + cycle * 40)
    vid_steps = min(300, 120 + cycle * 20)
    idx = stage_offset
    jobs = [
        (
            "image",
            img_steps,
            (
                f"import sys; sys.path.insert(0, r'{root}'); "
                f"from {pkg}.image.train import train; "
                f"train(finetune=True, finetune_steps={img_steps})"
            ),
            f"image_ft:{img_steps}",
        ),
        (
            "video",
            vid_steps,
            (
                f"import sys; sys.path.insert(0, r'{root}'); "
                f"from {pkg}.video.train import train; "
                f"train(finetune=True, finetune_steps={vid_steps})"
            ),
            f"video_ft:{vid_steps}",
        ),
        (
            "voice",
            4,
            (
                f"import sys; sys.path.insert(0, r'{root}'); "
                f"from {pkg}.voice.train import train_voice; "
                f"train_voice(steps=4)"
            ),
            "voice_calibrate",
        ),
    ]
    for stage, steps, script, label in jobs:
        if _stop_requested():
            return idx
        _run(
            root,
            ["-u", "-c", script],
            label,
            product=product,
            stage=stage,
            steps=steps,
            stage_index=idx,
            stage_total=stage_total,
        )
        idx += 1
    return idx


def _evaluate(root: Path, pkg: str) -> Dict[str, Any]:
    py = _python(root)
    script = (
        f"import json,sys; sys.path.insert(0, r'{root}'); "
        f"from {pkg}.enterprise.eval import eval_text_enterprise; "
        f"from {pkg}.doctor import check_chat_smoke, check_code_smoke; "
        f"from {pkg}.text.chat import chat_with_meta; "
        f"rep=eval_text_enterprise(); "
        f"chat=check_chat_smoke(); code=check_code_smoke(); "
        f"ans,_=chat_with_meta('What is 3 plus 5?'); "
        f"print(json.dumps({{'score':rep.get('score'),'passes':rep.get('passes'),"
        f"'chat':chat.passed,'code':code.passed,'sample':(ans or '')[:180]}}))"
    )
    try:
        proc = subprocess.run(
            [str(py), "-u", "-c", script],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        line = (proc.stdout or "").strip().splitlines()
        raw = line[-1] if line else "{}"
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc), "score": 0.0, "passes": False}


def _main_body(args: argparse.Namespace) -> int:
    state = _load_state()
    if args.start_cycle:
        state["cycle"] = max(int(state.get("cycle") or 0), args.start_cycle)

    deadline = time.time() + float(args.max_hours) * 3600.0
    spp = _stages_per_product(args.skip_image_video)
    stage_total = spp * len(PRODUCTS)

    _write_live(
        running=True,
        cycle=int(state.get("cycle") or 0),
        max_cycles=args.max_cycles,
        stage_total=stage_total,
        stage_index=0,
        label="starting",
        products={p["name"]: {"phase": "idle", "score": None} for p in PRODUCTS},
        cycle_started_at=None,
    )
    _log(
        f"THINK_TRAIN_BEGIN max_cycles={args.max_cycles} max_hours={args.max_hours} "
        f"text_target={args.text_target} pass_streak={args.pass_streak}"
    )
    _log("Create logs/think_train/STOP to halt cleanly.")

    for product in PRODUCTS:
        root: Path = product["root"]
        if not root.exists():
            _log(f"MISSING product root {root}")
            _write_live(running=False, label="missing product root")
            return 1
        if not _python(root).exists():
            _log(f"MISSING venv python for {root}")
            _write_live(running=False, label="missing venv")
            return 1
        (root / "logs" / "think_train").mkdir(parents=True, exist_ok=True)

    while int(state.get("cycle") or 0) < args.max_cycles:
        if _stop_requested():
            _log("STOP file detected")
            break
        if time.time() >= deadline:
            _log("max hours reached")
            break

        cycle = int(state.get("cycle") or 0) + 1
        state["cycle"] = cycle
        _log(f"===== CYCLE {cycle}/{args.max_cycles} =====")
        cycle_report: Dict[str, Any] = {"cycle": cycle, "products": {}}
        _write_live(
            running=True,
            cycle=cycle,
            max_cycles=args.max_cycles,
            stage_index=0,
            stage_total=stage_total,
            cycle_started_at=time.time(),
            label=f"cycle {cycle} begin",
        )

        all_pass = True
        stage_cursor = 0
        for product in PRODUCTS:
            if _stop_requested() or time.time() >= deadline:
                break
            name = product["name"]
            root = product["root"]
            pkg = product["pkg"]
            _log(f"PRODUCT {name} cycle {cycle}")
            _write_live(product=name, label=f"{name} cycle {cycle}")
            stage_cursor = _train_text_cycle(
                root, pkg, name, cycle, stage_cursor, stage_total
            )
            if not args.skip_image_video:
                stage_cursor = _train_image_video(
                    root, pkg, name, cycle, stage_cursor, stage_total
                )
            if _stop_requested():
                break
            _write_live(
                product=name,
                stage="eval",
                stage_index=stage_cursor,
                stage_total=stage_total,
                label=f"{name} eval",
                stage_started_at=time.time(),
            )
            t0 = time.time()
            report = _evaluate(root, pkg)
            _STAGE_DURATIONS.append(time.time() - t0)
            stage_cursor += 1
            score = float(report.get("score") or 0.0)
            best = float((state.get("best_scores") or {}).get(name) or 0.0)
            if score > best:
                state.setdefault("best_scores", {})[name] = score
            passed = bool(report.get("passes")) or score >= args.text_target
            streak = int((state.get("consecutive_pass") or {}).get(name) or 0)
            streak = streak + 1 if passed else 0
            state.setdefault("consecutive_pass", {})[name] = streak
            cycle_report["products"][name] = {
                "score": score,
                "passed": passed,
                "streak": streak,
                "chat": report.get("chat"),
                "code": report.get("code"),
                "sample": report.get("sample"),
            }
            products = dict(_LIVE.get("products") or {})
            products[name] = {
                "phase": "eval_done",
                "score": score,
                "streak": streak,
                "sample": report.get("sample"),
            }
            _write_live(products=products, stage_index=stage_cursor)
            _log(
                f"{name} eval score={score} passes={report.get('passes')} "
                f"chat={report.get('chat')} code={report.get('code')} streak={streak}"
            )
            if streak < args.pass_streak:
                all_pass = False

        state.setdefault("history", []).append(cycle_report)
        if len(state["history"]) > 200:
            state["history"] = state["history"][-200:]
        _save_state(state)
        (LOG_ROOT / f"cycle_{cycle:04d}.json").write_text(
            json.dumps(cycle_report, indent=2), encoding="utf-8"
        )
        _write_live(
            stage="cycle_done",
            stage_index=stage_total,
            percent=100.0,
            label=f"cycle {cycle} complete",
        )

        if all_pass and all(
            int((state.get("consecutive_pass") or {}).get(p["name"]) or 0) >= args.pass_streak
            for p in PRODUCTS
        ):
            _log(
                f"QUALITY TARGET MET for {args.pass_streak} consecutive cycles "
                f"at text_target={args.text_target}"
            )
            break

    _log(f"THINK_TRAIN_END cycle={state.get('cycle')} best={state.get('best_scores')}")
    _save_state(state)
    _write_live(
        running=False,
        stage="idle",
        label="think train ended",
        stage_started_at=None,
        product=None,
        percent=float(_LIVE.get("percent") or 0),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous dual Navine/Navuryx think train")
    parser.add_argument("--max-cycles", type=int, default=48)
    parser.add_argument("--max-hours", type=float, default=96.0)
    parser.add_argument("--text-target", type=float, default=0.7)
    parser.add_argument("--pass-streak", type=int, default=3)
    parser.add_argument("--skip-image-video", action="store_true")
    parser.add_argument("--start-cycle", type=int, default=0)
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not _acquire_loop_lock():
        _log("Another think-train loop is already running (loop.lock). Exit.")
        return 2

    _HEARTBEAT_STOP.clear()
    heartbeat = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat.start()
    try:
        return _main_body(args)
    finally:
        _HEARTBEAT_STOP.set()
        _release_loop_lock()


if __name__ == "__main__":
    raise SystemExit(main())
