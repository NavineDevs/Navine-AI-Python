import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "custom_only"
OUT_LOG = LOG_DIR / "train.log"
ERR_LOG = LOG_DIR / "train.err.log"
STATE_PATH = LOG_DIR / "state.json"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _load_settings() -> Dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from navine.utils.config import load_config

    navine = load_config("navine")
    training = navine.get("training") or {}
    return {
        "max_rounds": int(training.get("max_rounds", 100)),
        "text_pass_score": float(training.get("text_pass_score", 0.55)),
        "image_pass_score": float(training.get("image_pass_score", 0.62)),
        "base_steps": int(training.get("base_steps", 1200)),
        "step_growth": int(training.get("step_growth", 200)),
        "max_steps": int(training.get("max_steps", 3000)),
    }


def _save_state(state: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _clear_stale_locks() -> None:
    from navine.utils.training_lock import clear_stale_lock

    for name in ("image", "video", "text"):
        if clear_stale_lock(name):
            _log(f"Cleared stale {name} training lock")


def _run(label: str, cmd: list[str]) -> int:
    _log(f"START {label}: {' '.join(cmd)}")
    start = time.time()
    try:
        with OUT_LOG.open("a", encoding="utf-8") as out, ERR_LOG.open("a", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err)
        code = int(proc.returncode)
    except Exception as exc:
        _log(f"ERROR {label}: {exc}")
        code = 1
    elapsed = int(time.time() - start)
    _log(f"EXIT {label} code={code} elapsed={elapsed // 3600}h{(elapsed % 3600) // 60}m{elapsed % 60}s")
    return code


def _evaluate_quality(text_target: float, image_target: float) -> Tuple[bool, Dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from navine.enterprise.eval import eval_image_enterprise, eval_text_enterprise, eval_video_enterprise

    text = eval_text_enterprise()
    image = eval_image_enterprise()
    video = eval_video_enterprise()
    text_score = float(text.get("score", 0.0))
    image_score = float(image.get("average_score", 0.0))
    text_ok = bool(text.get("passes")) or text_score >= text_target
    image_ok = bool(image.get("all_pass")) or image_score >= image_target
    report = {
        "text": text,
        "image": image,
        "video": video,
        "text_ok": text_ok,
        "image_ok": image_ok,
        "ready": text_ok and image_ok,
    }
    return text_ok and image_ok, report


def _round_steps(settings: Dict[str, Any], round_num: int) -> int:
    steps = settings["base_steps"] + (round_num - 1) * settings["step_growth"]
    return min(steps, settings["max_steps"])


def _run_round(round_num: int, settings: Dict[str, Any]) -> None:
    steps = _round_steps(settings, round_num)
    _log(f"ROUND {round_num} begin (steps={steps})")

    _run("prepare_data", [str(PYTHON), str(ROOT / "scripts" / "prepare_data.py")])
    _run("learn_everything", [str(PYTHON), "-m", "navine.cli", "learn", "everything"])
    _run("nsfw_autolearn", [str(PYTHON), "-m", "navine.cli", "learn", "nsfw", "autolearn"])
    _run("train_full", [str(PYTHON), "-u", "-m", "navine.cli", "train", "full", "--steps", str(steps)])
    _run(
        "enterprise_pipeline",
        [str(PYTHON), "-u", "-m", "navine.cli", "enterprise", "train", "--resume"],
    )
    _run(
        "custom_image_lora",
        [str(PYTHON), "-u", "-m", "navine.cli", "train", "image-lora", "--steps", str(min(steps, 2000))],
    )
    _run(
        "image_train_loop",
        [
            str(PYTHON),
            "-c",
            "from navine.image.train_loop import run_train_loop; run_train_loop(max_cycles=8, finetune_steps=300)",
        ],
    )
    _run(
        "enterprise_dual",
        [
            str(PYTHON),
            "-u",
            "-m",
            "navine.cli",
            "enterprise",
            "dual",
            "--max-cycles",
            "15",
            "--finetune-steps",
            "300",
        ],
    )


def main() -> int:
    if not PYTHON.exists():
        _log("ERROR: venv python not found")
        return 1

    sys.path.insert(0, str(ROOT))
    settings = _load_settings()
    _log("CUSTOM_ONLY_TRAIN_BEGIN")
    _log("External backends disabled. Training Navine custom models only.")
    from navine.device_manager import configure_compute_environment

    compute = configure_compute_environment()
    _log(f"Compute: device={compute.get('device')} cpu_threads={compute.get('cpu_threads')} dataloader={compute.get('dataloader')}")
    _clear_stale_locks()

    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "round": 0,
        "ready": False,
        "settings": settings,
    }
    _save_state(state)

    for round_num in range(1, settings["max_rounds"] + 1):
        state["round"] = round_num
        _save_state(state)
        _run_round(round_num, settings)
        ready, report = _evaluate_quality(settings["text_pass_score"], settings["image_pass_score"])
        state["last_report"] = report
        state["ready"] = ready
        _save_state(state)
        _log(
            f"ROUND {round_num} eval text={report['text'].get('score')} "
            f"image={report['image'].get('average_score')} ready={ready}"
        )
        if ready:
            _run("enterprise_promote", [str(PYTHON), "-m", "navine.cli", "enterprise", "promote", "--tier", "enterprise"])
            _log("CUSTOM_ONLY_TRAIN_COMPLETE quality targets met")
            return 0

    _log("CUSTOM_ONLY_TRAIN_STOPPED max rounds reached without full quality pass")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
