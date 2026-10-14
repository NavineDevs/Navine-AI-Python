import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "real_ai_train"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    if not PYTHON.exists():
        _log("ERROR: venv python not found")
        return 1

    sys.path.insert(0, str(ROOT))
    from navine.device_manager import configure_compute_environment
    from navine.enterprise.eval import eval_text_enterprise
    from navine.text.infer import clear_model_cache
    from navine.train.registry import train_type
    from navine.utils.training_lock import clear_stale_lock

    for name in ("text", "voice", "image", "video"):
        if clear_stale_lock(name):
            _log(f"Cleared stale lock: {name}")

    compute = configure_compute_environment()
    _log(f"REAL_AI_TRAIN_BEGIN device={compute.get('device')}")

    stages = [
        ("chat", 1400),
        ("coding", 900),
        ("math", 500),
        ("general", 400),
    ]
    for label, steps in stages:
        _log(f"STAGE {label} steps={steps}")
        try:
            train_type(label, steps=steps)
        except Exception as exc:
            _log(f"STAGE {label} ERROR: {exc}")
        clear_model_cache()

    report = eval_text_enterprise()
    _log(f"TEXT_EVAL score={report.get('score')} passes={report.get('passes')}")
    (LOG_DIR / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        from navine.doctor import check_chat_smoke, check_code_smoke

        chat = check_chat_smoke()
        code = check_code_smoke()
        _log(f"DOCTOR chat={chat.passed} code={code.passed}")
    except Exception as exc:
        _log(f"DOCTOR ERROR: {exc}")

    _log("REAL_AI_TRAIN_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
