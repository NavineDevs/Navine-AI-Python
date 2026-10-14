import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "full_modal_train"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
SIBLING = ROOT.parent / "Navine AI"
PKG = "navine"


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _run(args: list, env: dict | None = None) -> int:
    _log("RUN " + " ".join(str(a) for a in args))
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    _log(f"EXIT {proc.returncode}")
    return int(proc.returncode)


def seed_from_sibling_if_missing() -> None:
    if not SIBLING.exists():
        return
    pairs = [
        ("text", "text"),
        ("text_enterprise", "text_enterprise"),
        ("image", "image"),
        ("video", "video"),
    ]
    for folder, _ in pairs:
        dst_dir = ROOT / "checkpoints" / folder
        src_dir = SIBLING / "checkpoints" / folder
        dst = dst_dir / "latest.pt"
        src = src_dir / "latest.pt"
        if dst.exists() or not src.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        best_src = src_dir / "best.pt"
        if best_src.exists():
            shutil.copy2(best_src, dst_dir / "best.pt")
        _log(f"SEEDED {folder} from {src}")


def main() -> int:
    if not PYTHON.exists():
        _log("ERROR: venv python not found")
        return 1

    seed_from_sibling_if_missing()
    _log(f"FULL_MODAL_TRAIN_BEGIN package={PKG}")

    code = _run([str(PYTHON), str(ROOT / "scripts" / "run_real_ai_train.py")])
    if code != 0:
        _log(f"WARN real_ai_train exit={code}")

    code = _run(
        [
            str(PYTHON),
            "-c",
            (
                f"import sys; sys.path.insert(0, r'{ROOT}'); "
                f"from {PKG}.image.train import train; "
                "train(finetune=True, finetune_steps=400)"
            ),
        ]
    )
    if code != 0:
        _log(f"WARN image FT exit={code}")

    code = _run(
        [
            str(PYTHON),
            "-c",
            (
                f"import sys; sys.path.insert(0, r'{ROOT}'); "
                f"from {PKG}.video.train import train; "
                "train(finetune=True, finetune_steps=250)"
            ),
        ]
    )
    if code != 0:
        _log(f"WARN video FT exit={code}")

    try:
        sys.path.insert(0, str(ROOT))
        from navine.doctor import check_chat_smoke, check_code_smoke
        from navine.games.router import handle_game_message
        from navine.games.store import clear_active_game

        chat = check_chat_smoke()
        code_ok = check_code_smoke()
        clear_active_game("smoke")
        gr, gm = handle_game_message("play chess anything goes", "smoke")
        _log(f"DOCTOR chat={chat.passed} code={code_ok.passed} game_freer={gm.get('freer')}")
        (LOG_DIR / "smoke.json").write_text(
            json.dumps(
                {
                    "chat": chat.passed,
                    "code": code_ok.passed,
                    "game_freer": gm.get("freer"),
                    "game_preview": (gr or "")[:200],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        _log(f"SMOKE ERROR: {exc}")

    _log("FULL_MODAL_TRAIN_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
