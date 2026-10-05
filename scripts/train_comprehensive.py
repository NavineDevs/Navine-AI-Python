from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _brand() -> Dict[str, Any]:
    path = ROOT / "configs" / "brand.yaml"
    data: Dict[str, Any] = {
        "display_name": "Navine AI - Python",
        "package": "navine",
        "default_text_model": "text_enterprise",
    }
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass
    return data


def _log_dir() -> Path:
    path = ROOT / "logs" / "comprehensive_train"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with (_log_dir() / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    try:
        from navine.api.train_progress_helper import report_train_progress

        stage = "comprehensive"
        match = re.search(r"\btrain\s+([A-Za-z0-9_\-]+)", msg, re.I)
        if match:
            stage = match.group(1)
        elif "learn" in msg.lower():
            stage = "learn"
        report_train_progress(
            stage=stage,
            steps=0,
            steps_total=0,
            label=msg[:120],
            last_line=line,
            force=True,
        )
    except Exception:
        pass


def _py() -> str:
    venv = ROOT / "venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def _run(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    _log("RUN " + " ".join(cmd))
    try:
        completed = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": completed.returncode, "ok": completed.returncode == 0}
    except subprocess.TimeoutExpired as exc:
        return {"cmd": cmd, "returncode": -1, "ok": False, "error": f"timeout:{exc}"}
    except Exception as exc:
        return {"cmd": cmd, "returncode": -1, "ok": False, "error": str(exc)}


def _cli(package: str, *args: str) -> List[str]:
    return [_py(), "-m", f"{package}.cli", *args]


def _safe_train_types(package: str) -> List[str]:
    types = [
        "general",
        "chat",
        "coding",
        "creative",
        "math",
        "thinking",
        "detective",
        "multimodal-text",
        "games",
        "nsfw",
        "unrestricted",
        "hitboyx23",
        "hitboyx23_python",
        "voice",
    ]
    listed = _run(_cli(package, "train", "list"))
    if not listed.get("ok"):
        return [t for t in types if t not in ("hitboyx23", "hitboyx23_python") or "hitboy" in package]
    return [t for t in types if t not in ("hitboyx23", "hitboyx23_python") or "hitboy" in package]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all data sources and modalities")
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--image-steps", type=int, default=None)
    parser.add_argument("--video-steps", type=int, default=None)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--skip-learn", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    brand = _brand()
    package = str(brand.get("package") or "navine")
    product = str(brand.get("display_name") or package)
    default_model = str(brand.get("default_text_model") or "text_enterprise")
    quick = args.mode == "quick"
    text_steps = args.steps if args.steps is not None else (200 if quick else 1800)
    image_steps = args.image_steps if args.image_steps is not None else (600 if quick else 6000)
    video_steps = args.video_steps if args.video_steps is not None else (500 if quick else 4500)

    started = datetime.now(timezone.utc).isoformat()
    deadline = None
    if args.hours:
        deadline = time.time() + float(args.hours) * 3600
    results: List[Dict[str, Any]] = []
    _log(f"{product} comprehensive train start mode={args.mode} text_steps={text_steps}")

    def timed_out() -> bool:
        return deadline is not None and time.time() >= deadline

    if not args.skip_learn:
        for cmd in (("learn", "ingest"), ("learn", "index")):
            if timed_out():
                break
            results.append(_run(_cli(package, *cmd)))

    for name in _safe_train_types(package):
        if timed_out():
            break
        extra = ["--steps", str(text_steps)]
        results.append(_run(_cli(package, "train", name, *extra)))

    if default_model == "text_code" and not timed_out():
        results.append(_run(_cli(package, "train", "text-code", "--steps", str(text_steps))))
    if default_model == "hitboyx23_ai" and not timed_out():
        results.append(_run(_cli(package, "train", "hitboyx23", "--steps", str(text_steps))))
    if default_model == "hitboyx23_ai_python" and not timed_out():
        results.append(_run(_cli(package, "train", "hitboyx23_python", "--steps", str(text_steps))))

    if not args.skip_image and not timed_out():
        results.append(
            _run(_cli(package, "train", "image", "--steps", str(image_steps), "--require-cuda"))
        )
    if not args.skip_video and not timed_out():
        results.append(
            _run(_cli(package, "train", "video", "--steps", str(video_steps), "--require-cuda"))
        )

    pack = _run([_py(), "-m", f"{package}.llm", "pack"])
    results.append(pack)
    verify = _run([_py(), "-m", f"{package}.llm", "verify"])
    results.append(verify)
    gguf = _run([_py(), "-m", f"{package}.gguf_export", "export"])
    results.append(gguf)

    smoke = ROOT / "scripts" / "run_iterative_smoke_eval.py"
    if smoke.exists():
        results.append(_run([_py(), str(smoke), "comprehensive"]))

    summary = {
        "product": product,
        "package": package,
        "mode": args.mode,
        "started": started,
        "ended": datetime.now(timezone.utc).isoformat(),
        "text_steps": text_steps,
        "image_steps": image_steps,
        "video_steps": video_steps,
        "ok": all(bool(r.get("ok")) for r in results if r.get("cmd")),
        "results": results,
    }
    out = _log_dir() / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log(f"Wrote {out}")
    try:
        from navine.api.train_progress_helper import report_train_progress

        report_train_progress(
            stage="idle",
            steps=0,
            steps_total=0,
            label="complete" if summary["ok"] else "finished with errors",
            running=False,
            percent=100.0,
            force=True,
        )
    except Exception:
        pass
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
