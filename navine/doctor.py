import importlib
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from navine.utils.config import load_config
from navine.utils.paths import get_checkpoint_dir, get_project_root


class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = "", fix_hint: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.fix_hint = fix_hint


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "PIL",
    "yaml",
    "tqdm",
    "einops",
    "fastapi",
    "uvicorn",
    "requests",
    "bs4",
    "cv2",
    "imageio",
]


def check_python_version() -> CheckResult:
    version = sys.version_info
    ok = version >= (3, 9)
    detail = f"{version.major}.{version.minor}.{version.micro}"
    hint = "Install Python 3.9 or newer from python.org" if not ok else ""
    return CheckResult("Python version", ok, detail, hint)


def check_imports() -> List[CheckResult]:
    results = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            results.append(CheckResult(f"import {module}", True))
        except ImportError as exc:
            results.append(
                CheckResult(
                    f"import {module}",
                    False,
                    str(exc),
                    "pip install -r requirements.txt",
                )
            )
    return results


def check_backends() -> List[CheckResult]:
    results: List[CheckResult] = []
    try:
        from navine.utils.hardware import cloud_provider, cuda_available, pick_tier

        tier = pick_tier()
        results.append(CheckResult("hardware tier", True, tier))
        results.append(CheckResult("cuda available", cuda_available(), "yes" if cuda_available() else "no"))
        results.append(CheckResult("cloud provider", True, cloud_provider() or "none"))
    except Exception as exc:
        results.append(CheckResult("hardware tier", False, str(exc)))

    try:
        from navine.utils.inference_policy import is_custom_only

        if is_custom_only():
            results.append(CheckResult("text backend", True, "navine-custom-only"))
        else:
            from navine.text.opensource import detect_backend

            backend = detect_backend() or "custom"
            results.append(CheckResult("text backend", True, backend))
    except Exception as exc:
        results.append(CheckResult("text backend", False, str(exc)))

    try:
        import importlib

        ok = importlib.util.find_spec("diffusers") is not None
        results.append(
            CheckResult(
                "diffusers installed",
                ok,
                "yes" if ok else "no",
                "pip install -r requirements-external.txt",
            )
        )
    except Exception:
        pass
    return results


def check_package_installed() -> CheckResult:
    try:
        import navine
        from navine.utils.brand import brand_package

        pkg = brand_package()
        root = Path(navine.__file__).resolve().parent.parent
        expected = get_project_root()
        ok = root == expected or (root / pkg).exists()
        detail = str(root)
        hint = "pip install -e ." if not ok else ""
        return CheckResult(f"{pkg} package", ok, detail, hint)
    except ImportError as exc:
        return CheckResult("navine package", False, str(exc), "pip install -e .")


def check_checkpoints() -> List[CheckResult]:
    from navine.utils.brand import load_brand

    results = []
    model_ids = list(load_brand().get("model_ids") or [
        "text_enterprise",
        "text_code",
        "hitboyx23_ai",
        "hitboyx23_ai_python",
        "image_enterprise",
        "video_enterprise",
    ])
    for model in model_ids:
        ckpt = get_checkpoint_dir(model) / "latest.pt"
        ok = ckpt.exists()
        detail = str(ckpt) if ok else "missing"
        hint = "python -m navine.cli llm pack" if not ok else ""
        results.append(CheckResult(f"{model} checkpoint", ok, detail, hint))
    return results


def check_configs() -> List[CheckResult]:
    results = []
    for name in ("text_enterprise", "image_enterprise", "video_enterprise", "api", "navine"):
        try:
            load_config(name)
            results.append(CheckResult(f"config {name}.yaml", True))
        except Exception as exc:
            results.append(
                CheckResult(
                    f"config {name}.yaml",
                    False,
                    str(exc),
                    f"Restore configs/{name}.yaml",
                )
            )
    return results


def check_data_files() -> List[CheckResult]:
    root = get_project_root()
    paths = [
        ("instruction chat", root / "data" / "text" / "instruction_chat.txt"),
        ("sample code", root / "data" / "text" / "sample_code.jsonl"),
        ("image samples dir", root / "data" / "image" / "samples"),
        ("video samples dir", root / "data" / "video" / "samples"),
    ]
    results = []
    for label, path in paths:
        ok = path.exists()
        hint = "python scripts/prepare_data.py" if not ok else ""
        results.append(CheckResult(label, ok, str(path), hint))
    return results


def check_port(host: str, port: int) -> CheckResult:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        in_use = sock.connect_ex((host, port)) == 0
    finally:
        sock.close()
    if in_use:
        return CheckResult(
            f"port {port}",
            True,
            f"{host}:{port} in use (server may already be running)",
            "",
        )
    return CheckResult(f"port {port}", True, f"{host}:{port} available", "")


def check_chat_smoke() -> CheckResult:
    try:
        from navine.text.chat import chat_with_meta

        hello, hello_meta = chat_with_meta("hello", use_rag=False, use_search=False)
        hello_ok = not hello_meta.get("searched") and len(hello.strip()) > 5
        factual, factual_meta = chat_with_meta(
            "who is president of the united states",
            use_rag=True,
            use_search=False,
        )
        factual_ok = len(factual.strip()) > 5
        ok = hello_ok and factual_ok
        detail = (
            f"hello searched={hello_meta.get('searched')} len={len(hello)} | "
            f"factual searched={factual_meta.get('searched')} len={len(factual)} "
            f"(web fallback after local attempt is ok when search.fallback_on_failure is true)"
        )
        hint = "Chat should answer locally first; enable search toggle or fallback for web answers" if not ok else ""
        return CheckResult("chat local-first test", ok, detail, hint)
    except Exception as exc:
        return CheckResult("chat smoke test", False, str(exc), "python -m navine.cli train text")


def check_code_smoke() -> CheckResult:
    try:
        from navine.code.generate import generate_code

        code, lang = generate_code("fibonacci function", "python")
        ok = "def fibonacci" in code or "def " in code
        detail = f"lang={lang} len={len(code)}"
        hint = "python -m navine.cli train text" if not ok else ""
        return CheckResult("code smoke test", ok, detail, hint)
    except Exception as exc:
        return CheckResult("code smoke test", False, str(exc), "python -m navine.cli train text")


def check_launcher_script() -> CheckResult:
    root = get_project_root()
    scripts = sorted((root / "scripts").glob("*-launcher.ps1"))
    script = scripts[0] if scripts else None
    if script is None or not script.exists():
        return CheckResult("launcher script", False, "missing", "Restore scripts/*-launcher.ps1")
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$null, [ref]$e); if ($e) {{ exit 1 }} else {{ exit 0 }}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ok = result.returncode == 0
        detail = "syntax ok" if ok else (result.stderr or result.stdout or "parse error")
        return CheckResult("launcher script", ok, detail, f"Fix {script.name} syntax")
    except Exception as exc:
        return CheckResult("launcher script", False, str(exc), "")


def run_doctor(fix: bool = False) -> Tuple[List[CheckResult], int]:
    results: List[CheckResult] = []
    try:
        import os
        if os.getenv("NAVINE_DOCTOR_FAST") == "1":
            results.append(check_python_version())
            results.append(check_package_installed())
            results.extend(check_imports())
            results.extend(check_backends())
            failed = sum(1 for r in results if not r.passed)
            return results, failed
    except Exception:
        pass
    results.append(check_python_version())
    results.append(check_package_installed())
    results.extend(check_imports())
    results.extend(check_backends())
    results.extend(check_configs())
    results.extend(check_data_files())
    results.extend(check_checkpoints())
    try:
        api_cfg = load_config("api")
        host = api_cfg.get("bind_host", "127.0.0.1")
        port = int(api_cfg.get("port", 8765))
        if host in ("0.0.0.0", ""):
            host = "127.0.0.1"
        results.append(check_port(host, port))
    except Exception:
        results.append(check_port("127.0.0.1", 8765))
    results.append(check_launcher_script())
    missing_ckpt = any(r.name.endswith("checkpoint") and not r.passed for r in results)
    if not missing_ckpt:
        try:
            results.append(check_chat_smoke())
            results.append(check_code_smoke())
        except Exception as exc:
            results.append(CheckResult("smoke tests", False, str(exc), "Run doctor again"))
    if fix:
        _attempt_fixes(results)
    failed = sum(1 for r in results if not r.passed)
    return results, failed


def _attempt_fixes(results: List[CheckResult]) -> None:
    root = get_project_root()
    import_failed = any(r.name.startswith("import ") and not r.passed for r in results)
    if import_failed:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt")],
            cwd=str(root),
            check=False,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(root)],
            cwd=str(root),
            check=False,
        )
    data_missing = any("instruction chat" in r.name or "sample code" in r.name for r in results if not r.passed)
    if data_missing:
        prepare = root / "scripts" / "prepare_data.py"
        if prepare.exists():
            subprocess.run([sys.executable, str(prepare)], cwd=str(root), check=False)
    for r in results:
        if r.name == "text checkpoint" and not r.passed:
            subprocess.run([sys.executable, "-m", "navine.cli", "train", "text"], cwd=str(root), check=False)


def print_report(results: List[CheckResult]) -> int:
    print("Navine AI - Python Doctor")
    print("=" * 50)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = f"[{status}] {r.name}"
        if r.detail:
            line += f" - {r.detail}"
        print(line)
        if not r.passed and r.fix_hint:
            print(f"       fix: {r.fix_hint}")
    print("=" * 50)
    passed = sum(1 for r in results if r.passed)
    print(f"{passed}/{len(results)} checks passed")
    failed = len(results) - passed
    return failed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Navine AI - Python health check")
    parser.add_argument("--fix", action="store_true", help="Attempt automatic fixes")
    args = parser.parse_args()
    results, failed = run_doctor(fix=args.fix)
    failed = print_report(results)
    sys.exit(1 if failed else 0)
