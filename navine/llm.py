from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from navine.utils.paths import get_checkpoint_dir, get_output_dir, get_project_root

FORMAT = "navine-llm-v1"

CATALOG: List[Dict[str, Any]] = [
    {
        "name": "text_enterprise",
        "modality": "multimodal",
        "modalities": ["text", "image", "video", "voice"],
        "role": "Main assist/chat brain",
        "config_name": "text_enterprise",
        "needs_tokenizer": True,
        "inference": {"max_new_tokens": 256, "temperature": 0.85, "modes": ["chat", "code", "think", "detective", "analyze"]},
    },
    {
        "name": "text_code",
        "modality": "multimodal",
        "modalities": ["text", "image", "video", "voice"],
        "role": "Coding specialist",
        "config_name": "text_code",
        "needs_tokenizer": True,
        "inference": {"max_new_tokens": 320, "temperature": 0.2, "mode": "code", "modes": ["chat", "code", "think", "detective", "analyze"]},
    },
    {
        "name": "hitboyx23_ai",
        "modality": "multimodal",
        "modalities": ["text", "image", "video", "voice"],
        "role": "HitBoyXx23 AI (all coding languages)",
        "config_name": "hitboyx23_ai",
        "needs_tokenizer": True,
        "inference": {"max_new_tokens": 320, "temperature": 0.5, "modes": ["chat", "code", "think", "detective", "analyze"]},
    },
    {
        "name": "hitboyx23_ai_python",
        "modality": "multimodal",
        "modalities": ["text", "image", "video", "voice"],
        "role": "HitBoyXx23 AI (Python only)",
        "config_name": "hitboyx23_ai_python",
        "needs_tokenizer": True,
        "inference": {"max_new_tokens": 320, "temperature": 0.45, "modes": ["chat", "code", "think", "detective", "analyze"]},
    },
    {
        "name": "image_enterprise",
        "modality": "multimodal",
        "modalities": ["text", "image", "video", "voice"],
        "role": "Custom image diffusion",
        "config_name": "image_enterprise_v2",
        "needs_tokenizer": False,
        "inference": {
            "num_steps": 200,
            "guidance_scale": 1.5,
            "seed": 0,
            "output_size": 512,
            "modes": ["default", "code", "think", "detective", "analyze"],
        },
    },
    {
        "name": "video_enterprise",
        "modality": "multimodal",
        "modalities": ["text", "image", "video", "voice"],
        "role": "Custom video model",
        "config_name": "video_enterprise",
        "needs_tokenizer": False,
        "inference": {"num_frames": 32, "fps": 24, "seed": 0, "modes": ["default", "code", "think", "detective", "analyze"]},
    },
]


def models_root() -> Path:
    path = get_project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def spec_by_name(name: str) -> Dict[str, Any]:
    key = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "text": "text_enterprise",
        "chat": "text_enterprise",
        "navine_text": "text_enterprise",
        "think": "text_enterprise",
        "thinking": "text_enterprise",
        "detective": "text_enterprise",
        "mystery": "text_enterprise",
        "code": "text_code",
        "coding": "text_code",
        "navine_code": "text_code",
        "hitboyx23": "hitboyx23_ai",
        "hitboyx23_ai": "hitboyx23_ai",
        "hitboy": "hitboyx23_ai",
        "hitboyx23_python": "hitboyx23_ai_python",
        "hitboyx23_ai_python": "hitboyx23_ai_python",
        "hitboy_python": "hitboyx23_ai_python",
        "image": "image_enterprise",
        "video": "video_enterprise",
    }
    key = aliases.get(key, key)
    for row in CATALOG:
        if row["name"] == key:
            return row
    raise KeyError(f"Unknown Navine AI - Python LLM: {name}")


def checkpoint_path(name: str) -> Path:
    return get_checkpoint_dir(spec_by_name(name)["name"]) / "latest.pt"


def _read_weight_config(pt_path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    cfg = payload.get("config") or {}
    extra = payload.get("extra") or {}
    params = None
    state = payload.get("model_state") or payload
    if isinstance(state, dict):
        try:
            params = int(sum(int(v.numel()) for v in state.values() if hasattr(v, "numel")))
        except Exception:
            params = None
    return {"architecture": dict(cfg), "extra": dict(extra) if isinstance(extra, dict) else {}, "parameters": params}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_runner(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "",
                "ROOT = Path(__file__).resolve().parents[1]",
                "if str(ROOT) not in sys.path:",
                "    sys.path.insert(0, str(ROOT))",
                "",
                "from navine.llm import main_for",
                "",
                "if __name__ == '__main__':",
                f"    raise SystemExit(main_for({name!r}, sys.argv[1:]))",
                "",
            ]
        ),
        encoding="utf-8",
    )


def pack_one(name: str) -> Dict[str, Any]:
    spec = spec_by_name(name)
    try:
        from navine.utils.config import load_config

        cfg = load_config(spec["config_name"])
        ns = str(cfg.get("checkpoint_namespace") or spec["name"])
        ckpt_dir = get_checkpoint_dir(ns)
        architecture = {**(cfg.get("model") or {}), **(cfg.get("diffusion") or {})}
    except Exception:
        cfg = {}
        ckpt_dir = get_checkpoint_dir(spec["name"])
        architecture = {}
    pt = ckpt_dir / "latest.pt"
    if not pt.exists():
        alt = get_checkpoint_dir(spec["name"]) / "latest.pt"
        if alt.exists():
            ckpt_dir = alt.parent
            pt = alt
        else:
            return {"name": spec["name"], "ok": False, "error": f"missing {pt}"}
    parameters = None
    existing = ckpt_dir / "config.json"
    if existing.exists():
        try:
            old = json.loads(existing.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                if not architecture:
                    architecture = dict(old.get("architecture") or {})
                parameters = old.get("parameters")
        except Exception:
            pass
    if parameters is None and "image" in spec["name"] and architecture.get("arch_version") == 3:
        parameters = 127314883
    tok_src = ckpt_dir / "tokenizer.json"
    if spec["needs_tokenizer"] and not tok_src.exists():
        fallback = get_checkpoint_dir("text_enterprise") / "tokenizer.json"
        if fallback.exists():
            shutil.copy2(fallback, tok_src)
    packed = {
        "format": FORMAT,
        "runnable": True,
        "name": spec["name"],
        "modality": spec["modality"],
        "role": spec["role"],
        "config_name": spec["config_name"],
        "weights": "latest.pt",
        "weights_path": str(pt),
        "tokenizer": "tokenizer.json" if spec["needs_tokenizer"] else None,
        "architecture": architecture,
        "parameters": parameters,
        "inference": spec["inference"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run": [
            f"python models/{spec['name']}.py \"your prompt\"",
            f"python -m navine.llm run {spec['name']} \"your prompt\"",
        ],
    }
    _write_json(ckpt_dir / "config.json", packed)
    dest = models_root() / spec["name"]
    dest.mkdir(parents=True, exist_ok=True)
    _write_json(dest / "config.json", packed)
    if spec["needs_tokenizer"] and tok_src.exists():
        shutil.copy2(tok_src, dest / "tokenizer.json")
    _write_runner(dest / "run.py", spec["name"])
    _write_runner(models_root() / f"{spec['name']}.py", spec["name"])
    return {
        "name": spec["name"],
        "ok": True,
        "checkpoint": str(pt),
        "llm_file": str(models_root() / f"{spec['name']}.py"),
        "config": str(dest / "config.json"),
        "parameters": parameters,
    }


def pack_all() -> Dict[str, Any]:
    rows = [pack_one(item["name"]) for item in CATALOG]
    summary = {
        "format": FORMAT,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": rows,
    }
    _write_json(models_root() / "index.json", summary)
    try:
        from navine.engine import get_engine

        summary["engine_manifest"] = str(get_engine().write_manifest())
    except Exception as exc:
        summary["engine_manifest_error"] = str(exc)
    try:
        from navine.gguf_export import export_all_gguf

        summary["gguf"] = export_all_gguf()
    except Exception as exc:
        summary["gguf_error"] = str(exc)
    return summary


def list_llms() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in CATALOG:
        pt = get_checkpoint_dir(item["name"]) / "latest.pt"
        cfg = models_root() / item["name"] / "config.json"
        runner = models_root() / f"{item['name']}.py"
        gguf = models_root() / "gguf" / f"{item['name']}.gguf"
        rows.append(
            {
                "id": item["name"],
                "object": "model",
                "owned_by": "navine-ai",
                "modality": item["modality"],
                "role": item["role"],
                "runnable": bool(pt.exists() and runner.exists()),
                "weights": str(pt) if pt.exists() else None,
                "llm_file": str(runner) if runner.exists() else None,
                "gguf": str(gguf) if gguf.exists() else None,
                "config": str(cfg) if cfg.exists() else None,
                "bytes": int(pt.stat().st_size) if pt.exists() else 0,
            }
        )
    return rows


def run_text(name: str, prompt: str, **kwargs: Any) -> str:
    spec = spec_by_name(name)
    infer = dict(spec["inference"])
    infer.update({k: v for k, v in kwargs.items() if v is not None})
    from navine.engine import get_engine

    return get_engine().generate_text(
        prompt,
        name=spec["name"],
        max_new_tokens=int(infer.get("max_new_tokens") or 256),
        temperature=float(infer.get("temperature") or 0.7),
    )


def run_image(prompt: str, output: Optional[str] = None, **kwargs: Any) -> Path:
    infer = dict(spec_by_name("image_enterprise")["inference"])
    infer.update({k: v for k, v in kwargs.items() if v is not None})
    from navine.engine import get_engine

    return get_engine().generate_image(
        prompt,
        output=output,
        seed=int(infer.get("seed") or 0),
        num_steps=infer.get("num_steps"),
        guidance_scale=infer.get("guidance_scale"),
    )


def run_video(prompt: str, output: Optional[str] = None, **kwargs: Any) -> Path:
    infer = dict(spec_by_name("video_enterprise")["inference"])
    infer.update({k: v for k, v in kwargs.items() if v is not None})
    from navine.engine import get_engine

    return get_engine().generate_video(
        prompt,
        output=output,
        seed=int(infer.get("seed") or 0),
        num_frames=infer.get("num_frames"),
        fps=infer.get("fps"),
    )


def run_model(name: str, prompt: str, output: Optional[str] = None, modality: Optional[str] = None, **kwargs: Any) -> Any:
    spec = spec_by_name(name)
    target = modality or "text"
    if target == "image":
        return run_image(prompt, output=output, **kwargs)
    if target == "video":
        return run_video(prompt, output=output, **kwargs)
    if target == "voice":
        from navine.engine import get_engine
        return get_engine().generate_voice(prompt, name=spec["name"], **kwargs)
    return run_text(spec["name"], prompt, **kwargs)


def _persist_meta(name: str, architecture: Dict[str, Any], parameters: Optional[int]) -> None:
    spec = spec_by_name(name)
    paths = [
        get_checkpoint_dir(spec["name"]) / "config.json",
        models_root() / spec["name"] / "config.json",
    ]
    for path in paths:
        data: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}
        if architecture:
            data["architecture"] = architecture
        if parameters is not None:
            data["parameters"] = parameters
        data["runnable"] = True
        data["format"] = FORMAT
        data["name"] = spec["name"]
        data["modality"] = spec["modality"]
        _write_json(path, data)


def verify_one(name: str) -> Dict[str, Any]:
    spec = spec_by_name(name)
    pt = checkpoint_path(spec["name"])
    if not pt.exists():
        return {"name": spec["name"], "ok": False, "error": "missing weights"}
    import torch

    payload = torch.load(str(pt), map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    if spec["modality"] == "text":
        from navine.text.model import NavineTextModel
        from navine.text.tokenizer import NavineTokenizer

        model, _ = NavineTextModel.load_checkpoint(pt, "cpu")
        tok_path = pt.parent / "tokenizer.json"
        tokenizer = NavineTokenizer.load(tok_path) if tok_path.exists() else None
        params = model.count_parameters()
        del model
        result = {
            "name": spec["name"],
            "ok": True,
            "modality": "text",
            "parameters": params,
            "architecture": cfg,
            "tokenizer": bool(tokenizer),
            "llm_file": str(models_root() / f"{spec['name']}.py"),
        }
        _persist_meta(spec["name"], cfg, params)
        return result
    if spec["modality"] == "image":
        from navine.image.model import NavineDiffusionModel
        from navine.utils.config import load_config

        config = load_config(spec["config_name"])
        model = NavineDiffusionModel.load_checkpoint(pt, config, "cpu")
        ok = bool(getattr(model, "checkpoint_compatible", True))
        params = model.count_parameters()
        del model
        result = {
            "name": spec["name"],
            "ok": ok,
            "modality": "image",
            "parameters": params,
            "architecture": cfg,
            "llm_file": str(models_root() / f"{spec['name']}.py"),
        }
        _persist_meta(spec["name"], cfg, params)
        return result
    from navine.utils.config import load_config
    from navine.video.model import NavineVideoModel

    config = load_config(spec["config_name"])
    model = NavineVideoModel.load_checkpoint(pt, config, "cpu")
    ok = bool(getattr(model, "checkpoint_compatible", True))
    params = model.count_parameters()
    del model
    result = {
        "name": spec["name"],
        "ok": ok,
        "modality": "video",
        "parameters": params,
        "architecture": cfg,
        "llm_file": str(models_root() / f"{spec['name']}.py"),
    }
    _persist_meta(spec["name"], cfg, params)
    return result


def verify_all() -> Dict[str, Any]:
    rows = [verify_one(item["name"]) for item in CATALOG]
    return {"ok": all(bool(r.get("ok")) for r in rows), "models": rows}


def main_for(name: str, argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"models/{name}.py")
    parser.add_argument("prompt", nargs="+", help="Prompt for this Navine AI - Python LLM")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_intermixed_args(argv)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("Prompt required", file=sys.stderr)
        return 2
    result = run_model(
        name,
        prompt,
        output=args.output,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(result)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pack and run Navine AI - Python LLM files")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pack")
    sub.add_parser("list")
    sub.add_parser("verify")
    sub.add_parser("gguf")
    run_p = sub.add_parser("run")
    run_p.add_argument("name")
    run_p.add_argument("prompt", nargs="+")
    run_p.add_argument("--output", default=None)
    run_p.add_argument("--max-tokens", type=int, default=None)
    run_p.add_argument("--temperature", type=float, default=None)
    run_p.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "pack":
        print(json.dumps(pack_all(), indent=2))
        return 0
    if args.cmd == "list":
        print(json.dumps(list_llms(), indent=2))
        return 0
    if args.cmd == "gguf":
        from navine.gguf_export import main as gguf_main

        return gguf_main(["export"])
    if args.cmd == "verify":
        report = verify_all()
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("ok") else 1
    return main_for(args.name, args.prompt + (
        (["--output", args.output] if args.output else [])
        + (["--max-tokens", str(args.max_tokens)] if args.max_tokens is not None else [])
        + (["--temperature", str(args.temperature)] if args.temperature is not None else [])
        + (["--seed", str(args.seed)] if args.seed is not None else [])
    ))


if __name__ == "__main__":
    raise SystemExit(main())
