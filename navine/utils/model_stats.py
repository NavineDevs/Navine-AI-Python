from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from navine.utils.brand import load_brand
from navine.utils.paths import get_checkpoint_dir, get_project_root


def format_parameters(count: Optional[int]) -> Optional[str]:
    if count is None:
        return None
    value = int(count)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def format_parameters_exact(count: Optional[int]) -> Optional[str]:
    if count is None:
        return None
    return f"{int(count):,}"


def _model_catalog() -> Dict[str, Dict[str, Any]]:
    try:
        from navine.llm import CATALOG

        return {str(row["name"]): dict(row) for row in CATALOG}
    except Exception:
        return {}


def resolve_model_checkpoint(name: str) -> Path:
    if name == "voice":
        neural_ckpt = get_checkpoint_dir("voice") / "latest.pt"
        if neural_ckpt.exists():
            return neural_ckpt
        voice_report = get_checkpoint_dir("voice") / "train_report.json"
        if voice_report.exists():
            return voice_report
        return neural_ckpt

    if name == "deepfake":
        return get_checkpoint_dir("deepfake") / "latest.pt"

    if name in (
        "text_enterprise",
        "text_code",
        "hitboyx23_ai",
        "hitboyx23_ai_python",
        "image_enterprise",
        "image_enterprise_v2",
        "video_enterprise",
    ):
        try:
            from navine.utils.tier import load_modality_config, resolve_checkpoint_dir

            if name in ("text_code", "hitboyx23_ai", "hitboyx23_ai_python", "image_enterprise_v2"):
                return get_checkpoint_dir(name) / "latest.pt"
            modality = "image" if name.startswith("image") else ("video" if name.startswith("video") else "text")
            cfg = load_modality_config(modality)
            ckpt = resolve_checkpoint_dir(modality, cfg) / "latest.pt"
            if ckpt.exists():
                return ckpt
        except Exception:
            pass
        return get_checkpoint_dir(name) / "latest.pt"

    try:
        from navine.utils.tier import load_modality_config, resolve_checkpoint_dir

        cfg = load_modality_config(name)
        ckpt = resolve_checkpoint_dir(name, cfg) / "latest.pt"
        if ckpt.exists():
            return ckpt
    except Exception:
        pass
    return get_checkpoint_dir(name) / "latest.pt"


def _load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.suffix.lower() != ".pt":
        return {}
    try:
        import torch

        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _count_tensors(state: Any) -> Optional[int]:
    if not isinstance(state, dict):
        return None
    total = 0
    seen: set = set()
    for value in state.values():
        if not hasattr(value, "numel"):
            continue
        try:
            ptr = None
            if hasattr(value, "data_ptr"):
                try:
                    ptr = int(value.data_ptr())
                except Exception:
                    ptr = None
            if ptr is not None:
                if ptr in seen:
                    continue
                seen.add(ptr)
            total += int(value.numel())
        except (TypeError, ValueError):
            continue
    return total if total > 0 else None


def _unique_text_parameters(state: Any, architecture: Dict[str, Any]) -> Optional[int]:
    stored = _count_tensors(state)
    if stored is None:
        return None
    estimated = _estimate_parameters(architecture)
    if estimated is not None:
        if abs(int(estimated) - int(stored)) <= max(1024, int(estimated) * 0.01):
            return int(stored)
        return int(estimated)
    return stored


def _live_checkpoint_stats(path: Path) -> Tuple[Optional[int], Dict[str, Any], str]:
    if not path.exists():
        return None, {}, "none"
    if path.suffix.lower() != ".pt":
        return None, {}, "none"

    sidecar = path.parent / "config.json"
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                architecture = dict(meta.get("architecture") or meta.get("config") or {})
                raw = meta.get("parameters")
                if raw is not None:
                    params = int(raw)
                    if params > 0:
                        return params, architecture, "live_checkpoint"
        except Exception:
            pass

    payload = _load_checkpoint_payload(path)
    architecture = dict(payload.get("config") or {})
    state = payload.get("model_state")
    if state is None and payload:
        state = {k: v for k, v in payload.items() if hasattr(v, "numel")}

    params = None
    meta_params = payload.get("parameters") if isinstance(payload, dict) else None
    if meta_params is not None:
        try:
            params = int(meta_params)
        except (TypeError, ValueError):
            params = None

    if params is None:
        params = _count_tensors(state)
        if architecture.get("vocab_size") and architecture.get("d_model"):
            unique = _unique_text_parameters(state, architecture)
            if unique is not None:
                params = unique
        elif isinstance(state, dict):
            buffer_keys = (
                "betas",
                "alphas_cumprod",
                "sqrt_alphas_cumprod",
                "sqrt_one_minus_alphas_cumprod",
            )
            buf = 0
            for key in buffer_keys:
                tensor = state.get(key)
                if hasattr(tensor, "numel"):
                    buf += int(tensor.numel())
            if params is not None and buf > 0:
                params = int(params) - buf

    if params is None:
        try:
            from navine.llm import _read_weight_config

            meta = _read_weight_config(path)
            architecture = dict(meta.get("architecture") or architecture)
            raw = meta.get("parameters")
            if raw is not None:
                params = int(raw)
        except Exception:
            pass

    if params is not None and params > 0:
        try:
            sidecar.write_text(
                json.dumps(
                    {
                        "parameters": int(params),
                        "architecture": architecture,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        return int(params), architecture, "live_checkpoint"
    return None, architecture, "none"


def _estimate_parameters(architecture: Dict[str, Any]) -> Optional[int]:
    if not architecture:
        return None
    try:
        from navine.text.model import NavineTextModel

        vocab_size = int(architecture.get("vocab_size") or 0)
        if vocab_size <= 0:
            return None
        model = NavineTextModel(
            vocab_size=vocab_size,
            d_model=int(architecture.get("d_model") or 256),
            n_heads=int(architecture.get("n_heads") or 4),
            n_layers=int(architecture.get("n_layers") or 4),
            d_ff=int(architecture.get("d_ff") or 1024),
            max_seq_len=int(architecture.get("max_seq_len") or 512),
            dropout=float(architecture.get("dropout") or 0.1),
            use_rope=bool(architecture.get("use_rope", False)),
            use_kv_cache=bool(architecture.get("use_kv_cache", False)),
            use_swiglu=bool(architecture.get("use_swiglu", False)),
            use_rms_norm=bool(architecture.get("use_rms_norm", False)),
            tie_embeddings=bool(architecture.get("tie_embeddings", False)),
        )
        return int(model.count_parameters())
    except Exception:
        return None


def _configured_architecture(name: str, catalog_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from navine.utils.config import load_config

        cfg_name = str((catalog_row or {}).get("config_name") or name)
        if name == "deepfake":
            cfg_name = "deepfake_enterprise"
        cfg = load_config(cfg_name)
        return {**(cfg.get("model") or {}), **(cfg.get("diffusion") or {})}
    except Exception:
        return {}


def _infer_modality(name: str, catalog_row: Optional[Dict[str, Any]] = None) -> str:
    if catalog_row and catalog_row.get("modality"):
        modality = str(catalog_row["modality"])
        if modality != "multimodal":
            return modality
    lower = name.lower()
    if lower.startswith("image"):
        return "image"
    if lower.startswith("video"):
        return "video"
    if lower == "voice":
        return "voice"
    if lower == "deepfake":
        return "deepfake"
    if lower == "music":
        return "music"
    return "text"


def _display_label(name: str, catalog_row: Optional[Dict[str, Any]] = None) -> str:
    if catalog_row and catalog_row.get("role"):
        return str(catalog_row["role"])
    labels = {
        "text_enterprise": "Chat text model",
        "text_code": "Code text model",
        "image_enterprise": "Image model",
        "image_enterprise_v2": "Image model v2",
        "video_enterprise": "Video model",
        "hitboyx23_ai": "HitBoy coding model",
        "hitboyx23_ai_python": "HitBoy Python model",
        "voice": "Voice TTS model",
        "deepfake": "Deepfake face model",
        "music": "Music generator",
    }
    return labels.get(name, name.replace("_", " "))


def _estimate_onnx_parameters(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    size = int(path.stat().st_size)
    if size <= 0:
        return None
    return max(1, int(size / 4))


def _resolve_piper_onnx_path(voice_key: str = "en_US-lessac-medium") -> Path:
    root = get_project_root()
    direct = root / "voices" / "piper" / voice_key / f"{voice_key}.onnx"
    if direct.exists():
        return direct
    fallback = root / "voices" / "piper" / "en_US-lessac-medium" / "en_US-lessac-medium.onnx"
    return fallback


def _load_voice_report(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _configured_param_count(name: str, architecture: Dict[str, Any], modality: str) -> Optional[int]:
    try:
        if modality == "text" or (architecture.get("d_model") and architecture.get("n_layers")):
            from navine.text.model import build_text_model

            vocab = int(architecture.get("vocab_size") or 12000)
            model = build_text_model(dict(architecture), vocab)
            return int(model.count_parameters())
        if modality == "deepfake" or architecture.get("latent_dim"):
            from navine.deepfake.model import NavineDeepfakeModel

            keys = (
                "image_size",
                "in_channels",
                "base_channels",
                "channel_mults",
                "num_res_blocks",
                "latent_dim",
                "arch_version",
            )
            kwargs = {k: architecture[k] for k in keys if k in architecture}
            return int(NavineDeepfakeModel(**kwargs).count_parameters())
        if modality == "image" or (
            architecture.get("base_channels") and architecture.get("time_emb_dim")
        ):
            from navine.image.model import NavineDiffusionModel

            keys = (
                "image_size",
                "in_channels",
                "base_channels",
                "channel_mults",
                "num_res_blocks",
                "time_emb_dim",
                "text_emb_dim",
                "attn_resolutions",
                "text_max_len",
                "text_layers",
                "text_heads",
                "min_snr_gamma",
                "arch_version",
                "timesteps",
                "beta_start",
                "beta_end",
                "beta_schedule",
            )
            kwargs = {k: architecture[k] for k in keys if k in architecture}
            return int(NavineDiffusionModel(**kwargs).count_parameters())
        if modality == "video" or architecture.get("hidden_dim") and architecture.get("frame_size"):
            from navine.video.model import NavineVideoModel

            keys = ("frame_size", "num_frames", "in_channels", "hidden_dim", "num_layers")
            kwargs = {k: architecture[k] for k in keys if k in architecture}
            return int(NavineVideoModel(**kwargs).count_parameters())
        if modality == "voice" or (architecture.get("hidden_dim") and architecture.get("mel_bins")):
            from navine.voice.neural import NavineVoiceModel

            keys = ("vocab_size", "hidden_dim", "num_layers", "mel_bins", "max_seq_len")
            kwargs = {k: architecture[k] for k in keys if k in architecture}
            return int(NavineVoiceModel(**kwargs).count_parameters())
    except Exception:
        return None
    return None


def _stat_voice_model() -> Dict[str, Any]:
    from navine.utils.training_lock import training_lock_active

    report_path = get_checkpoint_dir("voice") / "train_report.json"
    neural_path = get_checkpoint_dir("voice") / "latest.pt"
    report = _load_voice_report(report_path)
    architecture: Dict[str, Any] = {}
    parameters: Optional[int] = None
    parameter_source = "none"
    piper_voice = "en_US-ryan-high"
    default_voice = "navine_trained"
    backend = "piper"

    try:
        from navine.utils.config import load_config

        voice_cfg = load_config("voice")
        ent_cfg = load_config("voice_enterprise")
        model_cfg = dict(voice_cfg.get("model") or ent_cfg.get("model") or {})
        architecture = dict(model_cfg)
        tts = dict(voice_cfg.get("tts") or {})
        default_voice = str(tts.get("default_voice") or default_voice)
        backend = str(tts.get("backend") or backend)
        piper_voice = str(tts.get("piper_voice") or ent_cfg.get("piper_voice") or piper_voice)
        voices = dict(voice_cfg.get("voices") or {})
        voice_row = dict(voices.get(default_voice) or {})
        if voice_row.get("piper_voice"):
            piper_voice = str(voice_row.get("piper_voice"))
        architecture.update(
            {
                "backend": backend,
                "default_voice": default_voice,
                "piper_voice": piper_voice,
                "sample_rate": tts.get("sample_rate"),
                "language": tts.get("language"),
                "rate": tts.get("rate"),
            }
        )
        stt = dict(voice_cfg.get("stt") or {})
        if stt:
            architecture["stt_backend"] = stt.get("backend")
            architecture["stt_model_size"] = stt.get("model_size")
    except Exception:
        pass

    neural_params = None
    neural_arch: Dict[str, Any] = {}
    if neural_path.exists() and neural_path.suffix.lower() == ".pt":
        neural_params, neural_arch, neural_source = _live_checkpoint_stats(neural_path)
        if neural_arch:
            architecture = {**architecture, **neural_arch}
        if neural_params:
            parameters = int(neural_params)
            parameter_source = neural_source

    if parameters is None:
        configured = _configured_param_count("voice", architecture, "voice")
        if configured:
            parameters = int(configured)
            parameter_source = "configured_target"

    piper_path = _resolve_piper_onnx_path(piper_voice)
    if piper_path.exists():
        architecture["piper_voice_file"] = str(piper_path)

    samples = list(report.get("samples") or [])
    ok_samples = [
        row for row in samples
        if isinstance(row, dict) and (row.get("result") or {}).get("ok")
    ]
    registered = dict(report.get("registered_voice") or {})
    trained = bool(parameters is not None and parameter_source == "live_checkpoint")

    mtime = None
    age_seconds = None
    ckpt = neural_path if neural_path.exists() else report_path
    if ckpt.exists():
        ts = ckpt.stat().st_mtime
        mtime = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - ts)

    if ok_samples or registered.get("ok"):
        architecture["calibration_samples"] = len(ok_samples) or len(samples)
        architecture["voices_available"] = len(report.get("voices_available") or [])

    return {
        "name": "voice",
        "label": "Voice neural model",
        "modality": "voice",
        "category": "voice",
        "parameters": parameters,
        "parameters_human": format_parameters(parameters),
        "parameters_exact": format_parameters_exact(parameters),
        "parameter_source": parameter_source,
        "architecture": architecture,
        "architecture_source": "live_checkpoint" if parameter_source == "live_checkpoint" else (
            "configured" if architecture else "none"
        ),
        "trained": trained,
        "checkpoint": str(neural_path if neural_path.exists() else ckpt),
        "checkpoint_key": str(neural_path.resolve()).lower() if neural_path.exists() else f"configured:voice:{parameters}",
        "mtime": mtime,
        "age_seconds": age_seconds,
        "training": bool(training_lock_active("voice")),
        "config_name": "voice_enterprise",
    }


def stat_one_model(name: str) -> Dict[str, Any]:
    if name == "voice":
        return _stat_voice_model()

    from navine.utils.training_lock import training_lock_active

    catalog = _model_catalog()
    catalog_name = name
    if name not in catalog and name.endswith("_v2"):
        base = name[:-3]
        if base in catalog:
            catalog_name = base
    catalog_row = catalog.get(catalog_name)

    ckpt = resolve_model_checkpoint(name)
    parameters = None
    architecture: Dict[str, Any] = {}
    parameter_source = "none"

    if ckpt.exists() and ckpt.suffix.lower() == ".pt":
        parameters, architecture, parameter_source = _live_checkpoint_stats(ckpt)

    if not architecture:
        architecture = _configured_architecture(name, catalog_row)

    if parameters is None:
        modality_guess = _infer_modality(name, catalog_row)
        configured = _configured_param_count(name, architecture, modality_guess)
        if configured:
            parameters = int(configured)
            parameter_source = "configured_target"

    mtime = None
    age_seconds = None
    if ckpt.exists():
        ts = ckpt.stat().st_mtime
        mtime = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - ts)

    modality = _infer_modality(name, catalog_row)
    ckpt_key = (
        str(ckpt.resolve()).lower()
        if ckpt.exists()
        else f"configured:{name}:{parameters}"
    )

    return {
        "name": name,
        "label": _display_label(name, catalog_row),
        "modality": modality,
        "category": modality,
        "parameters": parameters,
        "parameters_human": format_parameters(parameters),
        "parameters_exact": format_parameters_exact(parameters),
        "parameter_source": parameter_source,
        "architecture": architecture,
        "architecture_source": "live_checkpoint" if parameter_source == "live_checkpoint" and architecture else (
            "configured" if architecture else "none"
        ),
        "trained": bool(parameters is not None and parameter_source == "live_checkpoint"),
        "checkpoint": str(ckpt),
        "checkpoint_key": ckpt_key,
        "mtime": mtime,
        "age_seconds": age_seconds,
        "training": bool(training_lock_active(name if name != "voice" else "voice")),
        "config_name": str((catalog_row or {}).get("config_name") or name),
    }


def _flatten_inference_base(infer: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(infer or {})
    base.pop("modes", None)
    return base


def _sanitize_profile_params(modality: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if modality == "text":
        keys = (
            "max_new_tokens",
            "min_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "frequency_penalty",
            "no_repeat_ngram_size",
            "num_candidates",
        )
    elif modality == "image":
        keys = ("num_steps", "guidance_scale", "output_size", "seed", "prompt_prefix", "scheduler")
    elif modality == "video":
        keys = ("num_frames", "fps", "frame_output_size", "output_format", "seed", "prompt_prefix", "hybrid")
    elif modality == "voice":
        keys = (
            "backend",
            "default_voice",
            "language",
            "rate",
            "volume",
            "sample_rate",
            "model_size",
            "record_seconds",
            "use_vad",
            "vad_max_seconds",
        )
    elif modality == "deepfake":
        keys = ("swap_strength", "blend", "face_detector", "output_size")
    elif modality == "music":
        keys = (
            "engine",
            "sample_rate",
            "duration_min",
            "duration_max",
            "duration_default",
            "styles",
            "output_format",
            "offline",
        )
    else:
        keys = tuple(params.keys())
    clean: Dict[str, Any] = {}
    for key in keys:
        if key in params and params[key] is not None and params[key] != "":
            value = params[key]
            if isinstance(value, (dict, list)):
                clean[key] = value
            else:
                clean[key] = value
    return clean


def _profile_rows(
    modality: str,
    config_name: str,
    infer: Dict[str, Any],
    mode_labels: Optional[Dict[str, str]] = None,
    allowed_modes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    labels = mode_labels or {}
    base = _flatten_inference_base(infer)
    rows: List[Dict[str, Any]] = []
    mode_map = dict((infer or {}).get("modes") or {})
    if base:
        rows.append(
            {
                "modality": modality,
                "mode": "default",
                "label": labels.get("default") or f"{modality.title()} · default",
                "config": config_name,
                "params": _sanitize_profile_params(modality, base),
            }
        )
    for mode_name, mode_params in sorted(mode_map.items(), key=lambda item: item[0]):
        if allowed_modes and mode_name not in allowed_modes:
            continue
        merged = {**base, **dict(mode_params or {})}
        rows.append(
            {
                "modality": modality,
                "mode": str(mode_name),
                "label": labels.get(str(mode_name)) or f"{modality.title()} · {mode_name}",
                "config": config_name,
                "params": _sanitize_profile_params(modality, merged),
            }
        )
    return rows


def _resolve_image_config_name() -> str:
    try:
        from navine.utils.tier import load_modality_config

        cfg = load_modality_config("image")
        return str(cfg.get("checkpoint_namespace") or "image_enterprise_v2")
    except Exception:
        return "image_enterprise_v2"


def collect_inference_profiles() -> List[Dict[str, Any]]:
    from navine.utils.config import load_config

    brand = load_brand()
    allowed_text_modes = set(brand.get("modes") or [])
    rows: List[Dict[str, Any]] = []

    text_labels = {
        "default": "Text · default",
        "chat": "Text · chat",
        "code": "Text · code",
        "think": "Text · think",
        "detective": "Text · detective",
        "analyze": "Text · analyze",
        "osint": "Text · OSINT",
    }

    try:
        text_cfg = load_config("text_enterprise")
        text_infer = dict(text_cfg.get("inference") or {})
        text_modes = dict(text_infer.get("modes") or {})
        filtered = {
            key: value
            for key, value in text_modes.items()
            if not allowed_text_modes or key in allowed_text_modes or key in ("chat", "think", "detective", "analyze", "osint")
        }
        text_infer["modes"] = filtered
        rows.extend(_profile_rows("text", "text_enterprise", text_infer, text_labels, None))
    except Exception:
        pass

    try:
        code_cfg = load_config("text_code")
        code_infer = dict(code_cfg.get("inference") or {})
        code_modes = dict(code_infer.get("modes") or {})
        if "code" not in code_modes and code_infer:
            code_modes["code"] = _flatten_inference_base(code_infer)
        code_infer["modes"] = code_modes
        rows.extend(
            _profile_rows(
                "text",
                "text_code",
                code_infer,
                {"code": "Text · code"},
                {"code"},
            )
        )
    except Exception:
        pass

    image_cfg_name = _resolve_image_config_name()
    try:
        image_cfg = load_config(image_cfg_name)
        image_infer = dict(image_cfg.get("inference") or {})
        image_infer.pop("modes", None)
        rows.extend(
            _profile_rows(
                "image",
                image_cfg_name,
                image_infer,
                {"default": "Image · default"},
            )
        )
    except Exception:
        pass

    try:
        video_cfg = load_config("video_enterprise")
        video_infer = dict(video_cfg.get("inference") or {})
        video_infer.pop("modes", None)
        rows.extend(
            _profile_rows(
                "video",
                "video_enterprise",
                video_infer,
                {"default": "Video · default"},
            )
        )
    except Exception:
        pass

    try:
        voice_cfg = load_config("voice")
        tts = dict(voice_cfg.get("tts") or {})
        stt = dict(voice_cfg.get("stt") or {})
        if tts:
            rows.append(
                {
                    "modality": "voice",
                    "mode": "tts",
                    "label": "Voice · TTS",
                    "config": "voice",
                    "params": tts,
                }
            )
        if stt:
            rows.append(
                {
                    "modality": "voice",
                    "mode": "stt",
                    "label": "Voice · STT",
                    "config": "voice",
                    "params": stt,
                }
            )
        deepfake = dict(voice_cfg.get("deepfake") or {})
        if deepfake:
            rows.append(
                {
                    "modality": "deepfake",
                    "mode": "default",
                    "label": "Deepfake · default",
                    "config": "voice",
                    "params": deepfake,
                }
            )
    except Exception:
        pass

    try:
        df_cfg = load_config("deepfake_enterprise")
        df_infer = dict(df_cfg.get("inference") or {})
        df_model = dict(df_cfg.get("model") or {})
        if df_infer or df_model:
            rows.append(
                {
                    "modality": "deepfake",
                    "mode": "neural",
                    "label": "Deepfake · neural",
                    "config": "deepfake_enterprise",
                    "params": {**df_model, **df_infer},
                }
            )
    except Exception:
        pass

    try:
        from navine.music.generate import (
            DEFAULT_DURATION,
            MAX_DURATION,
            MIN_DURATION,
            SAMPLE_RATE,
            STYLE_HINTS,
        )

        rows.append(
            {
                "modality": "music",
                "mode": "generate",
                "label": "Music · generate",
                "config": "navine.music",
                "params": {
                    "engine": "procedural",
                    "sample_rate": SAMPLE_RATE,
                    "duration_min": MIN_DURATION,
                    "duration_max": MAX_DURATION,
                    "duration_default": DEFAULT_DURATION,
                    "styles": sorted(STYLE_HINTS.keys()),
                    "output_format": "wav",
                    "offline": True,
                },
            }
        )
    except Exception:
        pass

    if not any(row.get("mode") == "osint" and row.get("modality") == "text" for row in rows):
        try:
            text_cfg = load_config("text_enterprise")
            osint_params = dict((text_cfg.get("inference") or {}).get("modes", {}).get("osint") or {})
            if osint_params:
                rows.append(
                    {
                        "modality": "text",
                        "mode": "osint",
                        "label": "Text · OSINT",
                        "config": "text_enterprise",
                        "params": osint_params,
                    }
                )
        except Exception:
            pass

    return rows


def music_generator_stats() -> Dict[str, Any]:
    from navine.music.generate import (
        DEFAULT_DURATION,
        MAX_DURATION,
        MIN_DURATION,
        SAMPLE_RATE,
        STYLE_HINTS,
        _output_dir,
    )

    out_dir = _output_dir()
    clips = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True) if out_dir.exists() else []
    latest = clips[0] if clips else None
    architecture = {
        "engine": "procedural",
        "sample_rate": SAMPLE_RATE,
        "duration_range_sec": f"{MIN_DURATION}-{MAX_DURATION}",
        "default_duration_sec": DEFAULT_DURATION,
        "styles": sorted(STYLE_HINTS.keys()),
        "output_format": "wav",
        "offline": True,
        "clips_generated": len(clips),
    }
    return {
        "name": "music",
        "label": "Music generator",
        "modality": "music",
        "category": "music",
        "parameters": None,
        "parameters_human": "procedural",
        "parameters_exact": "local composer",
        "parameter_source": "generator",
        "architecture": architecture,
        "architecture_source": "configured",
        "trained": True,
        "checkpoint": str(latest) if latest else str(out_dir),
        "checkpoint_key": f"generator:music:{SAMPLE_RATE}",
        "shares_checkpoint_with": None,
        "mtime": datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).isoformat() if latest else None,
        "age_seconds": (datetime.now(timezone.utc).timestamp() - latest.stat().st_mtime) if latest else None,
        "training": False,
        "config_name": "navine.music",
    }


def collect_model_stats() -> Dict[str, Any]:
    brand = load_brand()
    model_ids = list(brand.get("model_ids") or [])
    extras = [
        "text_enterprise",
        "text_code",
        "image_enterprise_v2",
        "video_enterprise",
        "voice",
        "deepfake",
    ]
    for name in extras:
        if name not in model_ids:
            model_ids.append(name)
    rows = [stat_one_model(name) for name in model_ids]
    try:
        rows.append(music_generator_stats())
    except Exception:
        pass

    seen_checkpoints: Dict[str, str] = {}
    modality_best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        params = row.get("parameters")
        key = str(row.get("checkpoint_key") or "")
        source = str(row.get("parameter_source") or "")
        modality = str(row.get("modality") or row.get("name") or "other")
        if params is None:
            continue
        if source not in ("live_checkpoint", "configured_target"):
            continue
        if key and key in seen_checkpoints:
            row["shares_checkpoint_with"] = seen_checkpoints[key]
            continue
        if key:
            seen_checkpoints[key] = str(row["name"])
        prev = modality_best.get(modality)
        if prev is None:
            modality_best[modality] = row
            continue
        prev_live = prev.get("parameter_source") == "live_checkpoint"
        cur_live = source == "live_checkpoint"
        if cur_live and not prev_live:
            modality_best[modality] = row
        elif cur_live == prev_live and int(params) >= int(prev.get("parameters") or 0):
            modality_best[modality] = row

    total = sum(int(row["parameters"]) for row in modality_best.values())
    live_total = sum(
        int(row["parameters"])
        for row in modality_best.values()
        if row.get("parameter_source") == "live_checkpoint"
    )

    trained_count = sum(1 for row in rows if row.get("trained"))
    source_label = "live_checkpoint" if live_total and live_total == total else (
        "mixed" if live_total else ("configured_target" if total else "none")
    )
    host_specs = None
    try:
        from navine.realtime.host_specs import collect_host_specs, format_host_specs_reply

        host_specs = {
            "specs": collect_host_specs(),
            "text": format_host_specs_reply(),
        }
    except Exception:
        host_specs = None
    result = {
        "product": str(brand.get("display_name") or "Navine AI - Python"),
        "port": int(brand.get("port") or 8765),
        "default_text_model": str(brand.get("default_text_model") or "text_enterprise"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_parameters": total or None,
        "total_parameters_human": format_parameters(total or None),
        "total_parameters_exact": format_parameters_exact(total or None),
        "parameter_source": source_label,
        "trained_models": trained_count,
        "models": rows,
        "inference_profiles": collect_inference_profiles(),
        "host_specs": host_specs,
        "capabilities": [
            "chat",
            "code",
            "image",
            "video",
            "voice",
            "deepfake",
            "music",
            "osint",
            "train",
        ],
    }
    try:
        cache_path = get_project_root() / "logs" / "model_stats_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return result


def load_cached_model_stats(max_age_seconds: int = 3600) -> Optional[Dict[str, Any]]:
    cache_path = get_project_root() / "logs" / "model_stats_cache.json"
    if not cache_path.exists():
        return None
    try:
        age = datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime
        if age > float(max_age_seconds):
            return None
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("models") else None
    except Exception:
        return None
