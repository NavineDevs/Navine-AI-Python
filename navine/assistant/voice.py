import logging
import os
import re
import shutil
import time
import urllib.request
import warnings
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from navine.utils.config import load_config
from navine.utils.hardware import cuda_available
from navine.utils.paths import get_project_root

_CONFIG_NAME = "voice"
_WHISPER_MODELS: Dict[str, Any] = {}
_PIPER_VOICES: Dict[str, Any] = {}
_HF_HUB_PREPARED = False
_DEFAULT_PIPER_VOICE = "en_US-ryan-high"
_PIPER_VOICE_FILES = {
    "en_US-lessac-medium": {
        "repo": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium",
        "onnx": "en_US-lessac-medium.onnx",
        "json": "en_US-lessac-medium.onnx.json",
    },
    "en_US-ryan-high": {
        "repo": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high",
        "onnx": "en_US-ryan-high.onnx",
        "json": "en_US-ryan-high.onnx.json",
    },
}


def load_voice_config() -> Dict[str, Any]:
    try:
        return load_config(_CONFIG_NAME)
    except FileNotFoundError:
        return {"enabled": True, "tts": {}, "stt": {}, "voices": {}, "paths": {}}


def _config_path() -> Path:
    return get_project_root() / "configs" / f"{_CONFIG_NAME}.yaml"


def save_voice_config(config: Dict[str, Any]) -> None:
    with _config_path().open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)


def voices_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or load_voice_config()
    rel = str((cfg.get("paths") or {}).get("voices_dir") or "voices")
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or load_voice_config()
    rel = str((cfg.get("paths") or {}).get("output_dir") or "outputs/voice")
    path = get_project_root() / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_builtin_voices(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_voice_config()
    voices = cfg.setdefault("voices", {})
    changed = False
    if "navine_ai" not in voices:
        voices["navine_ai"] = {
            "display_name": "Navine AI - Python",
            "backend": "piper",
            "language": "en",
            "piper_voice": _DEFAULT_PIPER_VOICE,
        }
        changed = True
    tts = cfg.setdefault("tts", {})
    if not tts.get("default_voice"):
        tts["default_voice"] = "navine_ai" if "navine_ai" in voices else next(iter(voices), "navine_ai")
        changed = True
    if "backend" not in tts:
        tts["backend"] = "auto"
        changed = True
    cfg["enabled"] = bool(cfg.get("enabled", True))
    if changed:
        try:
            save_voice_config(cfg)
        except Exception:
            pass
    return cfg


def list_voices(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfg = ensure_builtin_voices(config)
    voices = cfg.get("voices") or {}
    result = []
    theme = "navine"
    try:
        from navine.utils.brand import brand_theme_id, brand_package, is_foreign_brand_text

        theme = str(brand_theme_id() or brand_package() or "navine").lower()
    except Exception:
        theme = "navine"
        is_foreign_brand_text = lambda value, theme_id=None: (
            "hitboy" in str(value or "").lower() or "navuryx" in str(value or "").lower()
        )
    for name, meta in voices.items():
        key = str(name).lower()
        display = str((meta or {}).get("display_name") or "").lower()
        if is_foreign_brand_text(f"{key} {display}", theme):
            continue
        entry = {"name": name}
        entry.update(meta or {})
        ref = (meta or {}).get("reference")
        if ref:
            entry["reference_exists"] = (get_project_root() / ref).exists()
        result.append(entry)
    root = voices_dir(cfg)
    for folder in sorted(root.iterdir()) if root.exists() else []:
        if not folder.is_dir() or folder.name in ("piper",):
            continue
        name = folder.name
        key = name.lower()
        if is_foreign_brand_text(key, theme):
            continue
        if any(v.get("name") == name for v in result):
            continue
        refs = list(folder.glob("reference.*"))
        if not refs:
            continue
        rel = refs[0].relative_to(get_project_root()).as_posix()
        result.append(
            {
                "name": name,
                "backend": "auto",
                "language": "en",
                "reference": rel,
                "reference_exists": True,
                "cloned": True,
            }
        )
    return result


def voice_status(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = ensure_builtin_voices(config)
    if _piper_available():
        try:
            _ensure_piper_model(_DEFAULT_PIPER_VOICE)
        except Exception:
            pass
    mic = pick_input_device((cfg.get("stt") or {}).get("input_device", "auto"))
    mic_name = None
    try:
        import sounddevice as sd

        if mic is not None:
            mic_name = str(sd.query_devices(mic).get("name"))
    except Exception:
        mic_name = str(mic)
    voices = list_voices(cfg)
    default_voice = (cfg.get("tts") or {}).get("default_voice")
    try:
        from navine.utils.brand import is_foreign_brand_text

        if default_voice and is_foreign_brand_text(default_voice):
            default_voice = next(
                (v.get("name") for v in voices if "navine" in str(v.get("name") or "").lower()),
                (voices[0].get("name") if voices else "navine_ai"),
            )
    except Exception:
        pass
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "backends": {
            "piper": _piper_available(),
            "coqui": _coqui_available(),
            "pyttsx3": _pyttsx3_available(),
        },
        "default_voice": default_voice,
        "voices": voices,
        "piper_models": {
            key: (voices_dir(cfg) / "piper" / key / meta["onnx"]).exists()
            for key, meta in _PIPER_VOICE_FILES.items()
        },
        "input_device_index": mic,
        "input_device_name": mic_name,
        "input_devices": list_input_devices()[:20],
    }

def add_voice(
    name: str,
    sample_path: str,
    language: str = "en",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config or load_voice_config()
    source = Path(sample_path).expanduser()
    if not source.exists() or not source.is_file():
        return {"ok": False, "error": f"Sample not found: {source}"}

    voice_folder = voices_dir(cfg) / name
    voice_folder.mkdir(parents=True, exist_ok=True)
    target = voice_folder / ("reference" + source.suffix.lower())
    shutil.copyfile(source, target)

    rel = target.relative_to(get_project_root()).as_posix()
    backend = "pyttsx3"
    if _coqui_available():
        backend = "coqui"
    elif _piper_available():
        backend = "piper"
    cfg.setdefault("voices", {})[name] = {
        "backend": backend,
        "language": language,
        "reference": rel,
        "cloned": True,
    }
    if backend == "piper":
        cfg["voices"][name]["piper_voice"] = _DEFAULT_PIPER_VOICE
    save_voice_config(cfg)
    return {"ok": True, "name": name, "reference": rel, "backend": backend, "cloned": True}


def _piper_available() -> bool:
    try:
        from piper.voice import PiperVoice

        return PiperVoice is not None
    except Exception:
        return False


def _coqui_available() -> bool:
    try:
        from TTS.api import TTS

        return TTS is not None
    except Exception:
        return False


def _pyttsx3_available() -> bool:
    try:
        import pyttsx3

        return pyttsx3 is not None
    except Exception:
        return False


def _ensure_piper_model(voice_key: str) -> Tuple[Optional[Path], Optional[str]]:
    spec = _PIPER_VOICE_FILES.get(voice_key)
    if not spec:
        return None, f"Unknown Piper voice: {voice_key}"
    folder = voices_dir() / "piper" / voice_key
    folder.mkdir(parents=True, exist_ok=True)
    onnx_path = folder / spec["onnx"]
    json_path = folder / spec["json"]
    _prepare_hf_hub()
    try:
        for filename in (spec["onnx"], spec["json"]):
            dest = folder / filename
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = f"{spec['repo']}/{filename}"
            urllib.request.urlretrieve(url, dest)
        if not onnx_path.exists() or onnx_path.stat().st_size <= 0:
            return None, f"Piper model missing: {onnx_path}"
        if not json_path.exists():
            return None, f"Piper config missing: {json_path}"
        return onnx_path, None
    except Exception as exc:
        return None, f"Piper model download failed: {exc}"


def _get_piper_voice(voice_key: str):
    if voice_key in _PIPER_VOICES:
        return _PIPER_VOICES[voice_key], None
    onnx_path, error = _ensure_piper_model(voice_key)
    if error:
        return None, error
    try:
        from piper.voice import PiperVoice

        voice = PiperVoice.load(str(onnx_path))
        _PIPER_VOICES[voice_key] = voice
        return voice, None
    except Exception as exc:
        return None, f"Piper load failed: {exc}"


def _normalize_wav(path: Path, target_peak: float = 0.92) -> None:
    try:
        import soundfile as sf
    except Exception:
        return
    if not path.exists() or path.stat().st_size <= 44:
        return
    try:
        data, sample_rate = sf.read(str(path), dtype="float32")
        if data.size == 0:
            return
        peak = float(np.max(np.abs(data)))
        if peak <= 0.0001:
            return
        gain = min(target_peak / peak, 4.0)
        boosted = np.clip(data * gain, -1.0, 1.0)
        sf.write(str(path), boosted, sample_rate)
    except Exception:
        return


def _play_wav(path: Path, volume: float = 1.0) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32")
        if data.size == 0:
            return False
        if volume != 1.0:
            data = np.clip(data * float(volume), -1.0, 1.0)
        sd.play(data, sample_rate)
        sd.wait()
        return True
    except Exception:
        pass
    try:
        import platform

        if platform.system() == "Windows":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return True
    except Exception:
        pass
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


def _speak_pyttsx3_live(text: str, voice_meta: Dict[str, Any], rate: int) -> Dict[str, Any]:
    try:
        import pyttsx3
    except Exception as exc:
        return {"ok": False, "error": f"pyttsx3 not installed ({exc}). Run: pip install pyttsx3"}
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        index = voice_meta.get("system_voice_index")
        if index is not None:
            system_voices = engine.getProperty("voices")
            if 0 <= int(index) < len(system_voices):
                engine.setProperty("voice", system_voices[int(index)].id)
        engine.say(text)
        engine.runAndWait()
        return {"ok": True, "backend": "pyttsx3", "played": True}
    except Exception as exc:
        return {"ok": False, "error": f"pyttsx3 live speak failed: {exc}"}


def remove_voice(name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_voice_config()
    voices = cfg.get("voices") or {}
    if name not in voices:
        return {"ok": False, "error": f"Voice not found: {name}"}
    del voices[name]
    save_voice_config(cfg)
    folder = voices_dir(cfg) / name
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True, "removed": name}


def _resolve_tts_backend(requested: str, voice_meta: Dict[str, Any]) -> str:
    ref = voice_meta.get("reference")
    has_ref = bool(ref and (get_project_root() / ref).exists())
    order: List[str] = []
    if requested and requested not in ("", "auto"):
        order.append(requested)
    if has_ref and _coqui_available():
        order.append("coqui")
    voice_backend = voice_meta.get("backend")
    if voice_backend and voice_backend not in ("", "auto"):
        order.append(str(voice_backend))
    for name in ("piper", "coqui", "pyttsx3"):
        if name not in order:
            order.append(name)
    for name in order:
        if name == "piper" and _piper_available():
            return "piper"
        if name == "coqui" and _coqui_available():
            return "coqui"
        if name == "pyttsx3" and _pyttsx3_available():
            return "pyttsx3"
    return "pyttsx3"


def _tts_backend_available(name: str) -> bool:
    if name == "piper":
        return _piper_available()
    if name == "coqui":
        return _coqui_available()
    if name in ("pyttsx3", "sapi", "auto"):
        return _pyttsx3_available()
    return False


def clone_voice(
    name: str,
    sample_path: str = "",
    language: str = "en",
    sample_base64: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import base64
    import re

    cfg = ensure_builtin_voices(config)
    safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", (name or "").strip())[:48]
    if not safe:
        return {"ok": False, "error": "Voice name is required"}
    path = (sample_path or "").strip()
    if sample_base64:
        raw = sample_base64
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw)
        except Exception as exc:
            return {"ok": False, "error": f"Invalid sample_base64: {exc}"}
        upload = voices_dir(cfg) / safe / "upload_sample.wav"
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(data)
        path = str(upload)
    if not path:
        seed = output_dir(cfg) / f"clone_seed_{safe}.wav"
        spoken = speak(
            f"Hello, this is the {safe.replace('_', ' ')} voice for Navine AI - Python.",
            voice="navine_ai",
            play=False,
            out_path=str(seed),
            config=cfg,
        )
        if not spoken.get("ok"):
            return {
                "ok": False,
                "error": spoken.get("error") or "Could not create a seed sample for cloning",
            }
        path = str(seed)
    result = add_voice(safe, path, language=language, config=cfg)
    if result.get("ok"):
        result["mode"] = "clone"
        result["message"] = (
            f"Voice '{safe}' registered. "
            + (
                "True cloning uses Coqui XTTS when installed. "
                if _coqui_available()
                else "Coqui not installed; speech uses Piper/system TTS and keeps your sample. "
            )
            + "Use Speak with this name."
        )
    return result


def _speak_coqui(text: str, voice_meta: Dict[str, Any], out_path: Path, language: str) -> Dict[str, Any]:
    try:
        from TTS.api import TTS
    except Exception as exc:
        return {"ok": False, "error": f"Coqui TTS not installed ({exc}). Run: pip install TTS"}
    reference = voice_meta.get("reference")
    ref_path = get_project_root() / reference if reference else None
    try:
        model_name = voice_meta.get("model") or "tts_models/multilingual/multi-dataset/xtts_v2"
        engine = TTS(model_name)
        if ref_path and ref_path.exists():
            engine.tts_to_file(text=text, speaker_wav=str(ref_path), language=language, file_path=str(out_path))
        else:
            engine.tts_to_file(text=text, language=language, file_path=str(out_path))
        return {"ok": True, "path": str(out_path), "backend": "coqui"}
    except Exception as exc:
        return {"ok": False, "error": f"Coqui synthesis failed: {exc}"}


def _speak_sapi_windows(text: str, out_path: Path, rate: int = 175) -> Dict[str, Any]:
    try:
        import win32com.client  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"win32com unavailable: {exc}"}
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(out_path), 3)
        speaker.AudioOutputStream = stream
        try:
            speaker.Rate = max(-10, min(10, int((rate - 175) / 15)))
        except Exception:
            pass
        speaker.Speak(str(text))
        stream.Close()
        if out_path.exists() and out_path.stat().st_size > 44:
            return {"ok": True, "path": str(out_path), "backend": "sapi"}
        return {"ok": False, "error": "SAPI wrote empty audio"}
    except Exception as exc:
        return {"ok": False, "error": f"SAPI synthesis failed: {exc}"}


def _speak_piper(text: str, voice_meta: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    voice_key = voice_meta.get("piper_voice") or _DEFAULT_PIPER_VOICE
    model = voice_meta.get("model")
    try:
        from piper.voice import PiperVoice
    except Exception as exc:
        return {"ok": False, "error": f"piper not installed ({exc}). Run: pip install piper-tts"}

    voice = None
    if model:
        model_path = get_project_root() / model
        if model_path.exists():
            try:
                voice = PiperVoice.load(str(model_path))
            except Exception:
                voice = None
    if voice is None:
        voice, error = _get_piper_voice(voice_key)
        if error:
            return {"ok": False, "error": error}
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wav_file:
            if hasattr(voice, "synthesize_wav"):
                voice.synthesize_wav(text, wav_file)
            else:
                sample_rate = int(getattr(getattr(voice, "config", None), "sample_rate", 22050) or 22050)
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                for chunk in voice.synthesize(text):
                    samples = getattr(chunk, "audio_int16_bytes", None)
                    if samples is None:
                        arr = getattr(chunk, "audio_int16", None)
                        if arr is not None:
                            wav_file.writeframes(bytes(arr))
                    else:
                        wav_file.writeframes(samples)
        if not out_path.exists() or out_path.stat().st_size <= 44:
            return {"ok": False, "error": "Piper wrote empty audio"}
        return {"ok": True, "path": str(out_path), "backend": "piper", "piper_voice": voice_key}
    except Exception as exc:
        return {"ok": False, "error": f"Piper synthesis failed: {exc}"}


def _speak_pyttsx3(text: str, voice_meta: Dict[str, Any], out_path: Path, rate: int) -> Dict[str, Any]:
    sapi = _speak_sapi_windows(text, out_path, rate=rate)
    if sapi.get("ok"):
        return sapi
    try:
        import pyttsx3
    except Exception as exc:
        return {"ok": False, "error": f"pyttsx3 not installed ({exc}). Run: pip install pyttsx3"}
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        index = voice_meta.get("system_voice_index")
        if index is not None:
            system_voices = engine.getProperty("voices")
            if 0 <= int(index) < len(system_voices):
                engine.setProperty("voice", system_voices[int(index)].id)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        time.sleep(0.2)
        if not out_path.exists() or out_path.stat().st_size <= 44:
            live = _speak_pyttsx3_live(text, voice_meta, rate)
            if live.get("ok"):
                return live
            return sapi if sapi.get("error") else {"ok": False, "error": "pyttsx3 wrote empty audio"}
        return {"ok": True, "path": str(out_path), "backend": "pyttsx3"}
    except Exception as exc:
        if sapi.get("ok"):
            return sapi
        return {"ok": False, "error": f"pyttsx3 synthesis failed: {exc}"}


def speak(
    text: str,
    voice: Optional[str] = None,
    play: Optional[bool] = None,
    out_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {"ok": False, "error": "Empty text"}
    text = str(text).strip()
    cfg = ensure_builtin_voices(config)
    if not bool(cfg.get("enabled", True)):
        return {"ok": False, "error": "Voice is disabled in configs/voice.yaml"}
    tts_cfg = cfg.get("tts") or {}
    voices = cfg.get("voices") or {}
    voice_name = (voice or "").strip() or tts_cfg.get("default_voice") or "navine_ai"
    voice_meta = dict(voices.get(voice_name) or {})
    if not voice_meta:
        voice_meta = {
            "backend": "auto",
            "language": tts_cfg.get("language") or "en",
            "piper_voice": _DEFAULT_PIPER_VOICE,
        }
    language = voice_meta.get("language") or tts_cfg.get("language") or "en"
    rate = int(tts_cfg.get("rate", 175))

    if out_path:
        target = Path(out_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = output_dir(cfg) / f"tts_{voice_name}_{stamp}.wav"

    preferred = _resolve_tts_backend(tts_cfg.get("backend", "auto"), voice_meta)
    order = []
    for name in (preferred, "piper", "coqui", "pyttsx3"):
        if name and name not in order and _tts_backend_available(name):
            order.append(name)
    if not order:
        order = ["pyttsx3"]

    errors: List[str] = []
    result: Dict[str, Any] = {"ok": False, "error": "No TTS backend available"}
    for backend in order:
        if backend == "coqui":
            result = _speak_coqui(text, voice_meta, target, language)
        elif backend == "piper":
            result = _speak_piper(text, voice_meta, target)
        else:
            result = _speak_pyttsx3(text, voice_meta, target, rate)
        if result.get("ok"):
            break
        if result.get("error"):
            errors.append(f"{backend}: {result.get('error')}")

    if result.get("ok"):
        result["voice"] = voice_name
        should_play = tts_cfg.get("autoplay", True) if play is None else play
        if should_play and not result.get("played"):
            volume = float(tts_cfg.get("volume", 1.15))
            wav_path = Path(result.get("path") or target)
            if wav_path.exists():
                _normalize_wav(wav_path)
                played = _play_wav(wav_path, volume=volume)
                result["played"] = played
                if not played:
                    live = _speak_pyttsx3_live(text, voice_meta, rate)
                    result["played"] = bool(live.get("ok"))
                    if not live.get("ok") and live.get("error"):
                        result["play_error"] = live.get("error")
        return result

    result["errors"] = errors
    result["error"] = " | ".join(errors) if errors else result.get("error")
    return result


def list_input_devices() -> List[Dict[str, Any]]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    rows = []
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels") or 0) <= 0:
                continue
            host_idx = int(dev.get("hostapi") or 0)
            host_name = ""
            try:
                host_name = str(hostapis[host_idx].get("name") or "")
            except Exception:
                host_name = str(host_idx)
            rows.append(
                {
                    "index": idx,
                    "name": str(dev.get("name") or f"device_{idx}"),
                    "channels": int(dev.get("max_input_channels") or 0),
                    "hostapi": host_idx,
                    "hostapi_name": host_name,
                    "default_samplerate": float(dev.get("default_samplerate") or 16000),
                }
            )
    except Exception:
        return []
    return rows


_BAD_INPUT_HINTS = (
    "cable output",
    "cable in",
    "vb-audio",
    "steam streaming",
    "stereo mix",
    "what u hear",
    "loopback",
    "sonar - stream",
    "nahimic mirroring",
    "boomvad",
    "pc speaker",
    "sound mapper",
    "primary sound capture",
)


def _score_input_device(dev: Dict[str, Any]) -> float:
    name = dev["name"].lower()
    host = str(dev.get("hostapi_name") or "").lower()
    s = 0.0
    if any(b in name for b in _BAD_INPUT_HINTS):
        s -= 100.0
    good = (
        "microphone array",
        "intel",
        "realtek",
        "mic input",
        "microphone",
        "headset",
        "array",
    )
    for i, g in enumerate(good):
        if g in name:
            s += 45.0 - i * 2.5
    if "wasapi" in host:
        s += 12.0
    elif "mme" in host:
        s += 4.0
    elif "directsound" in host or "windows directsound" in host:
        s += 2.0
    elif "wdm-ks" in host:
        s -= 5.0
    s += min(3.0, float(dev.get("channels") or 1) * 0.5)
    return s


def probe_input_device(device_idx: int, sample_rate: int = 16000) -> bool:
    return resolve_capture_rate(device_idx, preferred_rate=sample_rate) is not None


def resolve_capture_rate(device_idx: Optional[int], preferred_rate: int = 16000) -> Optional[int]:
    try:
        import sounddevice as sd
    except Exception:
        return None
    rates: List[int] = []
    for rate in (
        preferred_rate,
        48000,
        44100,
        32000,
        24000,
        22050,
        16000,
        8000,
    ):
        r = int(rate)
        if r not in rates:
            rates.append(r)
    try:
        if device_idx is not None:
            info = sd.query_devices(int(device_idx))
            default_rate = int(float(info.get("default_samplerate") or 0) or 0)
            if default_rate > 0 and default_rate not in rates:
                rates.insert(1, default_rate)
    except Exception:
        pass
    for rate in rates:
        try:
            kwargs = {
                "channels": 1,
                "samplerate": int(rate),
                "dtype": "float32",
                "blocksize": 1024,
            }
            if device_idx is not None:
                kwargs["device"] = int(device_idx)
            with sd.InputStream(**kwargs):
                pass
            return int(rate)
        except Exception:
            continue
    return None


def resample_mono(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    src = int(src_rate)
    dst = int(dst_rate)
    if data.size == 0 or src == dst or src <= 0 or dst <= 0:
        return data
    duration = float(data.size) / float(src)
    n_out = max(1, int(round(duration * dst)))
    if n_out == data.size:
        return data
    x_old = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


def pick_input_device(preferred: Optional[Any] = None, sample_rate: int = 16000) -> Optional[int]:
    devices = list_input_devices()
    if not devices:
        return None

    preferred_candidates: List[int] = []
    if preferred is not None and str(preferred).strip() not in ("", "auto", "default"):
        pref = str(preferred).strip()
        if pref.isdigit():
            preferred_candidates.append(int(pref))
        else:
            lower = pref.lower()
            for d in devices:
                if lower in d["name"].lower():
                    preferred_candidates.append(int(d["index"]))

    ranked = sorted(devices, key=_score_input_device, reverse=True)
    ordered: List[int] = []
    for idx in preferred_candidates + [int(d["index"]) for d in ranked]:
        if idx not in ordered:
            ordered.append(idx)

    try:
        import sounddevice as sd

        default_in = sd.default.device
        if isinstance(default_in, (list, tuple)):
            default_in = default_in[0]
        if default_in is not None and int(default_in) >= 0:
            di = int(default_in)
            if di not in ordered:
                ordered.append(di)
            else:
                ordered.remove(di)
                ordered.insert(min(3, len(ordered)), di)
    except Exception:
        pass

    for idx in ordered:
        dev = next((d for d in devices if int(d["index"]) == int(idx)), None)
        if dev is not None and _score_input_device(dev) < -50 and idx not in preferred_candidates:
            continue
        if probe_input_device(idx, sample_rate=sample_rate):
            return int(idx)
    for idx in ordered:
        if probe_input_device(idx, sample_rate=sample_rate):
            return int(idx)
    return ordered[0] if ordered else None


def record_audio(
    seconds: int,
    sample_rate: int,
    out_path: Path,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Recording needs sounddevice+soundfile ({exc}). Run: pip install sounddevice soundfile",
        }

    devices = list_input_devices()
    device_idx = pick_input_device(device, sample_rate=sample_rate)
    candidates: List[Optional[int]] = []
    if device_idx is not None:
        candidates.append(device_idx)
    for d in sorted(devices, key=_score_input_device, reverse=True):
        idx = int(d["index"])
        if idx not in candidates:
            candidates.append(idx)
    if None not in candidates:
        candidates.append(None)

    last_error = None
    for try_idx in candidates:
        capture_rate = resolve_capture_rate(try_idx, preferred_rate=sample_rate)
        if capture_rate is None:
            last_error = RuntimeError(f"No supported sample rate for device {try_idx}")
            continue
        try:
            frames = int(max(1, seconds) * capture_rate)
            rec_kwargs = {
                "frames": frames,
                "samplerate": capture_rate,
                "channels": 1,
                "dtype": "float32",
            }
            if try_idx is not None:
                rec_kwargs["device"] = try_idx
            data = sd.rec(**rec_kwargs)
            sd.wait()
            data = np.asarray(data, dtype=np.float32).reshape(-1)
            data = resample_mono(data, capture_rate, sample_rate)
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if 0.008 <= peak < 0.25 and data.size:
                data = np.clip(data * (0.45 / max(peak, 1e-6)), -1.0, 1.0)
                peak = float(np.max(np.abs(data))) if data.size else 0.0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), data, sample_rate)
            device_name = None
            try:
                if try_idx is not None:
                    device_name = str(sd.query_devices(try_idx).get("name"))
                else:
                    device_name = "system-default"
            except Exception:
                device_name = str(try_idx)
            silent = peak < 0.008
            return {
                "ok": True,
                "path": str(out_path),
                "peak": peak,
                "silent": silent,
                "device_index": try_idx,
                "device_name": device_name,
                "capture_rate": capture_rate,
                "sample_rate": sample_rate,
            }
        except Exception as exc:
            last_error = exc
            continue
    return {"ok": False, "error": f"Recording failed: {last_error}"}


def play_chime(config: Optional[Dict[str, Any]] = None, kind: str = "listen") -> bool:
    cfg = config or load_voice_config()
    stt_cfg = cfg.get("stt") or {}
    if not bool(stt_cfg.get("chime", True)):
        return False
    try:
        import sounddevice as sd
    except Exception:
        return False
    try:
        sample_rate = 22050
        duration = 0.08 if kind == "listen" else 0.12
        freq = 880.0 if kind == "listen" else 660.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = 0.18 * np.sin(2 * np.pi * freq * t) * np.linspace(1.0, 0.05, t.size)
        sd.play(wave.astype(np.float32), sample_rate)
        sd.wait()
        return True
    except Exception:
        return False


def post_speak_cooldown(config: Optional[Dict[str, Any]] = None, seconds: Optional[float] = None) -> None:
    cfg = config or load_voice_config()
    stt_cfg = cfg.get("stt") or {}
    wait = float(seconds if seconds is not None else stt_cfg.get("post_speak_cooldown", 0.65))
    if wait > 0:
        time.sleep(wait)


def record_audio_vad(
    max_seconds: float = 8.0,
    sample_rate: int = 16000,
    out_path: Optional[Path] = None,
    device: Optional[Any] = None,
    silence_ms: int = 700,
    speech_threshold: float = 0.018,
    min_speech_ms: int = 250,
    start_timeout: float = 4.0,
) -> Dict[str, Any]:
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Recording needs sounddevice+soundfile ({exc}). Run: pip install sounddevice soundfile",
        }

    device_idx = pick_input_device(device, sample_rate=sample_rate)
    capture_rate = resolve_capture_rate(device_idx, preferred_rate=sample_rate)
    if capture_rate is None and device_idx is not None:
        capture_rate = resolve_capture_rate(None, preferred_rate=sample_rate)
        device_idx = None
    if capture_rate is None:
        return {
            "ok": False,
            "error": (
                "VAD recording failed: no microphone sample rate works. "
                "Set stt.input_device in configs/voice.yaml to your real mic, or disable exclusive mode."
            ),
        }
    block = max(256, int(capture_rate * 0.05))
    silence_blocks = max(1, int(silence_ms / 50))
    min_speech_blocks = max(1, int(min_speech_ms / 50))
    max_blocks = max(1, int(max_seconds * capture_rate / block))
    start_blocks = max(1, int(start_timeout * capture_rate / block))
    chunks: List[np.ndarray] = []
    speaking = False
    speech_blocks = 0
    silence_seen = 0
    device_name = None
    try:
        if device_idx is not None:
            device_name = str(sd.query_devices(device_idx).get("name"))
        else:
            device_name = "system-default"
    except Exception:
        device_name = str(device_idx)

    try:
        stream_kwargs = {
            "samplerate": capture_rate,
            "channels": 1,
            "dtype": "float32",
            "blocksize": block,
        }
        if device_idx is not None:
            stream_kwargs["device"] = device_idx
        with sd.InputStream(**stream_kwargs) as stream:
            for i in range(max_blocks + start_blocks):
                data, _ = stream.read(block)
                frame = np.asarray(data, dtype=np.float32).reshape(-1)
                peak = float(np.max(np.abs(frame))) if frame.size else 0.0
                if not speaking:
                    if peak >= speech_threshold:
                        speaking = True
                        speech_blocks = 1
                        chunks.append(frame.copy())
                    elif i >= start_blocks:
                        break
                else:
                    chunks.append(frame.copy())
                    if peak >= speech_threshold * 0.6:
                        speech_blocks += 1
                        silence_seen = 0
                    else:
                        silence_seen += 1
                        if speech_blocks >= min_speech_blocks and silence_seen >= silence_blocks:
                            break
                    if len(chunks) >= max_blocks:
                        break
    except Exception as exc:
        return {"ok": False, "error": f"VAD recording failed: {exc}"}

    if not chunks:
        return {
            "ok": True,
            "silent": True,
            "peak": 0.0,
            "path": None,
            "device_index": device_idx,
            "device_name": device_name,
            "error": "No speech detected before timeout.",
        }

    audio = np.concatenate(chunks).astype(np.float32)
    audio = resample_mono(audio, capture_rate, sample_rate)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if 0.008 <= peak < 0.25 and audio.size:
        audio = np.clip(audio * (0.45 / max(peak, 1e-6)), -1.0, 1.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = output_dir() / f"listen_vad_{stamp}.wav"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(out_path), audio, sample_rate)
    except Exception as exc:
        return {"ok": False, "error": f"Failed to write audio: {exc}"}
    silent = peak < 0.008 or speech_blocks < min_speech_blocks
    return {
        "ok": True,
        "path": str(out_path),
        "peak": peak,
        "silent": silent,
        "device_index": device_idx,
        "device_name": device_name,
        "speech_blocks": speech_blocks,
        "capture_rate": capture_rate,
        "sample_rate": sample_rate,
    }


def _prepare_hf_hub() -> None:
    global _HF_HUB_PREPARED
    if _HF_HUB_PREPARED:
        return
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
    _HF_HUB_PREPARED = True


def _resolve_stt_device(compute_type: str) -> tuple[str, str]:
    if cuda_available():
        return "cuda", "float16"
    return "cpu", compute_type or "int8"


def _get_faster_whisper_model(model_size: str, device: str, compute_type: str):
    key = f"{model_size}|{device}|{compute_type}"
    if key in _WHISPER_MODELS:
        return _WHISPER_MODELS[key]
    _prepare_hf_hub()
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _WHISPER_MODELS[key] = model
    return model


def _transcribe_faster_whisper(wav_path: Path, model_size: str, language: str, compute_type: str) -> Dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        return {"ok": False, "error": f"faster-whisper not installed ({exc}). Run: pip install faster-whisper"}
    try:
        device, resolved_compute = _resolve_stt_device(compute_type)
        model = _get_faster_whisper_model(model_size, device, resolved_compute)
        segments, info = model.transcribe(
            str(wav_path),
            language=language or None,
            vad_filter=True,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "ok": True,
            "text": text,
            "backend": "faster_whisper",
            "device": device,
            "language": getattr(info, "language", language),
        }
    except Exception as exc:
        return {"ok": False, "error": f"faster-whisper failed: {exc}"}


def _transcribe_whisper(wav_path: Path, model_size: str, language: str) -> Dict[str, Any]:
    try:
        import whisper
    except Exception as exc:
        return {"ok": False, "error": f"whisper not installed ({exc}). Run: pip install openai-whisper"}
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(str(wav_path), language=language)
        return {"ok": True, "text": str(result.get("text", "")).strip(), "backend": "whisper"}
    except Exception as exc:
        return {"ok": False, "error": f"whisper failed: {exc}"}


def _resolve_stt_backend(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    for module, backend in (("faster_whisper", "faster_whisper"), ("whisper", "whisper")):
        try:
            __import__(module)
            return backend
        except Exception:
            continue
    return "faster_whisper"


def transcribe_file(wav_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_voice_config()
    stt_cfg = cfg.get("stt") or {}
    path = Path(wav_path).expanduser()
    if not path.exists():
        return {"ok": False, "error": f"Audio file not found: {path}"}
    model_size = stt_cfg.get("model_size", "base")
    language = stt_cfg.get("language", "en")
    compute_type = stt_cfg.get("compute_type", "int8")
    backend = _resolve_stt_backend(stt_cfg.get("backend", "auto"))
    if backend == "whisper":
        return _transcribe_whisper(path, model_size, language)
    result = _transcribe_faster_whisper(path, model_size, language, compute_type)
    if not result.get("ok"):
        result = _transcribe_whisper(path, model_size, language)
    return result


def listen(
    seconds: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    vad: Optional[bool] = None,
    chime: Optional[bool] = None,
    prompt: bool = True,
) -> Dict[str, Any]:
    cfg = ensure_builtin_voices(config)
    stt_cfg = cfg.get("stt") or {}
    record_seconds = int(seconds if seconds is not None else stt_cfg.get("record_seconds", 6))
    sample_rate = int(stt_cfg.get("sample_rate", 16000))
    input_device = stt_cfg.get("input_device", "auto")
    use_vad = bool(stt_cfg.get("use_vad", True) if vad is None else vad)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    wav_path = output_dir(cfg) / f"listen_{stamp}.wav"
    if prompt:
        print(f"Listening ({'VAD' if use_vad else f'{record_seconds}s'})... speak now.")
    if chime is None:
        chime = bool(stt_cfg.get("chime", True))
    if chime:
        play_chime(cfg, kind="listen")
    if use_vad:
        recorded = record_audio_vad(
            max_seconds=float(stt_cfg.get("vad_max_seconds", max(record_seconds, 6))),
            sample_rate=sample_rate,
            out_path=wav_path,
            device=input_device,
            silence_ms=int(stt_cfg.get("vad_silence_ms", 700)),
            speech_threshold=float(stt_cfg.get("vad_threshold", 0.018)),
            min_speech_ms=int(stt_cfg.get("vad_min_speech_ms", 250)),
            start_timeout=float(stt_cfg.get("vad_start_timeout", 4.0)),
        )
    else:
        recorded = record_audio(record_seconds, sample_rate, wav_path, device=input_device)
    if not recorded.get("ok"):
        return recorded
    if recorded.get("device_name") and prompt:
        print(f"Mic: {recorded.get('device_name')}")
    if recorded.get("silent") or not recorded.get("path"):
        return {
            "ok": True,
            "text": "",
            "silent": True,
            "peak": recorded.get("peak"),
            "device_name": recorded.get("device_name"),
            "path": recorded.get("path"),
            "error": recorded.get("error")
            or (
                "Microphone was too quiet or wrong input device "
                f"({recorded.get('device_name')}). "
                "Set stt.input_device in configs/voice.yaml to your real mic name or index. "
                f"Candidates: {', '.join(d['name'] for d in list_input_devices()[:8])}"
            ),
        }
    result = transcribe_file(str(recorded["path"]), cfg)
    result["path"] = recorded.get("path")
    result["peak"] = recorded.get("peak")
    result["device_name"] = recorded.get("device_name")
    result["silent"] = False
    return result


def listen_idle(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = ensure_builtin_voices(config)
    stt_cfg = cfg.get("stt") or {}
    idle_seconds = int(stt_cfg.get("idle_record_seconds", 4))
    use_vad = bool(stt_cfg.get("idle_use_vad", True))
    return listen(
        seconds=idle_seconds,
        config=cfg,
        vad=use_vad,
        chime=False,
        prompt=False,
    )


_WAKE_ALIASES = (
    "navine",
    "navene",
    "naveen",
    "navin",
    "navy",
    "vine",
    "nine",
    "novine",
    "no vine",
    "hey vine",
    "hey nine",
    "hey navy",
    "a navine",
    "hey there navine",
)


def _normalize_spoken(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("’", "'")
    t = re.sub(r"[,.!?]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def expand_wake_phrases(wake_words: List[str]) -> List[str]:
    out: List[str] = []
    for item in list(wake_words or []) + list(_WAKE_ALIASES):
        n = _normalize_spoken(item)
        if n and n not in out:
            out.append(n)
    return out


def strip_wake_words(text: str, wake_words: List[str]) -> str:
    cleaned = (text or "").strip()
    lower = _normalize_spoken(cleaned)
    ordered = sorted(expand_wake_phrases(wake_words), key=len, reverse=True)
    for w in ordered:
        if not w:
            continue
        if lower == w:
            return ""
        if lower.startswith(w + " "):
            cleaned = re.sub(
                r"(?i)^" + re.escape(w).replace(r"\ ", r"[\s,.-]+") + r"[\s,.-]*",
                "",
                cleaned,
                count=1,
            ).strip(" ,.-:")
            lower = _normalize_spoken(cleaned)
            continue
        if w in lower:
            cleaned = re.sub(
                r"(?i)" + r"[\s,.-]+".join(re.escape(p) for p in w.split()) + r"[\s,.-]*",
                " ",
                cleaned,
                count=1,
            )
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-:")
            lower = _normalize_spoken(cleaned)
    return cleaned.strip()


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def matches_any_phrase(text: str, phrases: List[str]) -> bool:
    lower = _normalize_spoken(text)
    if not lower:
        return False
    expanded = expand_wake_phrases(list(phrases or []))
    for phrase in expanded:
        p = _normalize_spoken(phrase)
        if not p:
            continue
        if p == lower or p in lower:
            return True
    tokens = lower.split()
    brand_targets = {"navine", "navene", "naveen", "navin", "vine", "navy", "nine"}
    for tok in tokens:
        for target in brand_targets:
            if tok == target:
                return True
            if len(tok) >= 3 and len(target) >= 3 and _edit_distance(tok, target) <= 1:
                return True
    if tokens and tokens[0] in ("hey", "okay", "ok", "hi", "yo") and len(tokens) >= 2:
        for target in ("navine", "navene", "vine", "navy", "nine"):
            if _edit_distance(tokens[1], target) <= 1:
                return True
    return False


def looks_like_direct_command(text: str) -> bool:
    lower = _normalize_spoken(text)
    if not lower or len(lower) < 3:
        return False
    starters = (
        "open ",
        "launch ",
        "start ",
        "play ",
        "close ",
        "create ",
        "make ",
        "write ",
        "read ",
        "delete ",
        "train ",
        "observe ",
        "screenshot",
        "what time",
        "what day",
        "search ",
        "go to ",
        "browse ",
    )
    if any(lower.startswith(s) for s in starters):
        return True
    bare_apps = {
        "spotify",
        "chrome",
        "discord",
        "steam",
        "notepad",
        "explorer",
        "edge",
        "firefox",
        "code",
        "vscode",
        "calculator",
        "youtube",
        "fortnite",
    }
    if lower in bare_apps:
        return True
    if "open" in lower and any(app in lower for app in bare_apps):
        return True
    try:
        from navine.assistant.agent import parse_intent

        intent = parse_intent(text)
        action = str(intent.get("action") or "unknown")
        return action not in ("unknown", "")
    except Exception:
        return False


def normalize_direct_command(text: str) -> str:
    lower = _normalize_spoken(text)
    bare_apps = {
        "spotify",
        "chrome",
        "discord",
        "steam",
        "notepad",
        "explorer",
        "edge",
        "firefox",
        "code",
        "vscode",
        "calculator",
        "youtube",
    }
    if lower in bare_apps:
        return f"open {lower}"
    return (text or "").strip()
