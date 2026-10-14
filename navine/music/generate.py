from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from navine.utils.paths import get_output_dir

SAMPLE_RATE = 22050
DEFAULT_DURATION = 12.0
MIN_DURATION = 4.0
MAX_DURATION = 30.0

STYLE_HINTS = {
    "lofi": {"bpm": 78, "scale": "pentatonic", "drums": True, "warmth": 0.75, "swing": 0.14, "bass": True},
    "ambient": {"bpm": 56, "scale": "major", "drums": False, "warmth": 0.92, "swing": 0.0, "bass": False},
    "drums": {"bpm": 112, "scale": "minor", "drums": True, "warmth": 0.28, "swing": 0.04, "bass": True},
    "techno": {"bpm": 128, "scale": "minor", "drums": True, "warmth": 0.22, "swing": 0.0, "bass": True},
    "hiphop": {"bpm": 90, "scale": "minor", "drums": True, "warmth": 0.58, "swing": 0.1, "bass": True},
    "chill": {"bpm": 72, "scale": "major", "drums": False, "warmth": 0.82, "swing": 0.06, "bass": True},
    "piano": {"bpm": 84, "scale": "major", "drums": False, "warmth": 0.7, "swing": 0.02, "bass": False},
    "synth": {"bpm": 100, "scale": "minor", "drums": True, "warmth": 0.38, "swing": 0.0, "bass": True},
    "beat": {"bpm": 96, "scale": "minor", "drums": True, "warmth": 0.45, "swing": 0.06, "bass": True},
}

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
}


def _output_dir() -> Path:
    return get_output_dir("music")


def _seed_from_prompt(prompt: str, seed: Optional[int]) -> int:
    if seed is not None:
        return int(seed) & 0xFFFFFFFF
    digest = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _detect_styles(prompt: str) -> List[str]:
    lower = (prompt or "").lower()
    found = [name for name in STYLE_HINTS if re.search(rf"\b{re.escape(name)}\b", lower)]
    if not found:
        if any(w in lower for w in ("calm", "soft", "relax", "sleep", "rain")):
            found = ["ambient"]
        elif any(w in lower for w in ("party", "club", "dance", "rave")):
            found = ["techno"]
        elif any(w in lower for w in ("rap", "trap", "boom")):
            found = ["hiphop"]
        else:
            found = ["lofi"]
    return found


def _merge_style(styles: List[str]) -> Dict[str, Any]:
    base = dict(STYLE_HINTS[styles[0]])
    if len(styles) > 1:
        other = STYLE_HINTS[styles[1]]
        base["bpm"] = int(round((base["bpm"] + other["bpm"]) / 2))
        base["drums"] = bool(base["drums"] or other["drums"])
        base["bass"] = bool(base.get("bass") or other.get("bass"))
        base["warmth"] = (float(base["warmth"]) + float(other["warmth"])) / 2.0
        base["swing"] = (float(base["swing"]) + float(other["swing"])) / 2.0
    return base


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _env(n: int, attack: float, release: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    a = max(1, int(attack * sr))
    r = max(1, int(release * sr))
    env = np.ones(n, dtype=np.float32)
    if a < n:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    else:
        env[:] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return env
    if r < n:
        env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _tone(freq: float, n: int, kind: str = "sine", sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sr)
    phase = 2.0 * math.pi * float(freq) * t
    if kind == "square":
        wave = np.sign(np.sin(phase)).astype(np.float32) * 0.7
    elif kind == "saw":
        wave = (2.0 * (t * freq - np.floor(0.5 + t * freq))).astype(np.float32) * 0.65
    elif kind == "triangle":
        wave = (2.0 * np.abs(2.0 * (t * freq - np.floor(t * freq + 0.5))) - 1.0).astype(np.float32)
    else:
        wave = np.sin(phase).astype(np.float32)
    return wave


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _lowpass(signal: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    if signal.size == 0:
        return signal
    out = np.empty_like(signal)
    prev = float(signal[0])
    a = float(max(0.01, min(0.95, alpha)))
    for i, sample in enumerate(signal):
        prev = prev + a * (float(sample) - prev)
        out[i] = prev
    return out


def _place(buf: np.ndarray, start: int, signal: np.ndarray, gain: float = 1.0) -> None:
    if start >= len(buf) or len(signal) == 0:
        return
    end = min(len(buf), start + len(signal))
    chunk = signal[: end - start] * float(gain)
    buf[start:end] += chunk


def _compose(prompt: str, duration: float, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    styles = _detect_styles(prompt)
    style = _merge_style(styles)
    rng = np.random.default_rng(seed)
    sr = SAMPLE_RATE
    n = int(duration * sr)
    mix = np.zeros(n, dtype=np.float32)

    bpm = float(style["bpm"])
    beat = 60.0 / bpm
    swing = float(style["swing"])
    scale = SCALES.get(str(style["scale"]), SCALES["minor"])
    root = 45 + int(rng.integers(0, 10))
    warmth = float(style["warmth"])
    use_bass = bool(style.get("bass", True))
    hard = "techno" in styles or "synth" in styles or "drums" in styles

    chord_degrees = [0, 3, 4, 0] if style["scale"] != "pentatonic" else [0, 2, 4, 0]
    if "hiphop" in styles:
        chord_degrees = [0, 0, 5, 3]
    bar_len = int(4 * beat * sr)
    bars = max(1, int(math.ceil(n / max(1, bar_len))))

    for bar in range(bars):
        start = bar * bar_len
        if start >= n:
            break
        deg = chord_degrees[bar % len(chord_degrees)]
        length = min(bar_len, n - start)
        chord = [
            root + scale[deg % len(scale)],
            root + scale[(deg + 2) % len(scale)] + 12,
        ]
        if warmth > 0.45:
            chord.append(root + scale[(deg + 4) % len(scale)] + 7)
        for midi in chord:
            kind = "sine" if warmth > 0.55 else "triangle"
            wave = _tone(_midi_to_hz(midi), length, kind)
            wave *= _env(length, 0.03, 0.28)
            _place(mix, start, wave, 0.16 if warmth > 0.5 else 0.12)

        if use_bass:
            bass_midi = root + scale[deg % len(scale)] - 12
            bass = _tone(_midi_to_hz(bass_midi), length, "sine" if not hard else "triangle")
            bass = _lowpass(bass, 0.12 if hard else 0.08)
            bass *= _env(length, 0.01, 0.22)
            _place(mix, start, bass, 0.34 if hard else 0.28)

    step = beat * 0.5
    steps = int(duration / step) + 1
    melody_pattern = [0, 2, 4, 2, 3, 4, 0, 2]
    for i in range(steps):
        t0 = i * step
        if (i % 2) == 1:
            t0 += swing * beat * 0.28
        start = int(t0 * sr)
        if start >= n:
            break
        if hard and (i % 4 == 3) and rng.random() < 0.35:
            continue
        degree = scale[melody_pattern[i % len(melody_pattern)] % len(scale)]
        if rng.random() < 0.18:
            degree = scale[int(rng.integers(0, len(scale)))]
        octave = 12 * (2 if warmth > 0.6 else 1)
        midi = root + degree + octave
        length = int(min(0.42, step * (0.85 if not hard else 0.55)) * sr)
        length = min(length, n - start)
        kind = "saw" if hard else ("triangle" if "piano" in styles else "sine")
        wave = _tone(_midi_to_hz(midi), length, kind)
        if hard:
            wave = _lowpass(wave, 0.35)
        wave *= _env(length, 0.008, 0.14)
        _place(mix, start, wave, 0.11 if hard else 0.13)

    if style["drums"]:
        for i in range(int(duration / beat) + 1):
            kick_at = int(i * beat * sr)
            if kick_at < n:
                kn = min(int(0.14 * sr), n - kick_at)
                t = np.arange(kn, dtype=np.float32) / sr
                kick = np.sin(2 * math.pi * (95 * np.exp(-20 * t) + 42) * t).astype(np.float32)
                kick *= _env(kn, 0.001, 0.09)
                _place(mix, kick_at, kick, 0.62 if hard else 0.5)
            snare_at = int((i * beat + beat * 0.5) * sr)
            if snare_at < n and (i % 2 == 1 or "drums" in styles or "hiphop" in styles):
                sn = min(int(0.1 * sr), n - snare_at)
                snare = _noise(sn, rng) * _env(sn, 0.001, 0.07)
                snare = _lowpass(snare, 0.55)
                _place(mix, snare_at, snare, 0.26)
            for sub in (0.25, 0.75):
                hat_at = int((i * beat + beat * sub) * sr)
                if hat_at >= n:
                    continue
                hn = min(int(0.035 * sr), n - hat_at)
                hat = _noise(hn, rng) * _env(hn, 0.0004, 0.025)
                _place(mix, hat_at, hat, 0.07 if sub == 0.25 else 0.05)

    pad_len = min(n, int(3.2 * sr))
    for midi in (root + 19, root + 24, root + 28):
        pad = _tone(_midi_to_hz(midi), pad_len, "sine")
        pad *= _env(pad_len, 0.55, 1.1) * (0.06 + 0.07 * warmth)
        _place(mix, 0, pad, 1.0)

    if "ambient" in styles or "chill" in styles:
        wash = _lowpass(_noise(n, rng), 0.02) * 0.03
        wash *= np.linspace(0.4, 1.0, n, dtype=np.float32)
        mix += wash

    peak = float(np.max(np.abs(mix))) if mix.size else 1.0
    if peak < 1e-6:
        mix[:] = 0.0
    else:
        mix = mix / peak
        mix = np.tanh(mix * 1.35).astype(np.float32) * 0.9
    fade = min(int(0.2 * sr), n // 6)
    if fade > 1:
        mix[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        mix[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    meta = {
        "styles": styles,
        "bpm": bpm,
        "scale": style["scale"],
        "root_midi": root,
        "drums": bool(style["drums"]),
        "bass": use_bass,
        "sample_rate": sr,
        "duration": float(duration),
        "seed": seed,
        "prompt": prompt,
    }
    return mix.astype(np.float32), meta


def generate_music(
    prompt: str,
    duration: float = DEFAULT_DURATION,
    seed: Optional[int] = None,
    out_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    text = (prompt or "").strip()
    if not text:
        return {"ok": False, "error": "Prompt is required"}
    dur = float(duration or DEFAULT_DURATION)
    dur = max(MIN_DURATION, min(MAX_DURATION, dur))
    use_seed = _seed_from_prompt(text, seed)
    audio, meta = _compose(text, dur, use_seed)

    if out_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.lower())[:40].strip("_") or "track"
        out_path = _output_dir() / f"{stamp}_{safe}_{use_seed}.wav"
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf

        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
    except Exception:
        import wave

        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())

    return {
        "ok": True,
        "path": str(path.resolve()),
        "mime": "audio/wav",
        "bytes": path.stat().st_size if path.exists() else 0,
        "meta": meta,
    }
