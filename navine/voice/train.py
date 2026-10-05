import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from navine.assistant.voice import (
    add_voice,
    list_voices,
    load_voice_config,
    output_dir,
    speak,
    save_voice_config,
    voices_dir,
)
from navine.utils.paths import get_project_root
from navine.utils.training_lock import training_lock

CALIBRATION_LINES = [
    "Navine AI - Python voice calibration sample one.",
    "Hello, this is Navine AI - Python speaking clearly.",
    "Testing text to speech for custom voice training.",
    "The time and date request should sound natural.",
]


def _report_path() -> Path:
    path = get_project_root() / "checkpoints" / "voice"
    path.mkdir(parents=True, exist_ok=True)
    return path / "train_report.json"


def _checkpoint_path() -> Path:
    path = get_project_root() / "checkpoints" / "voice"
    path.mkdir(parents=True, exist_ok=True)
    return path / "latest.pt"


def _train_neural_voice(texts: list[str], steps: Optional[int] = None) -> Dict[str, Any]:
    from navine.device_manager import get_device, print_train_device_banner, to_device
    from navine.utils.config import load_config
    from navine.voice.neural import build_voice_model

    cfg = load_config("voice_enterprise")
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("training") or {})
    max_steps = int(steps or train_cfg.get("max_steps") or 600)
    print_train_device_banner()
    device = get_device()
    model = build_voice_model(model_cfg)
    model = to_device(model, device)
    params = model.count_parameters()
    print(f"Navine AI - Python Voice Neural | Parameters: {params:,} | Device: {device}")
    from navine.utils.param_verify import assert_real_param_count

    assert_real_param_count(
        model,
        "voice",
        size_tier=str((config.get("model") or {}).get("size_tier") or config.get("size_tier") or "gpt3_1b"),
    )
    optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate") or 0.00012))
    lines = [str(line).strip() for line in texts if str(line).strip()]
    if not lines:
        lines = list(CALIBRATION_LINES)
    model.train()
    import torch

    for step in range(1, max_steps + 1):
        line = lines[(step - 1) % len(lines)]
        chars = [min(ord(c), 255) for c in line[: int(model_cfg.get("max_seq_len") or 512) - 1]] or [0]
        ids = torch.tensor([chars], dtype=torch.long, device=device)
        mel_len = max(32, min(160, len(chars) * 4))
        target = torch.randn(1, mel_len, int(model_cfg.get("mel_bins") or 80), device=device) * 0.05
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip") or 1.0))
        optimizer.step()
        if step == 1 or step % max(1, max_steps // 10) == 0 or step == max_steps:
            print(f"Voice neural step {step}/{max_steps} loss={float(loss.item()):.4f}")
    ckpt = _checkpoint_path()
    model.save_checkpoint(ckpt)
    return {"parameters": params, "checkpoint": str(ckpt), "steps": max_steps}


def train_voice(
    steps: Optional[int] = None,
    language: Optional[str] = None,
    voice_name: Optional[str] = None,
    sample_path: Optional[str] = None,
    set_default: bool = True,
) -> Dict[str, Any]:
    with training_lock("voice"):
        config = load_voice_config()
        if not bool(config.get("enabled", True)):
            raise RuntimeError("Voice is disabled in configs/voice.yaml")

        name = str(voice_name or "navine_trained").strip() or "navine_trained"
        lang = str(language or (config.get("tts") or {}).get("language") or "en")
        out = output_dir(config)
        samples = []
        errors = []
        registered = None

        sample_file = Path(str(sample_path or "").strip()) if sample_path else None
        if sample_file and sample_file.exists():
            print(f"Step 1/1 voice sample clone")
            registered = add_voice(name, str(sample_file), language=lang, config=config)
            samples.append({"text": "custom sample", "path": str(sample_file), "result": registered})
        else:
            count = max(1, min(int(steps or len(CALIBRATION_LINES)), 32))
            lines = CALIBRATION_LINES * ((count // len(CALIBRATION_LINES)) + 1)
            for idx, line in enumerate(lines[:count]):
                target = out / f"calib_{idx:02d}.wav"
                print(f"Step {idx + 1}/{count} voice calibration")
                try:
                    result = speak(
                        line,
                        voice=str((config.get("tts") or {}).get("default_voice") or "navine_ai"),
                        out_path=str(target),
                        play=False,
                        config=config,
                    )
                    samples.append({"text": line, "path": str(target), "result": result})
                except Exception as exc:
                    errors.append({"text": line, "error": str(exc)})

            if samples:
                first_path = samples[0]["path"]
                if Path(first_path).exists():
                    registered = add_voice(name, first_path, language=lang, config=config)

        voices = list_voices(config)
        if registered and registered.get("ok") and set_default:
            tts_cfg = config.setdefault("tts", {})
            tts_cfg["default_voice"] = registered.get("name") or name
            save_voice_config(config)

        report = {
            "finished": datetime.now(timezone.utc).isoformat(),
            "voice_name": name,
            "samples": samples,
            "errors": errors,
            "registered_voice": registered,
            "voices_available": [v.get("name") for v in voices],
            "voices_dir": str(voices_dir(config)),
        }
        try:
            neural = _train_neural_voice([row.get("text") or "" for row in samples], steps=steps)
            report["neural_voice"] = neural
        except Exception as exc:
            report["neural_voice_error"] = str(exc)
        _report_path().write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Navine AI - Python voice training complete. Samples: {len(samples)} | Errors: {len(errors)}")
        if registered and registered.get("ok"):
            print(f"Registered cloned voice: {registered.get('name')} -> {registered.get('reference')}")
            if set_default:
                print(f"Default voice set to: {registered.get('name')}")
        return report
