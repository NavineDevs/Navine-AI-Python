from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from navine.device_manager import configure_compute_environment, get_autocast_context, get_device
from navine.utils.paths import get_checkpoint_dir, get_output_dir, get_project_root

ENGINE_NAME = "Navine Engine"
ENGINE_VERSION = "1"


def probe() -> Dict[str, Any]:
    configure_compute_environment()
    device = get_device()
    info: Dict[str, Any] = {
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "sdpa": bool(hasattr(F, "scaled_dot_product_attention")),
        "kv_cache": True,
        "custom_kernels": ["sdpa", "rope", "swiglu", "rmsnorm", "ddpm_sample", "video_lstm_decode"],
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = torch.cuda.get_device_name(device)
        info["vram_gb"] = round(float(props.total_memory) / (1024 ** 3), 2)
        info["bf16"] = bool(torch.cuda.is_bf16_supported())
        info["dtype"] = "bfloat16" if info["bf16"] else "float16"
    else:
        info["gpu"] = None
        info["vram_gb"] = None
        info["bf16"] = False
        info["dtype"] = "float32"
    return info


def _infer_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    generated: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty and repetition_penalty > 1.0:
        for token_id in set(generated[0].tolist()):
            val = logits[0, token_id]
            logits[0, token_id] = torch.where(val > 0, val / repetition_penalty, val * repetition_penalty)
    logits = logits / max(float(temperature), 1e-8)
    if top_k is not None and top_k > 0:
        keep = min(int(top_k), logits.size(-1))
        values, _ = torch.topk(logits, keep)
        logits[logits < values[:, [-1]]] = float("-inf")
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class NavineEngine:
    def __init__(self) -> None:
        configure_compute_environment()
        self.device = get_device()
        self.dtype = _infer_dtype(self.device)
        self.probe = probe()
        self._text: Dict[str, Any] = {}
        self._image = None
        self._image_cfg = None
        self._video = None
        self._video_cfg = None
        self._init_kernels()

    def _init_kernels(self) -> None:
        from navine.engine.kernels import native_available, get_backend
        self.native_kernels = native_available()
        self.kernel_backend = get_backend()
        self.probe["native_kernels"] = self.native_kernels
        self.probe["kernel_backend"] = self.kernel_backend

    def status(self) -> Dict[str, Any]:
        loaded = []
        loaded.extend(sorted(self._text.keys()))
        if self._image is not None:
            loaded.append("image_enterprise")
        if self._video is not None:
            loaded.append("video_enterprise")
        data = dict(self.probe)
        data["loaded"] = loaded
        data["ready"] = bool(loaded)
        return data

    def _maybe_cast(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        model.eval()
        return model

    def _cast_text(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        if self.dtype != torch.float32 and self.device.type == "cuda":
            model = model.to(dtype=self.dtype)
        model.eval()
        return model

    def load_text(self, name: str = "text_enterprise") -> None:
        from navine.text.model import NavineTextModel
        from navine.text.tokenizer import NavineTokenizer

        norm = str(name).replace("-", "_")
        if norm in ("text_code", "code", "coding"):
            key = "text_code"
        elif norm in ("hitboyx23_ai", "hitboyx23", "hitboy"):
            key = "hitboyx23_ai"
        elif norm in ("hitboyx23_ai_python", "hitboyx23_python", "hitboy_python"):
            key = "hitboyx23_ai_python"
        else:
            key = "text_enterprise"
        if key in self._text:
            return
        ckpt = get_checkpoint_dir(key) / "latest.pt"
        tok_path = get_checkpoint_dir(key) / "tokenizer.json"
        if not tok_path.exists():
            tok_path = get_checkpoint_dir("text_enterprise") / "tokenizer.json"
        if not ckpt.exists() or not tok_path.exists():
            raise FileNotFoundError(f"Navine Engine missing text weights for {key}")
        model, payload = NavineTextModel.load_checkpoint(ckpt, self.device)
        model = self._cast_text(model)
        tokenizer = NavineTokenizer.load(tok_path)
        backend = "python"
        if key == "hitboyx23_ai" and self.native_kernels:
            backend = "mixed"
        self._text[key] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": (payload or {}).get("config") or {},
            "checkpoint": str(ckpt),
            "backend": backend,
        }

    def load_image(self) -> None:
        if self._image is not None:
            return
        from navine.image.model import NavineDiffusionModel
        from navine.utils.config import load_config

        cfg = load_config("image_enterprise")
        ckpt = get_checkpoint_dir("image_enterprise") / "latest.pt"
        if not ckpt.exists():
            raise FileNotFoundError("Navine Engine missing image_enterprise weights")
        model = NavineDiffusionModel.load_checkpoint(ckpt, cfg, self.device)
        if not getattr(model, "checkpoint_compatible", True):
            raise RuntimeError("Navine Engine image checkpoint is not compatible")
        self._image = self._maybe_cast(model)
        self._image_cfg = cfg

    def load_video(self) -> None:
        if self._video is not None:
            return
        from navine.utils.config import load_config
        from navine.video.model import NavineVideoModel

        cfg = load_config("video_enterprise")
        ckpt = get_checkpoint_dir("video_enterprise") / "latest.pt"
        if not ckpt.exists():
            raise FileNotFoundError("Navine Engine missing video_enterprise weights")
        model = NavineVideoModel.load_checkpoint(ckpt, cfg, self.device)
        if not getattr(model, "checkpoint_compatible", True):
            raise RuntimeError("Navine Engine video checkpoint is not compatible")
        self._video = self._maybe_cast(model)
        self._video_cfg = cfg

    def load_all(self) -> Dict[str, Any]:
        errors: Dict[str, str] = {}
        for name in ("text_enterprise", "text_code", "hitboyx23_ai", "hitboyx23_ai_python"):
            try:
                self.load_text(name)
            except Exception as exc:
                errors[name] = str(exc)
        try:
            self.load_image()
        except Exception as exc:
            errors["image_enterprise"] = str(exc)
        try:
            self.load_video()
        except Exception as exc:
            errors["video_enterprise"] = str(exc)
        report = self.status()
        if errors:
            report["errors"] = errors
        return report

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        name: str = "text_enterprise",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        norm = str(name).replace("-", "_")
        if norm in ("text_code", "code", "coding"):
            key = "text_code"
        elif norm in ("hitboyx23_ai", "hitboyx23", "hitboy"):
            key = "hitboyx23_ai"
        elif norm in ("hitboyx23_ai_python", "hitboyx23_python", "hitboy_python"):
            key = "hitboyx23_ai_python"
        else:
            key = "text_enterprise"
        self.load_text(key)
        bundle = self._text[key]
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        ids = tokenizer.encode(prompt, add_special=True)
        if tokenizer.eos_id in ids:
            ids = ids[:-1]
        generated = torch.tensor([ids], dtype=torch.long, device=self.device)
        past = None
        eos = tokenizer.eos_id
        with get_autocast_context(self.device):
            for _ in range(max(1, int(max_new_tokens))):
                step_in = generated[:, -1:] if past is not None else generated
                logits = model(step_in, past_kvs=past)[:, -1, :].float()
                past = getattr(model, "_last_kvs", None)
                nxt = _sample_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generated=generated,
                    repetition_penalty=repetition_penalty,
                )
                generated = torch.cat([generated, nxt], dim=1)
                if int(nxt.item()) == eos and generated.size(1) > len(ids) + 2:
                    break
        new_ids = generated[0, len(ids) :].tolist()
        if eos in new_ids:
            new_ids = new_ids[: new_ids.index(eos)]
        return tokenizer.decode(new_ids)

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        output: Optional[str] = None,
        seed: int = 0,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        model_profile: Optional[str] = None,
    ) -> Path:
        from torchvision.utils import save_image
        from navine.modes.profiles import apply_image_profile

        self.load_image()
        infer = (self._image_cfg or {}).get("inference") or {}
        mode_cfg = dict((infer.get("modes") or {}).get(str(model_profile or "").lower(), {}) or {})
        steps = int(num_steps if num_steps is not None else mode_cfg.get("num_steps") or infer.get("num_steps") or 200)
        scale = float(
            guidance_scale
            if guidance_scale is not None
            else mode_cfg.get("guidance_scale")
            or infer.get("guidance_scale")
            or 1.5
        )
        text = apply_image_profile(prompt, model_profile)
        prefix = mode_cfg.get("prompt_prefix")
        if prefix and prefix not in text:
            text = f"{prefix} {text}".strip()
        torch.manual_seed(int(seed))
        out = Path(output) if output else get_output_dir("image") / "engine.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        with get_autocast_context(self.device):
            samples = self._image.sample(
                batch_size=1,
                text=text,
                device=self.device,
                num_steps=steps,
                guidance_scale=scale,
                scheduler=str(infer.get("scheduler") or "ddpm"),
            )
        save_image((samples + 1) / 2, out)
        size = int(infer.get("output_size") or self._image.image_size)
        if size != int(self._image.image_size):
            from PIL import Image

            img = Image.open(out).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
            img.save(out, format="PNG")
        return out

    @torch.inference_mode()
    def generate_video(
        self,
        prompt: str,
        output: Optional[str] = None,
        seed: int = 0,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        model_profile: Optional[str] = None,
    ) -> Path:
        from navine.modes.profiles import apply_video_profile

        self.load_video()
        infer = (self._video_cfg or {}).get("inference") or {}
        mode_cfg = dict((infer.get("modes") or {}).get(str(model_profile or "").lower(), {}) or {})
        n = int(num_frames if num_frames is not None else mode_cfg.get("num_frames") or infer.get("num_frames") or 16)
        frame_fps = int(fps if fps is not None else mode_cfg.get("fps") or infer.get("fps") or 24)
        text = apply_video_profile(prompt, model_profile)
        prefix = mode_cfg.get("prompt_prefix")
        if prefix and prefix not in text:
            text = f"{prefix} {text}".strip()
        torch.manual_seed(int(seed))
        with get_autocast_context(self.device):
            frames = self._video.generate(text, self.device, num_frames=n, seed=int(seed))
        out = Path(output) if output else get_output_dir("video") / "engine.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        imgs = []
        from PIL import Image
        import numpy as np

        clip = frames[0].detach().float().cpu()
        for idx in range(clip.size(0)):
            frame = ((clip[idx].clamp(-1, 1) + 1) / 2).numpy()
            frame = np.transpose(frame, (1, 2, 0))
            frame = (frame * 255).clip(0, 255).astype("uint8")
            imgs.append(Image.fromarray(frame, mode="RGB"))
        size = int(infer.get("frame_output_size") or self._video.frame_size)
        if imgs and imgs[0].size[0] != size:
            imgs = [im.resize((size, size), Image.Resampling.LANCZOS) for im in imgs]
        try:
            import imageio

            imageio.mimsave(str(out), imgs, fps=frame_fps)
        except Exception:
            png = out.with_suffix(".png")
            imgs[0].save(png)
            return png
        return out

    def generate_voice(self, prompt: str, name: str = "text_enterprise", **kwargs: Any) -> Path:
        self.load_text(name)
        text_output = self.generate_text(prompt, name=name, max_new_tokens=kwargs.get("max_new_tokens") or 256, temperature=kwargs.get("temperature") or 0.7)
        from navine.voice.tts import synthesize
        out = Path(kwargs.get("output") or str(get_output_dir("voice") / "engine.wav"))
        out.parent.mkdir(parents=True, exist_ok=True)
        synthesize(text_output, out)
        return out

    def _resolve_modality(self, key: str, kwargs: Dict[str, Any]) -> Optional[str]:
        modality = kwargs.pop("modality", None)
        if modality and modality in ("text", "image", "video", "voice"):
            return modality
        if key in ("image_enterprise", "image", "image_think", "image_detective"):
            return "image"
        if key in ("video_enterprise", "video", "video_think", "video_detective"):
            return "video"
        return None

    def run(self, name: str, prompt: str, output: Optional[str] = None, **kwargs: Any) -> Any:
        key = str(name or "").strip().lower().replace("-", "_")
        modality = self._resolve_modality(key, kwargs)

        if modality == "image":
            profile = "detective" if "detective" in key else ("think" if "think" in key else kwargs.get("model_profile"))
            return self.generate_image(prompt, output=output, seed=int(kwargs.get("seed") or 0), num_steps=kwargs.get("num_steps"), guidance_scale=kwargs.get("guidance_scale"), model_profile=profile)
        if modality == "video":
            profile = "detective" if "detective" in key else ("think" if "think" in key else kwargs.get("model_profile"))
            return self.generate_video(prompt, output=output, seed=int(kwargs.get("seed") or 0), num_frames=kwargs.get("num_frames"), fps=kwargs.get("fps"), model_profile=profile)
        if modality == "voice":
            return self.generate_voice(prompt, name=key, **kwargs)

        if key in ("hitboyx23_ai", "hitboyx23", "hitboy", "hitboyx23_ai_python", "hitboyx23_python", "hitboy_python"):
            from navine.text.chat import chat as text_chat

            profile = None
            if "detective" in str(kwargs.get("model_profile", "")):
                profile = "detective"
            elif "think" in str(kwargs.get("model_profile", "")):
                profile = "think"
            return text_chat(prompt, model_profile=profile, **{k: v for k, v in kwargs.items() if k in ("max_new_tokens", "temperature") and v is not None})
        if key in ("text_enterprise", "text", "chat", "think", "thinking", "detective", "mystery"):
            from navine.text.chat import chat

            profile = "detective" if key in ("detective", "mystery") else ("think" if key in ("think", "thinking") else None)
            return chat(prompt, model_profile=profile, **{k: v for k, v in kwargs.items() if k in ("max_new_tokens", "temperature") and v is not None})
        if key in ("text_code", "code", "coding"):
            return self.generate_text(prompt, name="text_code", **{k: v for k, v in kwargs.items() if k in ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty") and v is not None})
        if key in ("image_enterprise",):
            return self.generate_image(prompt, output=output, seed=int(kwargs.get("seed") or 0), num_steps=kwargs.get("num_steps"), guidance_scale=kwargs.get("guidance_scale"), model_profile=kwargs.get("model_profile"))
        if key in ("video_enterprise",):
            return self.generate_video(prompt, output=output, seed=int(kwargs.get("seed") or 0), num_frames=kwargs.get("num_frames"), fps=kwargs.get("fps"), model_profile=kwargs.get("model_profile"))
        raise KeyError(f"Unknown engine target: {name}")

    def write_manifest(self) -> Path:
        path = get_project_root() / "engine" / "ENGINE.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probe": self.probe,
            "models": {
                "text_enterprise": str(get_checkpoint_dir("text_enterprise") / "latest.pt"),
                "text_code": str(get_checkpoint_dir("text_code") / "latest.pt"),
                "hitboyx23_ai": str(get_checkpoint_dir("hitboyx23_ai") / "latest.pt"),
                "hitboyx23_ai_python": str(get_checkpoint_dir("hitboyx23_ai_python") / "latest.pt"),
                "image_enterprise": str(get_checkpoint_dir("image_enterprise") / "latest.pt"),
                "video_enterprise": str(get_checkpoint_dir("video_enterprise") / "latest.pt"),
            },
            "gguf": {
                "text_enterprise": str(get_project_root() / "models" / "gguf" / "text_enterprise.gguf"),
                "text_code": str(get_project_root() / "models" / "gguf" / "text_code.gguf"),
                "hitboyx23_ai": str(get_project_root() / "models" / "gguf" / "hitboyx23_ai.gguf"),
                "hitboyx23_ai_python": str(get_project_root() / "models" / "gguf" / "hitboyx23_ai_python.gguf"),
                "image_enterprise": str(get_project_root() / "models" / "gguf" / "image_enterprise.gguf"),
                "video_enterprise": str(get_project_root() / "models" / "gguf" / "video_enterprise.gguf"),
            },
            "kernels": self.probe.get("custom_kernels"),
            "run": [
                "python -m navine.engine probe",
                "python -m navine.engine run text_enterprise \"hello\"",
                "python -m navine.engine run text_code \"write a python hello world\"",
                "python -m navine.engine run image_enterprise \"photoreal portrait\"",
                "python -m navine.engine run video_enterprise \"slow camera pan\"",
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


_ENGINE: Optional[NavineEngine] = None


def get_engine() -> NavineEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NavineEngine()
    return _ENGINE
