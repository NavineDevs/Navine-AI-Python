from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ollama_router = APIRouter()


class OllamaGenerateRequest(BaseModel):
    model: str = "text_enterprise"
    prompt: str = ""
    stream: bool = False
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None


class OllamaChatMessage(BaseModel):
    role: str
    content: str = ""


class OllamaChatRequest(BaseModel):
    model: str = "text_enterprise"
    messages: List[OllamaChatMessage] = Field(default_factory=list)
    stream: bool = False
    options: Optional[Dict[str, Any]] = None


def _opts(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    options = options or {}
    out: Dict[str, Any] = {}
    if "num_predict" in options:
        out["max_new_tokens"] = int(options["num_predict"])
    if "temperature" in options:
        out["temperature"] = float(options["temperature"])
    if "top_k" in options:
        out["top_k"] = int(options["top_k"])
    if "top_p" in options:
        out["top_p"] = float(options["top_p"])
    if "seed" in options:
        out["seed"] = int(options["seed"])
    return out


def _model_name(name: str) -> str:
    key = str(name or "text_enterprise").strip().lower().replace("-", "_")
    aliases = {
        "navine": "text_enterprise",
        "navine_text": "text_enterprise",
        "llama3": "text_enterprise",
        "mistral": "text_enterprise",
        "think": "text_enterprise",
        "thinking": "text_enterprise",
        "detective": "text_enterprise",
        "mystery": "text_enterprise",
        "code": "text_code",
        "codellama": "text_code",
        "hitboyx23": "hitboyx23_ai",
        "hitboyx23_ai": "hitboyx23_ai",
        "hitboy": "hitboyx23_ai",
        "hitboyx23_python": "hitboyx23_ai_python",
        "hitboyx23_ai_python": "hitboyx23_ai_python",
        "hitboy_python": "hitboyx23_ai_python",
        "image": "image_enterprise",
        "video": "video_enterprise",
    }
    return aliases.get(key, key)


@ollama_router.get("/tags")
def ollama_tags():
    from navine.gguf_export import gguf_dir
    from navine.llm import list_llms

    models = []
    for row in list_llms():
        gguf = gguf_dir() / f"{row['id']}.gguf"
        models.append(
            {
                "name": row["id"],
                "model": row["id"],
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": int(gguf.stat().st_size) if gguf.exists() else int(row.get("bytes") or 0),
                "digest": "navine",
                "details": {
                    "format": "gguf",
                    "family": "navine",
                    "parameter_size": row["id"],
                    "quantization_level": "F16",
                },
            }
        )
    return {"models": models}


@ollama_router.get("/version")
def ollama_version():
    from navine import __version__

    return {"version": __version__}


@ollama_router.post("/show")
def ollama_show(body: Dict[str, Any]):
    name = _model_name(str(body.get("name") or body.get("model") or "text_enterprise"))
    from navine.gguf_export import gguf_dir

    gguf = gguf_dir() / f"{name}.gguf"
    return {
        "modelfile": f"FROM {gguf}" if gguf.exists() else f"FROM {name}",
        "parameters": "temperature 0.7\nnum_ctx 1024",
        "template": "{{ if .System }}### System: {{ .System }}\n{{ end }}### User: {{ .Prompt }}\n### Assistant:",
        "details": {"format": "gguf", "family": "navine", "quantization_level": "F16"},
        "model_info": {"general.architecture": "navine", "general.name": name},
    }


@ollama_router.post("/generate")
def ollama_generate(req: OllamaGenerateRequest):
    name = _model_name(req.model)
    from navine.engine import get_engine

    engine = get_engine()
    kwargs = _opts(req.options)
    try:
        if "image" in name:
            path = engine.generate_image(req.prompt, seed=int(kwargs.get("seed") or 0), model_profile=None)
            text = str(path)
        elif "video" in name:
            path = engine.generate_video(req.prompt, seed=int(kwargs.get("seed") or 0), model_profile=None)
            text = str(path)
        else:
            from navine.text.chat import chat

            profile = None
            if any(k in name for k in ("detective", "mystery", "cipher", "cicada")):
                profile = "detective"
            elif "think" in name:
                profile = "think"
            if "code" in name:
                from navine.text.infer import generate as generate_text
                from navine.utils.paths import get_checkpoint_dir

                ckpt = get_checkpoint_dir("text_code") / "latest.pt"
                text = generate_text(
                    req.prompt,
                    max_new_tokens=int(kwargs.get("max_new_tokens") or 256),
                    temperature=float(kwargs.get("temperature") or 0.2),
                    checkpoint=str(ckpt) if ckpt.exists() else None,
                    mode="code",
                )
            else:
                text = chat(
                    req.prompt,
                    max_new_tokens=int(kwargs.get("max_new_tokens") or 256),
                    temperature=float(kwargs.get("temperature") or 0.7),
                    model_profile=profile,
                )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    payload = {
        "model": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": text,
        "done": True,
        "done_reason": "stop",
    }
    if req.stream:
        import json

        def chunks():
            yield json.dumps(payload) + "\n"

        return StreamingResponse(chunks(), media_type="application/x-ndjson")
    return JSONResponse(payload)


@ollama_router.get("/ps")
def ollama_ps():
    return {"models": []}


@ollama_router.post("/chat")
def ollama_chat(req: OllamaChatRequest):
    name = _model_name(req.model)
    prompt_parts: List[str] = []
    for msg in req.messages:
        role = (msg.role or "user").lower()
        if role == "system":
            prompt_parts.append(f"<|system|>{msg.content}")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>{msg.content}")
        else:
            prompt_parts.append(f"<|user|>{msg.content}")
    prompt_parts.append("<|assistant|>")
    prompt = "\n".join(prompt_parts)
    gen = OllamaGenerateRequest(model=name, prompt=prompt, stream=req.stream, options=req.options)
    result = ollama_generate(gen)
    if isinstance(result, StreamingResponse):
        return result
    parsed = json.loads(result.body)
    return JSONResponse(
        {
            "model": name,
            "created_at": parsed.get("created_at"),
            "message": {"role": "assistant", "content": parsed.get("response") or ""},
            "done": True,
            "done_reason": "stop",
        }
    )
