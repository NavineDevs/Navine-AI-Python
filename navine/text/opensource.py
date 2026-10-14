import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from navine.utils.config import load_config

_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_BACKEND_CACHE: Optional[str] = None
_LLAMA_MODEL = None
_TRANSFORMERS_PIPE = None


def load_opensource_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    try:
        _CONFIG_CACHE = load_config("opensource")
    except FileNotFoundError:
        _CONFIG_CACHE = {"enabled": "auto", "prefer": "ollama"}
    return _CONFIG_CACHE


def _ollama_available(config: Dict[str, Any]) -> bool:
    host = (config.get("ollama") or {}).get("host", "http://localhost:11434")
    try:
        request = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _llama_cpp_available(config: Dict[str, Any]) -> bool:
    model_path = (config.get("llama_cpp") or {}).get("model_path", "")
    if not model_path:
        return False
    try:
        import llama_cpp  # noqa: F401
        from pathlib import Path

        return Path(model_path).exists()
    except Exception:
        return False


def _transformers_available(config: Dict[str, Any]) -> bool:
    model_id = (config.get("transformers") or {}).get("model_id", "")
    if not model_id:
        return False
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _cloud_available(config: Dict[str, Any]) -> bool:
    cloud_cfg = config.get("cloud") or {}
    if cloud_cfg.get("enabled", True) is False:
        return False
    provider = str(cloud_cfg.get("provider") or "").strip().lower()
    if provider in ("", "auto"):
        provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "huggingface"
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    if provider == "huggingface":
        return bool(os.getenv("HF_TOKEN"))
    if provider == "navine_cloud":
        return bool(os.getenv("NAVINE_CLOUD_API_KEY"))
    return False


def detect_backend(force: bool = False) -> Optional[str]:
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None and not force:
        return _BACKEND_CACHE or None
    config = load_opensource_config()
    prefer = config.get("prefer", "ollama")
    checks = {
        "cloud": _cloud_available,
        "ollama": _ollama_available,
        "llama_cpp": _llama_cpp_available,
        "transformers": _transformers_available,
    }
    order = [prefer] + [name for name in checks if name != prefer]
    for name in order:
        try:
            if checks[name](config):
                _BACKEND_CACHE = name
                return name
        except Exception:
            continue
    _BACKEND_CACHE = ""
    return None


def is_opensource_enabled() -> bool:
    config = load_opensource_config()
    setting = config.get("enabled", "auto")
    if setting is False or setting == "off":
        return False
    if setting is True or setting == "on":
        return detect_backend() is not None
    return detect_backend() is not None


def _request_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Optional[Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _generate_cloud(prompt: str, config: Dict[str, Any], max_tokens: int, temperature: float) -> Optional[str]:
    cloud_cfg = config.get("cloud") or {}
    if cloud_cfg.get("enabled", True) is False:
        return None
    provider = str(cloud_cfg.get("provider") or "auto").strip().lower()
    model = str(cloud_cfg.get("model") or "meta-llama/llama-3.1-8b-instruct").strip()
    timeout = int(cloud_cfg.get("timeout", 120))

    if provider in ("auto", ""):
        provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "huggingface"

    if provider == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
        url = str(cloud_cfg.get("base_url") or "https://openrouter.ai/api/v1/chat/completions")
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        }
        resp = _request_json(url, body, headers, timeout)
        if not resp:
            return None
        try:
            return str(resp["choices"][0]["message"]["content"]).strip() or None
        except Exception:
            return None

    if provider == "huggingface" and os.getenv("HF_TOKEN"):
        url = str(cloud_cfg.get("base_url") or f"https://api-inference.huggingface.co/models/{model}")
        body = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        }
        resp = _request_json(url, body, headers, timeout)
        if not resp:
            return None
        if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
            return str(resp[0]["generated_text"]).strip() or None
        if isinstance(resp, dict) and "generated_text" in resp:
            return str(resp["generated_text"]).strip() or None
        return None

    if provider == "navine_cloud" and os.getenv("NAVINE_CLOUD_API_KEY"):
        url = str(cloud_cfg.get("base_url") or "https://api.navine.ai/v1/chat/completions")
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('NAVINE_CLOUD_API_KEY')}",
        }
        resp = _request_json(url, body, headers, timeout)
        if not resp:
            return None
        try:
            return str(resp["choices"][0]["message"]["content"]).strip() or None
        except Exception:
            return None

    return None


def _generate_ollama(prompt: str, config: Dict[str, Any], max_tokens: int, temperature: float) -> Optional[str]:
    ollama_cfg = config.get("ollama") or {}
    host = ollama_cfg.get("host", "http://localhost:11434")
    model = ollama_cfg.get("model", "llama3.2")
    timeout = int(ollama_cfg.get("timeout", 120))
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return (body.get("response") or "").strip() or None
    except Exception:
        return None


def _generate_llama_cpp(prompt: str, config: Dict[str, Any], max_tokens: int, temperature: float) -> Optional[str]:
    global _LLAMA_MODEL
    llama_cfg = config.get("llama_cpp") or {}
    model_path = llama_cfg.get("model_path", "")
    if not model_path:
        return None
    try:
        if _LLAMA_MODEL is None:
            from llama_cpp import Llama

            _LLAMA_MODEL = Llama(
                model_path=model_path,
                n_ctx=int(llama_cfg.get("n_ctx", 4096)),
                verbose=False,
            )
        output = _LLAMA_MODEL(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["### User:", "### System:"],
        )
        return (output["choices"][0]["text"] or "").strip() or None
    except Exception:
        return None


def _generate_transformers(prompt: str, config: Dict[str, Any], max_tokens: int, temperature: float) -> Optional[str]:
    global _TRANSFORMERS_PIPE
    tf_cfg = config.get("transformers") or {}
    model_id = tf_cfg.get("model_id", "")
    if not model_id:
        return None
    try:
        if _TRANSFORMERS_PIPE is None:
            from transformers import pipeline

            _TRANSFORMERS_PIPE = pipeline(
                "text-generation",
                model=model_id,
                device_map=tf_cfg.get("device", "cpu"),
            )
        result = _TRANSFORMERS_PIPE(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )
        return (result[0]["generated_text"] or "").strip() or None
    except Exception:
        return None


def generate_opensource(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Optional[str]:
    config = load_opensource_config()
    gen_cfg = config.get("generation") or {}
    tokens = int(max_tokens if max_tokens is not None else gen_cfg.get("max_tokens", 512))
    temp = float(temperature if temperature is not None else gen_cfg.get("temperature", 0.7))
    backend = detect_backend()
    if backend == "cloud":
        return _generate_cloud(prompt, config, tokens, temp)
    if backend == "ollama":
        return _generate_ollama(prompt, config, tokens, temp)
    if backend == "llama_cpp":
        return _generate_llama_cpp(prompt, config, tokens, temp)
    if backend == "transformers":
        return _generate_transformers(prompt, config, tokens, temp)
    return None
