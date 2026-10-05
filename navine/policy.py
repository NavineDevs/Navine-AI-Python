import sys
from importlib.machinery import ModuleSpec
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import yaml

from navine.utils.paths import get_project_root

DEFAULT_POLICY: Dict[str, Any] = {
    "local_only": True,
}

BLOCKED_AI_HOST_SUFFIXES = (
    "api.openai.com",
    "openai.azure.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    "api.cohere.ai",
    "api.cohere.com",
    "api.replicate.com",
    "api.stability.ai",
    "api-inference.huggingface.co",
    "inference-endpoint.huggingface.cloud",
    "api.mistral.ai",
    "api.deepseek.com",
    "api.together.xyz",
    "api.groq.com",
    "api.perplexity.ai",
    "bedrock-runtime.amazonaws.com",
    "openrouter.ai",
)

ALLOWED_LEARNING_HOST_SUFFIXES = (
    "duckduckgo.com",
    "html.duckduckgo.com",
    "lite.duckduckgo.com",
    "api.github.com",
    "github.com",
    "gist.github.com",
    "raw.githubusercontent.com",
    "www.reddit.com",
    "reddit.com",
    "old.reddit.com",
    "api.reddit.com",
    "api.stackexchange.com",
    "stackexchange.com",
    "hacker-news.firebaseio.com",
    "hnrss.org",
    "rust-lang.org",
    "en.wikipedia.org",
    "wikipedia.org",
    "export.arxiv.org",
    "www.gutenberg.org",
    "gutenberg.org",
    "gitlab.com",
    "codeberg.org",
    "dev.to",
    "developer.mozilla.org",
    "docs.python.org",
    "medium.com",
    "commons.wikimedia.org",
    "i.redd.it",
    "preview.redd.it",
    "external-preview.redd.it",
    "i.imgur.com",
    "v.redd.it",
    "redgifs.com",
    "www.redgifs.com",
    "thumbs.redgifs.com",
    "thumbs2.redgifs.com",
    "imgur.com",
    "gfycat.com",
    "www.gfycat.com",
    "giant.gfycat.com",
    "thumbs.gfycat.com",
    "rule34.xxx",
    "danbooru.donmai.us",
    "gelbooru.com",
    "img3.gelbooru.com",
    "e621.net",
    "sankakucomplex.com",
    "nhentai.net",
    "erome.com",
    "www.erome.com",
    "coomer.st",
    "kemono.cr",
    "hanime.tv",
    "xbooru.com",
    "a.4cdn.org",
    "i.4cdn.org",
    "s.4cdn.org",
    "waifu.im",
    "www.waifu.im",
    "api.waifu.im",
    "cdn.waifu.im",
    "infini-atomic.w3spaces.com",
    "w3spaces.com",
    "bbci.co.uk",
    "theguardian.com",
    "nytimes.com",
    "arstechnica.com",
    "reuters.com",
    "rss.cnn.com",
    "apnews.com",
    "techcrunch.com",
    "huggingface.co",
    "www.huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "hf.co",
    "datasets-server.huggingface.co",
)

BLOCKED_AI_MODULE_PREFIXES = (
    "openai",
    "anthropic",
    "google.generativeai",
    "google.ai",
    "cohere",
    "replicate",
    "stability_sdk",
    "huggingface_hub.inference",
    "transformers.pipelines",
    "sentence_transformers",
)

_config_cache: Optional[Dict[str, Any]] = None
_enforced = False


class ExternalAICallBlocked(RuntimeError):
    pass


def load_policy_config() -> Dict[str, Any]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    merged = dict(DEFAULT_POLICY)
    config_path = get_project_root() / "configs" / "navine.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged.update(loaded)
    _config_cache = merged
    return merged


def is_local_only() -> bool:
    return bool(load_policy_config().get("local_only", True))


def _host_matches_blocked(host: str) -> bool:
    host = host.lower().split(":")[0]
    for suffix in BLOCKED_AI_HOST_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _host_matches_allowed_learning(host: str) -> bool:
    host = host.lower().split(":")[0]
    for suffix in ALLOWED_LEARNING_HOST_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _load_extra_learning_hosts() -> tuple:
    try:
        from navine.nsfw.config import load_nsfw_config

        cfg = load_nsfw_config()
        domains = cfg.get("allowed_learning_domains") or []
        return tuple(str(domain).strip().lower() for domain in domains if str(domain).strip())
    except Exception:
        return ()


def _host_matches_extra_learning(host: str) -> bool:
    for suffix in _load_extra_learning_hosts():
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def is_learning_url_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.netloc
    if not host:
        return False
    if _host_matches_blocked(host):
        return False
    if _host_matches_allowed_learning(host):
        return True
    return _host_matches_extra_learning(host)


def assert_no_external_ai_url(url: str) -> None:
    if not is_local_only():
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return
    host = parsed.netloc
    if host and _host_matches_blocked(host):
        raise ExternalAICallBlocked(
            f"Navine AI - Python local-only policy blocked external AI request to {host}"
        )


def assert_learning_url_allowed(url: str) -> None:
    assert_no_external_ai_url(url)
    if not is_learning_url_allowed(url):
        raise ExternalAICallBlocked(
            f"Navine AI - Python autolearn policy blocked URL (not on learning allowlist): {url}"
        )


def http_get(url: str, **kwargs):
    assert_no_external_ai_url(url)
    import requests

    return requests.get(url, **kwargs)


def http_post(url: str, **kwargs):
    assert_no_external_ai_url(url)
    import requests

    return requests.post(url, **kwargs)


class _ExternalAIImportBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if not is_local_only():
            return None
        for prefix in BLOCKED_AI_MODULE_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return ModuleSpec(fullname, self)
        return None

    def find_module(self, fullname, path=None):
        spec = self.find_spec(fullname, path)
        return self if spec else None

    def create_module(self, spec):
        raise ExternalAICallBlocked(
            f"Navine AI - Python local-only policy blocked import of external AI module: {spec.name}"
        )

    def exec_module(self, module):
        raise ExternalAICallBlocked(
            f"Navine AI - Python local-only policy blocked import of external AI module: {module.__name__}"
        )


def enforce_local_only() -> None:
    global _enforced
    if _enforced or not is_local_only():
        return
    if not any(isinstance(finder, _ExternalAIImportBlocker) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ExternalAIImportBlocker())
    _enforced = True
