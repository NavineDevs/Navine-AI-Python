import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional
from urllib.parse import urlparse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from navine.auth.keys import ApiKeyStore


def extract_api_key(request: Request) -> Optional[str]:
    header_key = request.headers.get("X-Navine-API-Key") or request.headers.get("X-API-Key")
    if header_key:
        return header_key.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def is_localhost_client(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    host = client.host
    return host in ("127.0.0.1", "::1", "localhost")


def _request_hostname(request: Request) -> str:
    host = (request.headers.get("Host") or "").strip().lower()
    if not host:
        return ""
    if host.startswith("[") and "]" in host:
        return host[1 : host.index("]")]
    return host.split(":")[0]


def is_same_origin_web_ui(request: Request) -> bool:
    fetch_site = (request.headers.get("Sec-Fetch-Site") or "").strip().lower()
    if fetch_site == "same-origin":
        return True
    host = _request_hostname(request)
    if not host:
        return False
    for header_name in ("Origin", "Referer"):
        value = (request.headers.get(header_name) or "").strip()
        if not value:
            continue
        try:
            parsed_host = urlparse(value).hostname
        except Exception:
            parsed_host = None
        if parsed_host and parsed_host.lower() == host:
            return True
    return False


def path_requires_auth(path: str) -> bool:
    if path in (
        "/api/health",
        "/api/info",
        "/api/models/stats",
        "/api/sandbox",
    ):
        return False
    if path.startswith("/api/"):
        return True
    if path.startswith("/v1/"):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_config: Dict[str, Any], key_store: ApiKeyStore):
        super().__init__(app)
        self.api_config = api_config
        self.key_store = key_store

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if request.method == "OPTIONS":
            return await call_next(request)
        if not path_requires_auth(path):
            return await call_next(request)
        if not self.api_config.get("require_auth", True):
            return await call_next(request)
        if self.api_config.get("allow_localhost_without_auth", True) and is_localhost_client(request):
            return await call_next(request)
        if (
            path.startswith("/api/")
            and self.api_config.get("allow_same_origin_without_auth", True)
            and is_same_origin_web_ui(request)
        ):
            return await call_next(request)
        raw_key = extract_api_key(request)
        if not raw_key or not self.key_store.verify(raw_key):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
        request.state.api_key = raw_key
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_config: Dict[str, Any]):
        super().__init__(app)
        self.api_config = api_config
        rate_cfg = api_config.get("rate_limit") or {}
        self.enabled = bool(rate_cfg.get("enabled", False))
        self.limit = int(rate_cfg.get("requests_per_minute", 60))
        self.window = 60.0
        self.hits: Dict[str, Deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        api_key = getattr(request.state, "api_key", None)
        if api_key:
            return ApiKeyStore.hash_key(api_key)
        client = request.client
        if client is None:
            return "unknown"
        return client.host

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled or not path_requires_auth(request.url.path):
            return await call_next(request)
        now = time.monotonic()
        key = self._client_key(request)
        bucket = self.hits[key]
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        bucket.append(now)
        return await call_next(request)
