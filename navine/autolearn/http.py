import time
from typing import Any, Dict, Optional

from navine.policy import assert_learning_url_allowed, http_get

USER_AGENT = "NavineAI/1.0 (local training bot)"
DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP = 1.5
_last_request = 0.0


def learning_http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    sleep: float = DEFAULT_SLEEP,
    **kwargs: Any,
):
    global _last_request
    assert_learning_url_allowed(url)
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    elapsed = time.time() - _last_request
    if elapsed < sleep:
        time.sleep(sleep - elapsed)
    response = http_get(url, headers=merged, timeout=timeout, **kwargs)
    _last_request = time.time()
    return response
