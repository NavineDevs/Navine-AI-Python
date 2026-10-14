from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_INFERENCE_COUNT = 0
_INFERENCE_COND = threading.Condition()
_TRAINING_ACTIVE = False


def inference_active() -> bool:
    with _INFERENCE_COND:
        return _INFERENCE_COUNT > 0


def training_active() -> bool:
    with _INFERENCE_COND:
        return _TRAINING_ACTIVE


def wait_for_inference_idle(timeout_seconds: float = 0.0) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    with _INFERENCE_COND:
        while _INFERENCE_COUNT > 0:
            if timeout_seconds <= 0:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _INFERENCE_COND.wait(timeout=min(remaining, 2.0))
        return True


@contextmanager
def inference_session(label: str = "inference") -> Iterator[None]:
    global _INFERENCE_COUNT
    del label
    with _INFERENCE_COND:
        _INFERENCE_COUNT += 1
        _INFERENCE_COND.notify_all()
    try:
        yield
    finally:
        with _INFERENCE_COND:
            _INFERENCE_COUNT = max(0, _INFERENCE_COUNT - 1)
            _INFERENCE_COND.notify_all()


@contextmanager
def training_session(label: str = "train", wait_for_idle_seconds: float = 120.0) -> Iterator[None]:
    global _TRAINING_ACTIVE
    if not wait_for_inference_idle(wait_for_idle_seconds):
        raise RuntimeError("Inference in progress; training deferred")
    with _INFERENCE_COND:
        while _TRAINING_ACTIVE:
            _INFERENCE_COND.wait(timeout=0.25)
        _TRAINING_ACTIVE = True
    try:
        yield
    finally:
        with _INFERENCE_COND:
            _TRAINING_ACTIVE = False
            _INFERENCE_COND.notify_all()
