"""Small helpers shared across the deep research agent modules."""

import sys
import threading
from datetime import datetime


def get_current_datetime() -> str:
    return datetime.now().strftime("%A %B %d, %Y")


class Console:
    """Progress goes to stderr so stdout carries only the clarification or report."""

    def __init__(self, quiet: bool) -> None:
        self._quiet = quiet
        self._lock = threading.Lock()

    def log(self, prefix: str, message: str) -> None:
        if self._quiet:
            return
        with self._lock:
            print(f"[{prefix}] {message}", file=sys.stderr, flush=True)

    def stream(self, text: str) -> None:
        if self._quiet:
            return
        with self._lock:
            print(text, end="", file=sys.stderr, flush=True)


def shorten(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
