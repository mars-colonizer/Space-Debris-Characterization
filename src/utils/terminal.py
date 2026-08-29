"""Terminal progress reporting for Phase 2 pipeline scripts."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

QUIET = False
VERBOSE = False


def add_verbosity_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verbose", action="store_true", help="Print extra diagnostic details")
    group.add_argument("--quiet", action="store_true", help="Print errors and final status only")


def configure_from_args(args: argparse.Namespace) -> None:
    global QUIET, VERBOSE
    QUIET = bool(getattr(args, "quiet", False))
    VERBOSE = bool(getattr(args, "verbose", False))
    level = logging.ERROR if QUIET else logging.WARNING
    logging.basicConfig(level=level, force=True)
    for name in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _emit(msg: str, *, force: bool = False) -> None:
    if QUIET and not force:
        return
    print(msg, flush=True)


def line(text: str = "") -> None:
    _emit(text)


def banner(title: str) -> None:
    _emit("=" * 60)
    _emit(title)
    _emit("=" * 60)
    _emit("")


def section(title: str) -> None:
    _emit("")
    _emit("-" * 60)
    _emit(title)
    _emit("-" * 60)
    _emit("")


def step(current: int, total: int, message: str) -> None:
    _emit(f"[{current}/{total}] {message}")


def ok(message: str, indent: int = 0, *, force: bool = False) -> None:
    prefix = " " * indent
    _emit(f"{prefix}[OK] {message}", force=force)


def info(message: str, indent: int = 0) -> None:
    prefix = " " * indent
    _emit(f"{prefix}[INFO] {message}")


def warn(message: str, indent: int = 0) -> None:
    prefix = " " * indent
    _emit(f"{prefix}[WARNING] {message}")


def detail(message: str, indent: int = 4) -> None:
    if VERBOSE:
        _emit(" " * indent + message)


def error_banner(title: str, reason: str, hints: list[str] | None = None) -> None:
    _emit("")
    _emit("=" * 60, force=True)
    _emit(f"[ERROR] {title}", force=True)
    _emit("=" * 60, force=True)
    _emit("", force=True)
    _emit("Reason:", force=True)
    _emit(f"  {reason}", force=True)
    if hints:
        _emit("", force=True)
        _emit("Check:", force=True)
        for h in hints:
            _emit(f"    {h}", force=True)
    _emit("=" * 60, force=True)


def fail(title: str, reason: str, hints: list[str] | None = None, code: int = 1) -> None:
    error_banner(title, reason, hints)
    sys.exit(code)


def fmt_n(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:,.2f}" if abs(n) >= 1 else f"{n:.4f}"
    return f"{n:,}"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    ok(f"{label} completed in {elapsed:.2f} seconds")


class ScriptTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def total(self) -> float:
        return time.perf_counter() - self._start

    def print_total(self) -> None:
        _emit("")
        _emit(f"Total execution time: {self.total():.2f} seconds")


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> None:
    if not rows:
        return
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for row in rows:
                w = max(w, len(row[i]))
            col_widths.append(w + 2)

    header_line = "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    _emit(header_line)
    _emit("-" * len(header_line))
    for row in rows:
        _emit("".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))


def progress_count(label: str, count: int, *, every: int = 1000) -> None:
    if count <= 0:
        return
    if count == 1 or count % every == 0 or count < every:
        detail(f"{label}: {fmt_n(count)}", indent=4)
