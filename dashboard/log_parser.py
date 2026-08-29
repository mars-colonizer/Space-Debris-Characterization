"""Parse pipeline log lines and sanitize sensitive output."""

from __future__ import annotations

import re

from dashboard.stages import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_WARNING,
)

_PREP_STEP_MAP = {
    1: "prep",
    2: "prep",
    3: "quality",
    4: "quality",
    5: "quality",
    6: "features",
    7: "leakage",
    8: "prep",
    9: "split",
    10: "split",
}

_RE_PREP = re.compile(r"\[(\d+)/10\]")

_REDACT = [
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(DISCOS_TOKEN\s*=\s*)\S+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(SPACE_TRACK_PASSWORD\s*=\s*)\S+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(password['\"]?\s*[:=]\s*)\S+", re.I), r"\1[REDACTED]"),
]


def sanitize_log_line(line: str) -> str:
    out = line.rstrip("\n")
    for pattern, repl in _REDACT:
        out = pattern.sub(repl, out)
    return out


def apply_log_line(line: str, stage_status: dict[str, str], warnings: list[str]) -> str | None:
    """Update stage_status from a log line. Returns current running step key if detected."""
    if "[WARNING]" in line:
        warnings.append(line.strip())

    if "[ERROR]" in line or "FAILED" in line:
        for key in list(stage_status.keys()):
            if stage_status[key] == STATUS_RUNNING:
                stage_status[key] = STATUS_FAILED
        return None

    m = _RE_PREP.search(line)
    if m:
        step_num = int(m.group(1))
        active = _PREP_STEP_MAP.get(step_num)
        if active:
            for k, v in stage_status.items():
                if v == STATUS_RUNNING and k != active:
                    if k in ("prep", "quality", "leakage", "features", "split"):
                        stage_status[k] = STATUS_COMPLETE
            stage_status[active] = STATUS_RUNNING
            return active

    if "DATA PREPARATION COMPLETE" in line:
        for k in ("prep", "quality", "leakage", "features", "split"):
            stage_status[k] = STATUS_COMPLETE
        return None

    if "INGESTION COMPLETE" in line:
        stage_status["ingestion"] = STATUS_COMPLETE
        return None

    if "STAGE 1 — MODEL COMPARISON" in line:
        stage_status["stage1"] = STATUS_COMPLETE

    if "PHASE 2 — REAL DATA INGESTION" in line or "[1/6]" in line:
        stage_status["ingestion"] = STATUS_RUNNING
        return "ingestion"

    if "[5/6]" in line and "Fetching GP" in line:
        stage_status["ingestion"] = STATUS_RUNNING
        return "ingestion"

    if "STAGE 2 — SUMMARY" in line:
        stage_status["stage2"] = STATUS_COMPLETE

    if "END-TO-END INFERENCE COMPLETE" in line:
        stage_status["inference"] = STATUS_COMPLETE
        stage_status["final"] = STATUS_COMPLETE

    if "MODEL 1/3" in line or "MODEL 2/3" in line or "MODEL 3/3" in line:
        stage_status["stage1"] = STATUS_RUNNING
        return "stage1"

    if "CLASS 1/" in line or "STAGE 2 — PHYSICAL" in line:
        stage_status["stage2"] = STATUS_RUNNING
        return "stage2"

    if "Insufficient samples" in line or "insufficient" in line.lower():
        if "[WARNING]" in line:
            stage_status.setdefault("_has_warnings", STATUS_WARNING)

    return None


def mark_exec_start(stage_status: dict[str, str], step_keys: list[str]) -> None:
    for key in step_keys:
        stage_status[key] = STATUS_RUNNING


def mark_exec_complete(stage_status: dict[str, str], step_keys: list[str], had_warnings: bool) -> None:
    status = STATUS_WARNING if had_warnings else STATUS_COMPLETE
    for key in step_keys:
        stage_status[key] = status


def mark_exec_failed(stage_status: dict[str, str], step_keys: list[str]) -> None:
    for key in step_keys:
        if stage_status.get(key) == STATUS_RUNNING:
            stage_status[key] = STATUS_FAILED
