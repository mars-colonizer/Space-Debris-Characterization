"""Parse pipeline log lines and sanitize sensitive output."""

from __future__ import annotations

import re

from dashboard.stages import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_WARNING,
)

_INGEST_STEP_MAP = {
    1: "ingestion",
    2: "ingestion",
    3: "ingestion",
    4: "identifiers",
    5: "metadata",
    6: "metadata",
    7: "database",
    8: "database",
}

_PROCESS_STEP_MAP = {
    1: "processing",
    2: "processing",
    3: "processing",
    4: "period",
    5: "period",
    6: "period",
    7: "database",
    8: "poc",
}

_RE_INGEST = re.compile(r"\[(\d+)/8\]")
_RE_PROCESS = re.compile(r"\[(\d+)/8\]")

_REDACT = [
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(DISCOS_TOKEN\s*=\s*)\S+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(SPACE_TRACK_PASSWORD\s*=\s*)\S+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(KEEPTRACK_API_KEY\s*=\s*)\S+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(X-API-Key:\s*)\S+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(password['\"]?\s*[:=]\s*)\S+", re.I), r"\1[REDACTED]"),
    (re.compile(r"\bkt_[A-Za-z0-9]+\b"), "[REDACTED]"),
]


def sanitize_log_line(line: str) -> str:
    out = line.rstrip("\n")
    for pattern, repl in _REDACT:
        out = pattern.sub(repl, out)
    return out


def _is_error_line(line: str) -> bool:
    if line.strip().startswith("WARNING:") or "[WARNING]" in line:
        return False
    return "[ERROR]" in line


def _set_running(stage_status: dict[str, str], active: str) -> None:
    for k, v in stage_status.items():
        if v == STATUS_RUNNING and k != active:
            pass  # allow parallel sub-steps within same exec stage
    stage_status[active] = STATUS_RUNNING


def _advance_complete(stage_status: dict[str, str], *keys: str) -> None:
    for key in keys:
        if stage_status.get(key) == STATUS_RUNNING:
            stage_status[key] = STATUS_COMPLETE


def apply_log_line(line: str, stage_status: dict[str, str], warnings: list[str]) -> str | None:
    if "[WARNING]" in line or line.strip().startswith("WARNING:"):
        warnings.append(line.strip())

    if _is_error_line(line):
        for key in list(stage_status.keys()):
            if stage_status[key] == STATUS_RUNNING:
                stage_status[key] = STATUS_FAILED
        return None

    m = _RE_INGEST.search(line)
    if m:
        active = _INGEST_STEP_MAP.get(int(m.group(1)))
        if active:
            _set_running(stage_status, active)
            return active

    m = _RE_PROCESS.search(line)
    if m:
        active = _PROCESS_STEP_MAP.get(int(m.group(1)))
        if active:
            _set_running(stage_status, active)
            return active

    if "INGESTION COMPLETE" in line:
        for k in ("ingestion", "identifiers", "metadata", "database"):
            if stage_status.get(k) != STATUS_FAILED:
                stage_status[k] = STATUS_COMPLETE
        return None

    if "PHOTOMETRIC PROCESSING COMPLETE" in line:
        for k in ("processing", "period", "poc", "database"):
            if stage_status.get(k) != STATUS_FAILED:
                stage_status[k] = STATUS_COMPLETE
        return None

    if "PHASE 2 — DATA INGESTION" in line or ("Active data mode" in line and "[1/" in line):
        stage_status["ingestion"] = STATUS_RUNNING
        return "ingestion"

    if "PHASE 2 — PHOTOMETRIC PROCESSING" in line:
        _advance_complete(stage_status, "ingestion", "identifiers", "metadata", "database")
        stage_status["processing"] = STATUS_RUNNING
        return "processing"

    if "Resolving COSPAR" in line:
        _advance_complete(stage_status, "ingestion")
        stage_status["identifiers"] = STATUS_RUNNING
        return "identifiers"

    if any(
        n in line
        for n in (
            "Downloading MMT-9",
            "MMT raw observations",
            "Loading MMT-9 candidate",
        )
    ):
        stage_status["ingestion"] = STATUS_RUNNING
        return "ingestion"

    if any(
        n in line
        for n in (
            "Fetching Space-Track",
            "Fetching DISCOS",
            "DISCOS batch",
            "Space-Track GP fetch",
            "KeepTrack-resolved",
        )
    ):
        _advance_complete(stage_status, "ingestion", "identifiers")
        stage_status["metadata"] = STATUS_RUNNING
        return "metadata"

    if "TLE and DISCOS will not be fetched" in line or "skipping Space-Track and DISCOS" in line:
        stage_status["metadata"] = STATUS_WARNING
        return "metadata"

    if any(
        n in line
        for n in (
            "Quality filtering",
            "Retained after quality filter",
        )
    ):
        stage_status["processing"] = STATUS_RUNNING
        return "processing"

    if any(
        n in line
        for n in (
            "Lomb–Scargle",
            "PDM-validated",
            "Phase-folded",
        )
    ):
        stage_status["period"] = STATUS_RUNNING
        return "period"

    if any(
        n in line
        for n in (
            "Persisting light curves",
            "object index",
            "poc_plots",
            "periodograms/",
        )
    ):
        stage_status["database"] = STATUS_RUNNING
        return "database"

    if "period_analysis_summary.json" in line or "PoC processing summary" in line:
        stage_status["poc"] = STATUS_RUNNING
        return "poc"

    return None


def mark_exec_start(stage_status: dict[str, str], step_keys: list[str]) -> None:
    if step_keys:
        stage_status[step_keys[0]] = STATUS_RUNNING


def mark_exec_complete(stage_status: dict[str, str], step_keys: list[str], had_warnings: bool) -> None:
    from dashboard.stages import STATUS_COMPLETE, STATUS_WARNING, STATUS_WAITING

    status = STATUS_WARNING if had_warnings else STATUS_COMPLETE
    for key in step_keys:
        if stage_status.get(key) not in (STATUS_RUNNING, STATUS_WAITING):
            continue
        # Central DB persist succeeded if the script exited 0 — don't paint it WARNING
        # for unrelated soft warnings (MMT 403 cache fallback, KeepTrack fills, etc.).
        if key == "database":
            stage_status[key] = STATUS_COMPLETE
        else:
            stage_status[key] = status


def mark_exec_failed(stage_status: dict[str, str], step_keys: list[str]) -> None:
    from dashboard.stages import STATUS_STOPPED

    for key in step_keys:
        if stage_status.get(key) == STATUS_RUNNING:
            stage_status[key] = STATUS_STOPPED
