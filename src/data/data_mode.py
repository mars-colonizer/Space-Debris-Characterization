"""Runtime DATA_MODE resolution (SYNTHETIC vs ACTUAL)."""

from __future__ import annotations

import os

from src.data.env import env_bool, load_project_env

VALID_MODES = frozenset({"SYNTHETIC", "ACTUAL"})
_runtime_override: str | None = None


def get_data_mode() -> str:
    """Return active data mode; honors dashboard runtime override then env."""
    load_project_env()
    if _runtime_override in VALID_MODES:
        return _runtime_override
    explicit = os.getenv("DATA_MODE", "").strip().upper()
    if explicit in VALID_MODES:
        return explicit
    # Legacy .env flag
    return "SYNTHETIC" if env_bool("USE_SYNTHETIC", default=True) else "ACTUAL"


def set_data_mode(mode: str) -> str:
    """Set runtime mode and mirror to process environment for subprocesses."""
    global _runtime_override
    normalized = mode.strip().upper()
    if normalized not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Expected SYNTHETIC or ACTUAL.")
    _runtime_override = normalized
    os.environ["DATA_MODE"] = normalized
    os.environ["USE_SYNTHETIC"] = "true" if normalized == "SYNTHETIC" else "false"
    return normalized


def is_synthetic() -> bool:
    return get_data_mode() == "SYNTHETIC"
