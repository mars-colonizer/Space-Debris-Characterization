"""Load environment variables from .env (never commit .env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.config import PROJECT_ROOT

_ENV_LOADED = False


def load_project_env() -> None:
    """Load .env from project root if present."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    _ENV_LOADED = True


def env_bool(name: str, default: bool = False) -> bool:
    load_project_env()
    val = os.getenv(name, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    load_project_env()
    raw = os.getenv(name)
    return int(raw) if raw else default


def env_present(name: str) -> bool:
    load_project_env()
    return bool(os.getenv(name, "").strip())


def require_env(name: str) -> str:
    load_project_env()
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(
            f"Missing {name}. Copy .env.example to .env and set credentials."
        )
    return val
