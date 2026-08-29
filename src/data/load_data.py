"""Load raw TLE history and DISCOS metadata from CSV files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW

logger = logging.getLogger(__name__)

DEFAULT_TLE_PATH = DATA_RAW / "tle_history.csv"
DEFAULT_DISCOS_PATH = DATA_RAW / "discos_metadata.csv"


def load_tle_history(path: Path | str | None = None) -> pd.DataFrame:
    """Load TLE/orbital history CSV."""
    path = Path(path or DEFAULT_TLE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"TLE history not found at {path}. "
            "Place tle_history.csv in data/raw/ or run scripts/fetch_data.py / scripts/generate_sample_data.py."
        )
    df = pd.read_csv(path)
    logger.info("Loaded TLE history: %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df


def load_discos_metadata(path: Path | str | None = None) -> pd.DataFrame:
    """Load DISCOS metadata CSV."""
    path = Path(path or DEFAULT_DISCOS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"DISCOS metadata not found at {path}. "
            "Place discos_metadata.csv in data/raw/ or run scripts/fetch_data.py / scripts/generate_sample_data.py."
        )
    df = pd.read_csv(path)
    logger.info("Loaded DISCOS metadata: %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df
