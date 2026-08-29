"""CSV and API data ingestion for TLE, DISCOS, and photometric sources."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW
from src.data.load_data import load_discos_metadata, load_tle_history
from src.data.storage import ensure_storage_dirs, save_to_sqlite

logger = logging.getLogger(__name__)


def load_photometric_observations(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path or DATA_RAW / "photometric_observations.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def ingest_raw_data(
    tle_path: Path | str | None = None,
    discos_path: Path | str | None = None,
    photometric_path: Path | str | None = None,
    persist_db: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files and optionally persist to unified SQLite storage."""
    ensure_storage_dirs()
    tle = load_tle_history(tle_path)
    discos = load_discos_metadata(discos_path)
    photo = load_photometric_observations(photometric_path)

    if persist_db:
        save_to_sqlite(tle, "tle_history")
        save_to_sqlite(discos, "discos_metadata")
        if not photo.empty:
            save_to_sqlite(photo, "photometric_observations")
        logger.info("Persisted raw data to data/database/rso_poc.db")

    return tle, discos, photo


def ingest_from_directory(raw_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load standard filenames from data/raw/."""
    raw_dir = Path(raw_dir or DATA_RAW)
    return ingest_raw_data(
        tle_path=raw_dir / "tle_history.csv",
        discos_path=raw_dir / "discos_metadata.csv",
        photometric_path=raw_dir / "photometric_observations.csv",
    )
