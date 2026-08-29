"""Unified file-based data storage for Phase 2 POC."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DATA_DATABASE, DATA_PROCESSED, DATA_RAW

logger = logging.getLogger(__name__)

DB_PATH = DATA_DATABASE / "rso_poc.db"


def ensure_storage_dirs() -> None:
    """Create data directories if missing."""
    for d in (DATA_RAW, DATA_PROCESSED, DATA_DATABASE):
        d.mkdir(parents=True, exist_ok=True)


def save_to_sqlite(df: pd.DataFrame, table: str, db_path: Path | None = None) -> None:
    """Persist a DataFrame to the project SQLite database."""
    db_path = db_path or DB_PATH
    ensure_storage_dirs()
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
    logger.info("Saved %d rows to %s:%s", len(df), db_path, table)


def load_from_sqlite(table: str, db_path: Path | None = None) -> pd.DataFrame:
    """Load a table from the project SQLite database."""
    db_path = db_path or DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def save_dataset_meta(meta: dict, path: Path | None = None) -> None:
    path = path or DATA_PROCESSED / "dataset_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
