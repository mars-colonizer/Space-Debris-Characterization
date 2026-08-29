"""SQLite schema manager for RSO catalog and photometric observations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DATA_DATABASE

DB_PATH = DATA_DATABASE / "rso_poc.db"

RSO_CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS rso_catalog (
    cospar_id TEXT PRIMARY KEY,
    object_class TEXT,
    true_length REAL,
    true_width REAL,
    true_height REAL,
    true_mass REAL,
    true_shape TEXT,
    true_period REAL,
    true_tumbling INTEGER
);
"""

PHOTOMETRIC_DDL = """
CREATE TABLE IF NOT EXISTS photometric_observations (
    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cospar_id TEXT NOT NULL,
    mag_mean REAL,
    mag_std REAL,
    delta_mag REAL,
    estimated_period_sec REAL,
    apparent_shape_score REAL,
    is_tumbling INTEGER,
    FOREIGN KEY (cospar_id) REFERENCES rso_catalog(cospar_id)
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database(db_path: Path | None = None) -> Path:
    """Create catalog + photometric tables if missing."""
    path = db_path or DB_PATH
    with get_connection(path) as conn:
        conn.executescript(RSO_CATALOG_DDL)
        conn.executescript(PHOTOMETRIC_DDL)
    return path


def upsert_catalog(df: pd.DataFrame, db_path: Path | None = None) -> None:
    init_database(db_path)
    cols = [
        "cospar_id", "object_class", "true_length", "true_width", "true_height",
        "true_mass", "true_shape", "true_period", "true_tumbling",
    ]
    data = df[[c for c in cols if c in df.columns]].copy()
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM rso_catalog")
        data.to_sql("rso_catalog", conn, if_exists="append", index=False)


def upsert_photometric(df: pd.DataFrame, db_path: Path | None = None) -> None:
    init_database(db_path)
    cols = [
        "cospar_id", "mag_mean", "mag_std", "delta_mag",
        "estimated_period_sec", "apparent_shape_score", "is_tumbling",
    ]
    data = df[[c for c in cols if c in df.columns]].copy()
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM photometric_observations")
        data.to_sql("photometric_observations", conn, if_exists="append", index=False)


def load_catalog(db_path: Path | None = None) -> pd.DataFrame:
    path = db_path or DB_PATH
    if not path.exists():
        return pd.DataFrame()
    init_database(path)
    with get_connection(path) as conn:
        return pd.read_sql("SELECT * FROM rso_catalog", conn)


def load_photometric(db_path: Path | None = None) -> pd.DataFrame:
    path = db_path or DB_PATH
    if not path.exists():
        return pd.DataFrame()
    init_database(path)
    with get_connection(path) as conn:
        return pd.read_sql("SELECT * FROM photometric_observations", conn)
