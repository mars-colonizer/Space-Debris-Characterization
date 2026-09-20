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

OBJECTS_DDL = """
CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT PRIMARY KEY,
    cospar_id TEXT,
    object_name TEXT
);
"""

PHOTOMETRIC_DDL = """
CREATE TABLE IF NOT EXISTS photometric_observations (
    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT,
    cospar_id TEXT,
    mag_mean REAL,
    mag_std REAL,
    delta_mag REAL,
    estimated_period_sec REAL,
    lsp_period_sec REAL,
    pdm_period_sec REAL,
    pdm_theta REAL,
    apparent_shape_score REAL,
    is_tumbling INTEGER,
    quality_points INTEGER,
    quality_span_sec REAL
);
"""

LIGHT_CURVES_DDL = """
CREATE TABLE IF NOT EXISTS light_curves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT NOT NULL,
    cospar_id TEXT,
    time TEXT NOT NULL,
    mag REAL,
    mag_err REAL
);
"""

PERIODOGRAMS_DDL = """
CREATE TABLE IF NOT EXISTS periodograms (
    object_id TEXT PRIMARY KEY,
    cospar_id TEXT,
    lsp_period_sec REAL,
    pdm_period_sec REAL,
    pdm_theta REAL,
    extracted_period_sec REAL,
    is_tumbling INTEGER,
    periodogram_json TEXT,
    folded_json TEXT
);
"""

FOLDED_LIGHT_CURVES_DDL = """
CREATE TABLE IF NOT EXISTS folded_light_curves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT NOT NULL,
    cospar_id TEXT,
    period_sec REAL,
    phase REAL,
    mag REAL
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_has_columns(conn: sqlite3.Connection, table: str, columns: set[str]) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not rows:
        return False
    existing = {row[1] for row in rows}
    return columns.issubset(existing)


def _ensure_table(conn: sqlite3.Connection, table: str, ddl: str, required_cols: set[str]) -> None:
    if not _table_has_columns(conn, table, required_cols):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(ddl)


def _light_curves_needs_migration(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA table_info(light_curves)").fetchall()
    if not rows:
        return False
    cols = {row[1]: row for row in rows}
    if "object_id" not in cols:
        return True
    if not {"time", "mag"}.issubset(cols):
        return True
    cospar = cols.get("cospar_id")
    return cospar is not None and bool(cospar[3])


def _ensure_object_name_column(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(objects)").fetchall()
    if rows and "object_name" not in {row[1] for row in rows}:
        conn.execute("ALTER TABLE objects ADD COLUMN object_name TEXT")


def init_database(db_path: Path | None = None) -> Path:
    """Create catalog + photometric tables if missing."""
    path = db_path or DB_PATH
    with get_connection(path) as conn:
        conn.executescript(RSO_CATALOG_DDL)
        conn.executescript(OBJECTS_DDL)
        _ensure_object_name_column(conn)
        if _light_curves_needs_migration(conn):
            conn.execute("DROP TABLE IF EXISTS light_curves")
        _ensure_table(
            conn,
            "photometric_observations",
            PHOTOMETRIC_DDL,
            {"object_id", "cospar_id", "lsp_period_sec", "pdm_period_sec", "pdm_theta", "quality_points"},
        )
        _ensure_table(
            conn,
            "light_curves",
            LIGHT_CURVES_DDL,
            {"object_id", "time", "mag"},
        )
        _ensure_table(
            conn,
            "periodograms",
            PERIODOGRAMS_DDL,
            {"object_id", "cospar_id", "periodogram_json", "folded_json"},
        )
        _ensure_table(
            conn,
            "folded_light_curves",
            FOLDED_LIGHT_CURVES_DDL,
            {"object_id", "phase", "mag"},
        )
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


def upsert_objects(df: pd.DataFrame, db_path: Path | None = None) -> None:
    """Object-level index (NORAD ID → COSPAR ID) per Phase 2 database design."""
    init_database(db_path)
    cols = ["object_id", "cospar_id", "object_name"]
    data = df[[c for c in cols if c in df.columns]].drop_duplicates(subset=["object_id"]).copy()
    if data.empty or "object_id" not in data.columns:
        return
    data["object_id"] = data["object_id"].astype(str)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM objects")
        data.to_sql("objects", conn, if_exists="append", index=False)


def upsert_photometric(df: pd.DataFrame, db_path: Path | None = None) -> None:
    init_database(db_path)
    cols = [
        "object_id", "cospar_id", "mag_mean", "mag_std", "delta_mag",
        "estimated_period_sec", "lsp_period_sec", "pdm_period_sec", "pdm_theta",
        "apparent_shape_score", "is_tumbling", "quality_points", "quality_span_sec",
    ]
    data = df[[c for c in cols if c in df.columns]].copy()
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM photometric_observations")
        data.to_sql("photometric_observations", conn, if_exists="append", index=False)


def upsert_light_curves(df: pd.DataFrame, db_path: Path | None = None) -> None:
    from src.data.lightcurve_columns import normalize_lightcurve_columns
    from src.config import LIGHTCURVE_ERR_COL, LIGHTCURVE_MAG_COL, LIGHTCURVE_TIME_COL

    init_database(db_path)
    data = normalize_lightcurve_columns(df)
    cols = ["object_id", "cospar_id", LIGHTCURVE_TIME_COL, LIGHTCURVE_MAG_COL, LIGHTCURVE_ERR_COL]
    data = data[[c for c in cols if c in data.columns]].copy()
    if "object_id" not in data.columns:
        raise ValueError("light_curves require object_id (NORAD)")
    data["object_id"] = data["object_id"].astype(str)
    if "cospar_id" in data.columns:
        data["cospar_id"] = data["cospar_id"].where(data["cospar_id"].notna(), None)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM light_curves")
        data.to_sql("light_curves", conn, if_exists="append", index=False)


def upsert_folded_light_curves(df: pd.DataFrame, db_path: Path | None = None) -> None:
    from src.config import LIGHTCURVE_MAG_COL

    init_database(db_path)
    data = df.copy()
    if "magnitude" in data.columns and LIGHTCURVE_MAG_COL not in data.columns:
        data = data.rename(columns={"magnitude": LIGHTCURVE_MAG_COL})
    cols = ["object_id", "cospar_id", "period_sec", "phase", LIGHTCURVE_MAG_COL]
    data = data[[c for c in cols if c in data.columns]].copy()
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM folded_light_curves")
        data.to_sql("folded_light_curves", conn, if_exists="append", index=False)


def upsert_periodograms(df: pd.DataFrame, db_path: Path | None = None) -> None:
    init_database(db_path)
    cols = [
        "object_id", "cospar_id", "lsp_period_sec", "pdm_period_sec", "pdm_theta",
        "extracted_period_sec", "is_tumbling", "periodogram_json", "folded_json",
    ]
    data = df[[c for c in cols if c in df.columns]].copy()
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM periodograms")
        data.to_sql("periodograms", conn, if_exists="append", index=False)


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


def load_light_curves(db_path: Path | None = None) -> pd.DataFrame:
    from src.data.lightcurve_columns import normalize_lightcurve_columns

    path = db_path or DB_PATH
    if not path.exists():
        return pd.DataFrame()
    init_database(path)
    with get_connection(path) as conn:
        df = pd.read_sql("SELECT * FROM light_curves", conn)
    return normalize_lightcurve_columns(df)


def load_objects(db_path: Path | None = None) -> pd.DataFrame:
    path = db_path or DB_PATH
    if not path.exists():
        return pd.DataFrame()
    init_database(path)
    with get_connection(path) as conn:
        return pd.read_sql("SELECT * FROM objects", conn)
