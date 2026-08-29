"""Data cleaning utilities for TLE and DISCOS records."""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.config import COSPAR_ID_COL, EPOCH_COL, ORBITAL_ELEMENT_COLS

logger = logging.getLogger(__name__)


def normalize_cospar_id(series: pd.Series) -> pd.Series:
    """Normalize COSPAR IDs to uppercase YYYY-NNN[A-Z] format."""

    def _norm(val: object) -> str | None:
        if pd.isna(val):
            return None
        s = str(val).strip().upper()
        s = re.sub(r"\s+", "", s)
        m = re.match(r"^(\d{4})[- ]?(\d{1,3})([A-Z]{1,3})?$", s)
        if m:
            piece = m.group(3) or "A"
            return f"{m.group(1)}-{int(m.group(2)):03d}{piece[0]}"
        return s

    return series.map(_norm)


def clean_tle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean TLE history: normalize IDs, parse epochs, coerce numerics."""
    out = df.copy()
    if COSPAR_ID_COL not in out.columns:
        raise ValueError(f"TLE data missing required column: {COSPAR_ID_COL}")

    out[COSPAR_ID_COL] = normalize_cospar_id(out[COSPAR_ID_COL])
    out = out.dropna(subset=[COSPAR_ID_COL])

    if EPOCH_COL in out.columns:
        out[EPOCH_COL] = pd.to_datetime(out[EPOCH_COL], errors="coerce", utc=True)
        out = out.dropna(subset=[EPOCH_COL])

    for col in ORBITAL_ELEMENT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=[c for c in ORBITAL_ELEMENT_COLS if c in out.columns], how="any")
    dropped = before - len(out)
    if dropped:
        logger.warning("Dropped %d TLE rows with invalid orbital elements", dropped)

    logger.info("Cleaned TLE data: %d rows", len(out))
    return out


def clean_discos_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DISCOS metadata: normalize IDs, coerce physical dimensions."""
    out = df.copy()
    if COSPAR_ID_COL not in out.columns:
        raise ValueError(f"DISCOS data missing required column: {COSPAR_ID_COL}")

    out[COSPAR_ID_COL] = normalize_cospar_id(out[COSPAR_ID_COL])
    out = out.dropna(subset=[COSPAR_ID_COL])

    for col in ("length", "width", "height", "mass"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    logger.info("Cleaned DISCOS data: %d rows", len(out))
    return out
