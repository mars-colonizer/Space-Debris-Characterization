"""Canonical light-curve column names (time series + mag)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LIGHTCURVE_ERR_COL, LIGHTCURVE_MAG_COL, LIGHTCURVE_TIME_COL

_TIME_ALIASES = frozenset({"time", "timestamp", "epoch", "mjd", "time_sec", "times"})
_MAG_ALIASES = frozenset({"mag", "magnitude", "visual_magnitude", "vmag"})
_ERR_ALIASES = frozenset({"mag_err", "error", "err", "mag_error", "magnitude_errors"})


def _pick_col(df: pd.DataFrame, aliases: frozenset[str], canonical: str) -> str | None:
    if canonical in df.columns:
        return canonical
    for col in df.columns:
        if col.lower() in aliases:
            return col
    return None


def normalize_lightcurve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy timestamp/magnitude columns to canonical time/mag/mag_err."""
    if df.empty:
        return df
    out = df.copy()
    renames: dict[str, str] = {}
    time_col = _pick_col(out, _TIME_ALIASES, LIGHTCURVE_TIME_COL)
    mag_col = _pick_col(out, _MAG_ALIASES, LIGHTCURVE_MAG_COL)
    err_col = _pick_col(out, _ERR_ALIASES, LIGHTCURVE_ERR_COL)
    if time_col and time_col != LIGHTCURVE_TIME_COL:
        renames[time_col] = LIGHTCURVE_TIME_COL
    if mag_col and mag_col != LIGHTCURVE_MAG_COL:
        renames[mag_col] = LIGHTCURVE_MAG_COL
    if err_col and err_col != LIGHTCURVE_ERR_COL:
        renames[err_col] = LIGHTCURVE_ERR_COL
    return out.rename(columns=renames)


def extract_lightcurve_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (time series, mag, optional mag_err) from a light-curve frame."""
    norm = normalize_lightcurve_columns(df)
    if LIGHTCURVE_TIME_COL not in norm.columns or LIGHTCURVE_MAG_COL not in norm.columns:
        raise ValueError(f"light curve requires {LIGHTCURVE_TIME_COL} and {LIGHTCURVE_MAG_COL}")
    times = norm[LIGHTCURVE_TIME_COL].to_numpy()
    mags = norm[LIGHTCURVE_MAG_COL].to_numpy(dtype=float)
    errs = (
        norm[LIGHTCURVE_ERR_COL].to_numpy(dtype=float)
        if LIGHTCURVE_ERR_COL in norm.columns
        else None
    )
    return times, mags, errs


if __name__ == "__main__":
    sample = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "magnitude": [10.5], "error": [0.05]})
    out = normalize_lightcurve_columns(sample)
    assert list(out.columns) == ["time", "mag", "mag_err"], list(out.columns)
    t, m, e = extract_lightcurve_arrays(out)
    assert len(t) == 1 and m[0] == 10.5
    print("lightcurve_columns self-check OK")
