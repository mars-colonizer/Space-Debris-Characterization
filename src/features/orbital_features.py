"""Orbital feature engineering from TLE history."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.config import COSPAR_ID_COL, EPOCH_COL, ORBITAL_ELEMENT_COLS

logger = logging.getLogger(__name__)

# Earth gravitational parameter (km³/s²) for orbital period
MU_EARTH = 398600.4418


def _orbital_period_days(semi_major_axis_km: pd.Series) -> pd.Series:
    """Compute orbital period in days from semi-major axis (km)."""
    # T = 2π sqrt(a³/μ); convert seconds to days
    period_sec = 2 * math.pi * np.sqrt(np.power(semi_major_axis_km, 3) / MU_EARTH)
    return period_sec / 86400.0


def compute_orbital_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer orbital features from merged TLE history.

    Assumptions:
    - Records are grouped by cospar_id (or object_id fallback).
    - epoch column is parseable datetime, sorted ascending per object.
    - Keplerian elements are in degrees (inclination, raan, arg_perigee, mean_anomaly)
      and km (semi_major_axis); eccentricity is dimensionless.

    Time-series features (inclination drift, SMA decay) require ≥2 epochs per object.
    Objects with insufficient history get NaN for drift/decay; imputed later on train only.
    """
    if COSPAR_ID_COL not in df.columns:
        raise ValueError(f"Missing {COSPAR_ID_COL} for orbital feature grouping")

    group_col = COSPAR_ID_COL
    out = df.copy()

    # Snapshot features from latest epoch per object
    out = out.sort_values([group_col, EPOCH_COL] if EPOCH_COL in out.columns else [group_col])
    latest = out.groupby(group_col, as_index=False).last()

    if "semi_major_axis" in latest.columns:
        latest["orbital_period_days"] = _orbital_period_days(latest["semi_major_axis"])

    # Time-series deltas per object
    if EPOCH_COL in out.columns:
        out = out.sort_values([group_col, EPOCH_COL])
        grouped = out.groupby(group_col)

        if "inclination" in out.columns:
            dt_days = grouped[EPOCH_COL].diff().dt.total_seconds().div(86400)
            out["inclination_drift_deg_per_day"] = grouped["inclination"].diff().div(dt_days)
            out["inclination_drift_deg_per_day"] = out["inclination_drift_deg_per_day"].replace([np.inf, -np.inf], np.nan)
        if "semi_major_axis" in out.columns:
            dt_days = grouped[EPOCH_COL].diff().dt.total_seconds().div(86400)
            out["sma_decay_km_per_day"] = grouped["semi_major_axis"].diff().div(dt_days)
            out["sma_decay_km_per_day"] = out["sma_decay_km_per_day"].replace([np.inf, -np.inf], np.nan)

        # Take latest delta values (most recent interval)
        delta_cols = [c for c in ("inclination_drift_deg_per_day", "sma_decay_km_per_day") if c in out.columns]
        if delta_cols:
            deltas = out.groupby(group_col)[delta_cols].last().reset_index()
            latest = latest.merge(deltas, on=group_col, how="left")

        # Epoch span and observation count
        span = out.groupby(group_col).agg(
            epoch_count=(EPOCH_COL, "count"),
            epoch_span_days=(EPOCH_COL, lambda s: (s.max() - s.min()).total_seconds() / 86400),
        ).reset_index()
        latest = latest.merge(span, on=group_col, how="left")
    else:
        logger.warning("No epoch column — skipping time-series orbital features")
        latest["epoch_count"] = 1
        latest["epoch_span_days"] = 0.0

    feature_cols = [group_col] + [c for c in latest.columns if c not in df.columns or c in ORBITAL_ELEMENT_COLS]
    feature_cols = list(dict.fromkeys(feature_cols))  # dedupe preserve order

    logger.info(
        "Orbital features: %d objects, %d feature columns",
        len(latest),
        len(latest.columns) - 1,
    )
    return latest
