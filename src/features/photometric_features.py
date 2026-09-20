"""Object-level photometric aggregates from per-track light curves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import LIGHTCURVE_MAG_COL, LIGHTCURVE_TIME_COL
from src.data.lightcurve_columns import extract_lightcurve_arrays, normalize_lightcurve_columns
from src.features.period_analysis import FAP_PERIODIC, _to_elapsed_seconds, lomb_scargle_periodogram

MIN_TRACK_POINTS = 20


def _peak_to_peak_amplitude(mags: np.ndarray) -> float:
    lo, hi = np.percentile(mags, [5.0, 95.0])
    return float(hi - lo)


def compute_object_photometry(track_dfs: list[pd.DataFrame]) -> dict[str, Any]:
    """
    Run LSP on each track (>20 points) and aggregate object-level photometry features.
    """
    periods: list[float] = []
    amplitudes: list[float] = []
    n_periodic = 0
    n_valid = 0

    for raw in track_dfs:
        if raw is None or len(raw) == 0:
            continue
        df = normalize_lightcurve_columns(raw)
        if LIGHTCURVE_TIME_COL not in df.columns or LIGHTCURVE_MAG_COL not in df.columns:
            continue
        ts_raw, mags_raw, _ = extract_lightcurve_arrays(df)
        raw_ts = np.asarray(ts_raw)
        if np.issubdtype(raw_ts.dtype, np.number):
            ts = raw_ts.astype(float)
            ts = ts - np.nanmin(ts)
        else:
            ts = _to_elapsed_seconds(raw_ts)
        mags = np.asarray(mags_raw, dtype=float)
        ok = np.isfinite(ts) & np.isfinite(mags)
        ts, mags = ts[ok], mags[ok]
        if len(mags) <= MIN_TRACK_POINTS:
            continue
        _, uniq = np.unique(ts, return_index=True)
        uniq = np.sort(uniq)
        ts, mags = ts[uniq], mags[uniq]
        if len(mags) <= MIN_TRACK_POINTS:
            continue

        n_valid += 1
        amplitudes.append(_peak_to_peak_amplitude(mags))
        span = float(ts.max() - ts.min()) if len(ts) > 1 else 1.0
        lsp = lomb_scargle_periodogram(
            ts,
            mags,
            min_period_sec=2.0,
            max_period_sec=max(span / 3.0, 2.0),
        )
        fap = float(lsp["fap"])
        if fap < FAP_PERIODIC and np.isfinite(lsp["lsp_period_sec"]):
            n_periodic += 1
            periods.append(float(lsp["lsp_period_sec"]))

    periodic_fraction = float(n_periodic / n_valid) if n_valid else 0.0
    median_period = float(np.median(periods)) if periods else None
    period_scatter = float(np.std(periods, ddof=0)) if len(periods) >= 2 else None
    median_amplitude = float(np.median(amplitudes)) if amplitudes else None

    return {
        "track_count": n_valid,
        "periodic_fraction": round(periodic_fraction, 4),
        "median_period_sec": round(median_period, 4) if median_period is not None else None,
        "period_scatter": round(period_scatter, 4) if period_scatter is not None else None,
        "median_amplitude": round(median_amplitude, 4) if median_amplitude is not None else None,
        # 1 when a majority of tracks are periodic (FAP < 0.01)
        "is_tumbling_consistent": int(periodic_fraction > 0.5),
    }
