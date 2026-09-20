"""Light-curve quality filtering (Phase 2 FR5)."""

from __future__ import annotations

import numpy as np

MIN_POINTS = 10
MIN_SPAN_SEC = 60.0
MAX_MAG = 25.0


def filter_lightcurve(
    times_sec: np.ndarray | list,
    magnitudes: np.ndarray | list,
    errors: np.ndarray | list | None = None,
    *,
    min_points: int = MIN_POINTS,
    min_span_sec: float = MIN_SPAN_SEC,
    max_mag: float = MAX_MAG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int | float | bool]]:
    """
    Remove incomplete, non-finite, or out-of-range photometric samples.

    Raw apparent magnitudes are typically ≥0; median-detrended / differential
    series center near 0 and may be negative — do not cull those with mag≥0.
    """
    ts = np.asarray(times_sec)
    mags = np.asarray(magnitudes, dtype=float)
    errs = np.asarray(errors, dtype=float) if errors is not None else None

    stats: dict[str, int | float | bool] = {
        "input_points": len(mags),
        "rejected_nonfinite": 0,
        "rejected_mag_range": 0,
        "accepted": 0,
        "span_sec": 0.0,
        "passed": False,
    }

    if len(mags) == 0:
        return np.array([]), np.array([]), None, stats

    if np.issubdtype(ts.dtype, np.number):
        t = ts.astype(float)
    else:
        import pandas as pd

        t = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
        if t.notna().sum() >= 2:
            t = (t - t.min()).dt.total_seconds().to_numpy(dtype=float)
        else:
            t = np.arange(len(mags), dtype=float)

    valid = np.isfinite(t) & np.isfinite(mags)
    stats["rejected_nonfinite"] = int((~valid).sum())
    t, mags = t[valid], mags[valid]
    if errs is not None:
        errs = errs[valid]

    if len(mags):
        # Detrended/differential: median ~0 or any negative samples → |mag| bound only.
        # Raw apparent mag: keep classic 0..max_mag window.
        med = float(np.median(mags))
        looks_detrended = med < 1.0 or bool(np.any(mags < 0.0))
        if looks_detrended:
            mag_ok = np.abs(mags) <= max_mag
        else:
            mag_ok = (mags >= 0.0) & (mags <= max_mag)
        stats["rejected_mag_range"] = int((~mag_ok).sum())
        t, mags = t[mag_ok], mags[mag_ok]
        if errs is not None:
            errs = errs[mag_ok]

    span = float(t.max() - t.min()) if len(t) > 1 else 0.0
    stats["span_sec"] = span
    stats["accepted"] = len(mags)
    stats["passed"] = len(mags) >= min_points and span >= min_span_sec
    return t, mags, errs, stats


if __name__ == "__main__":
    t = np.linspace(0, 300, 50)
    m = 10 + 0.5 * np.sin(t)
    m[0] = np.nan
    m[1] = 99.0
    ft, fm, _, st = filter_lightcurve(t, m)
    assert st["passed"] is True and len(fm) == 48, st
    ft2, fm2, _, st2 = filter_lightcurve(t[:5], m[:5])
    assert st2["passed"] is False, st2
    # median-detrended must keep negatives
    finite = np.isfinite(m)
    md = m[finite] - np.median(m[finite])
    md = md[np.abs(md) < 50]
    _, _, _, st3 = filter_lightcurve(np.arange(len(md), dtype=float), md)
    assert st3["rejected_mag_range"] == 0 and st3["accepted"] == len(md), st3
    print("lightcurve_quality self-check OK:", st)
