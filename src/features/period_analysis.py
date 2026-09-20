"""Rotation period analysis — Lomb-Scargle (Astropy), PDM validation, phase folding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

TUMBLING_POWER_RATIO = 0.03  # legacy alias
PDM_NBINS = 10
PDM_PERIOD_TOL = 0.15  # scan ±15% around LSP candidate
FAP_PERIODIC = 0.01  # tracks with FAP below this are treated as periodic
MAX_FREQ_BINS = 100_000  # ponytail: O(n·m) LSP; upgrade = coarse-to-fine zoom around peaks


def _frequency_grid(
    times_sec: np.ndarray,
    min_period_sec: float = 1.0,
    max_period_sec: float = 3600.0,
) -> np.ndarray:
    """Span-aware cyclic-frequency grid (Hz): spacing ≤ 1/(10·span)."""
    ts = np.asarray(times_sec, dtype=float)
    span = float(ts.max() - ts.min()) if len(ts) > 1 else 1.0
    if span < 1e-9:
        span = 1.0
    dt = float(np.median(np.diff(np.sort(ts)))) if len(ts) > 1 else 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0

    nyquist = 0.5 / dt
    f_min = max(1.0 / span, 1.0 / float(max_period_sec))
    f_max = min(nyquist, 1.0 / float(min_period_sec))
    if f_max <= f_min:
        f_max = f_min * 1.01

    df = 1.0 / (10.0 * span)
    n_freq = int(np.ceil((f_max - f_min) / df)) + 1
    n_freq = max(n_freq, 3)
    if n_freq > MAX_FREQ_BINS:
        # ponytail: long-span grids would be O(10^8); cap bins, keep endpoints
        n_freq = MAX_FREQ_BINS
    return np.linspace(f_min, f_max, n_freq)


def _to_elapsed_seconds(timestamps: np.ndarray | list) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if ts.notna().sum() >= 2:
        return (ts - ts.min()).dt.total_seconds().to_numpy(dtype=float)
    numeric = pd.to_numeric(pd.Series(timestamps), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).sum() >= 2:
        return numeric - np.nanmin(numeric)
    return np.arange(len(timestamps), dtype=float)


def lomb_scargle_periodogram(
    times_sec: np.ndarray,
    magnitudes: np.ndarray,
    min_period_sec: float = 1.0,
    max_period_sec: float = 3600.0,
) -> dict[str, np.ndarray | float]:
    """Astropy Lomb-Scargle periodogram with Baluev false-alarm probability."""
    ts = np.asarray(times_sec, dtype=float)
    mags = np.asarray(magnitudes, dtype=float)
    valid = np.isfinite(ts) & np.isfinite(mags)
    ts, mags = ts[valid], mags[valid]
    freqs = _frequency_grid(ts, min_period_sec=min_period_sec, max_period_sec=max_period_sec)

    empty = {
        "frequencies": freqs,
        "power": np.zeros(len(freqs)),
        "lsp_period_sec": float("nan"),
        "peak_power": 0.0,
        "fap": 1.0,
    }
    if len(mags) < 3:
        return empty

    t_span = float(ts.max() - ts.min())
    if t_span < 1.0:
        ts = np.linspace(0.0, max(len(ts) - 1, 1), len(ts))

    try:
        # Astropy frequencies are cyclic (1/time), not angular
        ls = LombScargle(ts, mags)
        power = np.asarray(ls.power(freqs), dtype=float)
        peak_idx = int(np.argmax(power))
        peak_power = float(power[peak_idx])
        fap = float(
            ls.false_alarm_probability(
                peak_power,
                method="baluev",
                minimum_frequency=float(freqs.min()),
                maximum_frequency=float(freqs.max()),
            )
        )
    except Exception:
        return empty

    f0 = float(freqs[peak_idx])
    p_min = 1.0 / freqs[-1]
    p_max = 1.0 / freqs[0]
    lsp_period = float(np.clip(1.0 / f0 if f0 > 0 else p_max, p_min, p_max))
    if not np.isfinite(fap):
        fap = 1.0
    return {
        "frequencies": freqs,
        "power": power,
        "lsp_period_sec": lsp_period,
        "peak_power": peak_power,
        "fap": fap,
    }


def pdm_theta(times_sec: np.ndarray, magnitudes: np.ndarray, period_sec: float) -> float:
    """
    Stellingwerf-style Phase Dispersion Minimization statistic.
    Lower theta => more coherent folding at this period.
    """
    ts = np.asarray(times_sec, dtype=float)
    mags = np.asarray(magnitudes, dtype=float)
    valid = np.isfinite(ts) & np.isfinite(mags)
    ts, mags = ts[valid], mags[valid]
    if len(mags) < PDM_NBINS or period_sec <= 0:
        return 1.0

    phases = (ts % period_sec) / period_sec
    total_var = float(np.var(mags))
    if total_var < 1e-12:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, PDM_NBINS + 1)
    within_var = 0.0
    for i in range(PDM_NBINS):
        mask = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
        if mask.sum() < 2:
            continue
        within_var += float(np.var(mags[mask])) * mask.sum()
    within_var /= len(mags)
    return within_var / total_var


def pdm_validate_period(
    times_sec: np.ndarray,
    magnitudes: np.ndarray,
    lsp_period_sec: float,
    n_scan: int = 40,
) -> dict[str, float]:
    """Scan periods near LSP candidate; return PDM-validated best period."""
    if not np.isfinite(lsp_period_sec) or lsp_period_sec <= 0:
        return {"pdm_period_sec": float("nan"), "pdm_theta": 1.0}

    lo = lsp_period_sec * (1.0 - PDM_PERIOD_TOL)
    hi = lsp_period_sec * (1.0 + PDM_PERIOD_TOL)
    candidates = np.linspace(lo, hi, n_scan)
    thetas = [pdm_theta(times_sec, magnitudes, p) for p in candidates]
    best_idx = int(np.argmin(thetas))
    return {
        "pdm_period_sec": float(candidates[best_idx]),
        "pdm_theta": float(thetas[best_idx]),
    }


def phase_fold(
    times_sec: np.ndarray,
    magnitudes: np.ndarray,
    period_sec: float,
) -> dict[str, np.ndarray]:
    """Fold light curve on validated rotation period."""
    ts = np.asarray(times_sec, dtype=float)
    mags = np.asarray(magnitudes, dtype=float)
    valid = np.isfinite(ts) & np.isfinite(mags)
    ts, mags = ts[valid], mags[valid]
    if period_sec <= 0 or len(mags) == 0:
        return {"phase": np.array([]), "magnitude": np.array([])}
    phase = (ts % period_sec) / period_sec
    order = np.argsort(phase)
    return {"phase": phase[order], "magnitude": mags[order]}


def _period_candidates(lsp_period: float, span: float) -> list[float]:
    """Harmonic de-aliasing: test multiples/submultiples of LSP peak."""
    if not np.isfinite(lsp_period) or lsp_period <= 0:
        return []
    max_p = max(span / 1.5, lsp_period)
    min_p = max(span / 500, 0.5)
    raw = [lsp_period * f for f in (0.25, 0.33, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)]
    return sorted({float(np.clip(p, min_p, max_p)) for p in raw if min_p <= p <= max_p})


def analyze_rotation_period(
    times_sec: np.ndarray,
    magnitudes: np.ndarray,
) -> dict[str, Any]:
    """
    Full PoC period pipeline: LSP candidate → PDM validation → phase fold → extracted period.
    """
    raw_ts = np.asarray(times_sec)
    mags = np.asarray(magnitudes, dtype=float)
    ts = _to_elapsed_seconds(raw_ts) if not np.issubdtype(raw_ts.dtype, np.number) else raw_ts.astype(float)
    span = float(ts.max() - ts.min()) if len(ts) > 1 else float(max(len(ts), 1))

    lsp = lomb_scargle_periodogram(ts, mags, max_period_sec=max(span / 3.0, 2.0))
    candidates = _period_candidates(float(lsp["lsp_period_sec"]), span)
    if not candidates:
        candidates = [float(lsp["lsp_period_sec"])]

    scored = [(p, pdm_theta(ts, mags, p)) for p in candidates]
    pdm_period, best_theta = min(scored, key=lambda item: item[1])

    fap = float(lsp["fap"])
    is_tumbling = 1 if (best_theta > 0.85 or fap >= FAP_PERIODIC) else 0

    extracted = float(pdm_period) if is_tumbling == 0 else float("nan")
    fold_period = extracted if np.isfinite(extracted) else float(lsp["lsp_period_sec"])
    folded = phase_fold(ts, mags, fold_period)

    return {
        "lsp_period_sec": round(float(lsp["lsp_period_sec"]), 3),
        "pdm_period_sec": round(float(pdm_period), 3),
        "pdm_theta": round(float(best_theta), 4),
        "extracted_period_sec": round(extracted, 3) if np.isfinite(extracted) else None,
        "is_tumbling": is_tumbling,
        "fap": float(fap),
        "frequencies": lsp["frequencies"],
        "power": lsp["power"],
        "folded_phase": folded["phase"],
        "folded_magnitude": folded["magnitude"],
    }


def save_periodogram_json(result: dict[str, Any], path: Path) -> None:
    """Persist periodogram arrays for one object (PoC artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lsp_period_sec": result.get("lsp_period_sec"),
        "pdm_period_sec": result.get("pdm_period_sec"),
        "pdm_theta": result.get("pdm_theta"),
        "extracted_period_sec": result.get("extracted_period_sec"),
        "is_tumbling": result.get("is_tumbling"),
        "fap": result.get("fap"),
        "frequencies": np.asarray(result["frequencies"]).tolist(),
        "power": np.asarray(result["power"]).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    # ponytail: self-check — synthetic 15 s period should recover within 20%
    t = np.linspace(0, 600, 400)
    m = 10.0 + 0.9 * np.sin(2 * np.pi * t / 15.0) + np.random.default_rng(42).normal(0, 0.03, len(t))
    out = analyze_rotation_period(t, m)
    assert out["is_tumbling"] == 0, out
    assert out["extracted_period_sec"] is not None
    assert abs(out["extracted_period_sec"] - 15.0) < 3.0, out
    assert out["fap"] < FAP_PERIODIC, out
    print(
        "period_analysis self-check OK:",
        {k: out[k] for k in ("lsp_period_sec", "pdm_period_sec", "extracted_period_sec", "fap")},
    )
