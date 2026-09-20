"""Period recovery on real MMT-9 track timestamps + FAP sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import MMT_RAW_DIR
from src.data.mmt9_client import _parse_track_text
from src.features.period_analysis import lomb_scargle_periodogram, pdm_validate_period

TRACK = MMT_RAW_DIR / "mmt9_tracks" / "26389_10269127.txt"
PERIODS = (5.0, 20.0, 60.0, 150.0)


def _load_track_times(path: Path) -> np.ndarray:
    ts_raw, _, _ = _parse_track_text(path.read_text(encoding="utf-8", errors="replace"), max_points=10**9)
    import pandas as pd

    tt = pd.to_datetime(pd.Series(ts_raw), utc=True, errors="coerce")
    t_sec = (tt - tt.min()).dt.total_seconds().to_numpy(dtype=float)
    ok = np.isfinite(t_sec)
    t_sec = t_sec[ok]
    _, uniq = np.unique(t_sec, return_index=True)
    return t_sec[np.sort(uniq)]


@pytest.mark.skipif(not TRACK.is_file(), reason="cached track 26389_10269127.txt missing")
@pytest.mark.parametrize("period_sec", PERIODS)
def test_lomb_scargle_recovers_clean_sinusoid(period_sec: float) -> None:
    t_sec = _load_track_times(TRACK)
    dur = float(t_sec.max() - t_sec.min())
    assert period_sec < dur / 3.0, "injected period must fit in track/3"

    mags = np.sin(2.0 * np.pi * t_sec / period_sec)
    lsp = lomb_scargle_periodogram(
        t_sec,
        mags,
        min_period_sec=2.0,
        max_period_sec=dur / 3.0,
    )
    refined = pdm_validate_period(t_sec, mags, float(lsp["lsp_period_sec"]))
    recovered = float(refined["pdm_period_sec"])
    err = abs(recovered - period_sec) / period_sec
    assert err < 0.02, f"P={period_sec} recovered={recovered} err={err:.4%} fap={lsp['fap']}"
    assert float(lsp["fap"]) < 0.01, f"expected significant FAP, got {lsp['fap']}"


def test_fap_rejects_pure_noise() -> None:
    rng = np.random.default_rng(0)
    t_sec = np.linspace(0.0, 900.0, 900)
    mags = rng.normal(0.0, 1.0, size=len(t_sec))
    lsp = lomb_scargle_periodogram(t_sec, mags, min_period_sec=2.0, max_period_sec=300.0)
    assert float(lsp["fap"]) > 0.05, f"noise FAP too small: {lsp['fap']}"
