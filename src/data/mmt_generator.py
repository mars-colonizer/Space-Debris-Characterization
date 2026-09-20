"""Generate synthetic MMT-9 raw light-curve time series for PoC offline use."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COSPAR_ID_COL, DATA_RAW, LIGHTCURVE_ERR_COL, LIGHTCURVE_MAG_COL, LIGHTCURVE_TIME_COL, MMT_RAW_DIR, OBJECT_ID_COL, RANDOM_SEED


def generate_light_curve_series(
    true_period: float,
    delta_mag: float,
    mag_mean: float,
    is_tumbling: int,
    rng: np.random.Generator,
    n_points: int = 200,
    duration_sec: float = 600.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physics-informed photometric time series (timestamps in seconds from start)."""
    ts = np.sort(rng.uniform(0, duration_sec, n_points))
    if is_tumbling:
        mags = mag_mean + rng.normal(0, delta_mag / 3.0, n_points)
    else:
        phase0 = float(rng.uniform(0, 2 * np.pi))
        mags = mag_mean + (delta_mag / 2.0) * np.sin(2 * np.pi * ts / true_period + phase0)
        mags += rng.normal(0, 0.04, n_points)
    errs = np.full(n_points, 0.05)
    return ts, mags, errs


def write_mmt_lightcurves(
    catalog: pd.DataFrame,
    tle: pd.DataFrame,
    photometric: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    """
    Write combined mmt_lightcurves.csv plus per-object CSVs under data/raw/mmt_lightcurves/.
    Uses catalog truths to generate realistic rotating/tumbling signatures.
    """
    out_dir = Path(out_dir or MMT_RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    id_map = tle[[COSPAR_ID_COL, OBJECT_ID_COL]].drop_duplicates(subset=[COSPAR_ID_COL])
    merged = catalog.merge(id_map, on=COSPAR_ID_COL, how="left")
    if photometric is not None and not photometric.empty:
        photo_cols = [c for c in photometric.columns if c != COSPAR_ID_COL]
        merged = merged.merge(photometric[[COSPAR_ID_COL, *photo_cols]], on=COSPAR_ID_COL, how="left")

    rows: list[dict] = []
    for _, obj in merged.iterrows():
        cospar = obj[COSPAR_ID_COL]
        norad = obj.get(OBJECT_ID_COL, f"OBJ-{cospar}")
        true_period = float(obj.get("true_period", 30.0))
        delta_mag = float(obj.get("delta_mag", 1.0)) if "delta_mag" in obj else 1.5
        mag_mean = float(obj.get("mag_mean", 11.0)) if "mag_mean" in obj else float(rng.uniform(9, 14))
        is_tumbling = int(obj.get("true_tumbling", 0))

        ts, mags, errs = generate_light_curve_series(
            true_period=true_period,
            delta_mag=delta_mag,
            mag_mean=mag_mean,
            is_tumbling=is_tumbling,
            rng=rng,
        )
        base_epoch = pd.Timestamp("2024-06-01", tz="UTC")
        for t, m, e in zip(ts, mags, errs):
            rows.append({
                COSPAR_ID_COL: cospar,
                OBJECT_ID_COL: norad,
                LIGHTCURVE_TIME_COL: (base_epoch + pd.Timedelta(seconds=float(t))).isoformat(),
                LIGHTCURVE_MAG_COL: round(float(m), 4),
                LIGHTCURVE_ERR_COL: round(float(e), 4),
            })

        per_obj = pd.DataFrame({
            LIGHTCURVE_TIME_COL: [(base_epoch + pd.Timedelta(seconds=float(t))).isoformat() for t in ts],
            LIGHTCURVE_MAG_COL: np.round(mags, 4),
            LIGHTCURVE_ERR_COL: np.round(errs, 4),
        })
        safe_cospar = str(cospar).replace("/", "_")
        per_obj.to_csv(out_dir / f"{norad}_{safe_cospar}.csv", index=False)

    combined = pd.DataFrame(rows)
    combined_path = out_dir / "mmt_lightcurves.csv"
    combined.to_csv(combined_path, index=False)
    return combined_path


def regenerate_from_raw_catalog(seed: int = RANDOM_SEED) -> Path:
    """Rebuild MMT offline files from existing rso_catalog.csv + tle_history.csv."""
    catalog = pd.read_csv(DATA_RAW / "rso_catalog.csv")
    tle = pd.read_csv(DATA_RAW / "tle_history.csv")
    return write_mmt_lightcurves(catalog, tle, seed=seed)
