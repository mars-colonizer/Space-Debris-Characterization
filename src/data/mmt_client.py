"""MMT-9 / Mini-MegaTORTORA light-curve client and photometric feature extraction."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.signal import lombscargle
from scipy.stats import skew

from src.config import COSPAR_ID_COL, DATA_RAW, OBJECT_ID_COL, PHOTOMETRIC_FEATURE_COLS
from src.data.env import load_project_env

logger = logging.getLogger(__name__)

MMT_RAW_DIR = DATA_RAW / "mmt_lightcurves"
DEFAULT_API_URL = "https://mmt.insap.org/api/lightcurve"  # ponytail: placeholder; offline CSV is primary POC path
FETCH_TIMEOUT_SEC = 15
TUMBLING_POWER_RATIO = 0.03  # max_power/total_power below this => no dominant period (tumbling)


def extract_photometric_features(
    timestamps: np.ndarray | list,
    magnitudes: np.ndarray | list,
    errors: np.ndarray | list | None = None,
) -> dict[str, float | int]:
    """
    Derive summary photometric features from a light-curve time series.

    Returns keys aligned with PHOTOMETRIC_FEATURE_COLS.
    """
    mags = np.asarray(magnitudes, dtype=float)
    ts = _to_elapsed_seconds(timestamps)
    if errors is not None:
        err = np.asarray(errors, dtype=float)
        valid = np.isfinite(mags) & np.isfinite(ts) & np.isfinite(err)
    else:
        err = None
        valid = np.isfinite(mags) & np.isfinite(ts)

    mags = mags[valid]
    ts = ts[valid]
    if err is not None:
        err = err[valid]

    empty = {col: np.nan for col in PHOTOMETRIC_FEATURE_COLS}
    empty["is_tumbling"] = 0
    if len(mags) < 3:
        return empty

    mag_mean = float(np.nanmean(mags))
    mag_std = float(np.nanstd(mags))
    delta_mag = float(np.nanpercentile(mags, 95) - np.nanpercentile(mags, 5))
    apparent_shape_score = float(skew(mags, nan_policy="omit"))

    centered = mags - np.mean(mags)
    t_span = float(ts.max() - ts.min())
    if t_span < 1.0:
        ts = np.linspace(0.0, max(len(ts) - 1, 1), len(ts))

    freqs = np.linspace(1.0 / 3600.0, 1.0 / 1.0, 500)
    try:
        power = lombscargle(ts, centered, freqs, normalize=True)
    except Exception:
        power = np.zeros_like(freqs)

    peak_idx = int(np.argmax(power))
    f0 = float(freqs[peak_idx])
    estimated_period_sec = float(np.clip(1.0 / f0 if f0 > 0 else 3600.0, 1.0, 3600.0))

    peak_power = float(power[peak_idx])
    total_power = float(np.sum(power)) + 1e-12
    is_tumbling = 1 if (peak_power / total_power) < TUMBLING_POWER_RATIO else 0

    return {
        "mag_mean": round(mag_mean, 4),
        "mag_std": round(mag_std, 4),
        "delta_mag": round(delta_mag, 4),
        "estimated_period_sec": round(estimated_period_sec, 3),
        "apparent_shape_score": round(apparent_shape_score, 4),
        "is_tumbling": int(is_tumbling),
    }


def _to_elapsed_seconds(timestamps: np.ndarray | list) -> np.ndarray:
    """Convert timestamps to seconds from first sample."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if ts.notna().sum() >= 2:
        sec = (ts - ts.min()).dt.total_seconds().to_numpy(dtype=float)
        return sec
    numeric = pd.to_numeric(pd.Series(timestamps), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).sum() >= 2:
        return numeric - np.nanmin(numeric)
    return np.arange(len(timestamps), dtype=float)


def _sanitize_id(value: str) -> str:
    return str(value).strip().upper().replace("/", "_").replace(" ", "")


class MMTClient:
    """Fetch MMT light curves via API with offline CSV/JSON fallback."""

    def __init__(
        self,
        api_url: str | None = None,
        offline_dir: Path | str | None = None,
        timeout_sec: int = FETCH_TIMEOUT_SEC,
    ):
        load_project_env()
        self.api_url = api_url or (os.getenv("MMT_API_URL") or DEFAULT_API_URL)
        self.offline_dir = Path(offline_dir or MMT_RAW_DIR)
        self.timeout_sec = timeout_sec

    def fetch_lightcurve(
        self,
        cospar_id: str | None = None,
        norad_id: str | int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (timestamps, visual_magnitudes, errors) for one object.
        Tries live API first, then local offline files.
        """
        try:
            return self._fetch_api(cospar_id=cospar_id, norad_id=norad_id)
        except Exception as exc:
            logger.warning("MMT API fetch failed (%s) — using offline fallback", exc)
        return self._fetch_offline(cospar_id=cospar_id, norad_id=norad_id)

    def build_photometric_table(
        self,
        objects: pd.DataFrame,
        cospar_col: str = COSPAR_ID_COL,
        norad_col: str = OBJECT_ID_COL,
    ) -> pd.DataFrame:
        """Extract photometric summary rows for all objects in frame."""
        rows: list[dict[str, Any]] = []
        for _, obj in objects.iterrows():
            cospar = obj.get(cospar_col)
            norad = obj.get(norad_col)
            if pd.isna(cospar) and pd.isna(norad):
                continue
            try:
                ts, mags, errs = self.fetch_lightcurve(
                    cospar_id=None if pd.isna(cospar) else str(cospar),
                    norad_id=None if pd.isna(norad) else norad,
                )
                feats = extract_photometric_features(ts, mags, errs)
            except FileNotFoundError:
                logger.debug("No MMT data for cospar=%s norad=%s", cospar, norad)
                continue
            rows.append({COSPAR_ID_COL: cospar, **feats})
        return pd.DataFrame(rows)

    def _fetch_api(
        self,
        cospar_id: str | None = None,
        norad_id: str | int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params: dict[str, str] = {}
        if norad_id is not None:
            params["norad_id"] = str(norad_id)
        if cospar_id is not None:
            params["cospar_id"] = str(cospar_id)
        if not params:
            raise ValueError("cospar_id or norad_id required")

        resp = requests.get(self.api_url, params=params, timeout=self.timeout_sec)
        resp.raise_for_status()
        payload = resp.json()
        return (
            np.asarray(payload.get("timestamps") or payload.get("times"), dtype=object),
            np.asarray(payload.get("visual_magnitudes") or payload.get("magnitudes"), dtype=float),
            np.asarray(payload.get("errors") or payload.get("magnitude_errors", []), dtype=float),
        )

    def _fetch_offline(
        self,
        cospar_id: str | None = None,
        norad_id: str | int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.offline_dir.exists():
            raise FileNotFoundError(f"MMT offline directory missing: {self.offline_dir}")

        candidates: list[Path] = []
        if norad_id is not None:
            n = _sanitize_id(str(norad_id))
            candidates.extend(self.offline_dir.glob(f"*{n}*"))
        if cospar_id is not None:
            c = _sanitize_id(cospar_id)
            candidates.extend(self.offline_dir.glob(f"*{c}*"))

        for path in sorted(set(candidates)):
            if path.suffix.lower() == ".csv":
                return self._parse_csv(path)
            if path.suffix.lower() == ".json":
                return self._parse_json(path)

        # Long-format combined CSV fallback
        combined = self.offline_dir / "mmt_lightcurves.csv"
        if combined.exists():
            return self._parse_combined_csv(combined, cospar_id=cospar_id, norad_id=norad_id)

        raise FileNotFoundError(
            f"No offline MMT light curve for cospar={cospar_id} norad={norad_id} in {self.offline_dir}"
        )

    def _parse_csv(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        ts_col = _pick_col(df, ("timestamp", "time", "epoch", "mjd"))
        mag_col = _pick_col(df, ("magnitude", "visual_magnitude", "mag", "vmag"))
        err_col = _pick_col(df, ("error", "mag_error", "sigma", "err"))
        if not ts_col or not mag_col:
            raise ValueError(f"CSV {path} missing timestamp/magnitude columns")
        ts = df[ts_col].to_numpy()
        mags = df[mag_col].to_numpy(dtype=float)
        errs = df[err_col].to_numpy(dtype=float) if err_col else np.zeros(len(df))
        return ts, mags, errs

    def _parse_json(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = json.loads(path.read_text())
        return (
            np.asarray(data["timestamps"], dtype=object),
            np.asarray(data["visual_magnitudes"], dtype=float),
            np.asarray(data.get("errors", []), dtype=float),
        )

    def _parse_combined_csv(
        self,
        path: Path,
        cospar_id: str | None,
        norad_id: str | int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        if cospar_id and COSPAR_ID_COL in df.columns:
            df = df[df[COSPAR_ID_COL].astype(str).str.upper() == str(cospar_id).upper()]
        elif norad_id and OBJECT_ID_COL in df.columns:
            df = df[df[OBJECT_ID_COL].astype(str) == str(norad_id)]
        if df.empty:
            raise FileNotFoundError("No rows in combined MMT CSV for object")
        ts_col = _pick_col(df, ("timestamp", "time", "epoch", "mjd"))
        mag_col = _pick_col(df, ("magnitude", "mag", "visual_magnitude", "vmag"))
        err_col = _pick_col(df, ("error", "err", "mag_error", "sigma"))
        if not ts_col or not mag_col:
            raise ValueError(f"Combined CSV missing columns: {path}")
        ts = df[ts_col].to_numpy()
        mags = df[mag_col].to_numpy(dtype=float)
        errs = df[err_col].to_numpy(dtype=float) if err_col else np.zeros(len(df))
        return ts, mags, errs


def _pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in lower:
            return lower[name]
    return None


def _env_present(name: str) -> bool:
    load_project_env()
    return bool(os.getenv(name, "").strip())


def save_photometric_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (DATA_RAW / "photometric_observations.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    # ponytail: minimal self-check — fails if Lomb-Scargle path breaks
    t = np.linspace(0, 120, 200)
    m = 10 + 0.8 * np.sin(2 * np.pi * t / 15.0)
    out = extract_photometric_features(t, m)
    assert out["delta_mag"] > 0.5 and out["is_tumbling"] == 0, out
    print("mmt_client self-check OK:", out)