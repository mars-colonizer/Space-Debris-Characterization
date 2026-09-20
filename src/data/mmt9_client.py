"""Fetch public MMT-9 photometry from mmt.favor2.info (linked from http://mmt9.ru/satellites/)."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from src.config import COSPAR_ID_COL, DATA_RAW, LIGHTCURVE_ERR_COL, LIGHTCURVE_MAG_COL, LIGHTCURVE_TIME_COL, MMT_RAW_DIR, OBJECT_ID_COL, OBJECT_NAME_COL, PROJECT_ROOT
from src.data.data_mode import get_data_mode
from src.data.env import load_project_env

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://mmt.favor2.info"
TRACK_LINK_RE = re.compile(r"/satellites/track/(\d+)/download")
CATALOG_ENTRY_RE = re.compile(
    r'^NORAD\s+(\d+)\s+"([^"]*)"\s+(\S+)\s+(\d+)\s+(\S+)\s'
)
CATALOG_MAX_AGE_SEC = 7 * 24 * 3600
REQUEST_DELAY_SEC = 0.15
MAX_POINTS_PER_TRACK = 2000
DEFAULT_MAG_ERROR = 0.05
DEFAULT_CANDIDATES_CSV = DATA_RAW / "mmt9_candidates.csv"
NORAD_ID_COLUMNS = ("norad_id", "norad", "satno", OBJECT_ID_COL)
MIN_CACHED_POINTS = 3
HTTP_RETRIES = 3
HTTP_TIMEOUT_SEC = 60
# After a failed catalog refresh, reuse on-disk catalog for the rest of the process.
_CATALOG_REFRESH_FAILED = False


def _numeric_norad_counts(series: pd.Series) -> dict[int, int]:
    counts: dict[int, int] = {}
    for val, cnt in series.value_counts().items():
        try:
            counts[int(val)] = int(cnt)
        except (TypeError, ValueError):
            continue
    return counts


def mmt9_force_refresh() -> bool:
    load_project_env()
    return (os.getenv("MMT9_FORCE_REFRESH") or "").strip().lower() in ("1", "true", "yes", "on")


def mmt9_enabled() -> bool:
    load_project_env()
    source = (os.getenv("MMT_SOURCE") or "").strip().lower()
    if source in ("off", "offline", "none", "false", "0"):
        return False
    if source in ("mmt9", "favor2", "live", "api"):
        return source != "api"
    return get_data_mode() == "ACTUAL"


def candidates_csv_path() -> Path:
    load_project_env()
    explicit = (os.getenv("MMT9_CANDIDATES_CSV") or "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_CANDIDATES_CSV


def require_candidate_norad_ids() -> list[int]:
    """Load candidate NORAD IDs or raise if the configured CSV is missing/empty."""
    path = candidates_csv_path()
    ids = load_candidate_norad_ids(path=path)
    if not ids:
        raise FileNotFoundError(
            f"MMT9 candidates file missing or empty: {path}\n"
            "Place mmt9_candidates.csv with a norad_id column under data/raw/."
        )
    return ids


def load_candidate_norad_ids(
    path: Path | str | None = None,
    max_objects: int | None = None,
) -> list[int]:
    """NORAD IDs from mmt9_candidates.csv — exact list when file exists (not FETCH_MAX_OBJECTS)."""
    from src.data.env import env_int

    csv_path = Path(path) if path else candidates_csv_path()
    if not csv_path.is_file():
        return []

    df = pd.read_csv(csv_path)
    col = next((c for c in NORAD_ID_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"{csv_path} missing NORAD column (expected one of {NORAD_ID_COLUMNS})")

    ids = [int(x) for x in df[col].dropna().unique()]
    cap = env_int("MMT9_CANDIDATES_LIMIT", 0)
    if cap > 0:
        ids = ids[:cap]
    elif max_objects is not None and max_objects > 0:
        # ponytail: candidates file wins — max_objects only applies to catalog fallback
        pass
    logger.info("MMT-9 candidates: %d NORAD IDs from %s", len(ids), csv_path)
    return ids


def mmt9_allowlist() -> frozenset[int] | None:
    """When candidates CSV exists, MMT downloads are restricted to these NORAD IDs only."""
    ids = load_candidate_norad_ids()
    return frozenset(ids) if ids else None


def resolve_mmt_norad_ids(max_objects: int) -> tuple[list[int], str]:
    """
    NORAD IDs for MMT-first ingest.
    Uses all rows from mmt9_candidates.csv when present; otherwise samples the MMT catalog.
    """
    path = candidates_csv_path()
    if path.is_file():
        return require_candidate_norad_ids(), str(path)
    client = MMT9Client()
    client.ensure_catalog()
    return client.select_norad_ids(max_objects), "MMT-9 catalog sample"


def mmt_cache_status(
    norad_ids: list[int],
    cache_dir: Path | str | None = None,
) -> tuple[bool, list[int], list[int]]:
    """
    Return (all_cached, cached_ids, missing_ids) for candidate NORAD objects.
    Checks mmt_lightcurves.csv then per-track files under mmt9_tracks/.
    """
    if mmt9_force_refresh():
        return False, [], list(norad_ids)

    cache_dir = Path(cache_dir or MMT_RAW_DIR)
    combined = cache_dir / "mmt_lightcurves.csv"
    combined_counts: dict[int, int] = {}
    if combined.is_file():
        df = pd.read_csv(combined)
        if OBJECT_ID_COL in df.columns:
            combined_counts = _numeric_norad_counts(df[OBJECT_ID_COL].dropna())

    client = MMT9Client(cache_dir=cache_dir)
    cached: list[int] = []
    missing: list[int] = []
    for norad in norad_ids:
        if combined_counts.get(norad, 0) >= MIN_CACHED_POINTS:
            cached.append(norad)
        elif client._lightcurve_from_track_cache(norad) is not None:
            cached.append(norad)
        else:
            missing.append(norad)
    return len(missing) == 0, cached, missing


class MMT9Client:
    """Public MMT satellite photometry DB — catalog + per-track downloads."""

    def __init__(
        self,
        base_url: str | None = None,
        cache_dir: Path | str | None = None,
        timeout_sec: int = HTTP_TIMEOUT_SEC,
        request_delay_sec: float = REQUEST_DELAY_SEC,
    ):
        load_project_env()
        self.base_url = (base_url or os.getenv("MMT9_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.cache_dir = Path(cache_dir or MMT_RAW_DIR)
        self.track_cache_dir = self.cache_dir / "mmt9_tracks"
        self.catalog_path = self.cache_dir / "mmt9_catalog.txt"
        self.timeout_sec = timeout_sec
        self.request_delay_sec = request_delay_sec
        self._norad_to_mmt_id: dict[int, int] | None = None
        self._norad_to_name: dict[int, str] | None = None
        self._session = requests.Session()
        self._session.headers.setdefault("User-Agent", "RSO-PoC/1.0 (research; contact via repo)")

    def fetch_lightcurve(
        self,
        norad_id: int | str,
        cospar_id: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (iso timestamps, magnitudes, errors) for one NORAD object."""
        norad = int(norad_id)
        allow = mmt9_allowlist()
        if allow is not None and norad not in allow:
            raise FileNotFoundError(f"NORAD {norad} not in mmt9_candidates.csv — skipped")

        if not mmt9_force_refresh():
            cached = self._lightcurve_from_track_cache(norad)
            if cached is not None:
                return cached
            cached = self._lightcurve_from_combined_csv(norad)
            if cached is not None:
                return cached

        track_ids = self._list_track_ids(norad, online=True)
        if not track_ids:
            raise FileNotFoundError(f"No MMT-9 tracks for NORAD {norad}")

        tracks = [self._fetch_track(norad, track_id) for track_id in track_ids]
        ts, mags, errs = _merge_tracks_with_detrend(tracks)
        if len(mags) < MIN_CACHED_POINTS:
            raise FileNotFoundError(f"MMT-9 tracks for NORAD {norad} had too few points")
        logger.debug(
            "MMT-9 NORAD %s: merged %d tracks -> %d points",
            norad,
            len(track_ids),
            len(mags),
        )
        return ts, mags, errs

    def fetch_many(
        self,
        objects: pd.DataFrame,
        norad_col: str = OBJECT_ID_COL,
        cospar_col: str = COSPAR_ID_COL,
        progress_callback: Callable[[int, int, int | str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        """
        Fetch light curves for objects with NORAD ids.
        Returns (long-format light_curves, id frame, failed NORAD ids).
        Saves incrementally after each object so a late failure does not discard prior downloads.
        """
        self.ensure_catalog()
        session_rows: list[dict] = []
        ok = 0
        failed: list[int] = []
        total = len(objects)
        for _, obj in objects.iterrows():
            norad = obj.get(norad_col)
            cospar = obj.get(cospar_col)
            if pd.isna(norad):
                continue
            norad_int = int(norad)
            try:
                ts, mags, errs = self.fetch_lightcurve(norad_int, None if pd.isna(cospar) else str(cospar))
            except FileNotFoundError:
                logger.debug("MMT-9 miss NORAD=%s cospar=%s", norad, cospar)
                failed.append(norad_int)
                continue
            except Exception as exc:
                logger.warning("MMT-9 fetch failed NORAD=%s: %s", norad, exc)
                failed.append(norad_int)
                continue

            ok += 1
            object_name = self.resolve_object_name(norad_int)
            batch: list[dict] = []
            for t, m, e in zip(ts, mags, errs):
                row = {
                    COSPAR_ID_COL: cospar,
                    OBJECT_ID_COL: norad_int,
                    OBJECT_NAME_COL: object_name,
                    LIGHTCURVE_TIME_COL: t,
                    LIGHTCURVE_MAG_COL: float(m),
                    LIGHTCURVE_ERR_COL: float(e),
                }
                batch.append(row)
                session_rows.append(row)
            try:
                self._merge_lightcurves_csv(batch)
            except OSError as exc:
                logger.warning("MMT-9 cache write failed NORAD=%s: %s", norad_int, exc)
            if progress_callback:
                progress_callback(ok, total, norad_int)

        try:
            lc_df = self._read_combined_csv()
        except OSError as exc:
            logger.warning("Could not read combined light-curve cache: %s", exc)
            lc_df = pd.DataFrame(session_rows)
        if lc_df.empty and session_rows:
            lc_df = pd.DataFrame(session_rows)
        photo_ids = (
            lc_df[[COSPAR_ID_COL]].drop_duplicates()
            if not lc_df.empty and COSPAR_ID_COL in lc_df.columns
            else pd.DataFrame(columns=[COSPAR_ID_COL])
        )
        return lc_df, photo_ids, failed

    def ensure_catalog(self, force_refresh: bool = False) -> Path:
        global _CATALOG_REFRESH_FAILED
        fresh = (
            self.catalog_path.exists()
            and (time.time() - self.catalog_path.stat().st_mtime) < CATALOG_MAX_AGE_SEC
        )
        if not force_refresh and self.catalog_path.exists() and (fresh or _CATALOG_REFRESH_FAILED):
            return self.catalog_path

        url = f"{self.base_url}/satellites/download"
        logger.info("Downloading MMT-9 catalog from %s", url)
        try:
            resp = self._get(url)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.catalog_path.write_bytes(resp.content)
            self._norad_to_mmt_id = None
            self._norad_to_name = None
            _CATALOG_REFRESH_FAILED = False
            return self.catalog_path
        except requests.RequestException as exc:
            if self.catalog_path.exists():
                # ponytail: favor2 often 403s; stale on-disk catalog is enough for NORAD→mmt_id
                _CATALOG_REFRESH_FAILED = True
                logger.warning(
                    "MMT-9 catalog download failed (%s) — using cached %s",
                    exc,
                    self.catalog_path,
                )
                return self.catalog_path
            raise

    def resolve_object_name(self, norad_id: int | str) -> str | None:
        """Human-readable name from the MMT-9 catalog, if present."""
        name = self._catalog_name_map().get(int(norad_id))
        return name if name else None

    def resolve_mmt_id(self, norad_id: int | str) -> int | None:
        mapping = self._catalog_map()
        return mapping.get(int(norad_id))

    def select_norad_ids(
        self,
        max_objects: int,
        seed: int | None = None,
        prefer_variable: bool = True,
    ) -> list[int]:
        """Pick NORAD IDs from the MMT-9 catalog (variable objects first)."""
        from src.config import RANDOM_SEED

        self.ensure_catalog()
        variable: list[int] = []
        stable: list[int] = []
        for line in self.catalog_path.read_text(encoding="utf-8", errors="replace").splitlines():
            entry = _parse_catalog_entry(line)
            if entry is None:
                continue
            if prefer_variable and entry["variability"] > 0:
                variable.append(entry["norad"])
            else:
                stable.append(entry["norad"])

        rng = np.random.default_rng(RANDOM_SEED if seed is None else seed)
        rng.shuffle(variable)
        rng.shuffle(stable)
        chosen = variable[:max_objects]
        if len(chosen) < max_objects:
            chosen.extend(stable[: max_objects - len(chosen)])
        return chosen[:max_objects]

    def _catalog_map(self) -> dict[int, int]:
        if self._norad_to_mmt_id is not None:
            return self._norad_to_mmt_id
        self.ensure_catalog()
        mapping: dict[int, int] = {}
        for line in self.catalog_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = _parse_catalog_line(line)
            if parsed:
                mapping[parsed[0]] = parsed[1]
        self._norad_to_mmt_id = mapping
        logger.info("MMT-9 catalog: %d NORAD entries", len(mapping))
        return mapping

    def _catalog_name_map(self) -> dict[int, str]:
        if self._norad_to_name is not None:
            return self._norad_to_name
        self.ensure_catalog()
        names: dict[int, str] = {}
        for line in self.catalog_path.read_text(encoding="utf-8", errors="replace").splitlines():
            entry = _parse_catalog_entry(line)
            if entry is None:
                continue
            raw = str(entry.get("name") or "").strip()
            if raw:
                names[int(entry["norad"])] = raw
        self._norad_to_name = names
        return names

    def _lightcurve_from_track_cache(
        self,
        norad_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """All cached tracks merged with per-track median detrend, or None if unusable."""
        if not self.track_cache_dir.exists():
            return None
        tracks = [
            _parse_track_text(path.read_text(encoding="utf-8", errors="replace"))
            for path in sorted(self.track_cache_dir.glob(f"{norad_id}_*.txt"))
        ]
        ts, mags, errs = _merge_tracks_with_detrend(tracks)
        return (ts, mags, errs) if len(mags) >= MIN_CACHED_POINTS else None

    def _lightcurve_from_combined_csv(
        self,
        norad_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        combined = self.cache_dir / "mmt_lightcurves.csv"
        if not combined.is_file():
            return None
        df = pd.read_csv(combined)
        if OBJECT_ID_COL not in df.columns:
            return None
        norad = int(norad_id)
        nums = pd.to_numeric(df[OBJECT_ID_COL], errors="coerce")
        df = df[nums == norad]
        if len(df) < MIN_CACHED_POINTS:
            return None
        ts_col = next((c for c in (LIGHTCURVE_TIME_COL, "timestamp", "time", "epoch") if c in df.columns), None)
        mag_col = next((c for c in (LIGHTCURVE_MAG_COL, "magnitude", "mag", "visual_magnitude") if c in df.columns), None)
        err_col = next((c for c in (LIGHTCURVE_ERR_COL, "error", "err", "mag_error") if c in df.columns), None)
        if not ts_col or not mag_col:
            return None
        ts = df[ts_col].to_numpy(dtype=object)
        mags = df[mag_col].to_numpy(dtype=float)
        errs = df[err_col].to_numpy(dtype=float) if err_col else np.full(len(df), DEFAULT_MAG_ERROR)
        return ts, mags, errs

    def _cached_track_ids(self, norad_id: int) -> list[int]:
        if not self.track_cache_dir.exists():
            return []
        out: list[int] = []
        for path in self.track_cache_dir.glob(f"{norad_id}_*.txt"):
            stem = path.stem
            if "_" not in stem:
                continue
            try:
                out.append(int(stem.split("_", 1)[1]))
            except ValueError:
                continue
        return out

    def _list_track_ids(self, norad_id: int, online: bool = True) -> list[int]:
        cached = self._cached_track_ids(norad_id)
        if cached:
            return cached
        if not online:
            return []
        mmt_id = self.resolve_mmt_id(norad_id)
        if mmt_id is None:
            return []
        url = f"{self.base_url}/satellites/{mmt_id}"
        resp = self._get(url)
        ids = [int(m) for m in TRACK_LINK_RE.findall(resp.text)]
        seen: set[int] = set()
        out: list[int] = []
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
        return out

    def _fetch_track(
        self,
        norad_id: int,
        track_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_path = self.track_cache_dir / f"{norad_id}_{track_id}.txt"
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8", errors="replace")
        else:
            url = f"{self.base_url}/satellites/track/{track_id}/download"
            resp = self._get(url)
            text = resp.text
            self.track_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        return _parse_track_text(text, max_points=MAX_POINTS_PER_TRACK)

    def _read_combined_csv(self) -> pd.DataFrame:
        from src.data.lightcurve_columns import normalize_lightcurve_columns

        combined = self.cache_dir / "mmt_lightcurves.csv"
        if not combined.is_file():
            return pd.DataFrame()
        return normalize_lightcurve_columns(
            pd.read_csv(combined, dtype={COSPAR_ID_COL: "string"}, low_memory=False)
        )

    def _merge_lightcurves_csv(self, new_rows: list[dict]) -> None:
        """Append/merge one object's points into the combined cache file."""
        from src.data.lightcurve_columns import normalize_lightcurve_columns

        if not new_rows:
            return
        combined = self.cache_dir / "mmt_lightcurves.csv"
        combined.parent.mkdir(parents=True, exist_ok=True)
        new_df = normalize_lightcurve_columns(pd.DataFrame(new_rows))
        if combined.is_file():
            old = normalize_lightcurve_columns(
                pd.read_csv(combined, dtype={COSPAR_ID_COL: "string"}, low_memory=False)
            )
            merged = pd.concat([old, new_df], ignore_index=True)
            dedupe_cols = [c for c in (OBJECT_ID_COL, LIGHTCURVE_TIME_COL) if c in merged.columns]
            if dedupe_cols:
                merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
            merged.to_csv(combined, index=False)
        else:
            new_df.to_csv(combined, index=False)

    def _get(self, url: str) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(HTTP_RETRIES):
            try:
                time.sleep(self.request_delay_sec * (attempt + 1))
                resp = self._session.get(url, timeout=self.timeout_sec)
                resp.raise_for_status()
                return resp
            except (requests.RequestException, OSError) as exc:
                last_exc = exc
                logger.warning(
                    "MMT-9 HTTP attempt %d/%d failed (%s): %s",
                    attempt + 1,
                    HTTP_RETRIES,
                    url,
                    exc,
                )
        assert last_exc is not None
        raise last_exc


def _parse_catalog_entry(line: str) -> dict[str, int | str] | None:
    line = line.strip()
    m = CATALOG_ENTRY_RE.match(line)
    if not m:
        parsed = _parse_catalog_line(line)
        if parsed is None:
            return None
        return {"norad": parsed[0], "name": "", "variability": 0, "mmt_id": parsed[1]}
    try:
        mmt_id = int(line.split()[-1])
    except ValueError:
        return None
    return {
        "norad": int(m.group(1)),
        "name": m.group(2),
        "variability": int(m.group(4)),
        "mmt_id": mmt_id,
    }


def _parse_catalog_line(line: str) -> tuple[int, int] | None:
    line = line.strip()
    if not line.startswith("NORAD "):
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        norad = int(parts[1])
        mmt_id = int(parts[-1])
    except ValueError:
        return None
    return norad, mmt_id


def _parse_track_text(
    text: str,
    max_points: int = MAX_POINTS_PER_TRACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[str] = []
    magnitudes: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("date "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        date_s, time_s, _stdmag, mag = parts[0], parts[1], parts[2], parts[3]
        try:
            mag_f = float(mag)
        except ValueError:
            continue
        if not np.isfinite(mag_f):
            continue
        timestamps.append(f"{date_s}T{time_s}Z")
        magnitudes.append(mag_f)

    if not magnitudes:
        return np.array([]), np.array([]), np.array([])

    idx = np.arange(len(magnitudes))
    if len(idx) > max_points:
        idx = np.linspace(0, len(idx) - 1, max_points, dtype=int)

    ts = np.asarray([timestamps[i] for i in idx], dtype=object)
    mags = np.asarray([magnitudes[i] for i in idx], dtype=float)
    errs = np.full(len(mags), DEFAULT_MAG_ERROR, dtype=float)
    return ts, mags, errs


def _merge_tracks_with_detrend(
    tracks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate tracks after subtracting each track's median magnitude."""
    all_ts: list = []
    all_mags: list[float] = []
    all_errs: list[float] = []
    for ts, mags, errs in tracks:
        if len(mags) < 1:
            continue
        detrended = mags - float(np.median(mags))
        all_ts.extend(ts.tolist())
        all_mags.extend(detrended.tolist())
        all_errs.extend(errs.tolist())
    if not all_mags:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(all_ts, kind="stable")
    ts_arr = np.asarray(all_ts, dtype=object)[order]
    mag_arr = np.asarray(all_mags, dtype=float)[order]
    err_arr = np.asarray(all_errs, dtype=float)[order]
    _, uniq = np.unique(ts_arr, return_index=True)
    uniq = np.sort(uniq)
    return ts_arr[uniq], mag_arr[uniq], err_arr[uniq]


if __name__ == "__main__":
    cand = load_candidate_norad_ids()
    if cand:
        print(f"candidates: {len(cand)} NORAD IDs, first={cand[0]}, last={cand[-1]}")
    client = MMT9Client()
    client.ensure_catalog()
    assert client.resolve_mmt_id(12) == 1839, client.resolve_mmt_id(12)
    assert client.resolve_object_name(12), "expected catalog name for NORAD 12"
    merged = _merge_tracks_with_detrend([
        (np.array(["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"]), np.array([10.0, 11.0]), np.array([0.05, 0.05])),
        (np.array(["2020-01-01T02:00:00Z"]), np.array([20.0]), np.array([0.05])),
    ])
    assert len(merged[1]) == 3 and np.isclose(merged[1].sum(), 0.0), merged[1]
    ts, mags, errs = client.fetch_lightcurve(12)
    assert len(mags) >= 100, len(mags)
    assert np.isfinite(mags).all()
    print(f"mmt9_client self-check OK: NORAD 12 -> {len(mags)} points, mag range {mags.min():.2f}-{mags.max():.2f}")
