"""Space-Track.org GP/TLE data fetcher."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import pandas as pd
import requests

from src.config import COSPAR_ID_COL, EPOCH_COL, OBJECT_ID_COL, ORBITAL_ELEMENT_COLS
from src.data.env import require_env

logger = logging.getLogger(__name__)

BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
MU_EARTH = 398600.4418  # km³/s²


def _login(session: requests.Session) -> None:
    username = require_env("SPACE_TRACK_USERNAME")
    password = require_env("SPACE_TRACK_PASSWORD")
    resp = session.post(
        LOGIN_URL,
        data={"identity": username, "password": password},
        timeout=60,
    )
    resp.raise_for_status()
    if "Login Failed" in resp.text:
        raise PermissionError("Space-Track login failed — check SPACE_TRACK_USERNAME/PASSWORD")


def _mean_motion_to_sma(mean_motion_rev_per_day: float) -> float:
    """Convert mean motion (rev/day) to semi-major axis (km)."""
    n_rad_s = mean_motion_rev_per_day * 2 * 3.141592653589793 / 86400.0
    return (MU_EARTH / (n_rad_s**2)) ** (1 / 3)


def _normalize_cospar(object_id: str) -> str:
    s = str(object_id).strip().upper()
    m = re.match(r"^(\d{4})[- ]?(\d{1,3})([A-Z]{1,3})?$", s)
    if m:
        piece = (m.group(3) or "A")[0]
        return f"{m.group(1)}-{int(m.group(2)):03d}{piece}"
    return s


def _records_to_dataframe(
    records: list[dict[str, Any]],
    progress_callback: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, rec in enumerate(records, start=1):
        cospar = rec.get("OBJECT_ID") or ""
        mean_motion = rec.get("MEAN_MOTION") or rec.get("MEAN_MOTION")
        sma = rec.get("SEMIMAJOR_AXIS") or rec.get("SEMIMAJOR_AXIS")
        if sma is None and mean_motion is not None:
            try:
                sma = _mean_motion_to_sma(float(mean_motion))
            except (TypeError, ValueError):
                sma = None

        rows.append({
            COSPAR_ID_COL: _normalize_cospar(cospar),
            OBJECT_ID_COL: str(rec.get("NORAD_CAT_ID", "")),
            EPOCH_COL: rec.get("EPOCH"),
            "inclination": rec.get("INCLINATION"),
            "eccentricity": rec.get("ECCENTRICITY"),
            "semi_major_axis": sma,
            "raan": rec.get("RA_OF_ASC_NODE"),
            "arg_perigee": rec.get("ARG_OF_PERICENTER"),
            "mean_anomaly": rec.get("MEAN_ANOMALY"),
        })

        if progress_callback and i % 1000 == 0:
            progress_callback(i)

    df = pd.DataFrame(rows)
    if progress_callback and records:
        progress_callback(len(records))
    df = df.dropna(subset=[COSPAR_ID_COL])
    logger.info("Space-Track: parsed %d TLE/GP records", len(df))
    return df


def fetch_gp_history(
    norad_ids: list[int | str],
    epoch_days: int = 60,
    limit: int = 5000,
    progress_callback: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """
    Fetch GP history for NORAD catalog IDs (single batched query).

    ponytail: one gp_history call for the whole ID list; upgrade path is per-object cache on disk.
    """
    if not norad_ids:
        raise ValueError("No NORAD IDs provided for Space-Track query")

    id_csv = ",".join(str(int(i)) for i in norad_ids[:200])  # cap batch size
    query = (
        f"{BASE_URL}/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{id_csv}/"
        f"EPOCH/>now-{epoch_days}/"
        f"orderby/EPOCH asc/limit/{limit}/format/json"
    )

    with requests.Session() as session:
        _login(session)
        resp = session.get(query, timeout=120)
        resp.raise_for_status()
        data = resp.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected Space-Track response type: {type(data)}")

    return _records_to_dataframe(data, progress_callback=progress_callback)


def fetch_recent_gp(
    epoch_days: int = 30,
    limit: int = 5000,
    progress_callback: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Fetch recent GP records for all on-orbit objects (fallback when no NORAD list)."""
    query = (
        f"{BASE_URL}/basicspacedata/query/class/gp/"
        f"DECAY_DATE/null-val/EPOCH/>now-{epoch_days}/"
        f"orderby/NORAD_CAT_ID asc/limit/{limit}/format/json"
    )

    with requests.Session() as session:
        _login(session)
        resp = session.get(query, timeout=120)
        resp.raise_for_status()
        data = resp.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected Space-Track response type: {type(data)}")

    return _records_to_dataframe(data, progress_callback=progress_callback)
