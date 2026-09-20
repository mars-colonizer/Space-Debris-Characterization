"""KeepTrack API client — NORAD ↔ COSPAR (International Designator) resolution.

Docs: https://api.keeptrack.space/v4/docs
Attribution: data via KeepTrack (CC BY-NC 4.0) — https://keeptrack.space
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import COSPAR_ID_COL, DATA_RAW, OBJECT_ID_COL
from src.data.clean_data import normalize_cospar_id
from src.data.env import load_project_env

logger = logging.getLogger(__name__)

BASE_URL = "https://api.keeptrack.space/v4"
BRIEF_CACHE_PATH = DATA_RAW / "keeptrack_brief.json"
BRIEF_META_PATH = DATA_RAW / "keeptrack_brief.meta.json"
BRIEF_MAX_AGE_SEC = 3600
INDIVIDUAL_LOOKUP_DELAY_SEC = 0.25
MAX_INDIVIDUAL_LOOKUPS = 25  # ponytail: small gap-fill only; bulk via /sats/brief


def keeptrack_enabled() -> bool:
    load_project_env()
    return bool(os.getenv("KEEPTRACK_API_KEY", "").strip())


def _api_key() -> str:
    load_project_env()
    key = (os.getenv("KEEPTRACK_API_KEY") or "").strip()
    if not key:
        raise ValueError("KEEPTRACK_API_KEY not set")
    return key


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"X-API-Key": _api_key(), "Accept": "application/json"}
    if extra:
        h.update(extra)
    return h


def _normalize_object_id(value: str | None) -> str | None:
    if not value or pd.isna(value):
        return None
    normalized = normalize_cospar_id(pd.Series([value])).iloc[0]
    return str(normalized) if normalized else None


def _parse_brief_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("sats", "data", "satellites", "results"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    return []


def _brief_to_map(records: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in records:
        norad = rec.get("NORAD_CAT_ID") or rec.get("norad_cat_id")
        cospar = rec.get("OBJECT_ID") or rec.get("object_id")
        if norad is None or not cospar:
            continue
        norm = _normalize_object_id(str(cospar))
        if norm:
            out[str(int(norad))] = norm
    return out


def _load_brief_meta() -> dict[str, Any]:
    if not BRIEF_META_PATH.is_file():
        return {}
    try:
        return json.loads(BRIEF_META_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def fetch_brief_catalog(force_refresh: bool = False) -> dict[str, str]:
    """
    Download (or load cached) /v4/sats/brief and return NORAD → COSPAR map.
    Respects hourly cache + ETag per KeepTrack API guidance.
    """
    load_project_env()
    meta = _load_brief_meta()
    age = time.time() - float(meta.get("fetched_at", 0))
    if (
        not force_refresh
        and BRIEF_CACHE_PATH.is_file()
        and age < BRIEF_MAX_AGE_SEC
    ):
        records = _parse_brief_records(json.loads(BRIEF_CACHE_PATH.read_text()))
        return _brief_to_map(records)

    req_headers = _headers()
    etag = meta.get("etag")
    if etag and not force_refresh:
        req_headers["If-None-Match"] = etag

    url = f"{BASE_URL}/sats/brief"
    resp = requests.get(url, headers=req_headers, timeout=120)

    if resp.status_code == 304 and BRIEF_CACHE_PATH.is_file():
        records = _parse_brief_records(json.loads(BRIEF_CACHE_PATH.read_text()))
        meta["fetched_at"] = time.time()
        BRIEF_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        BRIEF_META_PATH.write_text(json.dumps(meta))
        return _brief_to_map(records)

    resp.raise_for_status()
    payload = resp.json()
    records = _parse_brief_records(payload)

    BRIEF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIEF_CACHE_PATH.write_text(json.dumps(records))
    BRIEF_META_PATH.write_text(
        json.dumps({
            "fetched_at": time.time(),
            "etag": resp.headers.get("ETag"),
            "count": len(records),
        })
    )
    logger.info("KeepTrack brief catalog cached: %d records", len(records))
    return _brief_to_map(records)


def lookup_satellite(norad_id: int | str) -> dict[str, Any] | None:
    """Single-satellite metadata (use sparingly — prefer brief catalog)."""
    url = f"{BASE_URL}/sat/{int(norad_id)}"
    resp = requests.get(url, headers=_headers(), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def resolve_cospar_map(norad_ids: list[int]) -> dict[str, str]:
    """
    Resolve NORAD IDs to COSPAR / International Designator strings.
    Uses cached brief catalog first, then individual lookups for gaps.
    """
    if not norad_ids:
        return {}
    if not keeptrack_enabled():
        return {}

    wanted = {str(int(n)) for n in norad_ids}
    out: dict[str, str] = {}

    try:
        brief_map = fetch_brief_catalog()
        for norad in wanted:
            if norad in brief_map:
                out[norad] = brief_map[norad]
    except Exception as exc:
        logger.warning("KeepTrack brief catalog failed: %s", exc)

    missing = sorted(wanted - set(out.keys()))
    if not missing:
        return out

    # ponytail: gap-fill only — not a bulk catalog walk
    for norad in missing[:MAX_INDIVIDUAL_LOOKUPS]:
        try:
            time.sleep(INDIVIDUAL_LOOKUP_DELAY_SEC)
            rec = lookup_satellite(norad)
            if rec:
                norm = _normalize_object_id(str(rec.get("OBJECT_ID", "")))
                if norm:
                    out[norad] = norm
        except Exception as exc:
            logger.warning("KeepTrack lookup NORAD %s failed: %s", norad, exc)

    if len(missing) > MAX_INDIVIDUAL_LOOKUPS:
        logger.warning(
            "KeepTrack: %d NORAD IDs unresolved after brief cache (cap %d individual lookups)",
            len(missing) - MAX_INDIVIDUAL_LOOKUPS,
            MAX_INDIVIDUAL_LOOKUPS,
        )
    return out


def merge_cospar_from_tle(
    norad_ids: list[int],
    tle: pd.DataFrame | None,
) -> dict[str, str]:
    """Build NORAD→COSPAR map from TLE rows, then KeepTrack for gaps."""
    out: dict[str, str] = {}
    if tle is not None and not tle.empty and OBJECT_ID_COL in tle.columns and COSPAR_ID_COL in tle.columns:
        subset = (
            tle[[OBJECT_ID_COL, COSPAR_ID_COL]]
            .dropna(subset=[OBJECT_ID_COL, COSPAR_ID_COL])
            .astype({OBJECT_ID_COL: str})
            .drop_duplicates(subset=[OBJECT_ID_COL], keep="last")
        )
        for _, row in subset.iterrows():
            norm = _normalize_object_id(str(row[COSPAR_ID_COL]))
            if norm:
                out[str(int(float(row[OBJECT_ID_COL])))] = norm

    missing = [int(n) for n in norad_ids if str(int(n)) not in out]
    if missing and keeptrack_enabled():
        out.update(resolve_cospar_map(missing))
    return out


if __name__ == "__main__":
    load_project_env()
    if not keeptrack_enabled():
        print("keeptrack_client: KEEPTRACK_API_KEY not set — skip live check")
    else:
        m = resolve_cospar_map([25544, 12])
        assert m.get("25544") == "1998-067A", m
        print("keeptrack_client self-check OK:", m)
