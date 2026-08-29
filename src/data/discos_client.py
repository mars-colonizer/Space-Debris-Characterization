"""ESA DISCOSweb API client."""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd
import requests

from src.config import CLASS_COL, COSPAR_ID_COL
from src.data.env import require_env

logger = logging.getLogger(__name__)

BASE_URL = "https://discosweb.esoc.esa.int/api"
PAGE_SIZE = 100

# DISCOS objectClass filters for balanced POC sampling (actual labels, not forced to Phase 1 names)
DEFAULT_CLASS_FILTERS: list[tuple[str, str]] = [
    ("Rocket Body", "contains(objectClass,Rocket)"),
    ("Payload", "eq(objectClass,Payload)"),
    ("Fragment", "contains(objectClass,Fragment)"),
    ("Debris", "contains(objectClass,Debris)"),
]


def _headers() -> dict[str, str]:
    token = require_env("DISCOS_TOKEN")
    return {
        "Authorization": f"Bearer {token}",
        "DiscosWeb-Api-Version": "2",
        "Accept": "application/json",
    }


def _extract_dimensions(attrs: dict[str, Any]) -> dict[str, float | None]:
    """Map DISCOS v2 dimension fields to length/width/height."""
    height = attrs.get("height")
    width = attrs.get("width")
    depth = attrs.get("depth")
    diameter = attrs.get("diameter")
    span = attrs.get("span")

    length = depth or diameter or span
    if length is None and diameter is not None:
        length = diameter
    if width is None and diameter is not None:
        width = diameter
    if height is None and diameter is not None:
        height = diameter

    def _f(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "length": _f(length),
        "width": _f(width),
        "height": _f(height),
    }


def _object_to_row(item: dict[str, Any]) -> dict[str, Any]:
    attrs = item.get("attributes", {})
    dims = _extract_dimensions(attrs)
    cospar = attrs.get("cosparId") or attrs.get("cospar_id") or attrs.get("objectId")
    obj_class = attrs.get("objectClass") or attrs.get("object_class")

    return {
        COSPAR_ID_COL: cospar,
        "satno": attrs.get("satno"),
        CLASS_COL: obj_class,
        **dims,
        "mass": attrs.get("mass"),
        "shape": attrs.get("shape"),
    }


def _fetch_page(params: dict[str, Any]) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{BASE_URL}/objects",
        headers=_headers(),
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_discos_objects(
    max_objects: int = 500,
    class_filters: list[tuple[str, str]] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> pd.DataFrame:
    """Fetch DISCOS objects with balanced sampling across objectClass filters."""
    class_filters = class_filters or DEFAULT_CLASS_FILTERS
    per_class = max(1, max_objects // len(class_filters))
    rows: list[dict[str, Any]] = []

    for filter_label, filt in class_filters:
        fetched = 0
        page = 1
        while fetched < per_class:
            batch = _fetch_page({
                "filter": filt,
                "page[number]": page,
                "page[size]": min(PAGE_SIZE, per_class - fetched),
            })
            if progress_callback:
                progress_callback(len(rows), filter_label)
            if not batch:
                break
            for item in batch:
                row = _object_to_row(item)
                if row[COSPAR_ID_COL]:
                    rows.append(row)
                    fetched += 1
                if fetched >= per_class:
                    break
            page += 1
            if len(batch) < PAGE_SIZE:
                break

    df = pd.DataFrame(rows).drop_duplicates(subset=[COSPAR_ID_COL])
    logger.info("DISCOS: fetched %d unique objects", len(df))
    if CLASS_COL in df.columns:
        logger.info("DISCOS classes:\n%s", df[CLASS_COL].value_counts())
    return df
