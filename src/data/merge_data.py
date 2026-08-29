"""Merge TLE history with DISCOS metadata via COSPAR ID."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import COSPAR_ID_COL

logger = logging.getLogger(__name__)


@dataclass
class MergeSummary:
    total_tle_objects: int
    total_discos_objects: int
    matched_objects: int
    unmatched_tle_objects: int
    unmatched_discos_objects: int
    duplicate_tle_ids: int
    duplicate_discos_ids: int

    def __str__(self) -> str:
        return (
            f"Total TLE objects: {self.total_tle_objects}\n"
            f"Total DISCOS objects: {self.total_discos_objects}\n"
            f"Matched objects: {self.matched_objects}\n"
            f"Unmatched TLE objects: {self.unmatched_tle_objects}\n"
            f"Unmatched DISCOS objects: {self.unmatched_discos_objects}\n"
            f"Duplicate TLE IDs: {self.duplicate_tle_ids}\n"
            f"Duplicate DISCOS IDs: {self.duplicate_discos_ids}"
        )


def _count_duplicates(ids: pd.Series) -> int:
    counts = ids.dropna().value_counts()
    return int((counts > 1).sum())


def merge_tle_discos(
    tle_df: pd.DataFrame,
    discos_df: pd.DataFrame,
    how: str = "inner",
) -> tuple[pd.DataFrame, MergeSummary]:
    """Merge TLE and DISCOS on normalized COSPAR ID."""
    tle_ids = set(tle_df[COSPAR_ID_COL].dropna().unique())
    discos_ids = set(discos_df[COSPAR_ID_COL].dropna().unique())

    dup_discos = _count_duplicates(discos_df[COSPAR_ID_COL])
    if dup_discos:
        logger.warning(
            "Found %d duplicate COSPAR IDs in DISCOS; keeping first occurrence", dup_discos
        )
        discos_df = discos_df.drop_duplicates(subset=[COSPAR_ID_COL], keep="first")

    dup_tle = _count_duplicates(tle_df[COSPAR_ID_COL])
    merged = tle_df.merge(discos_df, on=COSPAR_ID_COL, how=how, suffixes=("_tle", "_discos"))
    matched = len(set(merged[COSPAR_ID_COL].unique()) & tle_ids & discos_ids)

    summary = MergeSummary(
        total_tle_objects=len(tle_ids),
        total_discos_objects=len(discos_ids),
        matched_objects=matched if how == "inner" else len(set(merged[COSPAR_ID_COL].unique())),
        unmatched_tle_objects=len(tle_ids - discos_ids),
        unmatched_discos_objects=len(discos_ids - tle_ids),
        duplicate_tle_ids=dup_tle,
        duplicate_discos_ids=dup_discos,
    )
    logger.info("\n%s", summary)
    return merged, summary
