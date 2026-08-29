"""Configurable data-leakage filter — removes target/leakage columns from X."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from src.config import LEAKAGE_COLUMNS, LEAKAGE_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class LeakageFilterResult:
    features: pd.DataFrame
    removed_columns: list[str] = field(default_factory=list)
    kept_target_columns: dict[str, pd.Series] = field(default_factory=dict)

    @property
    def remaining_feature_count(self) -> int:
        return len(self.features.columns)


def _match_leakage_columns(columns: list[str], extra: list[str] | None = None) -> list[str]:
    explicit = set(c.lower() for c in (LEAKAGE_COLUMNS + (extra or [])))
    patterns = [re.compile(p, re.IGNORECASE) for p in LEAKAGE_PATTERNS]
    removed: list[str] = []
    for col in columns:
        col_lower = col.lower()
        if col_lower in explicit:
            removed.append(col)
            continue
        for pat in patterns:
            if pat.search(col):
                removed.append(col)
                break
    return sorted(set(removed))


def apply_leakage_filter(
    df: pd.DataFrame,
    target_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
    extra_leakage: list[str] | None = None,
    fail_on_empty: bool = True,
) -> LeakageFilterResult:
    """Remove leakage columns from feature matrix X; extract targets separately."""
    target_columns = target_columns or []
    id_columns = id_columns or []
    all_cols = list(df.columns)

    removed = _match_leakage_columns(all_cols, extra=extra_leakage)
    for t in target_columns:
        if t in df.columns and t not in removed:
            removed.append(t)

    id_set = set(id_columns)
    feature_cols = [c for c in all_cols if c not in removed and c not in id_set]

    if fail_on_empty and not feature_cols:
        raise ValueError(
            "Leakage filter removed all columns — no features remain. "
            f"Removed: {removed}. Check LEAKAGE_COLUMNS config."
        )

    kept_targets = {t: df[t] for t in target_columns if t in df.columns}
    features = df[feature_cols].copy()

    leakage_removed = [c for c in removed if c not in id_set]
    logger.info("Removed leakage columns:")
    for col in leakage_removed:
        logger.info("  * %s", col)
    logger.info("Remaining feature count: %d", len(feature_cols))

    return LeakageFilterResult(
        features=features,
        removed_columns=leakage_removed,
        kept_target_columns=kept_targets,
    )
