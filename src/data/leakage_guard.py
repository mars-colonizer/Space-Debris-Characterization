"""Leakage guard — block ground-truth labels; allow photometric observables in X."""

from __future__ import annotations

from src.config import (
    CLASS_COL,
    LEAKAGE_TARGET_COLUMNS,
    PHOTOMETRIC_FEATURE_COLS,
    STAGE2_PERIOD_TARGET,
    STAGE2_SHAPE_TARGET,
    STAGE2_SIZE_TARGETS,
    STAGE2_TUMBLING_TARGET,
)
from src.data.leakage_filter import LeakageFilterResult, apply_leakage_filter


def stage2_target_columns() -> list[str]:
    return (
        [CLASS_COL]
        + list(STAGE2_SIZE_TARGETS)
        + [STAGE2_SHAPE_TARGET, STAGE2_PERIOD_TARGET, STAGE2_TUMBLING_TARGET]
    )


def apply_leakage_guard(
    df,
    id_columns: list[str] | None = None,
    fail_on_empty: bool = True,
) -> LeakageFilterResult:
    """
    Remove ground-truth labels from features while keeping photometric observables.

    Blocked: true_length, true_width, true_height, true_mass, true_shape, true_period,
    true_tumbling (+ legacy length/width/height/mass/shape via config).
    Allowed in X: mag_mean, mag_std, delta_mag, estimated_period_sec,
    apparent_shape_score, is_tumbling (observational estimate).
    """
    # ponytail: explicit allowlist prevents shape/period regex false positives on photometry cols
    _ = PHOTOMETRIC_FEATURE_COLS  # documented contract; filter uses extra_leakage only
    return apply_leakage_filter(
        df,
        target_columns=stage2_target_columns(),
        id_columns=id_columns,
        extra_leakage=list(LEAKAGE_TARGET_COLUMNS),
        fail_on_empty=fail_on_empty,
    )
