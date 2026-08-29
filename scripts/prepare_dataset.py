#!/usr/bin/env python3
"""Prepare merged, feature-engineered training dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_COL,
    COSPAR_ID_COL,
    DATA_PROCESSED,
    DATA_RAW,
    NON_FEATURE_COLUMNS,
    ORBITAL_ELEMENT_COLS,
    PHOTOMETRIC_FEATURE_COLS,
    RANDOM_SEED,
    STAGE1_TARGET,
    STAGE2_SIZE_TARGETS,
    TEST_SIZE,
)
from src.data.clean_data import clean_discos_data, clean_tle_data
from src.data.db_manager import load_catalog, load_photometric
from src.data.ingest import ingest_from_directory
from src.data.leakage_guard import apply_leakage_guard
from src.data.merge_data import merge_tle_discos
from src.data.storage import save_to_sqlite
from src.features.orbital_features import compute_orbital_features
from src.utils import terminal as term

STEPS = 10


def _missing_summary(df: pd.DataFrame, cols: list[str] | None = None) -> dict[str, int]:
    cols = cols or list(df.columns)
    return {c: int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().sum() > 0}


def _print_missing(label: str, missing: dict[str, int]) -> None:
    term.info(f"{label}:", indent=4)
    if not missing:
        term.detail("None", indent=8)
        return
    for col, n in sorted(missing.items()):
        term.detail(f"{col}: {term.fmt_n(n)}", indent=8)


def object_level_split(cospar_ids: pd.Series, test_size: float, seed: int) -> tuple[set, set]:
    unique_ids = cospar_ids.dropna().unique()
    rng = pd.Series(unique_ids).sample(frac=1, random_state=seed)
    n_test = max(1, int(len(rng) * test_size))
    test_ids = set(rng.iloc[:n_test])
    train_ids = set(rng.iloc[n_test:])
    return train_ids, test_ids


def _print_data_quality_summary(
    tle_raw: pd.DataFrame,
    discos_raw: pd.DataFrame,
    summary,
    leakage_removed: list[str],
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ids: set,
    test_ids: set,
) -> None:
    term.banner("DATA QUALITY SUMMARY")
    term.info("Source data")
    term.line("-" * 60)
    term.line(f"TLE records:                  {term.fmt_n(len(tle_raw)):>10}")
    term.line(f"Unique TLE objects:           {term.fmt_n(tle_raw[COSPAR_ID_COL].nunique()):>10}")
    term.line(f"DISCOS objects:               {term.fmt_n(len(discos_raw)):>10}")
    term.line(f"Matched objects:              {term.fmt_n(summary.matched_objects):>10}")

    term.line("")
    term.info("Class distribution (matched objects)")
    term.line("-" * 60)
    if CLASS_COL in train_df.columns or CLASS_COL in test_df.columns:
        combined = pd.concat([train_df[[CLASS_COL]], test_df[[CLASS_COL]]], ignore_index=True)
        counts = combined[CLASS_COL].value_counts()
        for cls, n in counts.items():
            term.line(f"{str(cls):<30}{term.fmt_n(n):>10}")
        term.line(f"{'Total':<30}{term.fmt_n(len(combined)):>10}")
        for cls, n in test_df[CLASS_COL].value_counts().items():
            if n < 5:
                term.warn(f"Class '{cls}' contains only {n} test sample(s). Per-class metrics may be unreliable.")

    term.line("")
    term.info("Feature summary")
    term.line("-" * 60)
    raw_feature_count = len([c for c in ORBITAL_ELEMENT_COLS if c in tle_raw.columns]) + 2
    term.line(f"Orbital element columns:      {term.fmt_n(raw_feature_count):>10}")
    term.line(f"Leakage features removed:     {term.fmt_n(len(leakage_removed)):>10}")
    term.line(f"Final model features:         {term.fmt_n(len(feature_cols)):>10}")
    if term.VERBOSE:
        term.detail("Feature names: " + ", ".join(feature_cols), indent=4)

    term.line("")
    term.info("Train/Test")
    term.line("-" * 60)
    term.line(f"Training objects:             {term.fmt_n(len(train_ids)):>10}")
    term.line(f"Testing objects:              {term.fmt_n(len(test_ids)):>10}")
    term.line(f"Training rows:                {term.fmt_n(len(train_df)):>10}")
    term.line(f"Testing rows:                 {term.fmt_n(len(test_df)):>10}")

    overlap = train_ids & test_ids
    term.line("")
    term.info("Leakage check")
    term.line("-" * 60)
    term.line(f"COSPAR overlap:               {term.fmt_n(len(overlap)):>10}")
    if overlap:
        term.warn("Object-level leakage detected — same COSPAR ID in train and test")
    else:
        term.ok("No object-level leakage detected")

    if summary.unmatched_discos_objects:
        term.warn(f"{summary.unmatched_discos_objects} DISCOS objects could not be matched to TLE data.")
    if summary.unmatched_tle_objects:
        term.warn(f"{summary.unmatched_tle_objects} TLE objects had no DISCOS metadata match.")

    term.line("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged training dataset")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 2 — DATA PREPARATION")

    try:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        term.step(1, STEPS, "Loading raw TLE data...")
        tle_raw, discos_raw, photo_raw = ingest_from_directory()
        term.ok(f"{term.fmt_n(len(tle_raw))} records loaded")
        term.detail(f"Shape: {tle_raw.shape}", indent=4)

        term.step(2, STEPS, "Loading DISCOS metadata...")
        term.ok(f"{term.fmt_n(len(discos_raw))} objects loaded")
        term.detail(f"Shape: {discos_raw.shape}", indent=4)
        if not photo_raw.empty:
            term.ok(f"{term.fmt_n(len(photo_raw))} photometric observations loaded")
        else:
            term.detail("No photometric_observations.csv — orbital-only features", indent=4)

        term.step(3, STEPS, "Validating identifiers...")
        if COSPAR_ID_COL not in tle_raw.columns or COSPAR_ID_COL not in discos_raw.columns:
            term.fail(
                "DATA PREPARATION FAILED",
                f"Missing required column: {COSPAR_ID_COL}",
                ["Expected cospar_id in data/raw/tle_history.csv", "Expected cospar_id in data/raw/discos_metadata.csv"],
            )
        term.ok("COSPAR IDs detected in both sources")

        term.step(4, STEPS, "Matching TLE and DISCOS objects...")
        merged_preview, summary = merge_tle_discos(tle_raw, discos_raw, how="inner")
        term.info(f"TLE objects:       {term.fmt_n(summary.total_tle_objects)}", indent=4)
        term.info(f"DISCOS objects:    {term.fmt_n(summary.total_discos_objects)}", indent=4)
        term.info(f"Matched:           {term.fmt_n(summary.matched_objects)}", indent=4)
        term.info(f"Unmatched TLE:     {term.fmt_n(summary.unmatched_tle_objects)}", indent=4)
        term.info(f"Unmatched DISCOS:  {term.fmt_n(summary.unmatched_discos_objects)}", indent=4)
        if summary.duplicate_tle_ids:
            term.detail(f"Duplicate TLE IDs (multi-epoch): {summary.duplicate_tle_ids}", indent=4)

        term.step(5, STEPS, "Cleaning data...")
        miss_before_tle = _missing_summary(tle_raw, ORBITAL_ELEMENT_COLS)
        miss_before_discos = _missing_summary(discos_raw, STAGE2_SIZE_TARGETS + ["mass", "shape", "true_mass", "true_shape"])
        _print_missing("Missing values before cleaning (TLE)", miss_before_tle)
        _print_missing("Missing values before cleaning (DISCOS)", miss_before_discos)

        with term.timed("Data cleaning"):
            tle = clean_tle_data(tle_raw)
            discos = clean_discos_data(discos_raw)

        miss_after_tle = _missing_summary(tle, ORBITAL_ELEMENT_COLS)
        miss_after_discos = _missing_summary(discos, STAGE2_SIZE_TARGETS + ["mass", "shape", "true_mass", "true_shape"])
        _print_missing("Missing values after cleaning (TLE)", miss_after_tle)
        _print_missing("Missing values after cleaning (DISCOS)", miss_after_discos)
        term.ok(f"TLE rows after cleaning: {term.fmt_n(len(tle))}")
        term.ok(f"DISCOS rows after cleaning: {term.fmt_n(len(discos))}")

        merged, summary = merge_tle_discos(tle, discos, how="inner")

        term.step(6, STEPS, "Calculating orbital features...")
        for feat in ["Semi-major axis", "Eccentricity", "Inclination", "Orbital period",
                     "Inclination drift", "Semi-major-axis decay", "Epoch count/span"]:
            term.detail(feat, indent=4)
        with term.timed("Feature engineering"):
            featured = compute_orbital_features(merged)
        term.ok("Orbital features generated")
        term.detail(f"Objects after feature aggregation: {term.fmt_n(len(featured))}", indent=4)

        # Merge ground-truth catalog + photometric observables (multi-modal)
        catalog = load_catalog()
        if catalog.empty and (DATA_RAW / "rso_catalog.csv").exists():
            catalog = pd.read_csv(DATA_RAW / "rso_catalog.csv")
        if not catalog.empty:
            truth_cols = [c for c in catalog.columns if c not in (COSPAR_ID_COL, CLASS_COL)]
            featured = featured.merge(catalog[[COSPAR_ID_COL] + truth_cols], on=COSPAR_ID_COL, how="left")
        else:
            rename_map = {
                "length": "true_length", "width": "true_width", "height": "true_height",
                "mass": "true_mass", "shape": "true_shape",
            }
            for old, new in rename_map.items():
                if old in featured.columns and new not in featured.columns:
                    featured[new] = featured[old]

        photo = photo_raw if not photo_raw.empty else load_photometric()
        if photo.empty and (DATA_RAW / "photometric_observations.csv").exists():
            photo = pd.read_csv(DATA_RAW / "photometric_observations.csv")
        if not photo.empty:
            photo_cols = [c for c in PHOTOMETRIC_FEATURE_COLS if c in photo.columns]
            photo_one = photo.groupby(COSPAR_ID_COL, as_index=False)[photo_cols].mean()
            featured = featured.merge(photo_one, on=COSPAR_ID_COL, how="left")
            term.ok(f"Merged photometric features: {', '.join(photo_cols)}")

        term.step(7, STEPS, "Applying data-leakage filter...")
        id_cols = [c for c in NON_FEATURE_COLUMNS if c in featured.columns]
        term.info("Removing:", indent=4)
        leakage_result = apply_leakage_guard(featured, id_columns=id_cols)
        for col in leakage_result.removed_columns:
            term.detail(f"- {col}", indent=8)
        term.ok(f"Leakage filtering complete — {term.fmt_n(leakage_result.remaining_feature_count)} model features remain")

        term.step(8, STEPS, "Preparing feature matrix...")
        term.ok("Categorical encoding deferred to sklearn Pipeline during model training")
        term.detail(f"Numeric/categorical imputation fit on train split only", indent=4)

        dataset = leakage_result.features.copy()
        for col in id_cols:
            if col in featured.columns:
                dataset[col] = featured[col].values
        for col, series in leakage_result.kept_target_columns.items():
            dataset[col] = series.values

        term.step(9, STEPS, "Creating object-level train/test split...")
        train_ids, test_ids = object_level_split(dataset[COSPAR_ID_COL], TEST_SIZE, RANDOM_SEED)
        term.info(f"Training objects: {term.fmt_n(len(train_ids))}", indent=4)
        term.info(f"Testing objects:  {term.fmt_n(len(test_ids))}", indent=4)
        overlap = train_ids & test_ids
        if overlap:
            term.warn("COSPAR IDs appear in both train and test")
        else:
            term.ok("No COSPAR ID appears in both train and test")

        train_df = dataset[dataset[COSPAR_ID_COL].isin(train_ids)].reset_index(drop=True)
        test_df = dataset[dataset[COSPAR_ID_COL].isin(test_ids)].reset_index(drop=True)

        term.step(10, STEPS, "Writing processed datasets...")
        train_path = DATA_PROCESSED / "train.csv"
        test_path = DATA_PROCESSED / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        (DATA_PROCESSED / "merge_summary.txt").write_text(str(summary))

        meta = {
            "train_objects": len(train_ids),
            "test_objects": len(test_ids),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "removed_leakage_columns": leakage_result.removed_columns,
            "feature_columns": list(leakage_result.features.columns),
            "photometric_features": [c for c in PHOTOMETRIC_FEATURE_COLS if c in leakage_result.features.columns],
            "split_strategy": "object_level_by_cospar_id",
        }
        (DATA_PROCESSED / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
        save_to_sqlite(train_df, "train")
        save_to_sqlite(test_df, "test")

        term.ok(str(train_path.relative_to(PROJECT_ROOT)))
        term.ok(str(test_path.relative_to(PROJECT_ROOT)))

        term.banner("DATA PREPARATION COMPLETE")
        _print_data_quality_summary(
            tle_raw, discos_raw, summary,
            leakage_result.removed_columns,
            list(leakage_result.features.columns),
            train_df, test_df, train_ids, test_ids,
        )
        timer.print_total()

    except FileNotFoundError as exc:
        term.fail("DATA PREPARATION FAILED", str(exc), [
            "data/raw/tle_history.csv",
            "data/raw/discos_metadata.csv",
            "Run scripts/fetch_data.py or scripts/generate_sample_data.py first",
        ])
    except Exception as exc:
        term.fail("DATA PREPARATION FAILED", str(exc))


if __name__ == "__main__":
    main()
