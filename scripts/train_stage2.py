#!/usr/bin/env python3
"""Phase 3 Stage 2: class-conditioned physical regression (LOOCV, small-N RF)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_COL,
    DATA_PROCESSED,
    MODELS_DIR,
    PHOTOMETRIC_FEATURE_COLS,
    RANDOM_SEED,
)
from src.utils import terminal as term

MATRIX_PATH = DATA_PROCESSED / "phase3_feature_matrix.csv"
OUT_DIR = MODELS_DIR / "stage2"
FILL_VALUE = -1.0
MIN_SAMPLES = 5

ORBITAL_FEATURES = [
    "inclination",
    "eccentricity",
    "semi_major_axis",
    "raan",
    "arg_perigee",
    "mean_anomaly",
    "orbital_period_days",
    "inclination_drift_deg_per_day",
    "sma_decay_km_per_day",
    "epoch_count",
    "epoch_span_days",
]
FUSED_FEATURES = ORBITAL_FEATURES + list(PHOTOMETRIC_FEATURE_COLS)

# Preference order for continuous DISCOS physical targets.
TARGET_CANDIDATES = [
    "mass",
    "true_mass",
    "length",
    "true_length",
    "width",
    "true_width",
    "height",
    "true_height",
    "cross_section",
    "rcs",
]

CLASS_ORDER = ["Payload", "Rocket Body", "Debris"]


def _safe_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "class"


def select_target_col(df: pd.DataFrame) -> str:
    """Pick one continuous numerical physical property present in the matrix."""
    found: list[tuple[str, int]] = []
    for col in TARGET_CANDIDATES:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        n = int(series.notna().sum())
        if n == 0:
            continue
        # Require real numeric variation (skip constant / all-NaN after coerce).
        if series.dropna().nunique() < 2:
            term.detail(f"skip {col}: <2 unique numeric values", indent=4)
            continue
        found.append((col, n))
        term.detail(f"candidate {col}: {n} non-null", indent=4)

    if not found:
        raise ValueError(
            "No continuous physical target found "
            f"(looked for {TARGET_CANDIDATES}). Check DISCOS columns in the matrix."
        )
    # Prefer mass* then first candidate with the most labels.
    for preferred in ("mass", "true_mass"):
        for col, n in found:
            if col == preferred:
                return col
    found.sort(key=lambda x: (-x[1], TARGET_CANDIDATES.index(x[0]) if x[0] in TARGET_CANDIDATES else 99))
    return found[0][0]


def build_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=50,
        max_depth=3,
        random_state=RANDOM_SEED,
    )


def loocv_metrics(X: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    model = build_regressor()
    preds = cross_val_predict(model, X, y, cv=LeaveOneOut())
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return mae, r2


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Stage 2 physical regression")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 3 — STAGE 2 PHYSICAL REGRESSION")

    try:
        if not MATRIX_PATH.is_file():
            term.fail(
                "STAGE 2 TRAINING FAILED",
                f"{MATRIX_PATH} not found",
                ["Run scripts/prepare_dataset.py first"],
            )

        df = pd.read_csv(MATRIX_PATH)
        term.ok(f"Loaded {MATRIX_PATH.relative_to(PROJECT_ROOT)} — shape {df.shape}")

        missing_feats = [c for c in FUSED_FEATURES if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Matrix missing fused features: {missing_feats}")
        if CLASS_COL not in df.columns:
            raise ValueError(f"Matrix missing {CLASS_COL}")

        term.info("Scanning DISCOS physical targets...")
        target_col = select_target_col(df)
        term.ok(f"Primary regression target: {target_col}")

        work = df.copy()
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        before = len(work)
        work = work.dropna(subset=[target_col]).reset_index(drop=True)
        term.ok(f"Rows with {target_col}: {term.fmt_n(len(work))} (dropped {term.fmt_n(before - len(work))})")

        X_all = work[FUSED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(FILL_VALUE)
        y_all = work[target_col].to_numpy(dtype=float)

        classes = [c for c in CLASS_ORDER if c in set(work[CLASS_COL].astype(str))]
        for extra in sorted(set(work[CLASS_COL].astype(str)) - set(classes)):
            classes.append(extra)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []

        for cls in classes:
            mask = work[CLASS_COL].astype(str) == cls
            n = int(mask.sum())
            term.section(f"{cls} (n={n})")

            if n < MIN_SAMPLES:
                term.warn(f"Skipping '{cls}' — {n} < {MIN_SAMPLES} valid samples")
                rows.append(
                    {
                        "class": cls,
                        "target": target_col,
                        "n": n,
                        "mae": None,
                        "r2": None,
                        "skipped": True,
                    }
                )
                continue

            X = X_all.loc[mask]
            y = y_all[mask.to_numpy()]

            term.step(1, 2, "Leave-One-Out CV...")
            mae, r2 = loocv_metrics(X, y)
            term.ok(f"LOOCV MAE={mae:.4f}  R²={r2:.4f}")

            term.step(2, 2, "Fit final regressor on all class samples...")
            final = build_regressor()
            final.fit(X, y)
            out_path = OUT_DIR / f"regressor_{_safe_class_name(cls)}.pkl"
            joblib.dump(
                {
                    "model": final,
                    "target_col": target_col,
                    "feature_names": list(FUSED_FEATURES),
                    "object_class": cls,
                    "fill_value": FILL_VALUE,
                    "n_samples": n,
                    "loocv_mae": mae,
                    "loocv_r2": r2,
                },
                out_path,
            )
            term.ok(f"Saved {out_path.relative_to(PROJECT_ROOT)}")

            rows.append(
                {
                    "class": cls,
                    "target": target_col,
                    "n": n,
                    "mae": mae,
                    "r2": r2,
                    "skipped": False,
                }
            )

        term.banner("STAGE 2 — LOOCV RESULTS")
        term.line("")
        term.line("| Class | Target Variable | N-samples | LOOCV MAE | LOOCV R2 |")
        term.line("|---|---|---:|---:|---:|")
        for row in rows:
            if row["skipped"]:
                term.line(
                    f"| {row['class']} | {row['target']} | {row['n']} | — (skipped) | — |"
                )
            else:
                term.line(
                    f"| {row['class']} | {row['target']} | {row['n']} | "
                    f"{row['mae']:.4f} | {row['r2']:.4f} |"
                )
        term.line("")

        trained = sum(1 for r in rows if not r["skipped"])
        if trained == 0:
            term.fail("STAGE 2 TRAINING FAILED", "No class had enough samples to train")

        term.banner("STAGE 2 TRAINING COMPLETE")
        timer.print_total()

    except Exception as exc:
        term.fail("STAGE 2 TRAINING FAILED", str(exc))


if __name__ == "__main__":
    main()
