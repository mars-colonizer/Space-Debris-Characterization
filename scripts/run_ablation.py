#!/usr/bin/env python3
"""Stage 1 Phase 3 ablation: orbital vs photometric vs fused under GroupKFold(launch_group)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_COL, DATA_PROCESSED, MODELS_DIR, PHOTOMETRIC_FEATURE_COLS, RANDOM_SEED
from src.utils import terminal as term

MATRIX_PATH = DATA_PROCESSED / "phase3_feature_matrix.csv"
LAUNCH_GROUP_COL = "launch_group"
FILL_VALUE = -1.0
N_SPLITS = 4

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

PHOTOMETRIC_FEATURES = list(PHOTOMETRIC_FEATURE_COLS)

LGBM_PARAMS = dict(
    random_state=RANDOM_SEED,
    verbose=-1,
    min_child_samples=2,
    num_leaves=7,
    max_depth=3,
    n_estimators=50,
)


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    missing_orb = [c for c in ORBITAL_FEATURES if c not in df.columns]
    missing_photo = [c for c in PHOTOMETRIC_FEATURES if c not in df.columns]
    if missing_orb:
        raise ValueError(f"Missing orbital features: {missing_orb}")
    if missing_photo:
        raise ValueError(
            f"Missing photometric features: {missing_photo}. "
            "Re-run scripts/prepare_dataset.py to rebuild phase3_feature_matrix.csv."
        )
    return {
        "orbital_features": list(ORBITAL_FEATURES),
        "photometric_features": list(PHOTOMETRIC_FEATURES),
        "fused_features": list(ORBITAL_FEATURES) + list(PHOTOMETRIC_FEATURES),
    }


def build_models() -> dict[str, object]:
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_SEED, n_estimators=30),
        "LightGBM": LGBMClassifier(**LGBM_PARAMS),
    }


def group_cv_macro_f1(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_SPLITS,
) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = model.__class__(**model.get_params())
        clf.fit(X.iloc[train_idx], y[train_idx])
        pred = clf.predict(X.iloc[test_idx])
        scores.append(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)))
    return float(np.mean(scores)) if scores else float("nan")


def print_markdown_table(rows: list[dict[str, float | str]]) -> None:
    term.line("")
    term.line("| Model | Orbital F1 | Photometric F1 | Fused F1 |")
    term.line("|---|---:|---:|---:|")
    for row in rows:
        term.line(
            f"| {row['model']} | {row['orbital']:.4f} | {row['photometric']:.4f} | {row['fused']:.4f} |"
        )
    term.line("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Phase 3 ablation study")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 3 — STAGE 1 ABLATION STUDY")

    try:
        if not MATRIX_PATH.is_file():
            term.fail(
                "ABLATION FAILED",
                f"{MATRIX_PATH} not found",
                ["Run scripts/prepare_dataset.py first"],
            )

        df = pd.read_csv(MATRIX_PATH)
        term.ok(f"Loaded {MATRIX_PATH.relative_to(PROJECT_ROOT)} — shape {df.shape}")

        if CLASS_COL not in df.columns or LAUNCH_GROUP_COL not in df.columns:
            raise ValueError(f"Matrix must contain {CLASS_COL} and {LAUNCH_GROUP_COL}")

        sets = feature_sets(df)
        for name, cols in sets.items():
            term.info(f"{name}: {len(cols)} cols", indent=4)
            if term.VERBOSE:
                term.detail(", ".join(cols), indent=8)

        y_raw = df[CLASS_COL].astype(str)
        groups = df[LAUNCH_GROUP_COL].astype(str).to_numpy()
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        term.ok(f"Classes: {list(le.classes_)} | launch groups: {pd.Series(groups).nunique()}")

        n_groups = int(pd.Series(groups).nunique())
        n_splits = min(N_SPLITS, n_groups)
        if n_splits < 2:
            raise ValueError(f"Need ≥2 launch groups for GroupKFold; got {n_groups}")
        if n_splits < N_SPLITS:
            term.warn(f"Only {n_groups} launch groups — using GroupKFold(n_splits={n_splits})")

        models = build_models()
        table_rows: list[dict[str, float | str]] = []

        term.step(1, 3, f"Running GroupKFold({n_splits}) ablation...")
        for model_name, model in models.items():
            row: dict[str, float | str] = {"model": model_name}
            for set_key, col_key in (
                ("orbital_features", "orbital"),
                ("photometric_features", "photometric"),
                ("fused_features", "fused"),
            ):
                cols = sets[set_key]
                X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(FILL_VALUE)
                score = group_cv_macro_f1(model, X, y, groups, n_splits=n_splits)
                row[col_key] = score
                term.detail(f"{model_name} / {set_key}: macro-F1={score:.4f}", indent=4)
            table_rows.append(row)

        term.step(2, 3, "Ablation results (macro F1)")
        print_markdown_table(table_rows)

        term.step(3, 3, "Fit final LightGBM on fused features...")
        fused_cols = sets["fused_features"]
        X_full = df[fused_cols].apply(pd.to_numeric, errors="coerce").fillna(FILL_VALUE)
        final = LGBMClassifier(**LGBM_PARAMS)
        final.fit(X_full, y)

        out_dir = MODELS_DIR / "stage1"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "lgbm_fused.pkl"
        joblib.dump(
            {
                "model": final,
                "label_encoder": le,
                "feature_names": fused_cols,
                "fill_value": FILL_VALUE,
                "classes": list(le.classes_),
            },
            out_path,
        )
        term.ok(f"Saved {out_path.relative_to(PROJECT_ROOT)}")

        importances = pd.Series(final.feature_importances_, index=fused_cols).sort_values(
            ascending=False
        )
        term.info("LightGBM fused feature importances (descending):", indent=4)
        for name, val in importances.items():
            term.detail(f"{name}: {val}", indent=8)

        term.banner("ABLATION STUDY COMPLETE")
        timer.print_total()

    except Exception as exc:
        term.fail("ABLATION FAILED", str(exc))


if __name__ == "__main__":
    main()
