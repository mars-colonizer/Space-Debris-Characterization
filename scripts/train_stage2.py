#!/usr/bin/env python3
"""Train class-conditioned Stage 2 sizing, shape, and rotation models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_COL,
    DATA_PROCESSED,
    MIN_STAGE2_SAMPLES_PER_CLASS,
    MODELS_DIR,
    NON_FEATURE_COLUMNS,
    RANDOM_SEED,
    RESULTS_DIR,
    STAGE2_PERIOD_TARGET,
    STAGE2_SHAPE_TARGET,
    STAGE2_SIZE_TARGETS,
    STAGE2_TUMBLING_TARGET,
    STAGE2_TARGETS,
)
from src.evaluation.metrics import evaluate_regressor, save_stage2_metrics
from src.models.stage1_classifier import build_preprocessor, load_stage1_model, predict_with_confidence
from src.models.stage2_regressor import (
    _available_targets,
    _safe_class_name,
    build_rotation_pipelines,
    build_shape_pipeline,
    build_size_pipeline,
    save_stage2_models,
)
from src.utils import terminal as term

ALL_LABEL_COLS = (
    STAGE2_SIZE_TARGETS
    + [STAGE2_SHAPE_TARGET, STAGE2_PERIOD_TARGET, STAGE2_TUMBLING_TARGET]
    + ["length", "width", "height", "mass", "shape"]
)


def _get_xy(df: pd.DataFrame, size_targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    drop_cols = NON_FEATURE_COLUMNS + [CLASS_COL] + [c for c in ALL_LABEL_COLS if c not in size_targets]
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in ALL_LABEL_COLS]
    return df[feature_cols], df[size_targets], feature_cols


def _fmt_metric(val: float | None) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return f"{val:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 models")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 2 — STAGE 2 PHYSICAL CHARACTERIZATION")

    try:
        train_path = DATA_PROCESSED / "train.csv"
        test_path = DATA_PROCESSED / "test.csv"
        if not train_path.exists():
            term.fail("STAGE 2 TRAINING FAILED", f"{train_path} not found", ["Run scripts/prepare_dataset.py first"])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        available_targets = [t for t in STAGE2_SIZE_TARGETS if t in train_df.columns and train_df[t].notna().sum() > 0]
        if not available_targets:
            # fallback legacy column names
            legacy = {"length": "true_length", "width": "true_width", "height": "true_height"}
            for old, new in legacy.items():
                if old in train_df.columns and new not in train_df.columns:
                    train_df[new] = train_df[old]
                    test_df[new] = test_df[old]
            available_targets = [t for t in STAGE2_SIZE_TARGETS if t in train_df.columns and train_df[t].notna().sum() > 0]
        if not available_targets:
            term.fail("STAGE 2 TRAINING FAILED", "No Stage 2 size targets available in training data")

        term.info("Size targets:")
        for t in available_targets:
            term.detail(t, indent=4)
        term.line()

        all_classes = sorted(train_df[CLASS_COL].dropna().unique())
        X_train, y_train, feature_names = _get_xy(train_df, available_targets)
        targets = _available_targets(y_train, available_targets)
        if not targets:
            term.fail("STAGE 2 TRAINING FAILED", "No valid regression targets with sufficient samples")

        size_models: dict[str, Pipeline] = {}
        shape_models: dict[str, Pipeline] = {}
        rotation_models: dict[str, dict] = {}
        skipped: list[str] = []
        class_metrics: list[dict] = []

        for idx, cls in enumerate(all_classes, start=1):
            term.section(f"CLASS {idx}/{len(all_classes)} — {cls.upper()}")
            mask = train_df[CLASS_COL] == cls
            n = int(mask.sum())
            term.line(f"Training samples: {term.fmt_n(n)}")

            if n < MIN_STAGE2_SAMPLES_PER_CLASS:
                term.warn(f"Insufficient samples for class '{cls}' ({n} < {MIN_STAGE2_SAMPLES_PER_CLASS})")
                skipped.append(cls)
                continue

            y_cls = y_train.loc[mask, targets]
            complete = y_cls.notna().all(axis=1)
            if complete.sum() < MIN_STAGE2_SAMPLES_PER_CLASS:
                term.warn(f"Insufficient complete target rows for class '{cls}'")
                skipped.append(cls)
                continue

            X_cls = X_train.loc[mask].loc[complete]
            y_cls = y_cls.loc[complete]

            term.step(1, 4, "Training size regressors...")
            size_pipe = build_size_pipeline(feature_names)
            with term.timed(f"{cls} size training"):
                size_pipe.fit(X_cls, y_cls.values)
            size_models[cls] = size_pipe
            term.ok("Size model trained")

            term.step(2, 4, "Training shape classifier...")
            if STAGE2_SHAPE_TARGET in train_df.columns:
                y_shape = train_df.loc[X_cls.index, STAGE2_SHAPE_TARGET].dropna()
                if len(y_shape) >= MIN_STAGE2_SAMPLES_PER_CLASS:
                    shape_pipe = build_shape_pipeline(feature_names)
                    if y_shape.nunique() < 2:
                        shape_pipe = Pipeline([
                            ("preprocessor", build_preprocessor(feature_names)),
                            ("clf", DummyClassifier(strategy="most_frequent")),
                        ])
                    shape_pipe.fit(X_cls.loc[y_shape.index], y_shape)
                    shape_models[cls] = shape_pipe
                    term.ok(f"Shape model trained ({y_shape.nunique()} label(s))")
                else:
                    term.warn("Shape target insufficient — skipped")
            else:
                term.warn("No true_shape column — shape model skipped")

            term.step(3, 4, "Training rotation estimator...")
            rot_bundle: dict = {}
            if STAGE2_PERIOD_TARGET in train_df.columns:
                y_period = train_df.loc[X_cls.index, STAGE2_PERIOD_TARGET].dropna()
                if len(y_period) >= MIN_STAGE2_SAMPLES_PER_CLASS:
                    pipes = build_rotation_pipelines(feature_names)
                    pipes["period"].fit(X_cls.loc[y_period.index], y_period)
                    rot_bundle["period"] = pipes["period"]
                    term.ok("Spin period regressor trained")
            if STAGE2_TUMBLING_TARGET in train_df.columns:
                y_tumble = train_df.loc[X_cls.index, STAGE2_TUMBLING_TARGET].dropna().astype(int)
                if len(y_tumble) >= MIN_STAGE2_SAMPLES_PER_CLASS:
                    pipes = build_rotation_pipelines(feature_names)
                    if y_tumble.nunique() < 2:
                        pipes["tumbling"] = Pipeline([
                            ("preprocessor", build_preprocessor(feature_names)),
                            ("clf", DummyClassifier(strategy="most_frequent")),
                        ])
                    pipes["tumbling"].fit(X_cls.loc[y_tumble.index], y_tumble)
                    rot_bundle["tumbling"] = pipes["tumbling"]
                    term.ok("Tumbling classifier trained")
            if rot_bundle:
                rotation_models[cls] = rot_bundle

            term.step(4, 4, "In-sample size metrics...")
            preds = size_pipe.predict(X_cls)
            for i, target in enumerate(targets):
                m = evaluate_regressor(y_cls[target], preds[:, i], cls, target)
                class_metrics.append(m)
                term.info(f"{target}: MAE={_fmt_metric(m['mae'])} R²={_fmt_metric(m['r2'])}", indent=4)

            term.ok(f"Models saved: models/stage2/{_safe_class_name(cls)}*.joblib")

        if not size_models:
            term.fail("STAGE 2 TRAINING FAILED", "No class-conditioned models could be trained")

        save_stage2_models(size_models, targets, MODELS_DIR, shape_models, rotation_models)

        X_test, y_test, _ = _get_xy(test_df, targets)
        metrics_list = []
        for cls in size_models:
            mask = test_df[CLASS_COL] == cls
            if mask.sum() == 0:
                continue
            preds = size_models[cls].predict(X_test.loc[mask])
            for i, target in enumerate(targets):
                m = evaluate_regressor(y_test.loc[mask, target], preds[:, i], cls, target)
                m["eval_mode"] = "true_class"
                metrics_list.append(m)

        metrics_path = RESULTS_DIR / "stage2_metrics.csv"
        save_stage2_metrics(metrics_list, metrics_path)

        term.banner("STAGE 2 — SUMMARY")
        term.print_table(
            ["Class", "Samples", "Size", "Shape", "Rotation"],
            [[
                cls,
                str(int((train_df[CLASS_COL] == cls).sum())),
                "OK" if cls in size_models else "SKIP",
                "OK" if cls in shape_models else "SKIP",
                "OK" if cls in rotation_models else "SKIP",
            ] for cls in all_classes],
        )
        term.ok(f"Size models: {term.fmt_n(len(size_models))}", force=True)
        term.ok(f"Shape models: {term.fmt_n(len(shape_models))}", force=True)
        term.ok(f"Rotation models: {term.fmt_n(len(rotation_models))}", force=True)
        timer.print_total()

    except Exception as exc:
        term.fail("STAGE 2 TRAINING FAILED", str(exc))


if __name__ == "__main__":
    main()
