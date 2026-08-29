#!/usr/bin/env python3
"""Train and evaluate Stage 1 RSO classification baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_COL,
    DATA_PROCESSED,
    MODELS_DIR,
    NON_FEATURE_COLUMNS,
    RESULTS_DIR,
    STAGE1_TARGET,
    STAGE2_PERIOD_TARGET,
    STAGE2_SHAPE_TARGET,
    STAGE2_SIZE_TARGETS,
    STAGE2_TUMBLING_TARGET,
    STAGE2_TARGETS,
)
from src.evaluation.metrics import evaluate_classifier, save_confusion_matrix, save_stage1_comparison
from src.models.stage1_classifier import (
    build_stage1_pipelines,
    predict_with_confidence,
    save_stage1_models,
)
from src.utils import terminal as term

MODEL_ORDER = [
    ("decision_tree", "DECISION TREE"),
    ("lightgbm", "LIGHTGBM"),
    ("adaboost", "ADABOOST"),
]


def _get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    label_drop = STAGE2_SIZE_TARGETS + [
        STAGE2_SHAPE_TARGET, STAGE2_PERIOD_TARGET, STAGE2_TUMBLING_TARGET,
        "length", "width", "height", "mass", "shape",
    ]
    drop_cols = NON_FEATURE_COLUMNS + [STAGE1_TARGET] + [c for c in label_drop if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols], df[STAGE1_TARGET], feature_cols


def _print_class_distribution(y: pd.Series, indent: int = 4) -> None:
    for cls, n in y.value_counts().items():
        term.info(f"{cls}: {term.fmt_n(n)}", indent=indent)


def _best_model(metrics_list: list[dict]) -> str:
    if not metrics_list:
        return "No models trained"
    top_f1 = max(m["f1"] for m in metrics_list)
    winners = [m["model"] for m in metrics_list if m["f1"] == top_f1]
    if len(winners) > 1:
        return "No clear winner (metrics tied)"
    return winners[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 classifiers")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 2 — STAGE 1 CLASSIFICATION")
    term.info(f"Target: {STAGE1_TARGET.replace('_', ' ').title()}")
    term.line()

    try:
        train_path = DATA_PROCESSED / "train.csv"
        test_path = DATA_PROCESSED / "test.csv"
        if not train_path.exists():
            term.fail("STAGE 1 TRAINING FAILED", f"{train_path} not found", ["Run scripts/prepare_dataset.py first"])

        term.info("Loading training data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        X_train, y_train, feature_names = _get_xy(train_df)
        X_test, y_test, _ = _get_xy(test_df)

        term.ok(f"Training samples: {term.fmt_n(len(X_train))}")
        term.ok(f"Test samples: {term.fmt_n(len(X_test))}")
        term.ok(f"Features: {term.fmt_n(len(feature_names))}")
        term.ok(f"Classes: {term.fmt_n(y_train.nunique())}")
        term.line()
        term.info("Class distribution (train):")
        _print_class_distribution(y_train)
        term.info("Class distribution (test):")
        _print_class_distribution(y_test)
        for cls, n in y_test.value_counts().items():
            if n < 5:
                term.warn(f"Class '{cls}' contains only {n} test sample(s). Per-class metrics may be unreliable.")
        term.detail(f"Feature names: {feature_names}")

        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y_train)
        pipelines = build_stage1_pipelines(feature_names)
        trained: dict = {}
        metrics_list = []

        for idx, (key, display_name) in enumerate(MODEL_ORDER, start=1):
            term.section(f"MODEL {idx}/{len(MODEL_ORDER)} — {display_name}")
            pipe = pipelines[key]

            term.step(1, 4, "Building model...")
            term.detail(repr(pipe.named_steps["clf"]), indent=4)
            term.ok("Pipeline ready")

            term.step(2, 4, "Training...")
            with term.timed(f"{display_name} training"):
                pipe.fit(X_train, y_enc)
            trained[key] = pipe
            term.ok("Training complete")

            term.step(3, 4, "Running inference...")
            y_pred, _ = predict_with_confidence(pipe, X_test, label_encoder)
            term.ok("Inference complete")

            term.step(4, 4, "Calculating metrics...")
            m = evaluate_classifier(y_test, y_pred, display_name.title())
            metrics_list.append(m)
            term.line()
            term.info("Results:", indent=0)
            term.info(f"Accuracy:  {m['accuracy']:.4f}", indent=4)
            term.info(f"Precision: {m['precision']:.4f}", indent=4)
            term.info(f"Recall:    {m['recall']:.4f}", indent=4)
            term.info(f"F1 Score:  {m['f1']:.4f}", indent=4)

            term.info("Generating confusion matrix...")
            labels = sorted(set(y_test) | set(y_pred))
            term.detail(f"Class labels: {', '.join(str(l) for l in labels)}", indent=4)
            cm_path = save_confusion_matrix(y_test, y_pred, key, RESULTS_DIR)
            term.ok(f"Saved: {cm_path.relative_to(PROJECT_ROOT)}")

        save_stage1_models(trained, label_encoder, MODELS_DIR)
        for key, display_name in MODEL_ORDER:
            term.ok(f"Model saved: models/stage1_{key}.joblib")

        metrics_path = RESULTS_DIR / "stage1_metrics.csv"
        save_stage1_comparison(metrics_list, metrics_path)
        (MODELS_DIR / "feature_names.json").write_text(
            json.dumps({"feature_names": feature_names}, indent=2)
        )

        term.banner("STAGE 1 — MODEL COMPARISON")
        rows = [
            [m["model"], f"{m['accuracy']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"]
            for m in metrics_list
        ]
        term.print_table(["Model", "Accuracy", "Precision", "Recall", "F1"], rows)
        term.line()
        term.ok(f"Best baseline: {_best_model(metrics_list)}", force=True)
        term.ok(str(metrics_path.relative_to(PROJECT_ROOT)), force=True)
        term.line("=" * 60)
        timer.print_total()

    except Exception as exc:
        term.fail("STAGE 1 TRAINING FAILED", str(exc))


if __name__ == "__main__":
    main()
