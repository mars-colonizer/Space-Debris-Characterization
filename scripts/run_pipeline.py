#!/usr/bin/env python3
"""Run end-to-end Stage 1 → Stage 2 pipeline on a test object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_COL,
    COSPAR_ID_COL,
    DATA_PROCESSED,
    MODELS_DIR,
    NON_FEATURE_COLUMNS,
    STAGE2_PERIOD_TARGET,
    STAGE2_SHAPE_TARGET,
    STAGE2_SIZE_TARGETS,
    STAGE2_TUMBLING_TARGET,
    STAGE2_TARGETS,
)
from src.models.pipeline import RSOPipeline
from src.models.stage1_classifier import load_stage1_model, predict_with_confidence
from src.utils import terminal as term

STAGE1_NAME = "lightgbm"
STEPS = 8

LABEL_COLS = STAGE2_SIZE_TARGETS + [
    STAGE2_SHAPE_TARGET, STAGE2_PERIOD_TARGET, STAGE2_TUMBLING_TARGET,
    "length", "width", "height", "mass", "shape",
]


def _feature_row(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = NON_FEATURE_COLUMNS + [CLASS_COL] + [c for c in LABEL_COLS if c in df.columns]
    return df[[c for c in df.columns if c not in drop_cols]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end RSO inference")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 2 — END-TO-END RSO INFERENCE")

    try:
        test_path = DATA_PROCESSED / "test.csv"
        if not test_path.exists():
            term.fail("INFERENCE FAILED", "test.csv not found", ["Run scripts/prepare_dataset.py first"])
        if not (MODELS_DIR / f"stage1_{STAGE1_NAME}.joblib").exists():
            term.fail("INFERENCE FAILED", "Trained models not found", ["Run train_stage1.py and train_stage2.py first"])

        test_df = pd.read_csv(test_path)
        sample = test_df.iloc[[0]]
        cospar = sample[COSPAR_ID_COL].iloc[0]
        true_class = sample[CLASS_COL].iloc[0] if CLASS_COL in sample.columns else None

        term.info("Input object:")
        term.info(f"COSPAR ID: {cospar}", indent=4)
        if true_class:
            term.info(f"True class: {true_class}", indent=4)
        term.line()

        term.step(1, STEPS, "Loading trained preprocessing pipeline...")
        pipeline = RSOPipeline.load(MODELS_DIR, stage1_name=STAGE1_NAME)
        term.ok("Feature schema loaded from saved models")

        term.step(2, STEPS, "Loading Stage 1 classifier...")
        term.ok(f"Loaded models/stage1_{STAGE1_NAME}.joblib")

        term.step(3, STEPS, "Preparing input features...")
        X = _feature_row(sample)
        term.ok(f"Feature count: {term.fmt_n(X.shape[1])}")

        term.step(4, STEPS, "Running Stage 1 classification...")
        stage1 = load_stage1_model(STAGE1_NAME, MODELS_DIR)
        label_encoder = joblib.load(MODELS_DIR / "stage1_label_encoder.joblib")
        y_pred, confidence = predict_with_confidence(stage1, X, label_encoder)
        predicted_class = y_pred[0]
        conf = float(confidence[0]) if confidence[0] == confidence[0] else None
        term.ok(f"Predicted class: {predicted_class}")
        if conf is not None:
            term.ok(f"Confidence: {conf * 100:.2f}%")

        term.step(5, STEPS, "Selecting Stage 2 class-conditioned models...")
        stage2_key = pipeline.stage2_class_map.get(predicted_class)
        if predicted_class in pipeline.stage2_models and stage2_key:
            term.ok(f"Selected model: models/stage2/{stage2_key}.joblib")
        else:
            term.warn(f"No Stage 2 model available for class '{predicted_class}'")

        term.step(6, STEPS, "Estimating physical properties...")
        result = pipeline.predict_rso(X)
        dims = result.get("dimensions") or {}
        for dim in ("length", "width", "height"):
            val = dims.get(dim)
            if val is not None:
                term.info(f"{dim.capitalize()}: {val:.4f} m", indent=4)

        term.step(7, STEPS, "Estimating shape and rotation...")
        if result.get("shape"):
            term.info(f"Shape: {result['shape']}", indent=4)
        rot = result.get("rotation") or {}
        if rot.get("period_sec") is not None:
            term.info(f"Spin period: {rot['period_sec']:.2f} s", indent=4)
        if rot.get("tumbling"):
            term.info(f"Tumbling: {rot['tumbling']}", indent=4)

        term.step(8, STEPS, "Comparing with ground truth (if available)...")
        for dim, col in zip(("length", "width", "height"), STAGE2_SIZE_TARGETS):
            pred = dims.get(dim)
            actual = sample[col].iloc[0] if col in sample.columns else None
            if pred is not None and actual is not None and pd.notna(actual):
                term.info(f"{dim}: pred={pred:.4f} true={float(actual):.4f}", indent=4)

        term.banner("END-TO-END INFERENCE COMPLETE")
        term.info(f"Latency: {result.get('latency_seconds', 0):.4f} s")
        if term.VERBOSE:
            term.line(json.dumps(result, indent=2, default=str))
        timer.print_total()

    except Exception as exc:
        term.fail("INFERENCE FAILED", str(exc))


if __name__ == "__main__":
    main()
