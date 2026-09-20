#!/usr/bin/env python3
"""End-to-end inference using Phase 3 fused LightGBM + class-conditioned mass regressor."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_COL, COSPAR_ID_COL, DATA_PROCESSED, MODELS_DIR
from src.utils import terminal as term

FUSED_MODEL = MODELS_DIR / "stage1" / "lgbm_fused.pkl"
MATRIX_PATH = DATA_PROCESSED / "phase3_feature_matrix.csv"
FILL_VALUE = -1.0
STEPS = 5
# Prefer classes that have a Stage 2 mass regressor for the dashboard demo object.
_PREFERRED_CLASSES = ("Payload", "Rocket Body")


def _safe_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "class"


def _pick_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer an object whose true class has a trained mass regressor."""
    if CLASS_COL not in df.columns:
        return df.iloc[[0]].copy()
    for cls in _PREFERRED_CLASSES:
        reg = MODELS_DIR / "stage2" / f"regressor_{_safe_class_name(cls)}.pkl"
        if not reg.is_file():
            continue
        subset = df[df[CLASS_COL].astype(str) == cls]
        if "mass" in subset.columns:
            subset = subset[pd.to_numeric(subset["mass"], errors="coerce").notna()]
        if len(subset):
            return subset.iloc[[0]].copy()
    return df.iloc[[0]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end RSO inference (Phase 3)")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()
    t0 = time.time()

    term.banner("PHASE 3 — END-TO-END RSO INFERENCE")

    try:
        if not MATRIX_PATH.is_file():
            term.fail(
                "INFERENCE FAILED",
                f"{MATRIX_PATH} not found",
                ["Run scripts/prepare_dataset.py first"],
            )
        if not FUSED_MODEL.is_file():
            term.fail(
                "INFERENCE FAILED",
                f"{FUSED_MODEL} not found",
                ["Run scripts/run_ablation.py first"],
            )

        term.step(1, STEPS, "Loading fused LightGBM classifier...")
        bundle = joblib.load(FUSED_MODEL)
        model = bundle["model"]
        le = bundle["label_encoder"]
        feature_names = list(bundle["feature_names"])
        fill = float(bundle.get("fill_value", FILL_VALUE))
        term.ok(f"Loaded {FUSED_MODEL.relative_to(PROJECT_ROOT)}")

        term.step(2, STEPS, "Selecting test object from Phase 3 matrix...")
        df = pd.read_csv(MATRIX_PATH)
        sample = _pick_sample(df)
        cospar = sample[COSPAR_ID_COL].iloc[0] if COSPAR_ID_COL in sample.columns else "—"
        true_class = sample[CLASS_COL].iloc[0] if CLASS_COL in sample.columns else None
        # Dashboard log parser keys (keep wording stable).
        term.info(f"COSPAR ID: {cospar}", indent=4)
        if true_class is not None:
            term.info(f"True class: {true_class}", indent=4)

        term.step(3, STEPS, "Running Stage 1 classification...")
        X = sample[feature_names].apply(pd.to_numeric, errors="coerce").fillna(fill)
        pred_idx = int(model.predict(X)[0])
        predicted_class = str(le.inverse_transform([pred_idx])[0])
        conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf = float(proba[pred_idx]) * 100.0
        term.ok(f"Predicted class: {predicted_class}")
        if conf is not None:
            term.ok(f"Confidence: {conf:.2f}%")

        term.step(4, STEPS, "Running Stage 2 mass regression...")
        reg_path = MODELS_DIR / "stage2" / f"regressor_{_safe_class_name(predicted_class)}.pkl"
        mass_pred = None
        target_col = "mass"
        if reg_path.is_file():
            reg_bundle = joblib.load(reg_path)
            reg = reg_bundle["model"]
            reg_feats = list(reg_bundle.get("feature_names", feature_names))
            target_col = str(reg_bundle.get("target_col", "mass"))
            Xr = sample.reindex(columns=reg_feats).apply(pd.to_numeric, errors="coerce").fillna(fill)
            mass_pred = float(reg.predict(Xr)[0])
            term.ok(f"Selected model: {reg_path.relative_to(PROJECT_ROOT)}")
            # Parseable line for dashboard: "Mass: 123.4567 kg"
            term.info(f"Mass: {mass_pred:.4f} kg", indent=4)
        else:
            term.warn(f"No Stage 2 regressor for class '{predicted_class}' ({reg_path.name})")
            term.info("Mass: n/a", indent=4)

        # Photometric cues from the feature matrix (not Stage 2 outputs).
        if "median_period_sec" in sample.columns and pd.notna(sample["median_period_sec"].iloc[0]):
            term.info(f"Spin period: {float(sample['median_period_sec'].iloc[0]):.2f} s", indent=4)
        else:
            term.info("Spin period: n/a", indent=4)
        if "is_tumbling_consistent" in sample.columns and pd.notna(sample["is_tumbling_consistent"].iloc[0]):
            tumble = "yes" if int(sample["is_tumbling_consistent"].iloc[0]) else "no"
            term.info(f"Tumbling: {tumble}", indent=4)
        else:
            term.info("Tumbling: n/a", indent=4)

        term.step(5, STEPS, "Comparing with ground truth (if available)...")
        if true_class is not None:
            term.info(f"class: pred={predicted_class} true={true_class}", indent=4)
        if mass_pred is not None and target_col in sample.columns and pd.notna(sample[target_col].iloc[0]):
            term.info(
                f"{target_col}: pred={mass_pred:.4f} true={float(sample[target_col].iloc[0]):.4f}",
                indent=4,
            )

        term.banner("END-TO-END INFERENCE COMPLETE")
        term.info(f"Latency: {time.time() - t0:.4f} s")
        timer.print_total()

    except Exception as exc:
        term.fail("INFERENCE FAILED", str(exc))


if __name__ == "__main__":
    main()
