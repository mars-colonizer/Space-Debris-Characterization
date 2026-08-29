"""Stage 2 class-conditioned sizing, shape, and rotation models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src.config import (
    MIN_STAGE2_SAMPLES_PER_CLASS,
    RANDOM_SEED,
    STAGE2_PERIOD_TARGET,
    STAGE2_SHAPE_TARGET,
    STAGE2_SIZE_TARGETS,
    STAGE2_TUMBLING_TARGET,
    STAGE2_TARGETS,
)
from src.models.stage1_classifier import build_preprocessor

logger = logging.getLogger(__name__)


def _available_targets(df: pd.DataFrame, targets: list[str] | None = None) -> list[str]:
    targets = targets or STAGE2_TARGETS
    out = []
    for t in targets:
        if t in df.columns and df[t].notna().sum() >= MIN_STAGE2_SAMPLES_PER_CLASS:
            out.append(t)
        elif t in df.columns:
            logger.warning("Target '%s' has insufficient non-null samples — skipped", t)
        else:
            logger.warning("Target '%s' not found in dataset — skipped", t)
    return out


def predict_size(
    models: dict[str, Pipeline],
    predicted_class: str,
    X: pd.DataFrame,
    target_names: list[str],
) -> dict[str, float | None]:
    if predicted_class not in models:
        logger.warning("No Stage 2 size model for class '%s'", predicted_class)
        return {t: None for t in target_names}
    preds = models[predicted_class].predict(X)[0]
    return {name: float(val) for name, val in zip(target_names, preds)}


def predict_shape(
    models: dict[str, Pipeline],
    predicted_class: str,
    X: pd.DataFrame,
) -> str | None:
    if predicted_class not in models:
        return None
    return str(models[predicted_class].predict(X)[0])


def predict_rotation(
    models: dict[str, dict[str, Any]],
    predicted_class: str,
    X: pd.DataFrame,
) -> dict[str, float | int | None]:
    if predicted_class not in models:
        return {"period_sec": None, "tumbling": None}
    bundle = models[predicted_class]
    period = float(bundle["period"].predict(X)[0]) if bundle.get("period") else None
    tumbling = int(bundle["tumbling"].predict(X)[0]) if bundle.get("tumbling") else None
    return {"period_sec": period, "tumbling": tumbling}


def _safe_class_name(cls: str) -> str:
    return cls.lower().replace(" ", "_").replace("/", "_")


def save_stage2_models(
    size_models: dict[str, Pipeline],
    size_targets: list[str],
    output_dir: Path | str,
    shape_models: dict[str, Pipeline] | None = None,
    rotation_models: dict[str, dict[str, Any]] | None = None,
) -> None:
    output_dir = Path(output_dir)
    stage2_dir = output_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    class_map: dict[str, str] = {}
    for cls, model in size_models.items():
        safe = _safe_class_name(cls)
        class_map[cls] = safe
        joblib.dump(model, stage2_dir / f"{safe}.joblib")
        logger.info("Saved Stage 2 size model: %s", stage2_dir / f"{safe}.joblib")
    joblib.dump(size_targets, output_dir / "stage2_targets.joblib")
    joblib.dump(class_map, output_dir / "stage2_class_map.joblib")
    if shape_models:
        joblib.dump(shape_models, output_dir / "stage2_shape_models.joblib")
    if rotation_models:
        joblib.dump(rotation_models, output_dir / "stage2_rotation_models.joblib")


def load_stage2_models(models_dir: Path | str) -> tuple[dict, list[str], dict, dict, dict]:
    models_dir = Path(models_dir)
    targets = joblib.load(models_dir / "stage2_targets.joblib")
    class_map: dict[str, str] = joblib.load(models_dir / "stage2_class_map.joblib")
    size_models = {}
    for display_cls, safe_key in class_map.items():
        path = models_dir / "stage2" / f"{safe_key}.joblib"
        if path.exists():
            size_models[display_cls] = joblib.load(path)
    shape_path = models_dir / "stage2_shape_models.joblib"
    rotation_path = models_dir / "stage2_rotation_models.joblib"
    shape_models = joblib.load(shape_path) if shape_path.exists() else {}
    rotation_models = joblib.load(rotation_path) if rotation_path.exists() else {}
    return size_models, targets, class_map, shape_models, rotation_models


def build_size_pipeline(feature_names: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_names)),
        ("regressor", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, max_depth=10)
        )),
    ])


def build_shape_pipeline(feature_names: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_names)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=8)),
    ])


def build_rotation_pipelines(feature_names: list[str]) -> dict[str, Pipeline]:
    pre = build_preprocessor(feature_names)
    return {
        "period": Pipeline([
            ("preprocessor", pre),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, max_depth=8)),
        ]),
        "tumbling": Pipeline([
            ("preprocessor", build_preprocessor(feature_names)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=8)),
        ]),
    }
