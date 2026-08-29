"""Sequential Stage 1 → Stage 2 inference pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.models.stage1_classifier import load_stage1_model, predict_with_confidence
from src.models.stage2_regressor import (
    load_stage2_models,
    predict_rotation,
    predict_shape,
    predict_size,
)

logger = logging.getLogger(__name__)

DEFAULT_STAGE1_MODEL = "lightgbm"


class RSOPipeline:
    """End-to-end RSO classification, sizing, shape, and rotation estimation."""

    def __init__(
        self,
        stage1_model: Any,
        label_encoder: Any,
        stage2_size_models: dict,
        stage2_targets: list[str],
        stage2_class_map: dict[str, str],
        stage2_shape_models: dict | None = None,
        stage2_rotation_models: dict | None = None,
        stage1_name: str = DEFAULT_STAGE1_MODEL,
    ):
        self.stage1_model = stage1_model
        self.label_encoder = label_encoder
        self.stage2_models = stage2_size_models
        self.stage2_targets = stage2_targets
        self.stage2_class_map = stage2_class_map
        self.stage2_shape_models = stage2_shape_models or {}
        self.stage2_rotation_models = stage2_rotation_models or {}
        self.stage1_name = stage1_name

    @classmethod
    def load(cls, models_dir: Path | str | None = None, stage1_name: str = DEFAULT_STAGE1_MODEL) -> "RSOPipeline":
        models_dir = Path(models_dir or MODELS_DIR)
        stage1 = load_stage1_model(stage1_name, models_dir)
        label_encoder = joblib.load(models_dir / "stage1_label_encoder.joblib")
        size_models, targets, class_map, shape_models, rotation_models = load_stage2_models(models_dir)
        return cls(
            stage1, label_encoder, size_models, targets, class_map,
            shape_models, rotation_models, stage1_name,
        )

    def predict_rso(self, input_data: pd.DataFrame) -> dict[str, Any]:
        """Run full pipeline on one or more objects."""
        t0 = time.perf_counter()
        if len(input_data) != 1:
            logger.warning("predict_rso called with %d rows; returning first row result only", len(input_data))

        X = input_data.iloc[[0]] if len(input_data) > 1 else input_data
        predicted_class, confidence = predict_with_confidence(self.stage1_model, X, self.label_encoder)

        cls = predicted_class[0]
        conf = float(confidence[0]) if not pd.isna(confidence[0]) else None

        size_raw = predict_size(self.stage2_models, cls, X, self.stage2_targets)
        dimensions = {
            "length": size_raw.get("true_length") or size_raw.get("length"),
            "width": size_raw.get("true_width") or size_raw.get("width"),
            "height": size_raw.get("true_height") or size_raw.get("height"),
        }
        shape = predict_shape(self.stage2_shape_models, cls, X)
        rotation_raw = predict_rotation(self.stage2_rotation_models, cls, X)
        tumbling_val = rotation_raw.get("tumbling")
        tumbling_label = None
        if tumbling_val is not None:
            tumbling_label = "Tumbling" if int(tumbling_val) == 1 else "Stable"

        latency = time.perf_counter() - t0
        stage2_model_key = self.stage2_class_map.get(cls, cls)

        return {
            "predicted_class": cls,
            "classification_confidence": conf,
            "stage1_model": self.stage1_name,
            "stage2_model": stage2_model_key if cls in self.stage2_models else None,
            "size_predictions": size_raw,
            "dimensions": dimensions,
            "shape": shape,
            "rotation": {
                "period_sec": rotation_raw.get("period_sec"),
                "tumbling": tumbling_label,
                "tumbling_flag": tumbling_val,
            },
            "latency_seconds": round(latency, 4),
        }


def predict_rso(input_data: pd.DataFrame, models_dir: Path | str | None = None) -> dict[str, Any]:
    pipeline = RSOPipeline.load(models_dir)
    return pipeline.predict_rso(input_data)
