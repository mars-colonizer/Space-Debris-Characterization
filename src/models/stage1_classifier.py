"""Stage 1 RSO classification baselines."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import CATEGORICAL_FEATURES, RANDOM_SEED

logger = logging.getLogger(__name__)


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Build sklearn preprocessor: impute numerics, one-hot categoricals."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    num_cols = [c for c in feature_names if c not in cat_cols]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _classifier_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "decision_tree": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=12)),
        ]),
        "lightgbm": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", lgb.LGBMClassifier(
                random_state=RANDOM_SEED,
                n_estimators=100,
                verbose=-1,
                class_weight="balanced",
            )),
        ]),
        "adaboost": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED),
                n_estimators=100,
                random_state=RANDOM_SEED,
            )),
        ]),
    }


def build_stage1_pipelines(feature_names: list[str]) -> dict[str, Pipeline]:
    """Build unfitted Stage 1 pipelines (one per baseline model)."""
    preprocessor = build_preprocessor(feature_names)
    return _classifier_pipelines(preprocessor)


def train_stage1_models(
    X_train,
    y_train,
    feature_names: list[str],
) -> tuple[dict[str, Pipeline], LabelEncoder]:
    """Train all Stage 1 baseline classifiers."""
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_train)

    models = build_stage1_pipelines(feature_names)
    trained: dict[str, Pipeline] = {}
    for name, pipe in models.items():
        logger.info("Training Stage 1 model: %s", name)
        pipe.fit(X_train, y_enc)
        trained[name] = pipe

    return trained, label_encoder


def predict_with_confidence(model: Pipeline, X, label_encoder: LabelEncoder) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class labels and max probability (when available)."""
    y_enc = model.predict(X)
    classes = label_encoder.inverse_transform(y_enc)

    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = proba.max(axis=1)
    else:
        confidence = np.full(len(classes), np.nan)
        logger.warning("Classifier has no predict_proba — confidence set to NaN")

    return classes, confidence


def save_stage1_models(
    models: dict[str, Pipeline],
    label_encoder: LabelEncoder,
    output_dir,
) -> None:
    """Serialize Stage 1 models and label encoder."""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = output_dir / f"stage1_{name}.joblib"
        joblib.dump(model, path)
        logger.info("Saved %s", path)
    joblib.dump(label_encoder, output_dir / "stage1_label_encoder.joblib")


def load_stage1_model(name: str, models_dir) -> Pipeline:
    from pathlib import Path
    return joblib.load(Path(models_dir) / f"stage1_{name}.joblib")
