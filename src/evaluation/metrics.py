"""Evaluation metrics for classification and regression."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate_classifier(y_true, y_pred, model_name: str) -> dict:
    """Compute multiclass classification metrics."""
    labels = sorted(set(y_true) | set(y_pred))
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    logger.info("\n%s classification report:\n%s", model_name, classification_report(y_true, y_pred, zero_division=0))
    return metrics


def save_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save confusion matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = output_dir / f"stage1_{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate_regressor(y_true, y_pred, class_label: str, target: str) -> dict:
    """Compute regression metrics for one class/target pair."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {"class": class_label, "target": target, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "class": class_label,
        "target": target,
        "mae": mean_absolute_error(yt, yp),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": r2_score(yt, yp) if len(yt) > 1 else np.nan,
        "n": int(mask.sum()),
    }


def save_stage1_comparison(metrics_list: list[dict], path: Path) -> None:
    """Save Stage 1 comparison table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_list).to_csv(path, index=False)
    logger.info("Saved Stage 1 metrics to %s", path)


def save_stage2_metrics(metrics_list: list[dict], path: Path) -> None:
    """Save Stage 2 per-class/target metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_list).to_csv(path, index=False)
    logger.info("Saved Stage 2 metrics to %s", path)
