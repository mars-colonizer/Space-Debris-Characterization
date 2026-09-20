#!/usr/bin/env python3
"""Phase 3 explainability: GroupKFold confusion matrix + SHAP for fused LightGBM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

try:
    import shap
except ImportError:  # optional — confusion matrix still runs without it
    shap = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_COL, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR
from src.utils import terminal as term

MODEL_PATH = MODELS_DIR / "stage1" / "lgbm_fused.pkl"
MATRIX_PATH = DATA_PROCESSED / "phase3_feature_matrix.csv"
OUT_DIR = RESULTS_DIR / "phase3"
LAUNCH_GROUP_COL = "launch_group"
N_SPLITS = 4
DEFAULT_FILL = -1.0


def load_bundle() -> dict:
    if not MODEL_PATH.is_file():
        term.fail(
            "EXPLAINABILITY FAILED",
            f"{MODEL_PATH} not found",
            ["Run scripts/run_ablation.py first"],
        )
    return joblib.load(MODEL_PATH)


def load_xy(bundle: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if not MATRIX_PATH.is_file():
        term.fail(
            "EXPLAINABILITY FAILED",
            f"{MATRIX_PATH} not found",
            ["Run scripts/prepare_dataset.py first"],
        )
    df = pd.read_csv(MATRIX_PATH)
    feature_names: list[str] = list(bundle["feature_names"])
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Matrix missing fused features: {missing}")
    if CLASS_COL not in df.columns or LAUNCH_GROUP_COL not in df.columns:
        raise ValueError(f"Matrix must contain {CLASS_COL} and {LAUNCH_GROUP_COL}")

    fill = float(bundle.get("fill_value", DEFAULT_FILL))
    X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(fill)
    le = bundle["label_encoder"]
    y = le.transform(df[CLASS_COL].astype(str))
    groups = df[LAUNCH_GROUP_COL].astype(str).to_numpy()
    return X, y, groups, list(le.classes_)


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Phase 3 LightGBM (fused) — GroupKFold OOF")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_shap_plots(model, X: pd.DataFrame, class_names: list[str], out_dir: Path) -> tuple[Path, Path]:
    if shap is None:
        raise ImportError(
            "shap is not installed in this environment. "
            "Install with: pip install 'shap>=0.44'"
        )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    bar_path = out_dir / "shap_feature_importance.png"
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        class_names=class_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    dot_path = out_dir / "shap_summary_dot.png"
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        class_names=class_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(dot_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return bar_path, dot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 explainability and evaluation")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 3 — EXPLAINABILITY & EVALUATION")

    try:
        bundle = load_bundle()
        model = bundle["model"]
        if not isinstance(model, LGBMClassifier):
            raise TypeError(f"Expected LGBMClassifier in {MODEL_PATH.name}, got {type(model)}")

        X, y, groups, class_names = load_xy(bundle)
        term.ok(f"Loaded model + matrix — n={len(X)}, features={X.shape[1]}, classes={class_names}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        term.step(1, 2, f"GroupKFold({N_SPLITS}) out-of-fold predictions...")
        n_groups = int(pd.Series(groups).nunique())
        n_splits = min(N_SPLITS, n_groups)
        if n_splits < 2:
            raise ValueError(f"Need ≥2 launch groups for GroupKFold; got {n_groups}")
        cv = GroupKFold(n_splits=n_splits)
        # Match training params; clone avoids mutating the saved estimator during CV.
        cv_estimator = clone(model)
        y_pred = cross_val_predict(cv_estimator, X, y, cv=cv, groups=groups)

        cm_path = OUT_DIR / "confusion_matrix.png"
        save_confusion_matrix_plot(y, y_pred, class_names, cm_path)
        term.ok(f"Saved {cm_path.relative_to(PROJECT_ROOT)}")

        term.step(2, 2, "SHAP TreeExplainer on fused LightGBM...")
        saved = [cm_path.name]
        try:
            bar_path, dot_path = save_shap_plots(model, X, class_names, OUT_DIR)
            term.ok(f"Saved {bar_path.relative_to(PROJECT_ROOT)}")
            term.ok(f"Saved {dot_path.relative_to(PROJECT_ROOT)}")
            saved.extend([bar_path.name, dot_path.name])
        except ImportError as exc:
            # Keep the pipeline moving — confusion matrix is enough for the dashboard panel.
            term.warn(str(exc))
            term.warn("Skipping SHAP plots; confusion matrix was written.")

        term.banner("EXPLAINABILITY COMPLETE")
        term.info(f"Wrote {', '.join(saved)} → {OUT_DIR.relative_to(PROJECT_ROOT)}")
        timer.print_total()

    except Exception as exc:
        term.fail("EXPLAINABILITY FAILED", str(exc))


if __name__ == "__main__":
    main()
