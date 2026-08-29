"""Load metrics and artifacts from pipeline outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT, RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "pipeline_runs"


def load_ingestion_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {}
    tle_path = DATA_RAW / "tle_history.csv"
    discos_path = DATA_RAW / "discos_metadata.csv"
    if tle_path.exists():
        try:
            tle = pd.read_csv(tle_path)
            stats["gp_records"] = len(tle)
            stats["norad_ids"] = tle["object_id"].nunique() if "object_id" in tle.columns else 0
            stats["tle_objects"] = tle["cospar_id"].nunique() if "cospar_id" in tle.columns else 0
            stats["tle_ok"] = True
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass
    if discos_path.exists():
        try:
            discos = pd.read_csv(discos_path)
            stats["discos_objects"] = len(discos)
            stats["discos_ok"] = True
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass
    return stats


def load_photometric_stats() -> dict[str, Any]:
    """Light-curve observation summary for dashboard photometry card."""
    stats: dict[str, Any] = {}
    photo_path = DATA_RAW / "photometric_observations.csv"
    if not photo_path.exists():
        return stats
    try:
        photo = pd.read_csv(photo_path)
        stats["obs_count"] = len(photo)
        stats["object_count"] = photo["cospar_id"].nunique() if "cospar_id" in photo.columns else 0
        if "delta_mag" in photo.columns:
            stats["avg_delta_mag"] = round(float(photo["delta_mag"].mean()), 3)
            stats["max_delta_mag"] = round(float(photo["delta_mag"].max()), 3)
        if "is_tumbling" in photo.columns:
            stats["tumbling_fraction"] = round(float(photo["is_tumbling"].mean()), 3)
        stats["photo_ok"] = True
        from src.data.data_mode import get_data_mode
        stats["source"] = get_data_mode()
    except (OSError, ValueError, pd.errors.EmptyDataError, KeyError):
        pass
    return stats


def load_dataset_meta() -> dict[str, Any]:
    meta_path = DATA_PROCESSED / "dataset_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_merge_summary() -> dict[str, int]:
    path = DATA_PROCESSED / "merge_summary.txt"
    if not path.exists():
        return {}
    try:
        text = path.read_text()
    except OSError:
        return {}
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.search(r":\s*(\d+)\s*$", line)
        if m:
            key = line.split(":")[0].strip().lower().replace(" ", "_")
            out[key] = int(m.group(1))
    return out


def load_leakage_info() -> dict[str, Any]:
    meta = load_dataset_meta()
    overlap = 0
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    if train_path.exists() and test_path.exists():
        try:
            train = pd.read_csv(train_path, usecols=["cospar_id"])
            test = pd.read_csv(test_path, usecols=["cospar_id"])
            overlap = len(set(train["cospar_id"]) & set(test["cospar_id"]))
        except (OSError, ValueError, pd.errors.EmptyDataError, KeyError):
            overlap = 0
    return {
        "removed_columns": meta.get("removed_leakage_columns", []),
        "feature_count": len(meta.get("feature_columns", [])),
        "cospar_overlap": overlap,
    }


def load_class_distribution() -> pd.Series | None:
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return None
    try:
        train = pd.read_csv(train_path, usecols=["object_class"])
        test = pd.read_csv(test_path, usecols=["object_class"])
        combined = pd.concat([train, test], ignore_index=True)
        return combined["object_class"].value_counts()
    except (OSError, ValueError, pd.errors.EmptyDataError, KeyError):
        return None


def load_stage1_metrics() -> pd.DataFrame | None:
    path = RESULTS_DIR / "stage1_metrics.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def load_stage2_metrics() -> pd.DataFrame | None:
    path = RESULTS_DIR / "stage2_metrics.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def load_confusion_matrix_paths() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("stage1_*_confusion_matrix.png"))


def parse_inference_from_log(log_text: str) -> dict[str, Any]:
    """Extract end-to-end inference fields from run_pipeline stdout."""
    result: dict[str, Any] = {}
    patterns = {
        "cospar_id": r"COSPAR ID:\s*(\S+)",
        "true_class": r"True class:\s*(.+?)(?:\s*$|\s*\[)",
        "predicted_class": r"Predicted class:\s*(.+?)(?:\s*$|\s*\[)",
        "confidence": r"Confidence:\s*([\d.]+)%",
        "length": r"Length:\s*([\d.]+)\s*m",
        "width": r"Width:\s*([\d.]+)\s*m",
        "height": r"Height:\s*([\d.]+)\s*m",
        "shape": r"Shape:\s*(.+?)(?:\s*$|\s*\[)",
        "spin_period": r"Spin period:\s*([\d.]+)\s*s",
        "tumbling": r"Tumbling:\s*(\S+)",
        "latency_seconds": r"Latency:\s*([\d.]+)\s*s",
    }
    for key, pat in patterns.items():
        m = re.search(pat, log_text, re.I | re.M)
        if m:
            result[key] = m.group(1).strip()
    m = re.search(r"models/stage2/(\S+?)(?:\.joblib)?", log_text)
    if m:
        result["stage2_model"] = m.group(1)
    return result


def list_previous_runs(limit: int = 10) -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for path in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        summary_path = path / "summary.json"
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text())
                data["run_id"] = path.name
                runs.append(data)
            except (OSError, json.JSONDecodeError):
                continue
        if len(runs) >= limit:
            break
    return runs


def save_run_summary(run_dir: Path, summary: dict[str, Any], log_lines: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "execution.log").write_text("\n".join(log_lines))


def _validate_run_id(run_id: str) -> Path:
    """Resolve run path and prevent directory traversal."""
    if not run_id or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run_id")
    run_path = (RUNS_DIR / run_id).resolve()
    if RUNS_DIR.resolve() not in run_path.parents and run_path != RUNS_DIR.resolve():
        raise ValueError("Invalid run_id path")
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return run_path


def delete_run(run_id: str) -> None:
    import shutil
    path = _validate_run_id(run_id)
    shutil.rmtree(path)


def delete_all_runs() -> int:
    import shutil
    if not RUNS_DIR.exists():
        return 0
    count = 0
    for path in RUNS_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
            count += 1
    return count


def purge_all_artifacts() -> int:
    """Delete generated pipeline outputs; keep directory trees intact."""
    from src.config import MODELS_DIR

    deleted = 0

    def _unlink_files(directory: Path, patterns: list[str]) -> None:
        nonlocal deleted
        if not directory.exists():
            return
        for pattern in patterns:
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()
                    deleted += 1

    _unlink_files(RESULTS_DIR, ["*.csv", "*.json", "*.txt", "*.png"])
    _unlink_files(DATA_PROCESSED, ["*.json", "*.txt", "*.csv"])
    _unlink_files(DATA_RAW, ["*.json", "*.csv"])

    for sub in ("stage1", "stage2"):
        stage_dir = MODELS_DIR / sub
        if stage_dir.exists():
            for path in stage_dir.iterdir():
                if path.is_file():
                    path.unlink()
                    deleted += 1

    _unlink_files(MODELS_DIR, ["*.joblib", "*.pkl"])
    return deleted


def load_all_metrics(inference_result: dict | None = None) -> dict[str, Any]:
    """Aggregate metrics for dashboard API."""
    from src.config import MIN_STAGE2_SAMPLES_PER_CLASS, MODELS_DIR

    metrics: dict[str, Any] = {
        "ingestion": load_ingestion_stats(),
        "photometry": load_photometric_stats(),
        "merge": load_merge_summary(),
        "dataset_meta": load_dataset_meta(),
        "leakage": load_leakage_info(),
        "inference": inference_result or {},
    }

    s1 = load_stage1_metrics()
    metrics["stage1"] = s1.to_dict(orient="records") if s1 is not None else []

    s2 = load_stage2_metrics()
    if s2 is not None and "eval_mode" in s2.columns:
        s2 = s2[s2["eval_mode"] == "true_class"]
    metrics["stage2"] = s2.to_dict(orient="records") if s2 is not None else []

    metrics["confusion_matrices"] = [p.name for p in load_confusion_matrix_paths()]

    stage2_classes = []
    train_path = DATA_PROCESSED / "train.csv"
    if train_path.exists():
        try:
            train = pd.read_csv(train_path, usecols=["object_class"])
            counts = train["object_class"].value_counts()
            stage2_dir = MODELS_DIR / "stage2"
            shape_models_path = MODELS_DIR / "stage2_shape_models.joblib"
            has_shape = shape_models_path.exists()
            for cls, n in counts.items():
                safe = cls.lower().replace(" ", "_").replace("/", "_")
                model_path = stage2_dir / f"{safe}.joblib"
                if model_path.exists():
                    status = "OK"
                elif n < MIN_STAGE2_SAMPLES_PER_CLASS:
                    status = "insufficient"
                else:
                    status = "skipped"
                stage2_classes.append({
                    "class": cls,
                    "samples": int(n),
                    "status": status,
                    "shape_model": has_shape and model_path.exists(),
                })
        except (OSError, ValueError, pd.errors.EmptyDataError, KeyError):
            pass
    metrics["stage2_classes"] = stage2_classes

    dist = load_class_distribution()
    metrics["class_count"] = len(dist) if dist is not None else 0

    return metrics
