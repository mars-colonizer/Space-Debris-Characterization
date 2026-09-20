"""Load metrics and artifacts from pipeline outputs."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, PROJECT_ROOT, RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "pipeline_runs"
_METRICS_CACHE: dict[str, Any] = {"at": 0.0, "data": None}
_METRICS_TTL_SEC = 3.0


def _scalar(conn, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _json_safe(value: Any) -> Any:
    """Make metrics JSON-serializable (NaN/Inf → null)."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    return value


_DB_SNAPSHOT_CACHE: dict[str, Any] = {"at": 0.0, "data": None}
_DB_SNAPSHOT_TTL_SEC = 10.0


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
    ).fetchone()
    return row is not None


def _db_snapshot() -> dict[str, Any]:
    """Single-connection read of DB counts (ponytail: one round-trip on slow volumes)."""
    now = time.time()
    cached = _DB_SNAPSHOT_CACHE.get("data")
    if cached is not None and now - float(_DB_SNAPSHOT_CACHE.get("at", 0)) < _DB_SNAPSHOT_TTL_SEC:
        return cached

    from src.data.db_manager import DB_PATH, get_connection

    snap: dict[str, Any] = {
        "lc_points": 0,
        "lc_objects": 0,
        "lc_objects_with_cospar": 0,
        "objects_total": 0,
        "cospar_resolved": 0,
        "gp_records": 0,
        "norad_ids": 0,
        "discos_objects": 0,
        "objects_count": 0,
        "light_curves_count": 0,
        "periodograms_count": 0,
        "folded_light_curves_count": 0,
        "photometric_observations_count": 0,
    }
    if DB_PATH.exists():
        try:
            with get_connection(DB_PATH) as conn:
                if _table_exists(conn, "light_curves"):
                    snap["lc_points"] = _scalar(conn, "SELECT COUNT(*) FROM light_curves")
                    snap["lc_objects"] = _scalar(conn, "SELECT COUNT(DISTINCT object_id) FROM light_curves")
                    snap["lc_objects_with_cospar"] = _scalar(
                        conn,
                        "SELECT COUNT(DISTINCT object_id) FROM light_curves "
                        "WHERE cospar_id IS NOT NULL AND cospar_id != ''",
                    )
                    snap["light_curves_count"] = snap["lc_points"]
                if _table_exists(conn, "objects"):
                    snap["objects_total"] = _scalar(conn, "SELECT COUNT(*) FROM objects")
                    snap["cospar_resolved"] = _scalar(
                        conn,
                        "SELECT COUNT(*) FROM objects WHERE cospar_id IS NOT NULL AND cospar_id != ''",
                    )
                    snap["objects_count"] = snap["objects_total"]
                for table, key in (
                    ("periodograms", "periodograms_count"),
                    ("folded_light_curves", "folded_light_curves_count"),
                    ("photometric_observations", "photometric_observations_count"),
                ):
                    if _table_exists(conn, table):
                        snap[key] = _scalar(conn, f"SELECT COUNT(*) FROM {table}")
                if _table_exists(conn, "tle_history"):
                    snap["gp_records"] = _scalar(conn, "SELECT COUNT(*) FROM tle_history")
                    snap["norad_ids"] = _scalar(conn, "SELECT COUNT(DISTINCT object_id) FROM tle_history")
                if _table_exists(conn, "discos_metadata"):
                    snap["discos_objects"] = _scalar(conn, "SELECT COUNT(*) FROM discos_metadata")
        except Exception:
            pass

    _DB_SNAPSHOT_CACHE["at"] = now
    _DB_SNAPSHOT_CACHE["data"] = snap
    return snap


def invalidate_metrics_cache() -> None:
    _METRICS_CACHE["at"] = 0.0
    _METRICS_CACHE["data"] = None
    _DB_SNAPSHOT_CACHE["at"] = 0.0
    _DB_SNAPSHOT_CACHE["data"] = None


def _load_db_lightcurve_stats() -> dict[str, int]:
    snap = _db_snapshot()
    return {
        "lc_points": snap["lc_points"],
        "lc_objects": snap["lc_objects"],
        "lc_objects_with_cospar": snap["lc_objects_with_cospar"],
        "objects_total": snap["objects_total"],
        "cospar_resolved": snap["cospar_resolved"],
    }


def load_ingestion_stats(db_lc: dict[str, int] | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    snap = _db_snapshot()

    if snap["gp_records"]:
        stats["gp_records"] = snap["gp_records"]
        stats["norad_ids"] = snap["norad_ids"]
        stats["tle_ok"] = True
    else:
        tle_path = DATA_RAW / "tle_history.csv"
        if tle_path.exists():
            try:
                stats["gp_records"] = _csv_rows(tle_path)
                tle = pd.read_csv(tle_path, usecols=lambda c: c in ("object_id",))
                stats["norad_ids"] = int(tle["object_id"].nunique()) if "object_id" in tle.columns else 0
                stats["tle_ok"] = True
            except (OSError, ValueError, pd.errors.EmptyDataError):
                pass

    if snap["discos_objects"]:
        stats["discos_objects"] = snap["discos_objects"]
        stats["discos_ok"] = True
    else:
        discos_path = DATA_RAW / "discos_metadata.csv"
        if discos_path.exists():
            try:
                stats["discos_objects"] = _csv_rows(discos_path)
                stats["discos_ok"] = True
            except OSError:
                pass

    db_lc = db_lc or _load_db_lightcurve_stats()
    if db_lc["lc_points"]:
        stats["mmt_points"] = db_lc["lc_points"]
        stats["mmt_objects"] = db_lc["lc_objects"]
        stats["mmt_ok"] = True
    else:
        mmt_path = DATA_RAW / "mmt_lightcurves" / "mmt_lightcurves.csv"
        if mmt_path.exists():
            try:
                stats["mmt_points"] = _csv_rows(mmt_path)
                mmt = pd.read_csv(mmt_path, usecols=lambda c: c in ("object_id",))
                stats["mmt_objects"] = int(mmt["object_id"].nunique())
                stats["mmt_ok"] = True
            except (OSError, ValueError, pd.errors.EmptyDataError):
                pass

    from src.data.data_mode import get_data_mode
    stats["source"] = get_data_mode()
    return stats


def load_identifier_stats(db_lc: dict[str, int] | None = None) -> dict[str, Any]:
    """NORAD → COSPAR resolution coverage from central object index."""
    stats: dict[str, Any] = {}
    db_lc = db_lc or _load_db_lightcurve_stats()
    if db_lc["objects_total"] or db_lc["lc_objects"]:
        total = db_lc["objects_total"] or db_lc["lc_objects"]
        resolved = db_lc["cospar_resolved"] or db_lc["lc_objects_with_cospar"]
        stats["total_objects"] = total
        stats["cospar_resolved"] = resolved
        stats["cospar_missing"] = max(0, total - resolved)
        stats["coverage_pct"] = round(100 * resolved / total, 1) if total else 0.0
        stats["identifiers_ok"] = True
        stats["lc_points"] = db_lc["lc_points"]
        stats["lc_objects"] = db_lc["lc_objects"]
        stats["lc_objects_with_cospar"] = db_lc["lc_objects_with_cospar"]
    return stats


def load_metadata_stats(ing: dict[str, Any] | None = None, ids: dict[str, Any] | None = None) -> dict[str, Any]:
    """Space-Track + DISCOS stats scoped to KeepTrack-resolved MMT objects."""
    from src.data.data_mode import get_data_mode
    from src.data.keeptrack_client import keeptrack_enabled

    ing = ing or load_ingestion_stats()
    ids = ids or load_identifier_stats()
    mode = get_data_mode()
    mmt_objects = ing.get("mmt_objects") or ids.get("lc_objects") or 0
    eligible = ids.get("cospar_resolved") or ids.get("lc_objects_with_cospar") or 0
    skipped = max(0, int(mmt_objects) - int(eligible)) if mmt_objects else 0

    stats: dict[str, Any] = {
        "source": mode,
        "gated": mode == "ACTUAL",
        "keeptrack_enabled": keeptrack_enabled(),
        "mmt_objects": mmt_objects,
        "metadata_eligible": eligible,
        "metadata_skipped": skipped,
        "tle_objects": ing.get("norad_ids", 0),
        "gp_records": ing.get("gp_records", 0),
        "discos_objects": ing.get("discos_objects", 0),
        "tle_ok": ing.get("tle_ok", False),
        "discos_ok": ing.get("discos_ok", False),
    }
    if eligible and stats["tle_ok"]:
        stats["tle_match_eligible"] = int(stats["tle_objects"]) == int(eligible)
    return stats


def load_database_stats() -> dict[str, Any]:
    """Row counts for Phase 2 central database tables."""
    from src.data.db_manager import DB_PATH

    snap = _db_snapshot()
    stats: dict[str, Any] = {"db_path": str(DB_PATH.relative_to(PROJECT_ROOT))}
    if not DB_PATH.exists():
        return stats
    stats.update({
        "objects_count": snap["objects_count"],
        "light_curves_count": snap["light_curves_count"],
        "periodograms_count": snap["periodograms_count"],
        "folded_light_curves_count": snap["folded_light_curves_count"],
        "photometric_observations_count": snap["photometric_observations_count"],
    })
    if any(snap[k] for k in (
        "objects_count", "light_curves_count", "periodograms_count",
        "folded_light_curves_count", "photometric_observations_count",
    )):
        stats["database_ok"] = True
    return stats


def load_period_analysis_stats() -> dict[str, Any]:
    """LSP/PDM period analysis summary and per-object results."""
    stats: dict[str, Any] = {}
    summary_path = RESULTS_DIR / "period_analysis_summary.json"
    if summary_path.exists():
        try:
            stats.update(json.loads(summary_path.read_text()))
            stats["period_ok"] = True
        except (OSError, json.JSONDecodeError):
            pass

    folded_csv = DATA_PROCESSED / "folded_lightcurves.csv"
    if folded_csv.exists():
        try:
            stats["folded_points"] = _scalar_from_db("folded_light_curves") or _csv_rows(folded_csv)
            stats["folded_objects"] = stats.get("period_objects", 0)
        except (OSError, ValueError):
            pass

    from src.data.db_manager import DB_PATH, get_connection
    if DB_PATH.exists():
        try:
            with get_connection(DB_PATH) as conn:
                periodo = pd.read_sql(
                    """
                    SELECT p.object_id, o.object_name, p.lsp_period_sec, p.pdm_period_sec,
                           p.extracted_period_sec, p.is_tumbling, p.pdm_theta
                    FROM periodograms p
                    LEFT JOIN objects o ON o.object_id = p.object_id
                    """,
                    conn,
                )
            if not periodo.empty:
                stats["period_objects"] = len(periodo)
                stats["objects"] = _json_safe(periodo.where(pd.notna(periodo), None).to_dict(orient="records"))
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass
    return stats


def _scalar_from_db(table: str) -> int:
    from src.data.db_manager import DB_PATH, get_connection

    if not DB_PATH.exists():
        return 0
    try:
        with get_connection(DB_PATH) as conn:
            return _scalar(conn, f"SELECT COUNT(*) FROM {table}")
    except Exception:
        return 0


def _csv_rows(path: Path) -> int:
    try:
        return sum(1 for _ in path.open()) - 1
    except OSError:
        return 0


def load_poc_artifacts() -> dict[str, Any]:
    """Evaluation artifact inventory: diagnostic plots, periodograms, folded curves."""
    plot_dir = RESULTS_DIR / "poc_plots"
    periodogram_dir = RESULTS_DIR / "periodograms"
    folded_dir = RESULTS_DIR / "folded_lightcurves"

    plots = sorted(p.name for p in plot_dir.glob("*.png")) if plot_dir.exists() else []
    periodograms = sorted(p.name for p in periodogram_dir.glob("*.json")) if periodogram_dir.exists() else []
    folded = sorted(p.name for p in folded_dir.glob("*.json")) if folded_dir.exists() else []

    return {
        "plot_count": len(plots),
        "periodogram_count": len(periodograms),
        "folded_count": len(folded),
        "plots": plots[:24],
        "periodograms": periodograms[:12],
        "folded": folded[:12],
        "poc_ok": bool(plots),
    }


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
        "mass": r"Mass:\s*([\d.]+)\s*kg",
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
    if "mass" not in result and re.search(r"Mass:\s*n/a", log_text, re.I):
        result["mass_status"] = "n/a (no Stage 2 model)"
    if "spin_period" not in result and re.search(r"Spin period:\s*n/a", log_text, re.I):
        result["spin_period"] = "n/a"
    if "tumbling" not in result and re.search(r"Tumbling:\s*n/a", log_text, re.I):
        result["tumbling"] = "n/a"
    m = re.search(r"models/stage2/([^\s/]+?)(?:\.pkl|\.joblib)\b", log_text)
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


def clear_derived_pipeline_data(
    *,
    clear_models: bool = True,
    clear_raw_metadata: bool = True,
    clear_database: bool = True,
    clear_processed: bool = True,
) -> int:
    """
    Wipe dashboard-visible derived outputs before a fresh run.

    Preserves MMT-9 cache under data/raw/mmt_lightcurves/ (tracks, catalog, combined CSV).
    Clears evaluation plots, period analysis artifacts, Phase 3 XAI images, and optionally
    the SQLite DB, processed CSVs, models, and gated TLE/DISCOS raw exports.
    """
    import shutil

    from src.config import DATA_DATABASE, MODELS_DIR

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

    def _clear_dir_contents(directory: Path) -> None:
        nonlocal deleted
        if not directory.exists():
            return
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
                deleted += 1
            elif path.is_dir():
                shutil.rmtree(path)
                deleted += 1

    # Evaluation / photometric gallery + Phase 3 XAI (always — matches dashboard images)
    for sub in ("poc_plots", "periodograms", "folded_lightcurves", "phase3"):
        _clear_dir_contents(RESULTS_DIR / sub)
    _unlink_files(RESULTS_DIR, ["*.csv", "*.json", "*.txt", "*.png"])

    if clear_processed:
        _clear_dir_contents(DATA_PROCESSED)

    if clear_database:
        db_dir = DATA_DATABASE
        if db_dir.exists():
            for path in db_dir.glob("*.db"):
                path.unlink()
                deleted += 1
            for path in db_dir.glob("*.db-*"):
                path.unlink()
                deleted += 1

    if clear_models:
        for sub in ("stage1", "stage2"):
            _clear_dir_contents(MODELS_DIR / sub)
        _unlink_files(MODELS_DIR, ["*.joblib", "*.pkl"])

    if clear_raw_metadata and DATA_RAW.exists():
        # Gated Space-Track / DISCOS / KeepTrack / photo summaries — not MMT-9.
        for path in DATA_RAW.iterdir():
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".txt"}:
                path.unlink()
                deleted += 1

    global _DB_SNAPSHOT_CACHE
    _DB_SNAPSHOT_CACHE = {"at": 0.0, "data": None}
    invalidate_metrics_cache()
    return deleted


def purge_all_artifacts() -> int:
    """Delete generated pipeline outputs; keep MMT-9 light-curve cache intact."""
    return clear_derived_pipeline_data(
        clear_models=True,
        clear_raw_metadata=True,
        clear_database=True,
        clear_processed=True,
    )


def load_phase2_metrics() -> dict[str, Any]:
    """Metrics for pipeline dashboard cards (ingest → photometry → evaluation)."""
    db_lc = _load_db_lightcurve_stats()
    ing = load_ingestion_stats(db_lc)
    ids = load_identifier_stats(db_lc)
    return {
        "ingestion": ing,
        "identifiers": ids,
        "metadata": load_metadata_stats(ing, ids),
        "database": load_database_stats(),
        "period_analysis": load_period_analysis_stats(),
        "poc_artifacts": load_poc_artifacts(),
    }


def load_phase3_metrics(inference_result: dict | None = None) -> dict[str, Any]:
    """Optional Phase 3 ML metrics (collapsed in dashboard)."""
    from src.config import MIN_STAGE2_SAMPLES_PER_CLASS, MODELS_DIR

    metrics: dict[str, Any] = {
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


def load_all_metrics(inference_result: dict | None = None) -> dict[str, Any]:
    """Aggregate metrics for dashboard API."""
    now = time.time()
    cached = _METRICS_CACHE.get("data")
    cached_at = float(_METRICS_CACHE.get("at", 0))
    if cached is not None and now - cached_at < _METRICS_TTL_SEC:
        return cached

    metrics = load_phase2_metrics()
    metrics["phase3"] = load_phase3_metrics(inference_result)
    metrics["photometry"] = load_photometric_stats()
    metrics["inference_result"] = inference_result or {}
    metrics = _json_safe(metrics)
    _METRICS_CACHE["at"] = time.time()
    _METRICS_CACHE["data"] = metrics
    return metrics
