#!/usr/bin/env python3
"""
Phase 2 photometric processing pipeline (per project document §5.3, §7.1).

Data flow after API ingestion:
  object selection → quality filtering → numerical light curves →
  LSP → PDM → candidate period → phase folding → persisted products.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COSPAR_ID_COL, DATA_PROCESSED, DATA_RAW, LIGHTCURVE_MAG_COL, MMT_RAW_DIR, OBJECT_ID_COL, OBJECT_NAME_COL, RESULTS_DIR
from src.data.lightcurve_columns import extract_lightcurve_arrays, normalize_lightcurve_columns
from src.data.db_manager import (
    init_database,
    load_light_curves,
    load_objects,
    upsert_folded_light_curves,
    upsert_periodograms,
    upsert_photometric,
)
from src.data.mmt_client import extract_photometric_features, save_photometric_csv
from src.features.lightcurve_quality import filter_lightcurve
from src.features.period_analysis import analyze_rotation_period, save_periodogram_json
from src.utils import terminal as term

STEPS = 8


def _load_lightcurve_table() -> pd.DataFrame:
    """Prefer central DB; fall back to combined CSV on disk."""
    lc = load_light_curves()
    if not lc.empty:
        return lc
    combined = MMT_RAW_DIR / "mmt_lightcurves.csv"
    if combined.is_file():
        return normalize_lightcurve_columns(pd.read_csv(combined))
    raise FileNotFoundError(
        "No light curves in database or data/raw/mmt_lightcurves/mmt_lightcurves.csv — run fetch_data.py first"
    )


def _object_groups(lc: pd.DataFrame) -> pd.DataFrame:
    """One row per object (NORAD ID preferred, else COSPAR)."""
    id_col = OBJECT_ID_COL if OBJECT_ID_COL in lc.columns and lc[OBJECT_ID_COL].notna().any() else COSPAR_ID_COL
    cols = [c for c in (id_col, COSPAR_ID_COL) if c in lc.columns]
    groups = lc[cols].drop_duplicates(subset=[id_col])
    obj_meta = load_objects()
    if not obj_meta.empty and OBJECT_NAME_COL in obj_meta.columns:
        name_lookup = obj_meta.set_index("object_id")[OBJECT_NAME_COL].to_dict()
        groups[OBJECT_NAME_COL] = groups[id_col].astype(str).map(name_lookup)
    elif OBJECT_NAME_COL in lc.columns:
        names = lc.groupby(id_col)[OBJECT_NAME_COL].first()
        groups[OBJECT_NAME_COL] = groups[id_col].map(names)
    return groups


def _plot_label(norad: str | int, object_name: str | None = None) -> str:
    if object_name and str(object_name).strip():
        return f"{str(object_name).strip()} — NORAD {norad}"
    return f"NORAD {norad}"


def _slice_object(lc: pd.DataFrame, norad, cospar) -> pd.DataFrame:
    df = lc
    if norad is not None and not pd.isna(norad) and OBJECT_ID_COL in df.columns:
        df = df[df[OBJECT_ID_COL].astype(str) == str(int(norad))]
    elif cospar is not None and not pd.isna(cospar) and COSPAR_ID_COL in df.columns:
        df = df[df[COSPAR_ID_COL].astype(str).str.upper() == str(cospar).upper()]
    return df


def _save_poc_plot(
    norad: str | int,
    times: np.ndarray,
    mags: np.ndarray,
    result: dict,
    out_dir: Path,
    object_name: str | None = None,
) -> Path | None:
    """Raw curve, LSP, and folded curve — PoC visual validation (document §8)."""
    if len(mags) < 3:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    label = _plot_label(norad, object_name)

    axes[0].plot(times, mags, ".", ms=2, alpha=0.7)
    axes[0].set_title(f"Clean light curve — {label}")
    axes[0].set_xlabel("Elapsed time (s)")
    axes[0].set_ylabel("Apparent Magnitude")
    axes[0].invert_yaxis()

    freqs = np.asarray(result.get("frequencies", []))
    power = np.asarray(result.get("power", []))
    if len(freqs) and len(power):
        periods = 1.0 / np.clip(freqs, 1e-9, None)
        axes[1].semilogx(periods, power, lw=0.8)
        if result.get("lsp_period_sec"):
            axes[1].axvline(result["lsp_period_sec"], color="C1", ls="--", label="LSP peak")
        axes[1].set_title("Lomb–Scargle periodogram")
        axes[1].set_xlabel("Period (s)")
        axes[1].set_ylabel("Power")
        axes[1].legend(fontsize=8)

    fold_phase = np.asarray(result.get("folded_phase", []))
    fold_mag = np.asarray(result.get("folded_magnitude", []))
    period = result.get("extracted_period_sec") or result.get("pdm_period_sec")
    if len(fold_phase):
        axes[2].plot(fold_phase, fold_mag, ".", ms=2, alpha=0.7)
        axes[2].set_title(f"Phase-folded (P={period}s)" if period else "Phase-folded")
        axes[2].set_xlabel("Phase")
        axes[2].set_ylabel("Apparent Magnitude")
        axes[2].invert_yaxis()

    fig.tight_layout()
    path = out_dir / f"norad_{norad}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_period_analysis() -> dict:
    """Process all ingested light curves through the Phase 2 PoC chain."""
    init_database()
    lc_all = _load_lightcurve_table()
    objects = _object_groups(lc_all)

    period_dir = RESULTS_DIR / "periodograms"
    folded_dir = RESULTS_DIR / "folded_lightcurves"
    plot_dir = RESULTS_DIR / "poc_plots"
    period_dir.mkdir(parents=True, exist_ok=True)
    folded_dir.mkdir(parents=True, exist_ok=True)

    photo_rows: list[dict] = []
    periodo_rows: list[dict] = []
    folded_rows: list[dict] = []
    analyzed = 0
    filtered_out = 0

    for _, obj in objects.iterrows():
        norad = obj.get(OBJECT_ID_COL)
        cospar = obj.get(COSPAR_ID_COL)
        object_name = obj.get(OBJECT_NAME_COL) if OBJECT_NAME_COL in obj.index else None
        if object_name is not None and pd.isna(object_name):
            object_name = None
        norad_key = str(int(norad)) if norad is not None and not pd.isna(norad) else None
        if norad_key is None and (cospar is None or pd.isna(cospar)):
            continue

        slice_df = _slice_object(lc_all, norad, cospar)
        if slice_df.empty:
            continue

        try:
            ts_raw, mags_raw, errs_raw = extract_lightcurve_arrays(slice_df)
        except ValueError:
            continue
        ts, mags, errs, qstats = filter_lightcurve(ts_raw, mags_raw, errs_raw)
        if not qstats["passed"]:
            filtered_out += 1
            continue

        result = analyze_rotation_period(ts, mags)
        feats = extract_photometric_features(ts, mags, errs)
        feats.update({
            "lsp_period_sec": result["lsp_period_sec"],
            "pdm_period_sec": result["pdm_period_sec"],
            "pdm_theta": result["pdm_theta"],
            "estimated_period_sec": result["extracted_period_sec"],
            "is_tumbling": result["is_tumbling"],
            "quality_points": int(qstats["accepted"]),
            "quality_span_sec": float(qstats["span_sec"]),
        })

        safe = norad_key or str(cospar).replace("/", "_")
        json_path = period_dir / f"norad_{safe}.json"
        save_periodogram_json(result, json_path)

        fold_period = result["extracted_period_sec"] or result["pdm_period_sec"]
        folded_payload = {
            "object_id": norad_key,
            "cospar_id": cospar,
            "object_name": object_name,
            "period_sec": fold_period,
            "phase": np.asarray(result["folded_phase"]).tolist(),
            "mag": np.asarray(result["folded_magnitude"]).tolist(),
        }
        folded_json_path = folded_dir / f"norad_{safe}.json"
        folded_json_path.write_text(json.dumps(folded_payload, indent=2))

        if norad_key and fold_period:
            for ph, mag in zip(result["folded_phase"], result["folded_magnitude"]):
                folded_rows.append({
                    "object_id": norad_key,
                    "cospar_id": cospar,
                    "period_sec": fold_period,
                    "phase": float(ph),
                    LIGHTCURVE_MAG_COL: float(mag),
                })

        if norad_key:
            _save_poc_plot(norad_key, ts, mags, result, plot_dir, object_name=object_name)

        row = {
            "object_id": norad_key,
            COSPAR_ID_COL: cospar,
            OBJECT_NAME_COL: object_name,
            **{k: v for k, v in feats.items() if k != "quality_points" and k != "quality_span_sec"},
            "quality_points": feats["quality_points"],
            "quality_span_sec": feats["quality_span_sec"],
        }
        photo_rows.append(row)
        periodo_rows.append({
            "object_id": norad_key,
            "cospar_id": cospar,
            "object_name": object_name,
            "lsp_period_sec": result["lsp_period_sec"],
            "pdm_period_sec": result["pdm_period_sec"],
            "pdm_theta": result["pdm_theta"],
            "extracted_period_sec": result["extracted_period_sec"],
            "is_tumbling": result["is_tumbling"],
            "periodogram_json": str(json_path.relative_to(PROJECT_ROOT)),
            "folded_json": str(folded_json_path.relative_to(PROJECT_ROOT)),
        })
        analyzed += 1

    photo_df = pd.DataFrame(photo_rows)
    periodo_df = pd.DataFrame(periodo_rows)
    folded_df = pd.DataFrame(folded_rows)

    if not photo_df.empty:
        save_photometric_csv(photo_df)
        upsert_photometric(photo_df)
    if not periodo_df.empty:
        upsert_periodograms(periodo_df)
    if not folded_df.empty:
        upsert_folded_light_curves(folded_df)
        folded_csv = DATA_PROCESSED / "folded_light_curves.csv"
        folded_csv.parent.mkdir(parents=True, exist_ok=True)
        folded_df.to_csv(folded_csv, index=False)

    summary_path = RESULTS_DIR / "period_analysis_summary.json"
    summary = {
        "objects_in_source": int(len(objects)),
        "objects_analyzed": analyzed,
        "objects_filtered_out": filtered_out,
        "stable_rotators": int((periodo_df["is_tumbling"] == 0).sum()) if not periodo_df.empty else 0,
        "tumbling": int((periodo_df["is_tumbling"] == 1).sum()) if not periodo_df.empty else 0,
        "mean_pdm_theta": float(periodo_df["pdm_theta"].mean()) if not periodo_df.empty else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 light-curve processing (LSP + PDM + phase fold)")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    term.banner("PHASE 2 — PHOTOMETRIC PROCESSING")
    try:
        term.step(1, STEPS, "Loading object-indexed light curves from central database...")
        lc = _load_lightcurve_table()
        id_col = OBJECT_ID_COL if OBJECT_ID_COL in lc.columns else COSPAR_ID_COL
        n_obj = _object_groups(lc)[id_col].nunique()
        term.ok(f"{term.fmt_n(len(lc))} observations, {term.fmt_n(n_obj)} objects")

        term.step(2, STEPS, "Selecting objects for PoC processing...")
        term.ok("Processing all objects with ingested light-curve data")

        term.step(3, STEPS, "Quality filtering → LSP → PDM → phase folding...")
        summary = run_period_analysis()
        term.ok(f"Retained after quality filter: {term.fmt_n(summary['objects_analyzed'])} objects")
        if summary["objects_filtered_out"]:
            term.warn(
                f"Filtered out: {term.fmt_n(summary['objects_filtered_out'])} objects "
                "(insufficient points or time span)"
            )

        term.step(4, STEPS, "Lomb–Scargle periodograms stored")
        term.step(5, STEPS, "PDM-validated candidate periods stored")
        term.step(6, STEPS, "Phase-folded light curves stored")
        term.ok("results/periodograms/, results/folded_lightcurves/, results/poc_plots/")

        term.step(7, STEPS, "Photometric summaries written to central database")
        term.ok(f"Stable rotators: {term.fmt_n(summary['stable_rotators'])}")
        term.ok(f"Tumbling objects: {term.fmt_n(summary['tumbling'])}")

        term.step(8, STEPS, "PoC processing summary")
        term.ok(f"Period analysis summary: results/period_analysis_summary.json")
        term.banner("PHOTOMETRIC PROCESSING COMPLETE")
        timer.print_total()
    except Exception as exc:
        term.fail("PHOTOMETRIC PROCESSING FAILED", str(exc))


if __name__ == "__main__":
    main()
