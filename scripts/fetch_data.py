#!/usr/bin/env python3
"""Unified data ingestion — SYNTHETIC (offline) or ACTUAL (MMT-9 → Space-Track + DISCOS)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from pandas.errors import DatabaseError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW, COSPAR_ID_COL, MMT_RAW_DIR, OBJECT_ID_COL, OBJECT_NAME_COL
from src.data.api_connectors import generate_synthetic_catalog
from src.data.data_mode import get_data_mode
from src.data.db_manager import init_database, upsert_catalog, upsert_light_curves, upsert_objects
from src.data.discos_client import fetch_discos_for_objects, fetch_discos_objects
from src.data.env import env_int, env_present, load_project_env
from src.data.env import MissingConfigError
from src.data.keeptrack_client import keeptrack_enabled, merge_cospar_from_tle, resolve_cospar_map
from src.data.mmt9_client import (
    MMT9Client,
    candidates_csv_path,
    mmt9_enabled,
    mmt_cache_status,
    resolve_mmt_norad_ids,
)
from src.data.mmt_generator import write_mmt_lightcurves
from src.data.spacetrack_client import fetch_gp_history, fetch_recent_gp
from src.data.storage import ensure_storage_dirs, save_dataset_meta, save_to_sqlite
from src.utils import terminal as term

STEPS = 8


def _write_ingest_meta(*, data_mode: str, source: str, **extra: object) -> None:
    """Persist data_mode/source (and optional counts) into dataset_meta.json."""
    ensure_storage_dirs()
    meta = {"data_mode": data_mode, "source": source, **extra}
    save_dataset_meta(meta)
    term.detail(f"dataset_meta.json ← data_mode={data_mode} source={source}", indent=4)


def _apply_cospar_to_lightcurves(cospar_map: dict[str, str]) -> None:
    """Write NORAD→COSPAR map onto combined MMT light-curve CSV."""
    combined = MMT_RAW_DIR / "mmt_lightcurves.csv"
    if not combined.exists() or not cospar_map:
        return
    lc = pd.read_csv(combined)
    if OBJECT_ID_COL not in lc.columns:
        return
    lc[OBJECT_ID_COL] = pd.to_numeric(lc[OBJECT_ID_COL], errors="coerce")
    mapped = lc[OBJECT_ID_COL].map(
        lambda v: cospar_map.get(str(int(v))) if pd.notna(v) else None
    )
    if COSPAR_ID_COL in lc.columns:
        lc[COSPAR_ID_COL] = mapped.fillna(lc[COSPAR_ID_COL])
    else:
        lc[COSPAR_ID_COL] = mapped
    lc.to_csv(combined, index=False)
    filled = int(lc[COSPAR_ID_COL].notna().sum())
    term.detail(f"Light-curve COSPAR tags: {term.fmt_n(filled)}/{term.fmt_n(len(lc))} rows", indent=4)


def _enrich_tle_cospar(tle: pd.DataFrame, cospar_map: dict[str, str]) -> pd.DataFrame:
    """Fill missing TLE COSPAR columns from the resolved NORAD→COSPAR map."""
    if tle.empty or not cospar_map or OBJECT_ID_COL not in tle.columns:
        return tle
    out = tle.copy()
    norad_keys = pd.to_numeric(out[OBJECT_ID_COL], errors="coerce").map(
        lambda v: str(int(v)) if pd.notna(v) else None
    )
    mapped = norad_keys.map(cospar_map)
    if COSPAR_ID_COL in out.columns:
        out[COSPAR_ID_COL] = mapped.fillna(out[COSPAR_ID_COL])
    else:
        out[COSPAR_ID_COL] = mapped
    return out


def _object_index_frame(
    norad_ids: list[int],
    cospar_map: dict[str, str],
    name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    name_map = name_map or {}
    rows = [
        {
            OBJECT_ID_COL: str(int(n)),
            COSPAR_ID_COL: cospar_map.get(str(int(n))),
            OBJECT_NAME_COL: name_map.get(int(n)) or name_map.get(str(int(n))),
        }
        for n in norad_ids
    ]
    return pd.DataFrame(rows)


def _check_credentials() -> None:
    missing = []
    if not env_present("SPACE_TRACK_USERNAME"):
        missing.append("SPACE_TRACK_USERNAME")
    if not env_present("SPACE_TRACK_PASSWORD"):
        missing.append("SPACE_TRACK_PASSWORD")
    if not env_present("DISCOS_TOKEN"):
        missing.append("DISCOS_TOKEN")
    if missing:
        term.fail(
            "DATA INGESTION FAILED",
            f"Missing credentials: {', '.join(missing)}",
            ["Copy .env.example to .env", "Set Space-Track and DISCOS credentials"],
        )
    term.ok("Space-Track credentials found")
    term.ok("DISCOS credentials found")


def _backfill_cospar_from_tle(photo: "pd.DataFrame", tle: "pd.DataFrame") -> "pd.DataFrame":
    """Attach COSPAR IDs from Space-Track TLE rows using NORAD object_id."""
    if photo.empty or tle.empty or OBJECT_ID_COL not in photo.columns:
        return photo
    cospar_map = (
        tle[[OBJECT_ID_COL, COSPAR_ID_COL]]
        .dropna(subset=[OBJECT_ID_COL])
        .astype({OBJECT_ID_COL: str})
        .drop_duplicates(subset=[OBJECT_ID_COL], keep="last")
        .set_index(OBJECT_ID_COL)[COSPAR_ID_COL]
    )
    out = photo.copy()
    out[OBJECT_ID_COL] = out[OBJECT_ID_COL].astype(str)
    mapped = out[OBJECT_ID_COL].map(cospar_map)
    if COSPAR_ID_COL in out.columns:
        out[COSPAR_ID_COL] = mapped.fillna(out[COSPAR_ID_COL])
    else:
        out[COSPAR_ID_COL] = mapped
    return out


def _cospar_map_for_norads(norad_ids: list[int], tle: pd.DataFrame) -> dict[str, str]:
    """Merge KeepTrack + Space-Track COSPAR mappings."""
    return merge_cospar_from_tle(norad_ids, tle)


def _backfill_lightcurve_cospar(norad_ids: list[int], tle: pd.DataFrame) -> None:
    """Legacy wrapper — apply merged COSPAR map to light curves."""
    cospar_map = _cospar_map_for_norads(norad_ids, tle)
    _apply_cospar_to_lightcurves(cospar_map)


def _ingest_mmt_lightcurves(
    catalog: "pd.DataFrame",
    tle: "pd.DataFrame",
    photometric: "pd.DataFrame | None" = None,
) -> None:
    """Generate/read MMT raw curves and persist to central database (no period analysis here)."""
    term.step(3, STEPS, "Generating / loading MMT-9 raw light curves...")
    mmt_path = write_mmt_lightcurves(catalog, tle, photometric=photometric)
    lc_df = pd.read_csv(mmt_path)
    upsert_light_curves(lc_df)
    term.ok(
        f"MMT raw observations: {term.fmt_n(len(lc_df))} points "
        f"({term.fmt_n(lc_df[COSPAR_ID_COL].nunique())} objects)"
    )
    term.detail("Period analysis runs in scripts/analyze_periods.py", indent=4)


def _write_synthetic() -> None:
    term.info("Mode: SYNTHETIC — generating physics-informed catalog", indent=4)
    data = generate_synthetic_catalog()
    ensure_storage_dirs()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    tle_path = DATA_RAW / "tle_history.csv"
    discos_path = DATA_RAW / "discos_metadata.csv"
    photo_path = DATA_RAW / "photometric_observations.csv"
    catalog_path = DATA_RAW / "rso_catalog.csv"

    data["tle"].to_csv(tle_path, index=False)
    data["discos"].to_csv(discos_path, index=False)
    data["catalog"].to_csv(catalog_path, index=False)

    init_database()
    upsert_catalog(data["catalog"])

    _ingest_mmt_lightcurves(data["catalog"], data["tle"], data["photometric"])

    save_to_sqlite(data["tle"], "tle_history")
    save_to_sqlite(data["discos"], "discos_metadata")
    upsert_objects(data["tle"][[OBJECT_ID_COL, COSPAR_ID_COL]].drop_duplicates(subset=[OBJECT_ID_COL]))

    term.ok(str(tle_path.relative_to(PROJECT_ROOT)))
    term.ok(str(discos_path.relative_to(PROJECT_ROOT)))
    term.ok(str(photo_path.relative_to(PROJECT_ROOT)))
    term.ok(f"Unique TLE objects: {term.fmt_n(data['tle']['cospar_id'].nunique())}")
    _write_ingest_meta(data_mode="SYNTHETIC", source="synthetic")


def _write_actual_mmt_first(max_objects: int, epoch_days: int) -> None:
    """MMT-9 catalog drives object selection; Space-Track + DISCOS follow NORAD IDs."""
    term.info("Mode: ACTUAL — MMT-9 catalog → Space-Track + DISCOS", indent=4)
    _check_credentials()

    term.step(2, STEPS, "Loading MMT-9 candidate NORAD IDs...")
    mmt9 = MMT9Client()
    norad_ids, id_source = resolve_mmt_norad_ids(max_objects)
    all_cached, cached_ids, missing_ids = mmt_cache_status(norad_ids)
    term.ok(f"Target: {term.fmt_n(len(norad_ids))} NORAD IDs from {id_source}")
    if all_cached:
        term.ok(f"MMT-9 cache hit — all {term.fmt_n(len(norad_ids))} candidates already on disk (skipping download)")
    elif cached_ids:
        term.info(
            f"MMT-9: {term.fmt_n(len(cached_ids))} cached, {term.fmt_n(len(missing_ids))} to download",
            indent=4,
        )
        mmt9.ensure_catalog()
    else:
        term.info("MMT source: mmt.favor2.info (public photometry DB)", indent=4)
        mmt9.ensure_catalog()

    in_catalog = [n for n in norad_ids if mmt9.resolve_mmt_id(n) is not None]
    not_in_catalog = [n for n in norad_ids if n not in set(in_catalog)]
    if not_in_catalog:
        term.warn(
            f"{term.fmt_n(len(not_in_catalog))} candidate NORAD IDs have no MMT-9 photometry "
            f"(not in catalog; max {term.fmt_n(len(in_catalog))} objects with light curves)"
        )
        if term.VERBOSE:
            term.detail(", ".join(str(n) for n in not_in_catalog), indent=4)

    term.step(3, STEPS, "Downloading MMT-9 light curves...")
    objects = pd.DataFrame({OBJECT_ID_COL: norad_ids})

    if all_cached:
        # Avoid re-reading/rewriting the ~60MB combined CSV once per object on the share.
        term.info("Loading light curves from on-disk cache (no MMT HTTP / no per-object merge)", indent=4)
        lc_df = pd.read_csv(
            MMT_RAW_DIR / "mmt_lightcurves.csv",
            dtype={COSPAR_ID_COL: "string"},
            low_memory=False,
        )
        lc_df[OBJECT_ID_COL] = pd.to_numeric(lc_df[OBJECT_ID_COL], errors="coerce")
        want = set(int(n) for n in norad_ids)
        lc_df = lc_df[lc_df[OBJECT_ID_COL].isin(want)].copy()
        mmt_failed: list[int] = []
    else:

        def mmt_progress(done: int, total: int, norad: str | int) -> None:
            term.info(f"MMT-9: {done}/{total} — NORAD {norad}", indent=4)

        lc_df, _, mmt_failed = mmt9.fetch_many(objects, progress_callback=mmt_progress)
    if lc_df.empty:
        term.warn("No MMT light curves retrieved — aborting MMT-first ingest")
        return
    n_ok = int(lc_df[OBJECT_ID_COL].nunique())
    term.ok(
        f"MMT raw observations: {term.fmt_n(len(lc_df))} points "
        f"({term.fmt_n(n_ok)} objects)"
    )
    mmt_norad_ids = sorted(
        pd.to_numeric(lc_df[OBJECT_ID_COL], errors="coerce").dropna().astype(int).unique().tolist()
    )
    name_map = {n: mmt9.resolve_object_name(n) for n in mmt_norad_ids}
    if mmt_failed:
        term.warn(
            f"MMT-9 partial download: {term.fmt_n(n_ok)} succeeded, "
            f"{term.fmt_n(len(mmt_failed))} failed/missing — continuing ingest"
        )
        if term.VERBOSE:
            term.detail(f"Failed NORAD IDs: {', '.join(str(n) for n in mmt_failed[:20])}", indent=4)

    # COSPAR comes from Space-Track GP OBJECT_ID; KeepTrack is optional gap-fill only.
    term.step(4, STEPS, "Fetching Space-Track GP/TLE for all MMT objects...")

    def tle_progress(count: int) -> None:
        term.info(f"Retrieved: {term.fmt_n(count)} GP records", indent=4)

    with term.timed("Space-Track GP fetch"):
        tle = fetch_gp_history(mmt_norad_ids, epoch_days=epoch_days, progress_callback=tle_progress)

    # Primary map: NORAD → COSPAR from Space-Track OBJECT_ID on each GP row
    cospar_map: dict[str, str] = {}
    if not tle.empty and OBJECT_ID_COL in tle.columns and COSPAR_ID_COL in tle.columns:
        subset = (
            tle[[OBJECT_ID_COL, COSPAR_ID_COL]]
            .dropna(subset=[OBJECT_ID_COL, COSPAR_ID_COL])
            .drop_duplicates(subset=[OBJECT_ID_COL], keep="last")
        )
        for _, row in subset.iterrows():
            try:
                cospar_map[str(int(float(row[OBJECT_ID_COL])))] = str(row[COSPAR_ID_COL])
            except (TypeError, ValueError):
                continue

    tle_norads = set(cospar_map.keys())
    missing_tle = [n for n in mmt_norad_ids if str(int(n)) not in tle_norads]
    if missing_tle:
        term.warn(
            f"{term.fmt_n(len(missing_tle))} MMT objects returned no Space-Track GP — "
            "trying KeepTrack COSPAR fallback for those only"
        )

    if missing_tle and keeptrack_enabled():
        kt_map = resolve_cospar_map(missing_tle)
        for k, v in kt_map.items():
            cospar_map.setdefault(k, v)
        term.ok(
            f"KeepTrack fallback filled {term.fmt_n(len(kt_map))} / "
            f"{term.fmt_n(len(missing_tle))} Space-Track misses"
        )
    elif missing_tle and not keeptrack_enabled():
        term.warn("KEEPTRACK_API_KEY not set — no fallback for Space-Track misses")

    resolved_norad_ids = [n for n in mmt_norad_ids if str(int(n)) in cospar_map]
    term.ok(
        f"COSPAR available for {term.fmt_n(len(resolved_norad_ids))}/"
        f"{term.fmt_n(len(mmt_norad_ids))} MMT objects "
        f"(Space-Track OBJECT_ID primary, KeepTrack optional)"
    )
    tle = _enrich_tle_cospar(tle, cospar_map)
    _apply_cospar_to_lightcurves(cospar_map)
    upsert_objects(_object_index_frame(mmt_norad_ids, cospar_map, name_map))
    term.ok(
        f"Retrieved {term.fmt_n(len(tle))} GP records for "
        f"{term.fmt_n(tle[OBJECT_ID_COL].nunique()) if not tle.empty else 0} NORAD IDs"
    )

    if not resolved_norad_ids:
        term.step(7, STEPS, "Persisting light curves and object index...")
        upsert_light_curves(pd.read_csv(MMT_RAW_DIR / "mmt_lightcurves.csv"))
        upsert_objects(_object_index_frame(mmt_norad_ids, cospar_map, name_map))
        term.step(8, STEPS, "Writing raw datasets...")
        ensure_storage_dirs()
        term.warn("No COSPAR IDs from Space-Track/KeepTrack — TLE/DISCOS CSVs not written")
        term.detail("Run scripts/analyze_periods.py for LSP/PDM/phase-fold processing", indent=4)
        _write_ingest_meta(data_mode="ACTUAL", source="mmt", n_mmt_objects=len(mmt_norad_ids), cospar_resolved=0)
        return

    term.step(6, STEPS, "Fetching DISCOS metadata for COSPAR-resolved MMT objects...")

    def discos_progress(done: int, total: int) -> None:
        term.info(f"DISCOS batch: {term.fmt_n(done)} / {term.fmt_n(total)} NORAD IDs", indent=4)

    with term.timed("DISCOS fetch"):
        discos = fetch_discos_for_objects(
            resolved_norad_ids, cospar_map, progress_callback=discos_progress
        )
    if discos.empty:
        term.warn("DISCOS returned no matches — continuing with TLE + MMT only")
        discos = pd.DataFrame(columns=[COSPAR_ID_COL, "satno", "object_class"])
    else:
        matched = discos["satno"].nunique() if "satno" in discos.columns else len(discos)
        term.ok(
            f"DISCOS matched {term.fmt_n(matched)} / "
            f"{term.fmt_n(len(resolved_norad_ids))} COSPAR-resolved objects"
        )

    term.step(7, STEPS, "Persisting light curves and object index...")
    _apply_cospar_to_lightcurves(cospar_map)
    upsert_light_curves(pd.read_csv(MMT_RAW_DIR / "mmt_lightcurves.csv"))
    upsert_objects(_object_index_frame(mmt_norad_ids, cospar_map, name_map))

    term.step(8, STEPS, "Writing raw datasets...")
    ensure_storage_dirs()
    tle_path = DATA_RAW / "tle_history.csv"
    discos_path = DATA_RAW / "discos_metadata.csv"
    tle.to_csv(tle_path, index=False)
    discos.drop(columns=["satno"], errors="ignore").to_csv(discos_path, index=False)

    save_to_sqlite(tle, "tle_history")
    save_to_sqlite(discos, "discos_metadata")

    term.ok(str(tle_path.relative_to(PROJECT_ROOT)))
    term.ok(str(discos_path.relative_to(PROJECT_ROOT)))
    term.ok(f"Unique TLE objects (COSPAR): {term.fmt_n(tle[COSPAR_ID_COL].nunique())}")
    term.detail("Run scripts/analyze_periods.py for LSP/PDM/phase-fold processing", indent=4)
    _write_ingest_meta(
        data_mode="ACTUAL",
        source="mmt",
        n_tle=len(tle),
        n_mmt_objects=len(mmt_norad_ids),
    )


def _write_actual_discos_first(max_objects: int, epoch_days: int) -> None:
    """Legacy path: balanced DISCOS sample, then Space-Track, then MMT."""
    term.info("Mode: ACTUAL — DISCOS → Space-Track → MMT light curves", indent=4)
    _check_credentials()

    def discos_progress(count: int, label: str) -> None:
        term.info(f"Retrieved: {term.fmt_n(count)} ({label})", indent=4)

    with term.timed("DISCOS fetch"):
        discos = fetch_discos_objects(max_objects=max_objects, progress_callback=discos_progress)
    term.ok(f"Retrieved {term.fmt_n(len(discos))} DISCOS objects")

    norad_ids = discos["satno"].dropna().astype(int).unique().tolist()

    def tle_progress(count: int) -> None:
        term.info(f"Retrieved: {term.fmt_n(count)}", indent=4)

    with term.timed("Space-Track GP fetch"):
        if norad_ids:
            tle = fetch_gp_history(norad_ids, epoch_days=epoch_days, progress_callback=tle_progress)
        else:
            term.warn("No NORAD IDs from DISCOS — falling back to recent GP query")
            tle = fetch_recent_gp(epoch_days=epoch_days, progress_callback=tle_progress)
    term.ok(f"Retrieved {term.fmt_n(len(tle))} GP records")

    term.step(6, STEPS, "Fetching MMT-9 light curves (raw download only)...")
    objects = tle[[COSPAR_ID_COL, OBJECT_ID_COL]].drop_duplicates(subset=[COSPAR_ID_COL])
    use_mmt9 = mmt9_enabled()
    if not use_mmt9:
        raise RuntimeError(
            "ACTUAL mode requires MMT-9 (set MMT9_ENABLED / credentials); "
            "demo light-curve fallback is disabled"
        )
    term.info("MMT source: mmt.favor2.info (public MMT-9 photometry DB)", indent=4)
    mmt9_client = MMT9Client()

    def mmt_progress(done: int, total: int, norad: str | int) -> None:
        term.info(f"MMT-9: {done}/{total} — NORAD {norad}", indent=4)

    lc_df, _, mmt_failed = mmt9_client.fetch_many(objects, progress_callback=mmt_progress)
    if lc_df.empty:
        raise RuntimeError(
            "ACTUAL mode: no MMT light curves retrieved — refusing demo/synthetic fallback"
        )
    term.ok(f"MMT raw observations: {term.fmt_n(len(lc_df))} points")
    upsert_light_curves(lc_df)
    if mmt_failed:
        term.warn(f"MMT-9 partial: {term.fmt_n(len(mmt_failed))} objects failed/missing")

    upsert_objects(tle[[OBJECT_ID_COL, COSPAR_ID_COL]].drop_duplicates(subset=[OBJECT_ID_COL]))

    term.step(7, STEPS, "Writing raw datasets...")
    ensure_storage_dirs()
    tle_path = DATA_RAW / "tle_history.csv"
    discos_path = DATA_RAW / "discos_metadata.csv"
    tle.to_csv(tle_path, index=False)
    discos.drop(columns=["satno"], errors="ignore").to_csv(discos_path, index=False)

    save_to_sqlite(tle, "tle_history")
    save_to_sqlite(discos, "discos_metadata")

    term.ok(str(tle_path.relative_to(PROJECT_ROOT)))
    term.ok(str(discos_path.relative_to(PROJECT_ROOT)))
    term.ok(f"Unique TLE objects (COSPAR): {term.fmt_n(tle['cospar_id'].nunique())}")
    _write_ingest_meta(
        data_mode="ACTUAL",
        source="discos",
        n_tle=len(tle),
        n_discos=len(discos),
    )


def _fetch_driver() -> str:
    load_project_env()
    import os

    if candidates_csv_path().is_file():
        return "mmt"
    driver = (os.getenv("FETCH_DRIVER") or "").strip().lower()
    if driver in ("mmt", "mmt9", "photometry"):
        return "mmt"
    if driver in ("discos", "metadata"):
        return "discos"
    return "mmt" if mmt9_enabled() else "discos"


def _write_actual(max_objects: int, epoch_days: int) -> None:
    if _fetch_driver() == "mmt":
        _write_actual_mmt_first(max_objects, epoch_days)
    else:
        _write_actual_discos_first(max_objects, epoch_days)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pipeline data (SYNTHETIC or ACTUAL)")
    term.add_verbosity_args(parser)
    args = parser.parse_args()
    term.configure_from_args(args)
    timer = term.ScriptTimer()

    load_project_env()
    mode = get_data_mode()
    term.banner(f"PHASE 2 — DATA INGESTION ({mode})")

    try:
        term.step(1, STEPS, f"Active data mode: {mode}")
        term.detail(f"Started at {term.timestamp()}")

        if mode == "SYNTHETIC":
            term.step(2, STEPS, "Generating synthetic multi-modal catalog...")
            _write_synthetic()
        else:
            max_objects = env_int("FETCH_MAX_OBJECTS", 200)
            epoch_days = env_int("FETCH_EPOCH_DAYS", 60)
            if _fetch_driver() == "discos":
                term.step(2, STEPS, "Connecting to ESA DISCOS...")
                term.ok("Authentication configured (Bearer token)")
                term.step(3, STEPS, "Fetching DISCOS metadata...")
                term.step(4, STEPS, "Connecting to Space-Track...")
                term.ok("Credentials loaded — authenticating on fetch")
                term.step(5, STEPS, "Fetching GP/TLE history...")
            _write_actual(max_objects, epoch_days)

        term.banner("INGESTION COMPLETE")
        timer.print_total()

    except MissingConfigError as exc:
        term.fail("DATA INGESTION FAILED", str(exc), ["Check your .env file"])
    except PermissionError as exc:
        term.fail("DATA INGESTION FAILED", str(exc), ["Verify Space-Track username/password"])
    except DatabaseError as exc:
        term.fail(
            "DATA INGESTION FAILED",
            f"Database write failed: {exc}",
            ["Check data/database/rso_poc.db is writable"],
        )
    except (requests.RequestException, OSError) as exc:
        term.fail(
            "DATA INGESTION FAILED",
            str(exc),
            ["Transient network error — re-run ingestion; cached MMT objects are kept on disk"],
        )
    except Exception as exc:
        term.fail("DATA INGESTION FAILED", str(exc))


if __name__ == "__main__":
    main()
