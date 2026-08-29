#!/usr/bin/env python3
"""Unified data ingestion — SYNTHETIC (offline) or ACTUAL (Space-Track + DISCOS + MMT)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW, COSPAR_ID_COL, OBJECT_ID_COL
from src.data.api_connectors import generate_synthetic_catalog
from src.data.data_mode import get_data_mode
from src.data.db_manager import init_database, upsert_catalog, upsert_photometric
from src.data.discos_client import fetch_discos_objects
from src.data.env import env_int, env_present, load_project_env
from src.data.mmt_client import MMTClient, save_photometric_csv
from src.data.spacetrack_client import fetch_gp_history, fetch_recent_gp
from src.data.storage import ensure_storage_dirs, save_to_sqlite
from src.utils import terminal as term

STEPS = 7


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
    data["photometric"].to_csv(photo_path, index=False)
    data["catalog"].to_csv(catalog_path, index=False)

    init_database()
    upsert_catalog(data["catalog"])
    upsert_photometric(data["photometric"])
    save_to_sqlite(data["tle"], "tle_history")
    save_to_sqlite(data["discos"], "discos_metadata")
    save_to_sqlite(data["photometric"], "photometric_observations")

    term.ok(str(tle_path.relative_to(PROJECT_ROOT)))
    term.ok(str(discos_path.relative_to(PROJECT_ROOT)))
    term.ok(str(photo_path.relative_to(PROJECT_ROOT)))
    term.ok(f"Photometric rows: {term.fmt_n(len(data['photometric']))}")
    term.ok(f"Unique TLE objects: {term.fmt_n(data['tle']['cospar_id'].nunique())}")


def _write_actual(max_objects: int, epoch_days: int) -> None:
    term.info("Mode: ACTUAL — Space-Track + DISCOS + MMT light curves", indent=4)
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

    term.step(6, STEPS, "Fetching MMT-9 light curves...")
    objects = tle[[COSPAR_ID_COL, OBJECT_ID_COL]].drop_duplicates(subset=[COSPAR_ID_COL])
    client = MMTClient()
    photo = client.build_photometric_table(objects)
    if photo.empty:
        term.warn("No MMT photometry retrieved — check data/raw/mmt_lightcurves/ offline files")
    else:
        term.ok(f"MMT photometric summaries: {term.fmt_n(len(photo))} objects")

    term.step(7, STEPS, "Writing raw datasets...")
    ensure_storage_dirs()
    tle_path = DATA_RAW / "tle_history.csv"
    discos_path = DATA_RAW / "discos_metadata.csv"
    tle.to_csv(tle_path, index=False)
    discos.drop(columns=["satno"], errors="ignore").to_csv(discos_path, index=False)
    save_photometric_csv(photo)

    save_to_sqlite(tle, "tle_history")
    save_to_sqlite(discos, "discos_metadata")
    if not photo.empty:
        save_to_sqlite(photo, "photometric_observations")
        upsert_photometric(photo)

    term.ok(str(tle_path.relative_to(PROJECT_ROOT)))
    term.ok(str(discos_path.relative_to(PROJECT_ROOT)))
    term.ok(f"Unique TLE objects (COSPAR): {term.fmt_n(tle['cospar_id'].nunique())}")


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
            term.step(2, STEPS, "Connecting to ESA DISCOS...")
            term.ok("Authentication configured (Bearer token)")
            term.step(3, STEPS, "Fetching DISCOS metadata...")
            term.step(4, STEPS, "Connecting to Space-Track...")
            term.ok("Credentials loaded — authenticating on fetch")
            term.step(5, STEPS, "Fetching GP/TLE history...")
            _write_actual(max_objects, epoch_days)

        term.banner("INGESTION COMPLETE")
        timer.print_total()

    except EnvironmentError as exc:
        term.fail("DATA INGESTION FAILED", str(exc), ["Check your .env file"])
    except PermissionError as exc:
        term.fail("DATA INGESTION FAILED", str(exc), ["Verify Space-Track username/password"])
    except Exception as exc:
        term.fail("DATA INGESTION FAILED", str(exc))


if __name__ == "__main__":
    main()
