#!/usr/bin/env python3
"""Generate synthetic sample TLE + DISCOS + photometric data for Phase 2 POC."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_RAW
from src.data.api_connectors import generate_synthetic_catalog
from src.data.db_manager import init_database, upsert_catalog, upsert_photometric
from src.data.storage import ensure_storage_dirs, save_to_sqlite


def generate_sample_data(n_objects: int = 80, epochs_per_object: int = 5):
    """Backward-compatible wrapper returning (tle_df, discos_df)."""
    data = generate_synthetic_catalog(n_objects, epochs_per_object)
    return data["tle"], data["discos"]


def main() -> None:
    ensure_storage_dirs()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_catalog()

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

    print(f"Generated {len(data['tle'])} TLE rows -> {tle_path}")
    print(f"Generated {len(data['discos'])} DISCOS rows -> {discos_path}")
    print(f"Generated {len(data['photometric'])} photometric rows -> {photo_path}")
    print(f"Unique objects: {data['tle']['cospar_id'].nunique()} TLE, {data['discos']['cospar_id'].nunique()} DISCOS")
    print("Class distribution:")
    print(data["discos"]["object_class"].value_counts())


if __name__ == "__main__":
    main()
