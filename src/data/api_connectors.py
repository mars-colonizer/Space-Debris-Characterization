"""Synthetic data connectors — physics-informed catalog + light-curve generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED

CLASSES = ["Rocket Body", "Defunct Satellite", "Mission-Related Object", "Fragment"]
CLASS_WEIGHTS = [0.25, 0.35, 0.20, 0.20]
SHAPES = ["Box-Wing", "Cylinder", "Flat-Plate", "Irregular"]


def _make_cospar(year: int, launch: int, piece: str) -> str:
    return f"{year}-{launch:03d}{piece}"


def _light_curve_profile(rng: np.random.Generator, object_class: str) -> dict:
    """Physics-informed photometric summary conditioned on class/shape."""
    if object_class == "Rocket Body":
        true_shape = "Cylinder"
        true_period = float(rng.uniform(5, 30))
        delta_mag = float(rng.uniform(1.5, 3.0))
        true_tumbling = 0
        is_tumbling = 0 if rng.random() > 0.1 else 1
        mag_std = float(rng.uniform(0.15, 0.45))
    elif object_class in ("Defunct Satellite", "Mission-Related Object"):
        true_shape = "Box-Wing"
        true_period = float(rng.uniform(60, 600))
        delta_mag = float(rng.uniform(0.2, 0.8))
        true_tumbling = 0
        is_tumbling = 0
        mag_std = float(rng.uniform(0.05, 0.15))
    else:  # Fragment debris
        true_shape = rng.choice(["Irregular", "Flat-Plate"])
        true_period = float(rng.uniform(0.5, 8))
        delta_mag = float(rng.uniform(2.0, 4.5))
        true_tumbling = 1
        is_tumbling = 1
        mag_std = float(rng.uniform(0.4, 1.2))

    mag_mean = float(rng.uniform(8.0, 15.0))
    # Observational period estimate: noisy version of truth
    period_noise = rng.normal(0, 0.15 * max(true_period, 1))
    estimated_period_sec = max(0.1, true_period + period_noise)
    apparent_shape_score = float(rng.normal(0.5 if true_shape == "Cylinder" else 0.0, 0.3))

    return {
        "true_shape": true_shape,
        "true_period": round(true_period, 3),
        "true_tumbling": int(true_tumbling),
        "mag_mean": round(mag_mean, 3),
        "mag_std": round(mag_std, 3),
        "delta_mag": round(delta_mag, 3),
        "estimated_period_sec": round(estimated_period_sec, 3),
        "apparent_shape_score": round(apparent_shape_score, 3),
        "is_tumbling": int(is_tumbling),
    }


def generate_synthetic_catalog(
    n_objects: int = 80,
    epochs_per_object: int = 5,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic TLE history, DISCOS metadata, RSO catalog truths,
    and photometric light-curve summaries.
    """
    rng = np.random.default_rng(seed)
    classes = rng.choice(CLASSES, size=n_objects, p=CLASS_WEIGHTS)

    tle_rows: list[dict] = []
    discos_rows: list[dict] = []
    catalog_rows: list[dict] = []
    photo_rows: list[dict] = []

    for i in range(n_objects):
        year = int(rng.integers(1990, 2023))
        launch = int(rng.integers(1, 999))
        piece = chr(int(rng.integers(ord("A"), ord("F") + 1)))
        cospar = _make_cospar(year, launch, piece)
        cls = classes[i]

        if cls == "Rocket Body":
            inc_base, sma_base, ecc_base = 28.0, 7200.0, 0.002
            length, width, height = rng.uniform(8, 15), rng.uniform(2, 4), rng.uniform(2, 4)
        elif cls == "Defunct Satellite":
            inc_base, sma_base, ecc_base = 51.6, 6800.0, 0.001
            length, width, height = rng.uniform(1, 5), rng.uniform(1, 3), rng.uniform(1, 3)
        elif cls == "Mission-Related Object":
            inc_base, sma_base, ecc_base = 45.0, 7000.0, 0.003
            length, width, height = rng.uniform(0.1, 2), rng.uniform(0.1, 1), rng.uniform(0.1, 1)
        else:
            inc_base, sma_base, ecc_base = 60.0, 7100.0, 0.008
            length, width, height = rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3)

        lc = _light_curve_profile(rng, cls)
        mass = round(float(rng.uniform(0.1, 5000)), 2)

        catalog_rows.append({
            "cospar_id": cospar,
            "object_class": cls,
            "true_length": round(float(length), 3),
            "true_width": round(float(width), 3),
            "true_height": round(float(height), 3),
            "true_mass": mass,
            **{k: lc[k] for k in ("true_shape", "true_period", "true_tumbling")},
        })

        discos_rows.append({
            "cospar_id": cospar,
            "object_class": cls,
            "length": round(float(length), 3),
            "width": round(float(width), 3),
            "height": round(float(height), 3),
            "mass": mass,
            "shape": lc["true_shape"],
        })

        photo_rows.append({
            "cospar_id": cospar,
            **{k: lc[k] for k in (
                "mag_mean", "mag_std", "delta_mag", "estimated_period_sec",
                "apparent_shape_score", "is_tumbling",
            )},
        })

        base_epoch = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 30)))
        for e in range(epochs_per_object):
            days_offset = e * int(rng.integers(1, 5))
            epoch = base_epoch + pd.Timedelta(days=days_offset)
            inc = inc_base + e * rng.normal(0.01, 0.005)
            sma = sma_base - e * rng.uniform(0.01, 0.05)
            tle_rows.append({
                "cospar_id": cospar,
                "object_id": f"OBJ-{i:04d}",
                "epoch": epoch.isoformat(),
                "inclination": round(float(inc), 6),
                "eccentricity": round(float(ecc_base + rng.normal(0, 0.0005)), 7),
                "semi_major_axis": round(float(sma), 3),
                "raan": round(float(rng.uniform(0, 360)), 4),
                "arg_perigee": round(float(rng.uniform(0, 360)), 4),
                "mean_anomaly": round(float(rng.uniform(0, 360)), 4),
            })

    # Unmatched audit rows
    tle_rows.append({
        "cospar_id": "2020-999Z", "object_id": "UNMATCHED-TLE",
        "epoch": "2024-06-01T00:00:00+00:00",
        "inclination": 90.0, "eccentricity": 0.001, "semi_major_axis": 7000.0,
        "raan": 0.0, "arg_perigee": 0.0, "mean_anomaly": 0.0,
    })
    discos_rows.append({
        "cospar_id": "2019-888Y", "object_class": "Fragment",
        "length": 0.1, "width": 0.1, "height": 0.1, "mass": 0.05, "shape": "Irregular",
    })

    return {
        "tle": pd.DataFrame(tle_rows),
        "discos": pd.DataFrame(discos_rows),
        "catalog": pd.DataFrame(catalog_rows),
        "photometric": pd.DataFrame(photo_rows),
    }
