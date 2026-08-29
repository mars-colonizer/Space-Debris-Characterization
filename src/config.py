"""Central configuration for Phase 2 POC pipeline."""

import os
from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_DATABASE = PROJECT_ROOT / "data" / "database"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data source mode: SYNTHETIC (offline generator) or ACTUAL (live APIs + MMT)
DATA_MODE = os.getenv("DATA_MODE", "SYNTHETIC").upper()
MMT_RAW_DIR = DATA_RAW / "mmt_lightcurves"

# Reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Column identifiers
COSPAR_ID_COL = "cospar_id"
OBJECT_ID_COL = "object_id"
EPOCH_COL = "epoch"
CLASS_COL = "object_class"

# TLE orbital element columns (keplerian)
ORBITAL_ELEMENT_COLS = [
    "inclination",
    "eccentricity",
    "semi_major_axis",
    "raan",
    "arg_perigee",
    "mean_anomaly",
]

# Stage 1 target
STAGE1_TARGET = CLASS_COL

# Stage 2 targets (ground truth — removed from X before training)
STAGE2_SIZE_TARGETS = ["true_length", "true_width", "true_height"]
STAGE2_SHAPE_TARGET = "true_shape"
STAGE2_PERIOD_TARGET = "true_period"
STAGE2_TUMBLING_TARGET = "true_tumbling"
STAGE2_TARGETS = STAGE2_SIZE_TARGETS  # backward-compatible alias for sizing regressors

# Photometric observables allowed as model features
PHOTOMETRIC_FEATURE_COLS = [
    "mag_mean",
    "mag_std",
    "delta_mag",
    "estimated_period_sec",
    "apparent_shape_score",
    "is_tumbling",
]

# Explicit ground-truth columns blocked from feature matrix
LEAKAGE_TARGET_COLUMNS = [
    "true_length",
    "true_width",
    "true_height",
    "true_mass",
    "true_shape",
    "true_period",
    "true_tumbling",
]

# Leakage prevention — columns that must never appear in feature matrix X
LEAKAGE_COLUMNS = [
    "length",
    "width",
    "height",
    "depth",
    "diameter",
    "span",
    "size",
    "mass",
    "shape",
    "dimensions",
    "object_class",
    "object_type",
    "class",
    "label",
] + LEAKAGE_TARGET_COLUMNS

# Pattern-based leakage detection (case-insensitive substring match)
LEAKAGE_PATTERNS = [
    r"^true_length$",
    r"^true_width$",
    r"^true_height$",
    r"^true_mass$",
    r"^true_shape$",
    r"^true_period$",
    r"^true_tumbling$",
    r"^length$",
    r"^width$",
    r"^height$",
    r"dimension",
    r"\bmass\b",
    r"^shape$",
    r"\bsize\b",
    r"object_class",
    r"object_type",
]

# Categorical feature columns (after feature engineering)
CATEGORICAL_FEATURES = ["object_type_code"]

# Columns kept in dataset but excluded from model features
NON_FEATURE_COLUMNS = [COSPAR_ID_COL, OBJECT_ID_COL, EPOCH_COL]

# Minimum samples per class for Stage 2 class-conditioned regression
MIN_STAGE2_SAMPLES_PER_CLASS = 5
