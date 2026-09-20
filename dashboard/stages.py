"""Pipeline stage definitions for the SSA dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATUS_WAITING = "WAITING"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETE = "COMPLETE"
STATUS_WARNING = "WARNING"
STATUS_FAILED = "FAILED"
STATUS_STOPPED = "STOPPED"

# Ingest → identifiers → process → period → database → evaluation artifacts
PIPELINE_STEPS: list = []  # filled below after StepDef


@dataclass
class StepDef:
    key: str
    label: str
    number: int


PIPELINE_STEPS = [
    StepDef("ingestion", "MMT-9 LIGHT CURVES", 1),
    StepDef("identifiers", "NORAD → COSPAR (KEEPTRACK)", 2),
    StepDef("metadata", "TLE + DISCOS (KEEPTRACK-GATED)", 3),
    StepDef("processing", "QUALITY FILTERING", 4),
    StepDef("period", "LSP + PDM PERIOD ANALYSIS", 5),
    StepDef("database", "CENTRAL DATABASE", 6),
    StepDef("poc", "EVALUATION & ARTIFACTS", 7),  # key retained for log-parser compatibility
]


@dataclass
class ExecStage:
    key: str
    label: str
    script: str
    step_keys: list[str]
    depends_on: list[str] = field(default_factory=list)
    phase: int = 2


# Photometry + evaluation pipeline
EXEC_STAGES: list[ExecStage] = [
    ExecStage(
        "fetch",
        "API Ingestion",
        "scripts/fetch_data.py",
        ["ingestion", "identifiers", "metadata", "database"],
        phase=2,
    ),
    ExecStage(
        "periods",
        "Photometric Processing",
        "scripts/analyze_periods.py",
        ["processing", "period", "poc"],
        depends_on=["fetch"],
        phase=2,
    ),
]

# ML characterization branch (Phase 3)
PHASE3_EXEC_STAGES: list[ExecStage] = [
    ExecStage(
        "prepare",
        "AI Dataset Preparation",
        "scripts/prepare_dataset.py",
        [],  # ML stage — do not remap Phase 2 "database" stepper on soft warnings
        depends_on=["periods"],
        phase=3,
    ),
    ExecStage(
        "stage1",
        "Stage 1 Ablation",
        "scripts/run_ablation.py",
        [],
        depends_on=["prepare"],
        phase=3,
    ),
    ExecStage(
        "explain",
        "Explainability (XAI)",
        "scripts/explain_models.py",
        [],
        depends_on=["stage1"],
        phase=3,
    ),
    ExecStage(
        "stage2",
        "Stage 2 Regression",
        "scripts/train_stage2.py",
        [],
        depends_on=["explain"],
        phase=3,
    ),
    ExecStage(
        "inference",
        "End-to-End Inference",
        "scripts/run_pipeline.py",
        [],
        depends_on=["stage2"],
        phase=3,
    ),
]

ALL_EXEC_STAGES = EXEC_STAGES + PHASE3_EXEC_STAGES
EXEC_BY_KEY = {e.key: e for e in ALL_EXEC_STAGES}
PHASE2_EXEC_KEYS = [e.key for e in EXEC_STAGES]
PHASE3_EXEC_KEYS = [e.key for e in PHASE3_EXEC_STAGES]
# Full SSA pipeline = photometry + ML characterization
FULL_PIPELINE_KEYS = PHASE2_EXEC_KEYS + PHASE3_EXEC_KEYS
ALL_EXEC_KEYS = FULL_PIPELINE_KEYS


def python_executable() -> str:
    """Interpreter for pipeline subprocesses (override with RSO_PYTHON on TrueNAS etc.)."""
    import sys

    explicit = (os.environ.get("RSO_PYTHON") or "").strip()
    if explicit:
        return explicit
    return sys.executable


def script_path(rel: str) -> Path:
    return PROJECT_ROOT / rel
