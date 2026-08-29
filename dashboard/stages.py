"""Pipeline stage definitions and script mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATUS_WAITING = "WAITING"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETE = "COMPLETE"
STATUS_WARNING = "WARNING"
STATUS_FAILED = "FAILED"
STATUS_STOPPED = "STOPPED"


@dataclass
class StepDef:
    key: str
    label: str
    number: int


PIPELINE_STEPS: list[StepDef] = [
    StepDef("ingestion", "API INGESTION", 1),
    StepDef("prep", "DATA PREPARATION", 2),
    StepDef("quality", "DATA QUALITY", 3),
    StepDef("leakage", "LEAKAGE FILTER", 4),
    StepDef("features", "FEATURE ENGINEERING", 5),
    StepDef("split", "TRAIN/TEST SPLIT", 6),
    StepDef("stage1", "STAGE 1 CLASSIFICATION", 7),
    StepDef("stage2", "STAGE 2 REGRESSION", 8),
    StepDef("inference", "END-TO-END INFERENCE", 9),
    StepDef("final", "FINAL RESULTS", 10),
]


@dataclass
class ExecStage:
    key: str
    label: str
    script: str
    step_keys: list[str]
    depends_on: list[str] = field(default_factory=list)


EXEC_STAGES: list[ExecStage] = [
    ExecStage("fetch", "API Ingestion", "scripts/fetch_data.py", ["ingestion"]),
    ExecStage(
        "prepare",
        "Data Preparation",
        "scripts/prepare_dataset.py",
        ["prep", "quality", "leakage", "features", "split"],
        depends_on=["fetch"],
    ),
    ExecStage("stage1", "Stage 1 Classification", "scripts/train_stage1.py", ["stage1"], depends_on=["prepare"]),
    ExecStage("stage2", "Stage 2 Regression", "scripts/train_stage2.py", ["stage2"], depends_on=["stage1"]),
    ExecStage(
        "inference",
        "End-to-End Inference",
        "scripts/run_pipeline.py",
        ["inference", "final"],
        depends_on=["stage2"],
    ),
]

EXEC_BY_KEY = {e.key: e for e in EXEC_STAGES}


def python_executable() -> str:
    venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return "python3"


def script_path(rel: str) -> Path:
    return PROJECT_ROOT / rel
