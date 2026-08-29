"""In-memory pipeline state manager for FastAPI dashboard."""

from __future__ import annotations

import threading
import time
from typing import Any

from dashboard.stages import EXEC_BY_KEY, PIPELINE_STEPS, PROJECT_ROOT, STATUS_WAITING

ALL_EXEC_KEYS = ["fetch", "prepare", "stage1", "stage2", "inference"]


def _data_mode_label() -> str:
    from src.data.data_mode import get_data_mode
    return get_data_mode()


class PipelineStateManager:
    """Thread-safe pipeline execution state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset_idle()

    def reset_idle(self) -> None:
        with self._lock:
            self.pipeline_status = "IDLE"
            self.current_exec_key: str | None = None
            self.current_step_key: str | None = None
            self.stage_status = {s.key: STATUS_WAITING for s in PIPELINE_STEPS}
            self.logs: list[str] = []
            self.warnings: list[str] = []
            self.errors: list[str] = []
            self.start_time: float | None = None
            self.end_time: float | None = None
            self.elapsed_sec: float | None = None
            self.failed_stage: str | None = None
            self.failed_error: str | None = None
            self.inference_result: dict[str, Any] = {}
            self.active_run_dir: str | None = None
            self.runner = None

    def reset(self) -> None:
        """Clear in-memory pipeline state (alias for reset_idle)."""
        self.reset_idle()

    def begin_run(self, exec_keys: list[str]) -> bool:
        with self._lock:
            if self.pipeline_status == "RUNNING":
                return False
            self.pipeline_status = "RUNNING"
            self.current_exec_key = None
            self.current_step_key = None
            self.stage_status = {s.key: STATUS_WAITING for s in PIPELINE_STEPS}
            self.logs = []
            self.warnings = []
            self.errors = []
            self.start_time = time.time()
            self.end_time = None
            self.elapsed_sec = None
            self.failed_stage = None
            self.failed_error = None
            self.inference_result = {}
            self.active_run_dir = None
            self._pending_exec_keys = exec_keys
            return True

    def sync_from_runner(self, runner) -> None:
        with self._lock:
            self.logs = list(runner.logs)
            self.warnings = list(runner.warnings)
            self.errors = list(runner.errors)
            self.stage_status.update(runner.stage_status)
            if runner.current_exec_key:
                self.current_exec_key = EXEC_BY_KEY[runner.current_exec_key].label
            running = [k for k, v in runner.stage_status.items() if v == "RUNNING" and not k.startswith("_")]
            if running:
                self.current_step_key = running[-1]
            if runner.run_dir:
                self.active_run_dir = str(runner.run_dir)

    def finish_run(self, result: dict[str, Any]) -> None:
        from dashboard.result_loader import parse_inference_from_log

        with self._lock:
            self.pipeline_status = result.get("status", "FAILED")
            self.stage_status = result.get("stage_status", self.stage_status)
            self.warnings = result.get("warnings", [])
            self.errors = result.get("errors", [])
            self.logs = result.get("logs", self.logs)
            self.failed_stage = result.get("failed_stage")
            self.failed_error = result.get("failed_error")
            self.end_time = time.time()
            if self.start_time:
                self.elapsed_sec = round(self.end_time - self.start_time, 2)
            if result.get("inference_log"):
                self.inference_result = parse_inference_from_log(result["inference_log"])
            if result.get("run_dir"):
                self.active_run_dir = result["run_dir"]
            self.runner = None

    def stop_requested(self) -> None:
        with self._lock:
            if self.pipeline_status == "RUNNING":
                self.pipeline_status = "STOPPED"

    def to_status_dict(self) -> dict[str, Any]:
        with self._lock:
            elapsed = None
            if self.start_time:
                end = self.end_time or time.time()
                elapsed = round(end - self.start_time, 2)
            return {
                "pipeline_status": self.pipeline_status,
                "current_exec_key": self.current_exec_key,
                "current_step_key": self.current_step_key,
                "stage_status": dict(self.stage_status),
                "start_time": self.start_time,
                "start_time_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)) if self.start_time else None,
                "elapsed_sec": elapsed or self.elapsed_sec,
                "failed_stage": self.failed_stage,
                "failed_error": self.failed_error,
                "warnings_count": len(self.warnings),
                "errors_count": len(self.errors),
                "active_run_dir": self.active_run_dir,
                "inference_result": self.inference_result,
                "data_mode": _data_mode_label(),
            }

    def is_running(self) -> bool:
        with self._lock:
            return self.pipeline_status == "RUNNING"

    def check_dependencies(self, exec_key: str) -> str | None:
        raw_tle = PROJECT_ROOT / "data" / "raw" / "tle_history.csv"
        raw_discos = PROJECT_ROOT / "data" / "raw" / "discos_metadata.csv"
        train = PROJECT_ROOT / "data" / "processed" / "train.csv"
        stage1 = PROJECT_ROOT / "models" / "stage1_lightgbm.joblib"
        deps = {
            "prepare": (raw_tle.exists() and raw_discos.exists(), "Run Ingestion first (raw CSVs missing)."),
            "stage1": (train.exists(), "Run Data Preparation first (train.csv missing)."),
            "stage2": (stage1.exists(), "Run Stage 1 first (models missing)."),
            "inference": (stage1.exists(), "Run Stage 1 and Stage 2 first."),
        }
        ok, msg = deps.get(exec_key, (True, ""))
        return None if ok else msg

    def can_delete_run(self, run_id: str) -> tuple[bool, str | None]:
        """Return (allowed, error_message)."""
        if self.is_running() and self.active_run_dir and run_id in self.active_run_dir:
            return False, "Cannot delete the run directory while pipeline is RUNNING."
        return True, None


_state = PipelineStateManager()


def get_state() -> PipelineStateManager:
    return _state
