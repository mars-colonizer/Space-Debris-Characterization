"""Subprocess pipeline runner with live log streaming and subscriber callbacks."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from typing import Callable

from dashboard.log_parser import (
    apply_log_line,
    mark_exec_complete,
    mark_exec_failed,
    mark_exec_start,
    sanitize_log_line,
)
from dashboard.result_loader import RUNS_DIR, save_run_summary
from dashboard.stages import EXEC_BY_KEY, PROJECT_ROOT, python_executable, script_path


class LogBroadcaster:
    """Fan-out log/status messages to WebSocket clients via thread-safe queues."""

    def __init__(self) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish_log(self, line: str) -> None:
        self._publish({"type": "log", "line": line})

    def publish_status(self, status: dict) -> None:
        self._publish({"type": "status", "data": status})

    def _publish(self, msg: dict) -> None:
        with self._lock:
            dead: list[queue.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)


broadcaster = LogBroadcaster()


class PipelineRunner:
    """Runs pipeline scripts sequentially in a background thread."""

    def __init__(
        self,
        exec_keys: list[str],
        on_log: Callable[[str], None] | None = None,
        on_status: Callable[[], None] | None = None,
    ):
        self.exec_keys = exec_keys
        self.on_log = on_log or (lambda _line: None)
        self.on_status = on_status or (lambda: None)
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.done_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.stage_status: dict[str, str] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.logs: list[str] = []
        self.start_time: float | None = None
        self.failed_stage: str | None = None
        self.failed_error: str | None = None
        self.inference_log: list[str] = []
        self.current_exec_key: str | None = None
        self.run_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def start(self) -> None:
        self.start_time = time.time()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def poll(self) -> dict | None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            self.logs.append(line)
            self.on_log(line)

        try:
            return self.done_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _emit(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"{ts} {sanitize_log_line(line)}"
        self.log_queue.put(formatted)

    def _run(self) -> None:
        from dashboard.stages import PIPELINE_STEPS, STATUS_WAITING

        self.stage_status = {s.key: STATUS_WAITING for s in PIPELINE_STEPS}
        overall_ok = True
        self._emit("[INFO] Starting Phase 2 pipeline execution")
        self.on_status()

        for exec_key in self.exec_keys:
            if self._stop.is_set():
                self.done_queue.put(self._result("STOPPED"))
                return

            stage = EXEC_BY_KEY[exec_key]
            self.current_exec_key = exec_key
            mark_exec_start(self.stage_status, stage.step_keys)
            self._emit(f"[INFO] >>> Starting {stage.label}: {stage.script}")
            self.on_status()

            rc = self._run_script(stage.script, stage.label)
            if rc != 0:
                overall_ok = False
                mark_exec_failed(self.stage_status, stage.step_keys)
                self.failed_stage = stage.label
                self.failed_error = self.errors[-1] if self.errors else f"Exit code {rc}"
                self._emit(f"[ERROR] Pipeline stopped at {stage.label}")
                break

            had_warnings = any("[WARNING]" in w for w in self.warnings)
            mark_exec_complete(self.stage_status, stage.step_keys, had_warnings)
            self._emit(f"[OK] Completed {stage.label}")
            self.on_status()

        status = "COMPLETED" if overall_ok and not self._stop.is_set() else (
            "STOPPED" if self._stop.is_set() else "FAILED"
        )
        elapsed = time.time() - (self.start_time or time.time())
        self._emit(f"[INFO] Pipeline finished with status: {status} ({elapsed:.2f}s)")

        summary = {
            "status": status,
            "start_time": datetime.fromtimestamp(self.start_time or time.time()).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_sec": round(elapsed, 2),
            "stage_status": self.stage_status,
            "warnings": self.warnings[-50:],
            "errors": self.errors,
            "failed_stage": self.failed_stage,
            "failed_error": self.failed_error,
            "exec_keys": self.exec_keys,
        }
        save_run_summary(self.run_dir, summary, self.logs)
        self.done_queue.put(self._result(status, summary))
        self.on_status()

    def _result(self, status: str, summary: dict | None = None) -> dict:
        return {
            "status": status,
            "stage_status": self.stage_status,
            "warnings": self.warnings,
            "errors": self.errors,
            "logs": self.logs,
            "failed_stage": self.failed_stage,
            "failed_error": self.failed_error if status != "STOPPED" else "Pipeline stopped by user",
            "inference_log": "".join(self.inference_log),
            "run_dir": str(self.run_dir),
            "summary": summary,
            "current_exec_key": self.current_exec_key,
        }

    def _run_script(self, script_rel: str, label: str) -> int:
        cmd = [python_executable(), str(script_path(script_rel))]
        self._emit(f"[INFO] Executing: {' '.join(cmd[:2])} {script_rel}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            msg = f"[ERROR] Failed to start {script_rel}: {exc}"
            self.errors.append(msg)
            self._emit(msg)
            return 1

        assert self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            if self._stop.is_set():
                self._proc.terminate()
                return 130

            line = raw_line.rstrip("\n")
            if not line:
                continue

            self._emit(line)
            apply_log_line(line, self.stage_status, self.warnings)
            self.on_status()

            if "[ERROR]" in line or "FAILED" in line:
                self.errors.append(line.strip())

            if "run_pipeline.py" in script_rel or label == "End-to-End Inference":
                self.inference_log.append(line + "\n")

        rc = self._proc.wait()
        self._proc = None
        return rc
