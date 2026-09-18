"""FastAPI server — AI-Enabled Space Situational Awareness dashboard."""

from __future__ import annotations

import asyncio
import queue as thread_queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard.pipeline_runner import PipelineRunner, broadcaster
from dashboard.result_loader import (
    clear_derived_pipeline_data,
    delete_all_runs,
    delete_run,
    invalidate_metrics_cache,
    list_previous_runs,
    load_all_metrics,
    load_poc_artifacts,
    purge_all_artifacts,
)
from dashboard.stages import (
    EXEC_BY_KEY,
    FULL_PIPELINE_KEYS,
    PHASE2_EXEC_KEYS,
    PIPELINE_STEPS,
)
from dashboard.state import get_state
from src.config import RESULTS_DIR
from src.data.data_mode import get_data_mode, set_data_mode

DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

app = FastAPI(title="AI-Enabled Space Situational Awareness")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PHASE3_IMAGES_DIR = Path(__file__).resolve().parents[1] / "results" / "phase3"
PHASE3_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/phase3_images",
    StaticFiles(directory=str(PHASE3_IMAGES_DIR)),
    name="phase3_images",
)

# Hardcoded Phase 3 results from the latest ablation + Stage 2 LOOCV runs.
PHASE3_METRICS = {
    "ablation": [
        {"model": "LightGBM", "orbital_f1": 0.5500, "fused_f1": 0.5686},
        {"model": "AdaBoost", "orbital_f1": 0.5763, "fused_f1": 0.5974},
        {"model": "DecisionTree", "orbital_f1": 0.4516, "fused_f1": 0.5159},
    ],
    "stage2": [
        {"class": "Payload", "target": "mass", "n_samples": 16, "mae": 893.62, "r2": 0.1218},
        {"class": "Rocket Body", "target": "mass", "n_samples": 12, "mae": 958.39, "r2": 0.1697},
    ],
    "images": {
        "confusion_matrix": "/phase3_images/confusion_matrix.png",
        "shap_feature_importance": "/phase3_images/shap_feature_importance.png",
    },
}

_poll_thread: threading.Thread | None = None
_poll_stop = threading.Event()


def _on_log(line: str) -> None:
    state = get_state()
    if state.runner:
        state.sync_from_runner(state.runner)
    broadcaster.publish_log(line)


def _on_status() -> None:
    state = get_state()
    if state.runner:
        state.sync_from_runner(state.runner)
    broadcaster.publish_status(state.to_status_dict())


def _poll_runner_loop(runner: PipelineRunner) -> None:
    state = get_state()
    result: dict | None = None
    try:
        # Wait until the worker thread is actually running — otherwise is_alive
        # is False before start() and we false-STOP the pipeline immediately.
        start_deadline = time.time() + 5.0
        while (
            not runner.is_alive
            and not _poll_stop.is_set()
            and time.time() < start_deadline
        ):
            result = runner.poll() or result
            if result:
                break
            time.sleep(0.05)

        while runner.is_alive and not _poll_stop.is_set():
            state.sync_from_runner(runner)
            broadcaster.publish_status(state.to_status_dict())
            result = runner.poll() or result
            if result:
                break
            time.sleep(0.15)

        # Final drain after the worker thread exits (do not discard earlier polls).
        # Brief grace if the worker just finished putting its result.
        drain_deadline = time.time() + 2.0
        while result is None and time.time() < drain_deadline:
            result = runner.poll()
            if result:
                break
            if not runner.is_alive:
                time.sleep(0.05)
            else:
                break

        if not result:
            result = {
                "status": "STOPPED",
                "stage_status": dict(runner.stage_status),
                "warnings": list(runner.warnings),
                "errors": list(runner.errors) or ["Pipeline exited without a completion status"],
                "logs": list(runner.logs),
                "failed_stage": runner.failed_stage,
                "failed_error": runner.failed_error or "Pipeline did not run to completion",
                "inference_log": "".join(runner.inference_log),
                "run_dir": str(runner.run_dir),
            }
        # Failures / aborts surface as STOPPED in the dashboard badge.
        if result.get("status") in (None, "", "FAILED"):
            result["status"] = "STOPPED"

        state.finish_run(result)
        # Ensure any lingering subprocess is terminated once the run is settled.
        try:
            runner.stop()
        except Exception:
            pass
        broadcaster.publish_status(state.to_status_dict())
        broadcaster.publish_log(f"[INFO] Dashboard: pipeline status -> {result['status']}")
        broadcaster.publish_metrics_refresh()
    finally:
        _poll_stop.set()


def _start_pipeline(exec_keys: list[str]) -> None:
    global _poll_thread
    state = get_state()
    if not state.begin_run(exec_keys):
        raise HTTPException(status_code=409, detail="Pipeline already running")

    # Fresh run: wipe prior derived outputs shown on the dashboard (keep MMT-9 cache).
    keys = set(exec_keys)
    is_photometry = bool(keys & {"fetch", "periods"}) or keys.issuperset(set(PHASE2_EXEC_KEYS))
    is_ml = bool(keys & {"prepare", "stage1", "explain", "stage2", "inference"})
    deleted = clear_derived_pipeline_data(
        clear_models=is_ml or is_photometry,
        clear_raw_metadata=is_photometry,
        clear_database=is_photometry,
        clear_processed=is_photometry or bool(keys & {"prepare"}),
    )
    broadcaster.publish_log(
        f"[INFO] Cleared {deleted} prior derived file(s) before run (MMT-9 cache preserved)"
    )
    broadcaster.publish_metrics_refresh()

    def run():
        runner = PipelineRunner(exec_keys, on_log=_on_log, on_status=_on_status)
        state.runner = runner
        _poll_stop.clear()
        import os
        os.environ["DATA_MODE"] = get_data_mode()
        # Start the worker before the poller so is_alive is True when polling begins.
        runner.start()
        global _poll_thread
        _poll_thread = threading.Thread(target=_poll_runner_loop, args=(runner,), daemon=True)
        _poll_thread.start()

    threading.Thread(target=run, daemon=True).start()


@app.get("/")
async def index():
    return FileResponse(TEMPLATES_DIR / "index.html")


class ModeRequest(BaseModel):
    mode: str


@app.get("/api/current-mode")
async def api_current_mode():
    return {"mode": get_data_mode()}


@app.post("/api/set-mode")
async def api_set_mode(body: ModeRequest):
    state = get_state()
    if state.is_running():
        raise HTTPException(status_code=400, detail="Cannot change mode while pipeline is RUNNING")
    try:
        mode = set_data_mode(body.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    broadcaster.publish_log(f"[INFO] Dashboard: data mode set to {mode}")
    broadcaster.publish_status(state.to_status_dict())
    return {"status": "success", "mode": mode}


@app.get("/api/status")
async def api_status():
    return get_state().to_status_dict()


@app.get("/api/metrics")
async def api_metrics():
    state = get_state()
    return load_all_metrics(state.inference_result)


@app.get("/api/phase3/metrics")
async def api_phase3_metrics():
    """Phase 3 ablation + Stage 2 mass-regression summary for the AI Characterization panel."""
    return PHASE3_METRICS


@app.get("/api/poc-artifacts")
async def api_poc_artifacts():
    return load_poc_artifacts()


_ARTIFACT_DIRS = {
    "poc_plots": RESULTS_DIR / "poc_plots",
    "periodograms": RESULTS_DIR / "periodograms",
    "folded_lightcurves": RESULTS_DIR / "folded_lightcurves",
}


@app.get("/api/artifacts/{category}/{filename}")
async def api_serve_artifact(category: str, filename: str):
    if category not in _ARTIFACT_DIRS:
        raise HTTPException(status_code=404, detail="Unknown artifact category")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = (_ARTIFACT_DIRS[category] / filename).resolve()
    base = _ARTIFACT_DIRS[category].resolve()
    if base not in path.parents and path != base:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


@app.get("/api/exec-stages")
async def api_exec_stages():
    return {
        "phase2": [{"key": e.key, "label": e.label, "script": e.script} for e in EXEC_BY_KEY.values() if e.phase == 2],
        "phase3": [{"key": e.key, "label": e.label, "script": e.script} for e in EXEC_BY_KEY.values() if e.phase == 3],
    }


@app.get("/api/previous-runs")
async def api_previous_runs():
    return list_previous_runs(limit=20)


@app.post("/api/run-pipeline")
async def api_run_pipeline():
    """Run the full SSA pipeline: photometry + prepare + ablation + XAI + Stage 2 + inference."""
    if get_state().is_running():
        raise HTTPException(status_code=409, detail="Pipeline already running")
    _start_pipeline(FULL_PIPELINE_KEYS)
    return {"status": "started", "stages": FULL_PIPELINE_KEYS}


@app.post("/api/run-phase3")
async def api_run_phase3():
    """Same as full SSA pipeline (photometry through ML characterization)."""
    if get_state().is_running():
        raise HTTPException(status_code=409, detail="Pipeline already running")
    _start_pipeline(FULL_PIPELINE_KEYS)
    return {"status": "started", "stages": FULL_PIPELINE_KEYS}


@app.post("/api/run-stage/{stage_name}")
async def api_run_stage(stage_name: str):
    if stage_name not in EXEC_BY_KEY:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_name}")
    state = get_state()
    if state.is_running():
        raise HTTPException(status_code=409, detail="Pipeline already running")
    warn = state.check_dependencies(stage_name)
    if warn:
        raise HTTPException(status_code=400, detail=warn)
    _start_pipeline([stage_name])
    return {"status": "started", "stage": stage_name}


@app.post("/api/stop-pipeline")
async def api_stop_pipeline():
    state = get_state()
    if not state.is_running() or not state.runner:
        raise HTTPException(status_code=400, detail="No pipeline running")
    state.runner.stop()
    state.stop_requested()
    return {"status": "stop_requested"}


@app.delete("/api/runs/{run_id}")
async def api_delete_run(run_id: str):
    state = get_state()
    ok, err = state.can_delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    try:
        delete_run(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete run: {exc}")
    return {"status": "success", "deleted": run_id}


@app.delete("/api/runs")
async def api_delete_all_runs():
    state = get_state()
    if state.is_running():
        raise HTTPException(status_code=400, detail="Cannot clear runs while pipeline is RUNNING")
    try:
        count = delete_all_runs()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to clear runs: {exc}")
    return {"status": "success", "count": count}


@app.post("/api/reset-all")
async def api_reset_all():
    state = get_state()
    if state.is_running():
        raise HTTPException(status_code=400, detail="Cannot reset while pipeline is actively running.")
    try:
        purge_all_artifacts()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to purge artifacts: {exc}")
    invalidate_metrics_cache()
    state.reset()
    broadcaster.publish_status(state.to_status_dict())
    broadcaster.publish_log("[INFO] Dashboard: all data, metrics, and models cleared.")
    return {"status": "success", "message": "All data, metrics, and models cleared — ready for a fresh SSA pipeline run."}


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    state = get_state()
    inbox = broadcaster.subscribe()
    try:
        for line in state.logs[-300:]:
            await websocket.send_json({"type": "log", "line": line})
        await websocket.send_json({"type": "status", "data": state.to_status_dict()})
        await websocket.send_json({"type": "metrics_refresh"})

        while True:
            try:
                msg = await asyncio.to_thread(inbox.get, True, 30)
            except thread_queue.Empty:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json(msg)
            if msg.get("type") == "status" and msg.get("data", {}).get("pipeline_status") in (
                "COMPLETED", "FAILED", "STOPPED"
            ):
                await websocket.send_json({"type": "metrics_refresh"})
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.unsubscribe(inbox)


@app.get("/api/steps")
async def api_steps():
    return [{"key": s.key, "label": s.label, "number": s.number} for s in PIPELINE_STEPS]
