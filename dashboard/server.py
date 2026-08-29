"""FastAPI server — Phase 2 control dashboard."""

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
from dashboard.result_loader import delete_all_runs, delete_run, list_previous_runs, load_all_metrics, purge_all_artifacts
from dashboard.stages import EXEC_BY_KEY, PIPELINE_STEPS
from dashboard.state import ALL_EXEC_KEYS, get_state
from src.data.data_mode import get_data_mode, set_data_mode

DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

app = FastAPI(title="Phase 2 Control Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_poll_thread: threading.Thread | None = None
_poll_stop = threading.Event()


def _on_log(line: str) -> None:
    state = get_state()
    state.sync_from_runner(state.runner)
    broadcaster.publish_log(line)


def _on_status() -> None:
    state = get_state()
    if state.runner:
        state.sync_from_runner(state.runner)
    broadcaster.publish_status(state.to_status_dict())


def _poll_runner_loop(runner: PipelineRunner) -> None:
    state = get_state()
    while runner.is_alive() and not _poll_stop.is_set():
        state.sync_from_runner(runner)
        broadcaster.publish_status(state.to_status_dict())
        runner.poll()
        time.sleep(0.15)

    result = runner.poll()
    if result:
        state.finish_run(result)
        broadcaster.publish_status(state.to_status_dict())
        broadcaster.publish_log(f"[INFO] Dashboard: pipeline status -> {result['status']}")


def _start_pipeline(exec_keys: list[str]) -> None:
    global _poll_thread
    state = get_state()
    if not state.begin_run(exec_keys):
        raise HTTPException(status_code=409, detail="Pipeline already running")

    def run():
        runner = PipelineRunner(exec_keys, on_log=_on_log, on_status=_on_status)
        state.runner = runner
        _poll_stop.clear()
        import os
        os.environ["DATA_MODE"] = get_data_mode()
        global _poll_thread
        _poll_thread = threading.Thread(target=_poll_runner_loop, args=(runner,), daemon=True)
        _poll_thread.start()
        runner.start()

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


@app.get("/api/previous-runs")
async def api_previous_runs():
    return list_previous_runs(limit=20)


@app.post("/api/run-pipeline")
async def api_run_pipeline():
    if get_state().is_running():
        raise HTTPException(status_code=409, detail="Pipeline already running")
    _start_pipeline(ALL_EXEC_KEYS)
    return {"status": "started", "stages": ALL_EXEC_KEYS}


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
    state.reset()
    broadcaster.publish_status(state.to_status_dict())
    broadcaster.publish_log("[INFO] Dashboard: all data, metrics, and models cleared.")
    return {"status": "success", "message": "All data, metrics, and models cleared."}


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
