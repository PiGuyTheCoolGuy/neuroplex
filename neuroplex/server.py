"""Local web application. Bind to loopback and reach it through an SSH tunnel."""

import asyncio
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt
from starlette.concurrency import run_in_threadpool

from .runner import Runner
from .lab import ExperimentRequest

STATIC = Path(__file__).parent / "static"


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "speed", "learning", "save", "new_life", "curriculum", "stage",
                    "memory", "auto_evaluate", "cancel_experiment", "adopt_champion", "auto_life", "auto_life_delay"]
    value: StrictBool | StrictInt | None = None


def create_app(data_dir: Path, seed: int = 7, untrained: bool = False, stage: int = 0) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        runner = Runner(data_dir, seed, untrained, stage)
        app.state.runner = runner
        runner.start()
        try:
            yield
        finally:
            await run_in_threadpool(runner.close)

    app = FastAPI(title="Neuroplex", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def local_controls(request: Request, call_next):
        # CORS is deliberately not enabled. Also reject cross-site forms/fetches.
        if request.method == "POST":
            if request.headers.get("x-neuroplex-client") != "dashboard":
                return JSONResponse({"detail": "Missing dashboard request header"}, status_code=403)
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin controls are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        runner = app.state.runner
        return JSONResponse({"status": "error" if runner.error else "ok", "error": runner.error},
                            status_code=503 if runner.error else 200)

    @app.get("/api/state")
    def state():
        return app.state.runner.snapshot()

    @app.post("/api/control")
    def control(command: Control):
        runner = app.state.runner
        if runner.error:
            raise HTTPException(503, runner.error)
        with runner.lock:
            sim = runner.sim
            match command.action:
                case "pause": sim.paused = True
                case "resume": sim.paused = False
                case "speed":
                    if type(command.value) is not int or command.value not in (1, 2, 5, 10):
                        raise HTTPException(422, "Speed must be 1, 2, 5, or 10")
                    sim.speed = command.value
                case "learning":
                    if type(command.value) is not bool:
                        raise HTTPException(422, "Learning requires a boolean")
                    if sim.brain.learning != command.value:
                        sim.brain.policy.reset_activity()
                        sim.brain.eligibility.fill(0)
                    sim.brain.learning = command.value
                case "save": sim.save(runner.checkpoint)
                case "auto_life":
                    if type(command.value) is not bool:
                        raise HTTPException(422, "Automatic lives require a boolean")
                    sim.auto_life = command.value
                    sim.save(runner.checkpoint)
                case "auto_life_delay":
                    if type(command.value) is not int or not 1 <= command.value <= 300:
                        raise HTTPException(422, "Delay must be an integer from 1 to 300 real seconds")
                    sim.auto_life_delay = command.value
                    sim.save(runner.checkpoint)
                case "new_life":
                    if sim.world.alive:
                        raise HTTPException(409, "This creature is still alive")
                    sim.new_life()
                    sim.save(runner.checkpoint)
                case "curriculum" | "memory" | "auto_evaluate":
                    if type(command.value) is not bool:
                        raise HTTPException(422, "This control requires a boolean")
                    if command.action == "memory":
                        sim.set_memory(command.value)
                    elif command.action == "curriculum":
                        sim.curriculum = command.value
                    else:
                        sim.auto_evaluate = command.value
                        sim.next_evaluation = sim.elapsed + 1800
                case "stage":
                    if type(command.value) is not int or not 0 <= command.value <= 4:
                        raise HTTPException(422, "Stage must be 0, 1, 2, 3, or 4")
                    sim.set_stage(command.value)
                case "cancel_experiment": runner.lab.cancel()
                case "adopt_champion":
                    try:
                        runner.adopt_champion()
                    except ValueError as exc:
                        raise HTTPException(409, str(exc)) from exc
            # Apply control transitions to the timer immediately; only the
            # simulation worker may trigger an automatic next life.
            runner.update_auto_life(allow_restart=False)
            return runner.snapshot()

    @app.post("/api/experiments")
    def experiment(request: ExperimentRequest):
        runner = app.state.runner
        if runner.error:
            raise HTTPException(503, runner.error)
        with runner.lock:
            try:
                runner.lab.start(runner.sim, request.model_dump())
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return runner.snapshot()

    @app.get("/api/metrics.csv")
    def metrics_csv():
        runner = app.state.runner
        with runner.lock:
            rows = list(runner.sim.metrics)
        output = io.StringIO()
        columns = list(rows[0]) if rows else ["sample", "time", "life", "energy", "hydration", "health", "food_per_minute"]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="neuroplex-metrics.csv"'})

    @app.get("/api/experiments/result")
    def experiment_result():
        runner = app.state.runner
        with runner.lock:
            runner.lab.refresh(force=True)
            path = runner.lab.current / "result.json" if runner.lab.current else None
            if path is None or not path.is_file():
                raise HTTPException(404, "No completed experiment result")
            content = path.read_text()
        return Response(content, media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="neuroplex-experiment.json"'})

    @app.websocket("/ws")
    async def stream(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        scheme = "https" if websocket.url.scheme == "wss" else "http"
        if origin and origin != f"{scheme}://{websocket.headers.get('host')}":
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            initial = await run_in_threadpool(app.state.runner.snapshot)
            await asyncio.wait_for(websocket.send_json(initial), timeout=5)
            seen = {kind: initial["stream_time"] for kind in ("frame", "telemetry", "history", "charts")}
            while True:
                started = asyncio.get_running_loop().time()
                messages = await run_in_threadpool(app.state.runner.stream_messages, seen)
                for kind, stamp, payload in messages:
                    await asyncio.wait_for(websocket.send_text(payload), timeout=5)
                    seen[kind] = stamp
                await asyncio.sleep(max(.001, .05 - (asyncio.get_running_loop().time() - started)))
        except (WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            pass

    return app
