"""Local web application. Bind to loopback and reach it through an SSH tunnel."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from .runner import Runner

STATIC = Path(__file__).parent / "static"


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "speed", "learning", "save", "new_life"]
    value: bool | int | None = None


def create_app(data_dir: Path, seed: int = 7, untrained: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        runner = Runner(data_dir, seed, untrained)
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
                    sim.brain.learning = command.value
                case "save": sim.save(runner.checkpoint)
                case "new_life":
                    if sim.world.alive:
                        raise HTTPException(409, "This creature is still alive")
                    sim.new_life()
                    sim.save(runner.checkpoint)
            return runner.snapshot()

    @app.websocket("/ws")
    async def stream(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        scheme = "https" if websocket.url.scheme == "wss" else "http"
        if origin and origin != f"{scheme}://{websocket.headers.get('host')}":
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                state = await run_in_threadpool(app.state.runner.snapshot)
                await websocket.send_json(state)
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass

    return app
