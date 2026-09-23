"""One low-priority experiment process at a time, with durable bounded results."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .experiments import atomic_json


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["evaluate", "evolve", "escape", "shelter"] = "evaluate"
    episodes: int = Field(default=40, ge=1, le=200)
    stage: int = Field(default=0, ge=0, le=4)
    seconds: float = Field(default=60.0, ge=5, le=600)
    evaluation_seconds: float = Field(default=30.0, ge=5, le=300)
    trials: int = Field(default=3, ge=1, le=5)
    population: int = Field(default=4, ge=2, le=8)
    generations: int = Field(default=3, ge=1, le=10)
    seed: int = Field(default=9001, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def practice_bounds(self):
        if self.kind in ("escape", "shelter"):
            self.stage = 3
            if self.seconds > 60:
                raise ValueError("Practice episodes must be 5–60 seconds")
        return self


class Laboratory:
    def __init__(self, directory):
        self.directory = Path(directory) / "experiments"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.current = None
        self.champion_id = None
        self.history = []
        self.status = {"state": "idle"}
        self._last_read = 0.0
        self._cancel_requested = False
        self.index = self.directory / "index.json"
        if self.index.exists():
            data = json.loads(self.index.read_text())
            self.history = data.get("history", [])[-100:]
            champion_id = data.get("champion")
            if champion_id and len(champion_id) == 32 and all(c in "0123456789abcdef" for c in champion_id):
                self.champion_id = champion_id
            identity = data.get("current")
            if identity and len(identity) == 32 and all(c in "0123456789abcdef" for c in identity):
                self.current = self.directory / identity
                self.refresh(force=True)
                if self.status.get("state") in ("running", "queued", "cancelling"):
                    self.status["state"] = "interrupted"
                    self.status["error"] = "Server restarted during the experiment. Start a new run."
                    atomic_json(self.current / "status.json", self.status)

    def _save_index(self):
        atomic_json(self.index, {"current": self.current.name if self.current else None,
                                "champion": self.champion_id, "history": self.history[-100:]})

    def running(self):
        return self.process is not None and self.process.poll() is None

    def refresh(self, force=False):
        if not force and time.monotonic() - self._last_read < 1:
            return
        self._last_read = time.monotonic()
        if self.current and (self.current / "status.json").exists():
            self.status = json.loads((self.current / "status.json").read_text())
        if (self.current and self.status.get("champion_available") and (self.current / "champion.npz").is_file()
                and self.champion_id != self.current.name):
            self.champion_id = self.current.name
            self._save_index()
        if self.process is not None and self.process.poll() is not None:
            if self.status.get("state") in ("running", "queued", "cancelling"):
                if self._cancel_requested:
                    self.status.update(state="cancelled", phase="cancelled")
                else:
                    self.status.update(state="failed", error=f"Experiment worker exited ({self.process.returncode}); see worker.log")
                atomic_json(self.current / "status.json", self.status)
            self.process = None
            self._save_index()
        if (self.current and self.status.get("state") == "completed"
                and not any(h["id"] == self.current.name for h in self.history)):
            self.history.append({"id": self.current.name, "kind": self.status["kind"],
                                 "stage": self.status["stage"], "finished_at": self.status["finished_at"],
                                 "summary": self.status["summary"]})
            self.history = self.history[-100:]
            self._save_index()

    def snapshot(self):
        self.refresh()
        return {"current": self.status, "history": self.history[-30:], "running": self.running(),
                "champion_available": bool(self.champion_id and (self.directory / self.champion_id / "champion.npz").is_file())}

    def start(self, sim, request):
        self.refresh(force=True)
        if self.running():
            raise ValueError("An experiment is already running")
        request = ExperimentRequest(**request).model_dump()
        self._cancel_requested = False
        self.current = self.directory / uuid.uuid4().hex
        self.current.mkdir()
        saved_at = sim.saved_at
        try:
            sim.save(self.current / "source.npz")
        finally:
            sim.saved_at = saved_at
        atomic_json(self.current / "request.json", request)
        self.status = {"state": "queued", "kind": request["kind"], "stage": request["stage"], "request": request}
        atomic_json(self.current / "status.json", self.status)
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with (self.current / "worker.log").open("a") as output:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "neuroplex.experiments", "--request", str(self.current / "request.json"),
                 "--source", str(self.current / "source.npz"), "--output", str(self.current)],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, env=environment)
        self._save_index()
        # Only generated job directories with the ownership marker are eligible.
        jobs = [p for p in self.directory.iterdir() if p.is_dir() and not p.is_symlink() and len(p.name) == 32
                and all(c in "0123456789abcdef" for c in p.name) and (p / "request.json").is_file()]
        for old in sorted(jobs, key=lambda p: p.stat().st_mtime, reverse=True)[20:]:
            if old != self.current and old.name != self.champion_id:
                shutil.rmtree(old)
        return self.snapshot()

    def cancel(self):
        if self.running():
            self._cancel_requested = True
            self.process.terminate()
            self.status["state"] = "cancelling"
        return self.snapshot()

    def champion(self):
        self.refresh(force=True)
        if self.running() or not self.champion_id:
            raise ValueError("Finish evolution or escape practice first")
        path = self.directory / self.champion_id / "champion.npz"
        if not path.is_file():
            raise ValueError("No completed generation has produced a champion")
        return path

    def close(self):
        if self.running():
            self._cancel_requested = True
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.refresh(force=True)
