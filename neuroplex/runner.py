"""One simulation worker regardless of browser count; fixed steps, wall-clock pacing."""

import fcntl
import logging
import os
import shutil
from pathlib import Path
import threading
import time

from .config import Config
from .simulation import Simulation

log = logging.getLogger(__name__)


class Runner:
    def __init__(self, directory: Path, seed: int = 7, untrained: bool = False):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file_lock = (self.directory / "process.lock").open("a")
        try:
            fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file_lock.close()
            raise RuntimeError(f"Neuroplex is already using {self.directory}. Stop the other process first.") from None
        self.checkpoint = self.directory / "checkpoint.npz"
        try:
            self.sim = (Simulation.load(self.checkpoint) if self.checkpoint.exists()
                        else Simulation(Config(seed=seed, pretrained_policy=not untrained)))
            if self.sim.migrated_from == 1:
                # One permanent original survives ordinary autosave rotation.
                backup = self.directory / "checkpoint.v1.npz"
                if not backup.exists():
                    temporary = backup.with_suffix(".tmp")
                    with self.checkpoint.open("rb") as source, temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                        destination.flush()
                        os.fsync(destination.fileno())
                    os.replace(temporary, backup)
                self.sim.save(self.checkpoint)
                log.info("Migrated v0.1 checkpoint; original retained at %s", backup)
        except Exception:
            self.file_lock.close()
            raise
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.actual_speed = 0.0
        self.tick_ms = 0.0

    def start(self):
        self.thread = threading.Thread(target=self._run, name="neuroplex-simulation", daemon=True)
        self.thread.start()

    def _run(self):
        last_save = time.monotonic()
        measured_at = time.monotonic()
        measured_steps = 0
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                # Batch size changes real-time pace, never the physical/neural dt.
                with self.lock:
                    batch = self.sim.speed
                for _ in range(batch):
                    if self.stop_event.is_set():
                        break
                    with self.lock:
                        before = self.sim.ticks
                        tick_start = time.monotonic()
                        self.sim.tick()
                        if self.sim.ticks != before:
                            self.tick_ms = self.tick_ms * 0.9 + (time.monotonic() - tick_start) * 100
                            measured_steps += 1
                now = time.monotonic()
                if now - measured_at >= 1:
                    self.actual_speed = measured_steps * self.sim.config.world_dt / (now - measured_at)
                    measured_at, measured_steps = now, 0
                if now - last_save >= 30:
                    with self.lock:
                        self.sim.save(self.checkpoint)
                    last_save = time.monotonic()
                self.stop_event.wait(max(0.001, self.sim.config.world_dt - (time.monotonic() - started)))
        except Exception as exc:
            log.exception("Simulation stopped after an error")
            self.error = f"{type(exc).__name__}: {exc}"

    def snapshot(self):
        with self.lock:
            state = self.sim.snapshot()
            state["runtime"] = {"actual_speed": self.actual_speed, "tick_ms": self.tick_ms,
                                "error": self.error}
            return state

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        try:
            # Do not replace a known-good checkpoint with potentially bad state.
            if not self.error:
                with self.lock:
                    self.sim.save(self.checkpoint)
        finally:
            self.file_lock.close()
