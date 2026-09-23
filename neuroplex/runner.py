"""One simulation worker regardless of browser count; fixed steps, wall-clock pacing."""

import fcntl
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import replace
import json
import os
import shutil
from pathlib import Path
import threading
import time

from .config import Config
from .simulation import Simulation
from .lab import Laboratory
from .experiments import GENES, copy_model
from .policy import ESCAPE_START, BUILD_START

log = logging.getLogger(__name__)


class Runner:
    def __init__(self, directory: Path, seed: int = 7, untrained: bool = False, stage: int = 0):
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
                        else Simulation(Config(seed=seed, pretrained_policy=not untrained, habitat_stage=stage)))
            if self.sim.migrated_from is not None:
                # One permanent original survives ordinary autosave rotation.
                backup = self.directory / f"checkpoint.v{self.sim.migrated_from}.npz"
                if not backup.exists():
                    temporary = backup.with_suffix(".tmp")
                    with self.checkpoint.open("rb") as source, temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                        destination.flush()
                        os.fsync(destination.fileno())
                    os.replace(temporary, backup)
                self.sim.expand_habitat()
                self.sim.save(self.checkpoint)
                log.info("Migrated checkpoint; original retained at %s", backup)
            self.lab = Laboratory(self.directory)
            self.metric_log = RotatingFileHandler(self.directory / "metrics.jsonl", maxBytes=5_000_000,
                                                  backupCount=3, encoding="utf-8")
            self.metric_log.setFormatter(logging.Formatter("%(message)s"))
            self.last_metric = self.sim.metric_sequence
        except Exception:
            self.file_lock.close()
            raise
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.actual_speed = 0.0
        self.tick_ms = 0.0
        self._life_timer_key = None
        self._life_remaining = None
        self._life_clock = time.monotonic()
        self._life_counting = False
        self._stream_cache = {}
        self._chart_revision = None
        self._history_revision = None
        self.update_auto_life(allow_restart=False)

    def update_auto_life(self, now=None, *, allow_restart=True):
        """Called under lock. Wall time only; pause freezes, disable cancels.

        Nothing in Simulation.tick() respawns, so evaluation episodes still end
        at death. Restarting the service starts a fresh delay, not offline time.
        """
        now = time.monotonic() if now is None else now
        sim = self.sim
        if sim.world.alive or not sim.auto_life:
            self._life_timer_key, self._life_remaining = None, None
            self._life_counting = False
        else:
            key = (sim.life, sim.auto_life_delay)
            if self._life_timer_key != key:
                self._life_timer_key = key
                self._life_remaining = float(sim.auto_life_delay)
            elif self._life_counting:
                self._life_remaining = max(0.0, self._life_remaining - max(0.0, now - self._life_clock))
            self._life_counting = not sim.paused and not self.error
            if allow_restart and self._life_counting and self._life_remaining <= 0:
                # Retain the completed life's final learning in the backup too.
                sim.save(self.checkpoint)
                sim.new_life()
                sim.event("Automatic next life: learned weights and skills retained.")
                sim.save(self.checkpoint)
                self._life_timer_key, self._life_remaining = None, None
                self._life_counting = False
        self._life_clock = now

    def auto_life_status(self, now=None):
        now = time.monotonic() if now is None else now
        remaining = self._life_remaining
        if remaining is not None and self._life_counting and not self.error:
            remaining = max(0.0, remaining - max(0.0, now - self._life_clock))
        return {"remaining": remaining, "waiting": self._life_timer_key is not None,
                "paused": bool(self.sim.paused or self.error)}

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
                        if self.sim.metric_sequence != self.last_metric:
                            self.metric_log.emit(logging.LogRecord("neuroplex.metrics", logging.INFO, "", 0,
                                                                  json.dumps(self.sim.metrics[-1]), (), None))
                            self.last_metric = self.sim.metric_sequence
                        if self.sim.ticks != before:
                            self.tick_ms = self.tick_ms * 0.9 + (time.monotonic() - tick_start) * 100
                            measured_steps += 1
                now = time.monotonic()
                with self.lock:
                    self.update_auto_life(now)
                    self.lab.refresh()
                    if (self.sim.auto_evaluate and self.sim.elapsed >= self.sim.next_evaluation
                            and not self.lab.running() and not self.sim.paused):
                        self.sim.next_evaluation = self.sim.elapsed + 1800
                        try:
                            self.lab.start(self.sim, {"kind": "evaluate", "stage": self.sim.world.stage,
                                                     "seed": (self.sim.config.seed + self.sim.metric_sequence) % 2**32,
                                                     "seconds": 60.0, "trials": 2})
                            self.sim.event("Started a frozen evaluation of a copy on two new worlds.")
                        except (OSError, ValueError) as exc:
                            log.exception("Could not launch evaluation")
                            self.sim.event(f"Evaluation could not start: {exc}")
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

    def snapshot(self, include_charts=True):
        with self.lock:
            state = self.sim.snapshot(include_charts=include_charts)
            state["stream_time"] = time.monotonic()
            state["auto_life_status"] = self.auto_life_status(state["stream_time"])
            state["runtime"] = {"actual_speed": self.actual_speed, "tick_ms": self.tick_ms,
                                "error": self.error}
            state["lab"] = self.lab.snapshot()
            return state

    def stream_messages(self, seen):
        """Latest-only, serialized-once caches shared by all browser clients.

        No per-client queue/backlog: slow clients skip old frames. Chart arrays
        travel only when changed, at most once a second, not with every frame.
        Nothing is built if no browser is asking for updates.
        """
        with self.lock:
            now = time.monotonic()
            for kind, interval in (("frame", .05), ("telemetry", .2), ("history", 1.0), ("charts", 1.0)):
                previous = self._stream_cache.get(kind)
                # Shared time buckets avoid halving the frame rate when a
                # client's next poll arrives a fraction of a millisecond early.
                if previous and int(now / interval) == int(previous[0] / interval):
                    continue
                if kind == "frame":
                    packet = {"world": self.sim.world.render_state(), "life": self.sim.life,
                              "paused": self.sim.paused, "speed": self.sim.speed,
                              "auto_life": self.sim.auto_life, "auto_life_delay": self.sim.auto_life_delay,
                              "auto_life_status": self.auto_life_status(now)}
                elif kind == "telemetry":
                    packet = self.snapshot(include_charts=False)
                elif kind == "history":
                    revision = (self.sim.life, self.sim.ticks // round(1 / self.sim.config.world_dt),
                                self.sim.world.alive)
                    if revision == self._history_revision:
                        continue
                    self._history_revision = revision
                    packet = {"history": list(self.sim.history)}
                else:
                    revision = (self.sim.life, self.sim.metric_sequence)
                    if revision == self._chart_revision:
                        continue
                    self._chart_revision = revision
                    packet = self.sim.chart_snapshot()
                    packet.pop("history")
                packet.update(kind=kind, stream_time=now)
                self._stream_cache[kind] = (now, json.dumps(packet, separators=(",", ":"), allow_nan=False))
            return [(kind, stamp, payload) for kind, (stamp, payload) in self._stream_cache.items()
                    if stamp > seen.get(kind, -1)]

    def adopt_champion(self):
        if self.sim.world.alive:
            raise ValueError("A champion can start a new life only after the current creature has died")
        champion_path = self.lab.champion()
        champion = Simulation.load(champion_path)
        practice = champion.brain.policy.source.startswith("escape-practice:")
        baseline = Simulation.load(champion_path.with_name("source.npz")) if practice else None
        self.sim.save(self.directory / f"checkpoint.before-evolution-life-{self.sim.life}.npz")
        self.sim.new_life()
        if practice:
            # Practice may have run while the main creature kept learning.
            # Merge only the trained skill/manager entries; never roll back
            # today's food, water, construction, or recurrent synaptic learning.
            import numpy as np
            current, trained, original = self.sim.brain.policy, champion.brain.policy, baseline.brain.policy
            section = slice(ESCAPE_START, BUILD_START)
            current.values[section] = trained.values[section]
            current.visits[section] += np.maximum(0, trained.visits[section] - original.visits[section])
            changed = trained.goal_values != original.goal_values
            changed[:, 3] = False
            current.goal_values[changed] = trained.goal_values[changed]
            current.goal_visits += np.maximum(0, trained.goal_visits - original.goal_visits)
            additional = int(trained.skill_updates[2] - original.skill_updates[2])
            current.skill_updates[2] += additional
            current.updates += additional
            current.goal_updates += trained.goal_updates - original.goal_updates
            current.source = trained.source
        else:
            genes = {name: getattr(champion.config, name) for name in GENES}
            self.sim.config = replace(self.sim.config, **genes)
            self.sim.brain.config = self.sim.brain.policy.config = self.sim.memory.config = self.sim.config
            self.sim.world.config = replace(self.sim.world.config, **genes)
            copy_model(champion, self.sim)
        self.sim.event("A trained candidate began a new life; the previous checkpoint was archived.")
        self.sim.save(self.checkpoint)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self.lab.close()
            self.metric_log.close()
            # Do not replace a known-good checkpoint with potentially bad state.
            if not self.error:
                with self.lock:
                    self.sim.save(self.checkpoint)
        finally:
            self.file_lock.close()
