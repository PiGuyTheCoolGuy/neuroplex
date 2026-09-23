"""Fixed-step simulation and complete, atomic checkpoints (no pickle)."""

from collections import deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .brain import Brain
from .config import Config
from .world import World
from .policy import MotorPolicy
from .memory import SensoryMemory
from .curriculum import STAGES, ready_to_advance


class Simulation:
    FORMAT_VERSION = 4

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.brain = Brain(self.config)
        self.world = World(self.config)
        self.memory = SensoryMemory(self.config)
        self.curriculum = self.config.curriculum_enabled
        self.stage_started = 0.0
        self.elapsed = 0.0
        self.metrics = deque(maxlen=4320)  # six hours, sampled every five seconds
        self.metric_sequence = 0
        self.life_records = deque(maxlen=256)
        self.auto_evaluate = True
        self.next_evaluation = 1800.0
        self.auto_life = True
        self.auto_life_delay = 10  # real seconds; the Runner, not trial physics, owns the timer
        self.paused = False
        self.speed = 1
        self.life = 1
        self.ticks = 0
        self.total_reward = 0.0
        self.total_drive_reward = 0.0
        self.history = deque(maxlen=240)
        self.events = deque(maxlen=30)
        self.saved_at = None
        self.migrated_from = None
        self.event("A new creature is awake. " + (
            "Its motor policy starts with learned food-seeking values."
            if self.config.pretrained_policy else "Its motor policy starts untrained."
        ))

    def event(self, message: str):
        self.events.appendleft({"time": self.world.time, "message": message})

    def tick(self):
        if self.paused or not self.world.alive:
            return
        senses = self.memory.observe(self.world.sense(), self.world.vision_range)
        speed, turn = self.brain.advance(senses)
        outcome = self.world.step(speed, turn)
        self.memory.advance(outcome["forward"], outcome["lateral"], outcome["rotation"], self.config.world_dt)
        if outcome["eaten"]:
            self.memory.traces[0] = 0
        next_senses = self.memory.observe(self.world.sense(), self.world.vision_range)
        self.brain.observe(next_senses, outcome["eaten"], self.world.touch, self.world.alive,
                           water_gain=outcome["water_gain"], damage=outcome["damage"], food_gain=outcome["food_gain"],
                           construction_gain=outcome["construction_gain"], pushed=outcome["pushed"])
        self.total_drive_reward += outcome["reward"]
        self.total_reward += self.brain.last_reward
        self.ticks += 1
        self.elapsed += self.config.world_dt
        if outcome["eaten"]:
            self.event(f"Ate {outcome['eaten']} food · learning reward {self.brain.last_reward:+.3f}")
        if outcome["drank"]:
            self.event("Drinking at a water source.")
        if outcome["damage"]:
            self.event(f"Predator attack · {outcome['damage']:.0f} health lost.")
        if outcome["construction_gain"]:
            self.event(f"Improved physical cover · construction reward {outcome['construction_gain'] * self.config.construction_reward:.2f}.")
        if outcome["died"]:
            self.event(f"Life ended: {self.world.death_reason}. Learned values are preserved.")
            self.life_records.append({"life": self.life, "survival_seconds": self.world.time,
                                      "food": self.world.eaten, "drinks": self.world.drinks,
                                      "stage": self.world.stage, "cause": self.world.death_reason,
                                      "push_distance": self.world.push_distance,
                                      "construction_reward_total": self.world.construction_reward_total})
        if self.ticks % round(1 / self.config.world_dt) == 0 or outcome["died"]:
            self.history.append({"time": self.world.time, "energy": self.world.energy,
                                 "hydration": self.world.hydration, "health": self.world.health,
                                 "eaten": self.world.eaten,
                                 "rate": float(self.brain.rates.mean())})
        if self.ticks % round(5 / self.config.world_dt) == 0 or outcome["died"]:
            self.record_metrics()
            recent = [p for p in list(self.metrics)[-14:] if p["life"] == self.life]
            if (self.curriculum and self.brain.learning and self.world.alive
                    and ready_to_advance(self.world, recent, self.stage_started, self.config.stage_seconds)):
                self.set_stage(self.world.stage + 1, automatic=True)

    def record_metrics(self):
        w = self.world
        recent = [p for p in list(self.metrics)[-13:] if p["life"] == self.life and p["age"] >= w.time - 60]
        first = recent[0] if recent else {"age": 0, "eaten": 0, "reward_total": 0, "drinks": 0}
        seconds = max(self.config.world_dt, w.time - first["age"])
        self.metric_sequence += 1
        self.metrics.append({"sample": self.metric_sequence, "time": round(self.elapsed, 6),
                             "age": round(w.time, 6), "life": self.life, "stage": w.stage,
                             "energy": w.energy, "hydration": w.hydration, "health": w.health,
                             "eaten": w.eaten, "drinks": w.drinks, "alive": w.alive,
                             "food_per_minute": (w.eaten - first["eaten"]) * 60 / seconds,
                             "drinks_per_minute": (w.drinks - first["drinks"]) * 60 / seconds,
                             "reward_per_second": (self.total_reward - first["reward_total"]) / seconds,
                             "reward_total": self.total_reward, "attacks": w.attacks,
                             "updates": self.brain.policy.updates,
                             "push_distance": w.push_distance, "best_shelter": w.best_shelter,
                             "construction_reward_total": w.construction_reward_total,
                             "protected_seconds": w.protected_seconds,
                             "weight_change": float(np.abs(self.brain.weights - self.brain.initial_weights).mean())})

    def set_stage(self, stage, automatic=False):
        if type(stage) is not int or not 0 <= stage < len(STAGES):
            raise ValueError("Stage must be an integer from 0 to 4")
        self.world.set_stage(stage)
        self.stage_started = self.world.time
        # A habitat change starts a new action-credit window, not a new policy.
        self.brain.policy.reset_activity()
        self.event(f"{'Curriculum advanced' if automatic else 'Habitat changed'}: {STAGES[stage]['name']}.")

    def set_memory(self, enabled):
        self.config = replace(self.config, memory_enabled=enabled)
        self.brain.config = self.brain.policy.config = self.memory.config = self.config
        self.world.config = replace(self.world.config, memory_enabled=enabled)
        self.memory.traces.fill(0)
        self.brain.policy.reset_activity()

    def expand_habitat(self):
        """One-time live-world upgrade; keep the body, clock, learning and needs."""
        c = self.config
        new = replace(c, world_width=c.world_width * 1.5, world_height=c.world_height * 1.5,
                      food_count=max(1, int(c.food_count * .75)) if c.food_count else 0,
                      water_count=max(1, int(c.water_count * 2 / 3)) if c.water_count else 0,
                      food_regrow_seconds=max(45.0, c.food_regrow_seconds))
        w = self.world
        w.x *= 1.5
        w.y *= 1.5
        w.food = w.food[:new.food_count] * 1.5
        w.regrow_at = w.regrow_at[:new.food_count].copy()
        w.water = w.water[:new.water_count] * 1.5
        w.predators *= 1.5
        w.config = replace(w.config, world_width=new.world_width, world_height=new.world_height,
                           food_count=new.food_count, water_count=new.water_count,
                           food_regrow_seconds=new.food_regrow_seconds)
        self.config = new
        self.brain.config = self.brain.policy.config = self.memory.config = new
        self.memory.traces.fill(0)
        self.brain.policy.reset_activity()
        w.place_blocks()
        w.sense()
        self.event("Habitat expanded 1.5× in each direction; fewer resources, slower regrowth, and movable blocks.")

    def new_life(self):
        if self.world.alive:
            raise ValueError("A new life is available only after this creature has died.")
        self.life += 1
        new_config = replace(self.config, seed=self.config.seed + self.life - 1, habitat_stage=self.world.stage)
        self.world = World(new_config)
        self.stage_started = 0.0
        self.memory.traces.fill(0)
        self.brain.reset_activity()
        self.brain.last_reward = 0
        self.paused = False
        self.ticks = 0
        self.total_reward = 0
        self.total_drive_reward = 0
        self.history.clear()
        self.events.clear()
        self.event(f"Life {self.life} began with the previous life's learned synaptic weights.")

    def chart_snapshot(self):
        return {"history": list(self.history),
                "metrics": list(self.metrics)[::max(1, (len(self.metrics) + 719) // 720)],
                "latest_metric": self.metrics[-1] if self.metrics else None,
                "life_records": list(self.life_records)}

    def snapshot(self, include_charts=True):
        return {
            "world": self.world.summary(), "brain": self.brain.summary(),
            "paused": self.paused, "speed": self.speed, "life": self.life,
            "events": list(self.events),
            "total_reward": self.total_reward, "saved_at": self.saved_at,
            "total_drive_reward": self.total_drive_reward,
            "memory": self.memory.summary(),
            "curriculum": {"enabled": self.curriculum, "stage": self.world.stage,
                           "stages": [s["name"] for s in STAGES], "stage_seconds": self.world.time - self.stage_started,
                           "minimum_seconds": self.config.stage_seconds},
            **(self.chart_snapshot() if include_charts else {}),
            "elapsed": self.elapsed,
            "auto_evaluate": self.auto_evaluate, "next_evaluation": self.next_evaluation,
            "auto_life": self.auto_life, "auto_life_delay": self.auto_life_delay,
        }

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        world_meta = {k: v for k, v in vars(self.world).items()
                      if k not in ("config", "rng", "ecology_rng", *World.ARRAY_NAMES)}
        metadata = {
            "version": self.FORMAT_VERSION, "config": asdict(self.config),
            "world_config": asdict(self.world.config), "world": world_meta,
            "world_rng": self.world.rng.bit_generator.state,
            "ecology_rng": self.world.ecology_rng.bit_generator.state,
            "brain_rng": self.brain.rng.bit_generator.state,
            "policy_rng": self.brain.policy.rng.bit_generator.state,
            "policy": {key: getattr(self.brain.policy, key) for key in MotorPolicy.STATE_NAMES},
            "brain": {k: getattr(self.brain, k) for k in
                      ("learning", "last_reward", "total_abs_change", "steps")},
            "simulation": {"paused": self.paused, "speed": self.speed, "life": self.life,
                           "ticks": self.ticks, "total_reward": self.total_reward,
                           "total_drive_reward": self.total_drive_reward,
                           "history": list(self.history), "events": list(self.events),
                           "curriculum": self.curriculum, "stage_started": self.stage_started,
                           "elapsed": self.elapsed, "metric_sequence": self.metric_sequence,
                           "auto_evaluate": self.auto_evaluate, "next_evaluation": self.next_evaluation,
                           "auto_life": self.auto_life, "auto_life_delay": self.auto_life_delay,
                           "saved_at": saved_at},
            "metrics": list(self.metrics), "life_records": list(self.life_records),
        }
        arrays = {"brain_" + k: getattr(self.brain, k) for k in Brain.ARRAY_NAMES}
        arrays.update({"policy_" + k: getattr(self.brain.policy, k) for k in MotorPolicy.ARRAY_NAMES})
        arrays.update({"world_" + k: getattr(self.world, k) for k in World.ARRAY_NAMES})
        arrays.update(memory_traces=self.memory.traces, metadata=np.array(json.dumps(metadata, allow_nan=False)))
        temp = path.with_suffix(".tmp")
        try:
            with temp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            # Keep one complete previous file without removing the current file.
            if path.exists():
                backup_temp = path.with_suffix(".backup.tmp")
                shutil.copyfile(path, backup_temp)
                os.replace(backup_temp, path.with_suffix(".previous.npz"))
            os.replace(temp, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp.unlink(missing_ok=True)
        self.saved_at = saved_at

    @classmethod
    def load(cls, path: Path):
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                version = metadata["version"]
                if version not in (1, 2, 3, cls.FORMAT_VERSION):
                    raise ValueError("unsupported checkpoint version")
                config = metadata["config"]
                if version == 1:
                    config = {**config, "learning_rate": Config().learning_rate}
                sim = cls(Config(**config))
                for name in Brain.ARRAY_NAMES:
                    if version == 1 and name == "motor_current":
                        continue
                    value = archive["brain_" + name]
                    expected = getattr(sim.brain, name)
                    if (value.shape != expected.shape or value.dtype != expected.dtype
                            or not np.isfinite(value).all()):
                        raise ValueError(f"invalid brain array: {name}")
                    setattr(sim.brain, name, value.copy())
                for key, value in metadata["brain"].items():
                    setattr(sim.brain, key, value)
                sim.brain.rng.bit_generator.state = metadata["brain_rng"]
                if version >= 2:
                    for name in MotorPolicy.ARRAY_NAMES if version >= 3 else ("values", "eligibility", "visits"):
                        value = archive["policy_" + name]
                        expected = getattr(sim.brain.policy, name)
                        if version == 2:
                            expected = expected[:153]
                        elif version == 3:
                            if name in ("values", "visits", "eligibility"):
                                expected = expected[:459]
                            elif name in ("goal_values", "goal_visits", "goal_eligibility"):
                                expected = expected[:, :3]
                            else:
                                expected = expected[:3]
                        if (value.shape != expected.shape or value.dtype != expected.dtype
                                or not np.isfinite(value).all()):
                            raise ValueError(f"invalid motor policy array: {name}")
                        if version == 2:
                            getattr(sim.brain.policy, name)[:153] = value
                            if name == "values":
                                sim.brain.policy.values[153:306] = value
                        elif version == 3:
                            target = getattr(sim.brain.policy, name)
                            if name in ("values", "visits", "eligibility"):
                                target[:459] = value
                            elif name in ("goal_values", "goal_visits", "goal_eligibility"):
                                target[:, :3] = value
                            else:
                                target[:3] = value
                        else:
                            setattr(sim.brain.policy, name, value.copy())
                    required = set(MotorPolicy.STATE_NAMES)
                    if version == 2:
                        required -= {"goal", "goal_state", "goal_updates"}
                    if version < 4:
                        required -= {"risk_scale"}
                    if set(metadata["policy"]) != required:
                        raise ValueError("invalid policy metadata fields")
                    for name, value in metadata["policy"].items():
                        setattr(sim.brain.policy, name, value)
                    sim.brain.policy.rng.bit_generator.state = metadata["policy_rng"]
                    if version == 2:
                        sim.brain.policy.skill_updates[0] = sim.brain.policy.updates
                    if (not 0 <= sim.brain.policy.state < len(sim.brain.policy.values) or not 0 <= sim.brain.policy.action < 6
                            or not 0 <= sim.brain.policy.goal < 4 or not 0 <= sim.brain.policy.goal_state < 108):
                        raise ValueError("invalid policy state or action")
                else:
                    # Preserve the recurrent memories. The new motor interface
                    # needs its documented fixed thresholds instead of old adaptation.
                    sim.brain.threshold[400:] = 1.0
                sim.world = World(Config(**metadata["world_config"]))
                for key, value in metadata["world"].items():
                    if key not in vars(sim.world) or not isinstance(value, (float, int, bool)):
                        raise ValueError("invalid world metadata")
                    if not np.isfinite(value):
                        raise ValueError("non-finite world state")
                    setattr(sim.world, key, value)
                world_arrays = World.ARRAY_NAMES if version >= 3 else ("food", "regrow_at", "retina")
                if version < 4:
                    world_arrays = tuple(k for k in world_arrays if k not in ("blocks", "block_retina", "obstacle_retina", "block_cover_best"))
                for key in world_arrays:
                    value = archive["world_" + key]
                    if value.shape != getattr(sim.world, key).shape or not np.isfinite(value).all():
                        raise ValueError(f"invalid world array: {key}")
                    setattr(sim.world, key, value.copy())
                sim.world.rng.bit_generator.state = metadata["world_rng"]
                if version >= 3:
                    sim.world.ecology_rng.bit_generator.state = metadata["ecology_rng"]
                    traces = archive["memory_traces"]
                    if traces.shape != (2, 4) or not np.isfinite(traces).all():
                        raise ValueError("invalid sensory memory")
                    sim.memory.traces[:] = traces
                    sim.metrics = deque(metadata["metrics"], maxlen=4320)
                    sim.life_records = deque(metadata["life_records"], maxlen=256)
                else:
                    sim.world.death_code = 0 if sim.world.alive else 1
                    sim.stage_started = sim.world.time
                    sim.elapsed = sim.world.time
                    sim.next_evaluation = sim.elapsed + 1800
                if not 0 <= sim.world.stage <= 4 or not 0 <= sim.world.death_code <= 3:
                    raise ValueError("invalid habitat stage or death cause")
                state = metadata["simulation"]
                sim.history = deque(state.pop("history"), maxlen=240)
                sim.events = deque(state.pop("events"), maxlen=30)
                for key, value in state.items():
                    if key not in ("paused", "speed", "life", "ticks", "total_reward", "total_drive_reward", "saved_at",
                                   "curriculum", "stage_started", "elapsed", "metric_sequence", "auto_evaluate", "next_evaluation",
                                   "auto_life", "auto_life_delay"):
                        raise ValueError("invalid simulation metadata")
                    setattr(sim, key, value)
                if sim.speed not in (1, 2, 5, 10):
                    raise ValueError("invalid simulation speed")
                if (type(sim.auto_life) is not bool or type(sim.auto_life_delay) is not int
                        or not 1 <= sim.auto_life_delay <= 300):
                    raise ValueError("invalid automatic life settings")
                if version == 1:
                    sim.total_drive_reward = sim.total_reward
                    sim.total_reward = 0.0
                    sim.brain.last_reward = 0.0
                if version < cls.FORMAT_VERSION:
                    sim.brain.policy.initialize_new_skills()
                    sim.brain.policy.reset_activity()
                    sim.world.place_blocks()
                    sim.world.sense()
                    sim.migrated_from = version
                    sim.event(f"Upgraded v{version} save: preserved learned values; added obstacle-aware escape and construction.")
                return sim
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Cannot load checkpoint {path}: {exc}. The file was not overwritten.") from exc
