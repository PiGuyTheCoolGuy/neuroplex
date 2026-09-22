"""Fixed-step simulation and complete, atomic checkpoints (no pickle)."""

from collections import deque
from dataclasses import asdict
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


class Simulation:
    FORMAT_VERSION = 2

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.brain = Brain(self.config)
        self.world = World(self.config)
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
        senses = self.world.sense()
        speed, turn = self.brain.advance(senses)
        outcome = self.world.step(speed, turn)
        next_senses = self.world.sense()
        self.brain.observe(next_senses, outcome["eaten"], self.world.touch, self.world.alive)
        self.total_drive_reward += outcome["reward"]
        self.total_reward += self.brain.last_reward
        self.ticks += 1
        if outcome["eaten"]:
            self.event(f"Ate {outcome['eaten']} food · learning reward {self.brain.last_reward:+.3f}")
        if outcome["died"]:
            self.event("The creature starved. Brain and world are preserved; start a new life manually.")
        if self.ticks % round(1 / self.config.world_dt) == 0 or outcome["died"]:
            self.history.append({"time": self.world.time, "energy": self.world.energy,
                                 "eaten": self.world.eaten,
                                 "rate": float(self.brain.rates.mean())})

    def new_life(self):
        if self.world.alive:
            raise ValueError("A new life is available only after this creature has died.")
        self.life += 1
        new_config = Config(**{**asdict(self.config), "seed": self.config.seed + self.life - 1})
        self.world = World(new_config)
        self.brain.reset_activity()
        self.brain.last_reward = 0
        self.paused = False
        self.ticks = 0
        self.total_reward = 0
        self.total_drive_reward = 0
        self.history.clear()
        self.events.clear()
        self.event(f"Life {self.life} began with the previous life's learned synaptic weights.")

    def snapshot(self):
        return {
            "world": self.world.summary(), "brain": self.brain.summary(),
            "paused": self.paused, "speed": self.speed, "life": self.life,
            "history": list(self.history), "events": list(self.events),
            "total_reward": self.total_reward, "saved_at": self.saved_at,
            "total_drive_reward": self.total_drive_reward,
        }

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        world_meta = {k: v for k, v in vars(self.world).items()
                      if k not in ("config", "rng", "food", "regrow_at", "retina")}
        metadata = {
            "version": self.FORMAT_VERSION, "config": asdict(self.config),
            "world_config": asdict(self.world.config), "world": world_meta,
            "world_rng": self.world.rng.bit_generator.state,
            "brain_rng": self.brain.rng.bit_generator.state,
            "policy_rng": self.brain.policy.rng.bit_generator.state,
            "policy": {key: getattr(self.brain.policy, key) for key in MotorPolicy.STATE_NAMES},
            "brain": {k: getattr(self.brain, k) for k in
                      ("learning", "last_reward", "total_abs_change", "steps")},
            "simulation": {"paused": self.paused, "speed": self.speed, "life": self.life,
                           "ticks": self.ticks, "total_reward": self.total_reward,
                           "total_drive_reward": self.total_drive_reward,
                           "history": list(self.history), "events": list(self.events),
                           "saved_at": saved_at},
        }
        arrays = {"brain_" + k: getattr(self.brain, k) for k in Brain.ARRAY_NAMES}
        arrays.update({"policy_" + k: getattr(self.brain.policy, k) for k in MotorPolicy.ARRAY_NAMES})
        arrays.update(world_food=self.world.food, world_regrow_at=self.world.regrow_at,
                      world_retina=self.world.retina,
                      metadata=np.array(json.dumps(metadata)))
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
                if version not in (1, cls.FORMAT_VERSION):
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
                if version == 2:
                    for name in MotorPolicy.ARRAY_NAMES:
                        value = archive["policy_" + name]
                        expected = getattr(sim.brain.policy, name)
                        if (value.shape != expected.shape or value.dtype != expected.dtype
                                or not np.isfinite(value).all()):
                            raise ValueError(f"invalid motor policy array: {name}")
                        setattr(sim.brain.policy, name, value.copy())
                    if set(metadata["policy"]) != set(MotorPolicy.STATE_NAMES):
                        raise ValueError("invalid policy metadata fields")
                    for name, value in metadata["policy"].items():
                        setattr(sim.brain.policy, name, value)
                    sim.brain.policy.rng.bit_generator.state = metadata["policy_rng"]
                    if not 0 <= sim.brain.policy.state < 153 or not 0 <= sim.brain.policy.action < 6:
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
                for key in ("food", "regrow_at", "retina"):
                    value = archive["world_" + key]
                    if value.shape != getattr(sim.world, key).shape or not np.isfinite(value).all():
                        raise ValueError(f"invalid world array: {key}")
                    setattr(sim.world, key, value.copy())
                sim.world.rng.bit_generator.state = metadata["world_rng"]
                state = metadata["simulation"]
                sim.history = deque(state.pop("history"), maxlen=240)
                sim.events = deque(state.pop("events"), maxlen=30)
                for key, value in state.items():
                    if key not in ("paused", "speed", "life", "ticks", "total_reward", "total_drive_reward", "saved_at"):
                        raise ValueError("invalid simulation metadata")
                    setattr(sim, key, value)
                if sim.speed not in (1, 2, 5, 10):
                    raise ValueError("invalid simulation speed")
                if version == 1:
                    sim.total_drive_reward = sim.total_reward
                    sim.total_reward = 0.0
                    sim.brain.last_reward = 0.0
                    sim.migrated_from = 1
                    sim.event("Upgraded to v0.2: preserved this world and recurrent weights; added the trained motor policy.")
                return sim
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Cannot load checkpoint {path}: {exc}. The file was not overwritten.") from exc
