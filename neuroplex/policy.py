"""A small online TD motor policy, coupled to the spiking body's motor neurons.

This is explicitly a hybrid: action values are a 153 x 6 table, not LIF neurons.
Local TD/eligibility updates require no backpropagation, replay, or action teacher.
Food coordinates and a desired heading never enter this module.
"""

import json
from pathlib import Path

import numpy as np

from .config import Config


ACTION_NAMES = ("Forward", "Curve left", "Curve right", "Turn left", "Turn right", "Backward")
# Currents specify a body's motor primitives, never a food-directed action.
# Columns: forward, backward, left, right; movement still depends on LIF spikes.
MOTOR_CURRENTS = np.array([
    [1.60, 0, 0, 0], [1.25, 0, 1.15, 0], [1.25, 0, 0, 1.15],
    [0, 0, 1.60, 0], [0, 0, 0, 1.60], [0, 1.22, 0, 0],
], dtype=np.float32)


class MotorPolicy:
    ARRAY_NAMES = ("values", "eligibility", "visits")
    STATE_NAMES = ("state", "action", "pending", "ticks", "discount", "return_sum",
                   "before_potential", "decisions", "updates", "last_td_error",
                   "last_reward", "last_base_reward", "last_shaping", "source", "bootstrap_updates")

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed + 2000)
        self.values = np.zeros((153, 6), dtype=np.float64)
        self.visits = np.zeros((153, 6), dtype=np.int64)
        self.decisions = 0
        self.updates = 0
        self.bootstrap_updates = 0
        self.source = "untrained"
        if config.pretrained_policy:
            path = Path(__file__).parent / "assets" / "foraging-v0.2.json"
            data = json.loads(path.read_text())
            values = np.asarray(data["values"], dtype=np.float64)
            visits = np.asarray(data["visits"], dtype=np.int64)
            if (data["format"] != 1 or values.shape != (153, 6) or visits.shape != (153, 6)
                    or not np.isfinite(values).all() or np.any(visits < 0)):
                raise ValueError("Invalid bundled motor policy")
            self.values[:] = values
            self.visits[:] = visits
            self.updates = self.bootstrap_updates = int(data["training"]["updates"])
            self.source = "bundled:foraging-v0.2"
        self.reset_activity()

    def reset_activity(self):
        self.eligibility = np.zeros((153, 6), dtype=np.float64)
        self.state = 0
        self.action = 0
        self.pending = False
        self.ticks = 0
        self.discount = 1.0
        self.return_sum = 0.0
        self.before_potential = 0.0
        self.last_td_error = 0.0
        self.last_reward = 0.0
        self.last_base_reward = 0.0
        self.last_shaping = 0.0

    @staticmethod
    def encode(senses: np.ndarray) -> int:
        """Discretize observations, with no built-in state -> correct action map."""
        brightness = float(np.max(senses[:16]))
        sector = int(np.argmax(senses[:16])) if brightness > 0.001 else 16
        proximity = int(brightness > 0.45) + int(brightness > 0.80)
        wall = 0
        if float(np.max(senses[22:26])) > 0.87 or senses[64] > 0.5:
            wall = 1 if np.mean(senses[16:24]) > np.mean(senses[24:32]) else 2
        return (wall * 17 + sector) * 3 + proximity

    def potential(self, senses: np.ndarray) -> float:
        angles = (np.arange(16) + 0.5) / 16 * self.config.vision_fov - self.config.vision_fov / 2
        alignment = 0.3 + 0.7 * np.maximum(0, np.cos(angles))
        return self.config.shaping_scale * float(np.max(senses[:16] * alignment))

    def exploration(self, learning: bool) -> float:
        if not learning:
            # Freezing weights does not remove behavioral exploration. A small,
            # observation-independent floor lets the body escape repeated actions.
            return self.config.exploration_floor
        c = self.config
        return c.exploration_floor + (c.exploration_start - c.exploration_floor) * np.exp(
            -self.updates / c.exploration_decay_decisions
        )

    def begin(self, senses: np.ndarray, learning: bool) -> np.ndarray:
        self.before_potential = self.potential(senses)
        if not self.pending:
            self.state = self.encode(senses)
            candidates = np.flatnonzero(self.values[self.state] >= self.values[self.state].max() - 1e-10)
            greedy = int(self.rng.choice(candidates))
            self.action = greedy
            if self.rng.random() < self.exploration(learning):
                self.action = int(self.rng.integers(6))
            # Watkins's trace cut: credit must not cross a nongreedy action.
            if self.action not in candidates:
                self.eligibility.fill(0)
            self.pending = True
            self.ticks = 0
            self.return_sum = 0.0
            self.discount = 1.0
            self.decisions += 1
        return MOTOR_CURRENTS[self.action]

    def observe(self, senses: np.ndarray, eaten: int, touch: float, alive: bool,
                learning: bool) -> float | None:
        """Reward a transition; update only after the held action finishes."""
        c = self.config
        gamma_tick = c.policy_discount ** (c.world_dt / c.action_seconds)
        self.last_base_reward = c.eating_reward * eaten - 0.2 * c.world_dt - 2.4 * c.world_dt * touch
        if not alive:
            self.last_base_reward -= c.eating_reward
        next_potential = self.potential(senses) if alive else 0.0
        self.last_shaping = gamma_tick * next_potential - self.before_potential
        self.last_reward = self.last_base_reward + self.last_shaping
        self.return_sum += self.discount * self.last_reward
        self.discount *= gamma_tick
        self.ticks += 1
        if self.ticks < round(c.action_seconds / c.world_dt) and alive:
            return None
        bootstrap = self.discount * float(self.values[self.encode(senses)].max()) if alive else 0.0
        td = self.return_sum + bootstrap - self.values[self.state, self.action]
        self.last_td_error = float(td)
        if learning:
            self.eligibility *= self.discount * c.policy_trace_decay
            self.eligibility[self.state, self.action] = 1.0
            self.values += c.policy_learning_rate * np.clip(td, -10, 10) * self.eligibility
            np.clip(self.values, -100, 200, out=self.values)
            self.visits[self.state, self.action] += 1
            self.updates += 1
        else:
            self.eligibility.fill(0)
        self.pending = False
        return float(td)

    def summary(self, learning: bool):
        return {
            "action": ACTION_NAMES[self.action], "state": self.state,
            "decisions": self.decisions, "updates": self.updates,
            "exploration": float(self.exploration(learning)),
            "td_error": self.last_td_error,
            "reward": self.last_reward, "base_reward": self.last_base_reward,
            "shaping_reward": self.last_shaping,
            "visited_pairs": int(np.count_nonzero(self.visits)),
            "parameters": int(self.values.size), "source": self.source,
            "bootstrap_updates": self.bootstrap_updates,
            "live_updates": self.updates - self.bootstrap_updates,
        }
