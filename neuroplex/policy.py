"""A small online TD motor policy, coupled to the spiking body's motor neurons.

This is explicitly a hybrid: learned attention and motor tables, not LIF neurons.
Local TD/eligibility updates require no backpropagation, replay, or action teacher.
Food coordinates and a desired heading never enter this module.
"""

import json
from pathlib import Path

import numpy as np

from .config import Config


ACTION_NAMES = ("Forward", "Curve left", "Curve right", "Turn left", "Turn right", "Backward")
GOAL_NAMES = ("Food", "Water", "Escape")
# Currents specify a body's motor primitives, never a food-directed action.
# Columns: forward, backward, left, right; movement still depends on LIF spikes.
MOTOR_CURRENTS = np.array([
    [1.60, 0, 0, 0], [1.25, 0, 1.15, 0], [1.25, 0, 0, 1.15],
    [0, 0, 1.60, 0], [0, 0, 0, 1.60], [0, 1.22, 0, 0],
], dtype=np.float32)


class MotorPolicy:
    ARRAY_NAMES = ("values", "eligibility", "visits", "goal_values", "goal_eligibility",
                   "goal_visits", "skill_updates")
    STATE_NAMES = ("state", "action", "pending", "ticks", "discount", "return_sum",
                   "before_potential", "decisions", "updates", "last_td_error",
                   "last_reward", "last_base_reward", "last_shaping", "source", "bootstrap_updates",
                   "goal", "goal_state", "goal_updates")

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed + 2000)
        self.values = np.zeros((459, 6), dtype=np.float64)
        self.visits = np.zeros((459, 6), dtype=np.int64)
        self.goal_values = np.zeros((108, 3), dtype=np.float64)
        self.goal_visits = np.zeros((108, 3), dtype=np.int64)
        self.skill_updates = np.zeros(3, dtype=np.int64)
        self.goal_updates = 0
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
            self.values[:153] = values
            # Explicit transfer learning: approaching water reuses food's motor
            # geometry. Choosing water versus food still has to be learned.
            self.values[153:306] = values
            self.visits[:153] = visits
            self.updates = self.bootstrap_updates = int(data["training"]["updates"])
            self.skill_updates[0] = self.updates
            self.source = "bundled:foraging-v0.2"
        self.reset_activity()

    def reset_activity(self):
        self.eligibility = np.zeros_like(self.values)
        self.goal_eligibility = np.zeros_like(self.goal_values)
        self.goal = 0
        self.goal_state = 0
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
        food = self.target_senses(senses, 0)[:16]
        potential = float(np.max(food * alignment))
        if len(senses) >= 146:
            if senses[144]:
                water = self.target_senses(senses, 1)[:16]
                potential *= 0.25 + 0.75 * float(senses[32])
                potential += (0.25 + 0.75 * float(senses[48])) * float(np.max(water * alignment))
            potential -= 1.5 * float(senses[96:112].max())
        return self.config.shaping_scale * potential

    @staticmethod
    def target_senses(senses, goal):
        result = senses[:80].copy()
        if len(senses) >= 146:
            if goal == 0:
                result[:16] = np.maximum(senses[:16], senses[112:128])
            elif goal == 1:
                result[:16] = np.maximum(senses[80:96], senses[128:144])
            else:
                result[:16] = senses[96:112]
        return result

    @staticmethod
    def available_goals(senses):
        available = [0]
        if len(senses) >= 146:
            if senses[144]:
                available.append(1)
            if senses[96:112].max() > 0.001:
                available.append(2)
        return np.asarray(available)

    def encode_goal(self, senses):
        hunger = min(2, int(float(senses[32]) * 3))
        thirst = min(2, int(float(senses[48]) * 3))
        threat = float(senses[96:112].max()) if len(senses) >= 146 else 0
        danger = int(threat > 0.001) + int(threat > 0.65)
        food = int(self.target_senses(senses, 0)[:16].max() > 0.001)
        water = int(len(senses) >= 146 and self.target_senses(senses, 1)[:16].max() > 0.001)
        return (((hunger * 3 + thirst) * 3 + danger) * 2 + food) * 2 + water

    def _epsilon(self, updates, learning):
        c = self.config
        if not learning:
            return c.exploration_floor
        return c.exploration_floor + (c.exploration_start - c.exploration_floor) * np.exp(
            -updates / c.exploration_decay_decisions)

    def exploration(self, learning: bool) -> float:
        if not learning:
            # Freezing weights does not remove behavioral exploration. A small,
            # observation-independent floor lets the body escape repeated actions.
            return self.config.exploration_floor
        return self._epsilon(self.skill_updates[self.goal], learning)

    def begin(self, senses: np.ndarray, learning: bool) -> np.ndarray:
        self.before_potential = self.potential(senses)
        if not self.pending:
            previous_goal = self.goal
            self.goal_state = self.encode_goal(senses)
            available = self.available_goals(senses)
            if len(available) == 1:
                self.goal = 0
            else:
                values = self.goal_values[self.goal_state, available]
                best = available[values >= values.max() - 1e-10]
                self.goal = int(self.rng.choice(best))
                if self.rng.random() < self._epsilon(self.goal_updates, learning):
                    self.goal = int(self.rng.choice(available))
                if self.goal not in best:
                    self.goal_eligibility.fill(0)
            if self.goal != previous_goal:
                self.eligibility.fill(0)
            self.state = self.goal * 153 + self.encode(self.target_senses(senses, self.goal))
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
                learning: bool, water_gain: float = 0.0, damage: float = 0.0,
                food_gain: float | None = None) -> float | None:
        """Reward a transition; update only after the held action finishes."""
        c = self.config
        gamma_tick = c.policy_discount ** (c.world_dt / c.action_seconds)
        food_units = eaten
        if len(senses) >= 146 and senses[144] and food_gain is not None:
            food_units = food_gain / c.food_energy
        self.last_base_reward = (c.eating_reward * food_units + c.drinking_reward * water_gain / 24.0
                                 - damage * 0.5 - 0.2 * c.world_dt - 2.4 * c.world_dt * touch)
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
        next_state = self.goal * 153 + self.encode(self.target_senses(senses, self.goal))
        bootstrap = self.discount * float(self.values[next_state].max()) if alive else 0.0
        td = self.return_sum + bootstrap - self.values[self.state, self.action]
        self.last_td_error = float(td)
        if learning:
            self.eligibility *= self.discount * c.policy_trace_decay
            self.eligibility[self.state, self.action] = 1.0
            self.values += c.policy_learning_rate * np.clip(td, -10, 10) * self.eligibility
            np.clip(self.values, -100, 200, out=self.values)
            self.visits[self.state, self.action] += 1
            self.updates += 1
            self.skill_updates[self.goal] += 1
            if len(self.available_goals(senses)) > 1 or self.goal != 0:
                available = self.available_goals(senses)
                best = float(self.goal_values[self.encode_goal(senses), available].max()) if alive else 0.0
                goal_td = self.return_sum + self.discount * best - self.goal_values[self.goal_state, self.goal]
                self.goal_eligibility *= self.discount * c.policy_trace_decay
                self.goal_eligibility[self.goal_state, self.goal] = 1
                self.goal_values += c.policy_learning_rate * np.clip(goal_td, -10, 10) * self.goal_eligibility
                np.clip(self.goal_values, -100, 200, out=self.goal_values)
                self.goal_visits[self.goal_state, self.goal] += 1
                self.goal_updates += 1
        else:
            self.eligibility.fill(0)
            self.goal_eligibility.fill(0)
        self.pending = False
        return float(td)

    def summary(self, learning: bool):
        return {
            "action": ACTION_NAMES[self.action], "state": self.state,
            "goal": GOAL_NAMES[self.goal], "goal_updates": self.goal_updates,
            "skill_updates": self.skill_updates.tolist(),
            "decisions": self.decisions, "updates": self.updates,
            "exploration": float(self.exploration(learning)),
            "td_error": self.last_td_error,
            "reward": self.last_reward, "base_reward": self.last_base_reward,
            "shaping_reward": self.last_shaping,
            "visited_pairs": int(np.count_nonzero(self.visits)),
            "parameters": int(self.values.size + self.goal_values.size), "source": self.source,
            "bootstrap_updates": self.bootstrap_updates,
            "live_updates": self.updates - self.bootstrap_updates,
        }
