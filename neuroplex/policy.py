"""A small online TD motor policy, coupled to the spiking body's motor neurons.

This is explicitly a hybrid: learned attention and motor tables, not LIF neurons.
Local TD/eligibility and feature updates require no backpropagation or action teacher.
The cover residual reuses only a bounded buffer of this creature's experience.
Food coordinates and a desired heading never enter this module.
"""

import json
from pathlib import Path

import numpy as np

from .config import Config
from .cover import CoverLearner, SENSES, cover_features


ACTION_NAMES = ("Forward", "Curve left", "Curve right", "Turn left", "Turn right", "Backward", "Rest")
GOAL_NAMES = ("Food", "Water", "Escape", "Build cover", "Use cover")
ESCAPE_START = 459  # retain all v3 rows verbatim for migration / legacy sensors
ESCAPE_ROWS = 17 * 3 * 16
BUILD_START = ESCAPE_START + ESCAPE_ROWS
BUILD_CONTEXTS = 4 * 5  # cover/contact state × relative bearing of another visible block
COVER_START = BUILD_START + 153 * BUILD_CONTEXTS
POLICY_ROWS = COVER_START + 153
GOAL_STATES = 108 * 4  # original need state × remembered cover/current occupancy
# Currents specify a body's motor primitives, never a food-directed action.
# Columns: forward, backward, left, right; movement still depends on LIF spikes.
MOTOR_CURRENTS = np.array([
    [1.60, 0, 0, 0], [1.25, 0, 1.15, 0], [1.25, 0, 0, 1.15],
    [0, 0, 1.60, 0], [0, 0, 0, 1.60], [0, 1.22, 0, 0], [0, 0, 0, 0],
], dtype=np.float32)


class MotorPolicy:
    ARRAY_NAMES = ("values", "eligibility", "visits", "goal_values", "goal_eligibility",
                   "goal_visits", "skill_updates")
    STATE_NAMES = ("state", "action", "pending", "ticks", "discount", "return_sum",
                   "before_potential", "decisions", "updates", "last_td_error",
                   "last_reward", "last_base_reward", "last_shaping", "source", "bootstrap_updates",
                   "goal", "goal_state", "goal_updates", "risk_scale")

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed + 2000)
        self.values = np.zeros((POLICY_ROWS, len(ACTION_NAMES)), dtype=np.float64)
        self.visits = np.zeros_like(self.values, dtype=np.int64)
        self.goal_values = np.zeros((GOAL_STATES, 5), dtype=np.float64)
        self.goal_visits = np.zeros((GOAL_STATES, 5), dtype=np.int64)
        self.skill_updates = np.zeros(5, dtype=np.int64)
        self.cover_learner = CoverLearner(config.seed)
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
            columns = 6 if data["format"] == 1 else 7
            if (data["format"] not in (1, 2) or values.shape != (153, columns) or visits.shape != (153, columns)
                    or not np.isfinite(values).all() or np.any(visits < 0)):
                raise ValueError("Invalid bundled motor policy")
            self.values[:153, :columns] = values
            # Explicit transfer learning: approaching water reuses food's motor
            # geometry. Choosing water versus food still has to be learned.
            self.values[153:306, :columns] = values
            self.visits[:153, :columns] = visits
            self.updates = self.bootstrap_updates = int(data["training"]["updates"])
            self.skill_updates[0] = self.updates
            self.source = "bundled:foraging-v0.2"
        self.initialize_new_skills()
        self.reset_activity()

    def initialize_new_skills(self):
        # Preserve old escape experience as a prior in every richer context.
        # Nothing here encodes an action label or a shelter arrangement.
        for mask in range(16):
            old_wall = (1 if mask & 2 else 2) if mask & 1 else 0
            start = ESCAPE_START + mask * 51
            self.values[start:start + 51] = self.values[306 + old_wall * 51:306 + (old_wall + 1) * 51]
        # Explicit transfer: approaching a visible block reuses food navigation.
        for context in range(BUILD_CONTEXTS):
            self.values[BUILD_START + context * 153:BUILD_START + (context + 1) * 153] = self.values[:153]
        self.initialize_cover_skill()

    def initialize_cover_skill(self):
        # Existing navigation values are a prior, not a shelter route or a plan.
        self.values[COVER_START:] = self.values[:153]
        for context in range(1, 4):
            self.goal_values[context * 108:(context + 1) * 108, :4] = self.goal_values[:108, :4]

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
        self.risk_scale = 1.0
        self.cover_learner.current_features.fill(0)

    def encode_motor(self, senses, goal):
        if len(senses) >= 181 and goal == 2:
            threat = self.threat_senses(senses)
            brightness = float(threat.max())
            sector = int(threat.argmax()) if brightness > .001 else 16
            proximity = int(brightness > .4) + int(brightness > .65)
            walls = senses[162:178]
            # Front, left, right, back: sense a dead end before physical contact.
            mask = sum(int(float(walls[index].max()) > .4) << bit for bit, index in enumerate(
                ([6, 7, 8, 9], [2, 3, 4, 5], [10, 11, 12, 13], [14, 15, 0, 1])))
            return ESCAPE_START + mask * 51 + sector * 3 + proximity
        if len(senses) >= 181 and goal == 3:
            context = int(senses[178] >= .25) * 2 + int(senses[179] > 0)
            blocks = senses[146:162]
            first, second = np.argsort(blocks)[-2:][::-1]
            relative = (int(second) - int(first) + 8) % 16
            neighbor = relative // 4 if blocks[second] > .001 else 4
            context = context * 5 + neighbor
            return BUILD_START + context * 153 + self.encode(self.target_senses(senses, goal))
        if goal == 4:
            return COVER_START + self.encode(self.target_senses(senses, goal))
        return goal * 153 + self.encode(self.target_senses(senses, goal))

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
            potential -= 1.5 * float(self.threat_senses(senses).max())
        if len(senses) >= 181:
            threat = self.threat_senses(senses)
            danger = float(threat.max())
            bearings = (np.arange(16) + .5) * 2 * np.pi / 16 - np.pi
            away = -float(np.dot(threat, np.cos(bearings))) / max(float(threat.sum()), .001)
            front = float(senses[168:172].max())
            # Disclosed shaping prior: distance, open escape direction, and not
            # facing a nearby obstacle. It never chooses a motor command.
            potential += danger * (-1.5 + .8 * away * (1 - front) - .6 * front)
        if len(senses) >= SENSES:
            # A bounded observation-only safety potential. It cannot pay a
            # perpetual camping/peekaboo bonus: the same discounted difference
            # is applied at every tick, including terminal states.
            need = max(float(senses[32]), float(senses[48]) if senses[144] else 0)
            recent_danger = max(float(senses[96:112].max()), float(senses[247]))
            potential += self.config.cover_shaping * recent_danger * senses[178] * (1 - need)
        return float(self.config.shaping_scale * potential)

    def target_senses(self, senses, goal):
        result = senses[:80].copy()
        if len(senses) >= 146:
            if goal == 0:
                result[:16] = np.maximum(senses[:16], senses[112:128])
            elif goal == 1:
                result[:16] = np.maximum(senses[80:96], senses[128:144])
            elif goal == 2:
                result[:16] = self.threat_senses(senses)
            elif goal == 3 and len(senses) >= 181:
                result[:16] = senses[146:162]
            elif goal == 4 and len(senses) >= SENSES:
                # Re-bin a remembered 360° body-relative bearing into the
                # established 240° navigation interface, even if behind us.
                result[:16] = 0
                for i, strength in enumerate(senses[213:229]):
                    angle = (i + .5) * 2 * np.pi / 16 - np.pi
                    sector = int(np.clip((angle / self.config.vision_fov + .5) * 16, 0, 15))
                    result[sector] = max(result[sector], strength)
        return result

    @staticmethod
    def threat_senses(senses):
        visible = senses[96:112]
        return np.maximum(visible, senses[229:245]) if len(senses) >= SENSES else visible

    @staticmethod
    def available_goals(senses):
        available = [0]
        if len(senses) >= 146:
            if senses[144]:
                available.append(1)
            if MotorPolicy.threat_senses(senses).max() > 0.001:
                available.append(2)
        if len(senses) >= 181 and senses[180] and senses[146:162].max() > .001:
            available.append(3)
        if len(senses) >= SENSES and senses[245] > .02:
            available.append(4)
        return np.asarray(available)

    def encode_goal(self, senses):
        hunger = min(2, int(float(senses[32]) * 3))
        thirst = min(2, int(float(senses[48]) * 3))
        threat = float(self.threat_senses(senses).max()) if len(senses) >= 146 else 0
        danger = int(threat > 0.001) + int(threat > 0.65)
        food = int(self.target_senses(senses, 0)[:16].max() > 0.001)
        water = int(len(senses) >= 146 and self.target_senses(senses, 1)[:16].max() > 0.001)
        base = (((hunger * 3 + thirst) * 3 + danger) * 2 + food) * 2 + water
        context = (int(senses[245] > .02) + 2 * int(senses[178] >= .25)) if len(senses) >= SENSES else 0
        return context * 108 + base

    def action_values(self, state, goal, features):
        return self.values[state] + self.cover_learner.scores(goal, features)

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
            return self.config.exploration_floor * self.risk_scale
        return self._epsilon(self.skill_updates[self.goal], learning) * self.risk_scale

    def begin(self, senses: np.ndarray, learning: bool) -> np.ndarray:
        self.before_potential = self.potential(senses)
        self.risk_scale = .2 if len(senses) >= 181 and self.threat_senses(senses).max() > .65 else 1.0
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
                if (previous_goal in available and len(senses) >= 181
                        and self.goal_values[self.goal_state, previous_goal] >= values.max() - self.config.goal_switch_margin):
                    self.goal = previous_goal
                if self.rng.random() < self._epsilon(self.goal_updates, learning) * self.risk_scale:
                    self.goal = int(self.rng.choice(available))
                if self.goal not in best:
                    self.goal_eligibility.fill(0)
            if self.goal != previous_goal:
                self.eligibility.fill(0)
            self.state = self.encode_motor(senses, self.goal)
            self.cover_learner.current_features[:] = cover_features(senses, self.config.vision_fov)
            scores = self.action_values(self.state, self.goal, self.cover_learner.current_features)
            candidates = np.flatnonzero(scores >= scores.max() - 1e-10)
            greedy = int(self.rng.choice(candidates))
            self.action = greedy
            if self.rng.random() < self.exploration(learning):
                self.action = int(self.rng.integers(len(ACTION_NAMES)))
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
                food_gain: float | None = None, construction_gain: float = 0.0, pushed: float = 0.0) -> float | None:
        """Reward a transition; update only after the held action finishes."""
        c = self.config
        gamma_tick = c.policy_discount ** (c.world_dt / c.action_seconds)
        food_units = eaten
        if len(senses) >= 146 and senses[144] and food_gain is not None:
            food_units = food_gain / c.food_energy
        self.last_base_reward = (c.eating_reward * food_units + c.drinking_reward * water_gain / 24.0
                                 - damage * 0.5 - 0.2 * c.world_dt - 2.4 * c.world_dt * (touch if not pushed else 0))
        self.last_base_reward += c.construction_reward * construction_gain
        if len(senses) >= 181:
            danger = float(self.threat_senses(senses).max())
            self.last_base_reward -= c.world_dt * (3 * danger * danger + 4 * touch * danger)
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
        next_state = self.encode_motor(senses, self.goal)
        features = cover_features(senses, self.config.vision_fov)
        bootstrap = self.discount * float(self.action_values(next_state, self.goal, features).max()) if alive else 0.0
        td = self.return_sum + bootstrap - self.action_values(self.state, self.goal, self.cover_learner.current_features)[self.action]
        self.last_td_error = float(td)
        if learning:
            self.cover_learner.observe(self.goal, self.action, self.state, next_state,
                                       self.return_sum, self.discount, alive, features, self.values, c)
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
            "cover_learning": self.cover_learner.summary(),
        }
