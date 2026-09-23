"""Small, shared action-value features and replay of the creature's own experience.

No map, hidden object coordinates, expert actions, or imagined successful pushes.
The residual augments retained motor tables for escape, build, and use-cover.
Food/water tables are not updated by replay. Storage is bounded on the OptiPlex.
"""

import numpy as np


SENSES = 248
FEATURES = 102
CAPACITY = 512


def cover_features(senses, fov=4 * np.pi / 3):
    result = np.zeros(FEATURES, dtype=np.float64)
    if len(senses) < SENSES:
        return result
    result[:10] = [1, senses[32], senses[48], senses[178], senses[179],
                   senses[144], senses[145], senses[245], senses[246], senses[247]]
    result[10:90] = np.concatenate((senses[146:162], senses[181:213], senses[213:245]))
    blocks = senses[146:162]
    first, second = np.argsort(blocks)[-2:][::-1]
    # Relative geometry from two visible bearings; no global block identities.
    angles = ((np.array([first, second]) + .5) / 16 - .5) * fov
    brightness = blocks[[first, second]]
    if brightness[0] > .001:
        result[90:93] = [brightness[0], np.sin(angles[0]), np.cos(angles[0])]
    if brightness[1] > .001:
        result[93:98] = [brightness[1], np.sin(angles[1]), np.cos(angles[1]),
                         np.sin(angles[1] - angles[0]), brightness[1] - brightness[0]]
    danger = max(float(senses[96:112].max()), float(senses[247]))
    result[98:] = [danger * senses[178], danger * senses[245],
                   senses[32] * senses[245], senses[48] * senses[245]]
    # Bounded step sizes despite differently populated visual fields.
    return result / max(1.0, float(np.linalg.norm(result)))


class CoverLearner:
    ARRAY_NAMES = ("weights", "updates", "replay_features", "replay_info", "current_features")
    LEARNED = ("weights", "updates")

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed + 7000)
        self.weights = np.zeros((3, 7, FEATURES), dtype=np.float64)
        self.updates = np.zeros(3, dtype=np.int64)
        self.replay_features = np.zeros((CAPACITY, 2, FEATURES), dtype=np.float64)
        # goal (2..4), action, current/next table row, return, discount, terminal
        self.replay_info = np.zeros((CAPACITY, 7), dtype=np.float64)
        self.current_features = np.zeros(FEATURES, dtype=np.float64)
        self.cursor = self.count = 0

    def scores(self, goal, features):
        return self.weights[goal - 2] @ features if goal >= 2 else np.zeros(7)

    def clear_replay(self):
        self.replay_features.fill(0)
        self.replay_info.fill(0)
        self.cursor = self.count = 0

    def _learn(self, features, info, values, rate):
        goal, action, state, next_state = map(int, info[:4])
        reward, discount, terminal = info[4:]
        before = values[state, action] + self.scores(goal, features[0])[action]
        after = 0 if terminal else np.max(values[next_state] + self.scores(goal, features[1]))
        error = float(np.clip(reward + discount * after - before, -10, 10))
        self.weights[goal - 2, action] += rate * error * features[0]
        np.clip(self.weights[goal - 2], -25, 25, out=self.weights[goal - 2])
        self.updates[goal - 2] += 1

    def observe(self, goal, action, state, next_state, reward, discount, alive,
                next_features, values, config):
        if goal < 2 or not self.current_features.any():
            return
        index = self.cursor
        self.replay_features[index] = [self.current_features, next_features]
        self.replay_info[index] = [goal, action, state, next_state, reward, discount, not alive]
        self.cursor = (index + 1) % CAPACITY
        self.count = min(CAPACITY, self.count + 1)
        self._learn(self.replay_features[index], self.replay_info[index], values, config.cover_learning_rate)
        # Same skill only: an escape drill cannot silently train building values.
        eligible = np.flatnonzero(self.replay_info[:self.count, 0] == goal)
        for _ in range(config.cover_replay_steps):
            replay = int(self.rng.choice(eligible))
            self._learn(self.replay_features[replay], self.replay_info[replay], values,
                        config.cover_learning_rate * .5)

    def summary(self):
        return {"experiences": self.count, "capacity": CAPACITY,
                "updates": self.updates.tolist(), "parameters": int(self.weights.size)}
