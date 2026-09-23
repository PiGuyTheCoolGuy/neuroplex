"""Sparse recurrent LIF neurons with three-factor, reward-modulated STDP.

No autodiff, backpropagation, batches, replay buffer, or target action labels.
This is a deliberately small research sandbox, not a validated biological model.
"""

import numpy as np

from .config import Config, GROUPS
from .policy import MotorPolicy


class Brain:
    ARRAY_NAMES = (
        "pre", "post", "inhibitory", "plastic", "weights", "initial_weights",
        "v", "spikes", "refractory", "threshold", "pre_trace", "post_trace",
        "eligibility", "rates", "motor_noise", "motor_rates", "motor_current",
    )

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed + 1000)
        n = config.neurons
        self.inhibitory = np.zeros(n, dtype=bool)
        self.inhibitory[self.rng.choice(np.arange(80, 400), 64, replace=False)] = True
        self.pre = np.repeat(np.arange(n, dtype=np.int32), config.fan_out)
        targets = np.arange(80, n, dtype=np.int32)
        self.post = np.concatenate([
            self.rng.choice(targets[targets != i], config.fan_out, replace=False)
            for i in range(n)
        ])
        self.plastic = ~self.inhibitory[self.pre]
        self.weights = self.rng.uniform(0.015, 0.05, len(self.pre)).astype(np.float32)
        self.weights[~self.plastic] *= -4.0  # fixed inhibitory scaffold; preserve Dale's sign
        self.initial_weights = self.weights.copy()
        self.policy = MotorPolicy(config)
        self.reset_activity()
        self.learning = True
        self.last_reward = 0.0
        self.total_abs_change = 0.0
        self.steps = 0

    def reset_activity(self):
        """A new life resets transient activity, never learned weights or values."""
        n = self.config.neurons
        self.v = self.rng.uniform(0.0, 0.7, n).astype(np.float32)
        self.spikes = np.zeros(n, dtype=bool)
        self.refractory = np.zeros(n, dtype=np.float32)
        self.threshold = np.ones(n, dtype=np.float32)
        self.pre_trace = np.zeros(n, dtype=np.float32)
        self.post_trace = np.zeros(n, dtype=np.float32)
        self.eligibility = np.zeros(len(self.pre), dtype=np.float32)
        self.rates = np.zeros(n, dtype=np.float32)
        self.motor_noise = np.zeros(3, dtype=np.float32)
        self.motor_rates = np.zeros(4, dtype=np.float32)
        self.motor_current = np.zeros(4, dtype=np.float32)
        self.policy.reset_activity()

    def step(self, senses: np.ndarray):
        dt = self.config.brain_dt
        drive = np.full(500, 0.92, dtype=np.float32)
        drive[:80] = 0.05 + 2.6 * senses[:80]
        if len(senses) >= 146:
            # Additional sensory and persistent memory signals use association
            # cells; the original 500-neuron topology and memories are retained.
            drive[80:112] += 1.2 * senses[112:144]
            drive[112:144] += 1.2 * senses[80:112]
        if len(senses) >= 181:
            drive[144:176] += 1.2 * senses[146:178]
            drive[176:184] += float(senses[178])
        for i, (_, a, b) in enumerate(GROUPS[-4:]):
            drive[a:b] = self.motor_current[i]
        recurrent = np.bincount(
            self.post, weights=self.weights * self.spikes[self.pre], minlength=500
        ).astype(np.float32)
        # A stable actuator interface prevents changing recurrent activity from
        # undoing the action the policy is learning to associate with reward.
        recurrent[400:] *= 0.05
        self.refractory = np.maximum(0, self.refractory - dt)
        available = self.refractory <= 0
        self.v += (dt / 0.020) * (drive - self.v) + recurrent
        self.v += self.rng.normal(0.0, 0.035, 500).astype(np.float32)
        self.v[~available] = 0
        np.clip(self.v, -1.0, 3.0, out=self.v)
        fired = (self.v >= self.threshold) & available
        self.v[fired] = 0
        self.refractory[fired] = 0.005

        # Pair traces contain only *earlier* spikes; same-bin spikes do not
        # invent an ordering. Eligibility bridges neural and behavioral time.
        decay = np.exp(-dt / 0.020)
        self.pre_trace *= decay
        self.post_trace *= decay
        self.pre_trace[self.pre_trace < 1e-10] = 0
        self.post_trace[self.post_trace < 1e-10] = 0
        self.eligibility *= np.exp(-dt / self.config.eligibility_tau)
        self.eligibility += (
            fired[self.post] * self.pre_trace[self.pre]
            - 1.05 * fired[self.pre] * self.post_trace[self.post]
        ) * self.plastic
        np.clip(self.eligibility, -8, 8, out=self.eligibility)
        self.pre_trace += fired
        self.post_trace += fired
        self.spikes = fired
        alpha = np.full(500, 1 - np.exp(-dt / 0.25), dtype=np.float32)
        alpha[400:] = 1 - np.exp(-dt / 0.04)
        self.rates += alpha * (fired / dt - self.rates)
        self.rates[self.rates < 1e-8] = 0
        # Bounded intrinsic homeostasis stabilizes recurrent activity. This
        # changes excitability, not synaptic weights, even with learning frozen.
        self.threshold[80:400] += dt * 0.003 * (self.rates[80:400] - 12.0)
        np.clip(self.threshold, 0.7, 1.8, out=self.threshold)
        self.steps += 1

    def advance(self, senses: np.ndarray) -> tuple[float, float]:
        self.motor_current[:] = self.policy.begin(senses, self.learning)
        for _ in range(round(self.config.world_dt / self.config.brain_dt)):
            self.step(senses)
        self.motor_rates[:] = [self.rates[a:b].mean() for _, a, b in GROUPS[-4:]]
        forward, backward, left, right = self.motor_rates
        # Continuous population-rate decoding, never a direct food steering rule.
        speed = np.clip((forward - backward) / 50.0, -1, 1)
        turn = np.clip((right - left) / 50.0, -1, 1)
        return float(speed), float(turn)

    def observe(self, senses: np.ndarray, eaten: int, touch: float, alive: bool, **outcome):
        td = self.policy.observe(senses, eaten, touch, alive, self.learning, **outcome)
        if td is not None:
            self.reinforce(float(np.clip(td, -1, 1)))
        self.last_reward = self.policy.last_reward

    def reinforce(self, reward: float):
        self.last_reward = float(reward)
        if not self.learning or reward == 0:
            return
        updated = self.weights + self.config.learning_rate * reward * self.eligibility
        updated[self.plastic] = np.clip(updated[self.plastic], 0, self.config.weight_max)
        updated[~self.plastic] = self.weights[~self.plastic]
        self.total_abs_change += float(np.abs(updated - self.weights).sum())
        self.weights[:] = updated

    def summary(self):
        return {
            "neurons": 500,
            "synapses": len(self.weights),
            "plastic_synapses": int(self.plastic.sum()),
            "mean_rate": float(self.rates.mean()),
            "rates": np.round(self.rates, 1).tolist(),
            "spikes": np.flatnonzero(self.spikes).tolist(),
            "groups": [{"name": name, "start": a, "end": b,
                        "rate": float(self.rates[a:b].mean())} for name, a, b in GROUPS],
            "motor_rates": self.motor_rates.tolist(),
            "learning": self.learning,
            "reward": self.last_reward,
            "weight_change": float(np.abs(self.weights - self.initial_weights).mean()),
            "total_abs_change": self.total_abs_change,
            "eligibility_mean": float(np.abs(self.eligibility).mean()),
            "policy": self.policy.summary(self.learning),
        }
