"""Finite sensory memory from observations and body motion, never hidden targets.

This is an engineered working-memory system, not learned recurrent recall.
Its persistent signals also drive association neurons in the spiking network.
"""

import math
import numpy as np


class SensoryMemory:
    def __init__(self, config):
        self.config = config
        # Food and water: estimated egocentric x, y, confidence, age.
        self.traces = np.zeros((2, 4), dtype=np.float64)

    def advance(self, forward, lateral, rotation, dt):
        cosine, sine = math.cos(rotation), math.sin(rotation)
        x, y = self.traces[:, 0].copy(), self.traces[:, 1].copy()
        self.traces[:, 0] = cosine * x + sine * y - forward
        self.traces[:, 1] = -sine * x + cosine * y - lateral
        self.traces[:, 2] *= math.exp(-dt / self.config.memory_seconds)
        self.traces[:, 3] += dt
        expired = self.traces[:, 3] >= self.config.memory_seconds
        self.traces[expired] = 0

    def observe(self, senses, vision_range):
        result = senses.copy()
        if not self.config.memory_enabled:
            return result
        for i, (start, destination) in enumerate(((0, 112), (80, 128))):
            retina = senses[start:start + 16]
            brightness = float(retina.max())
            if brightness > 0.001:
                sector = int(retina.argmax())
                angle = ((sector + 0.5) / 16 - 0.5) * self.config.vision_fov
                distance = (1 - brightness) * vision_range
                self.traces[i] = [distance * math.cos(angle), distance * math.sin(angle), 1, 0]
            else:
                x, y, confidence, age = self.traces[i]
                if confidence <= 0 or age >= self.config.memory_seconds:
                    continue
                angle = math.atan2(y, x)
                sector = int(np.clip((angle / self.config.vision_fov + 0.5) * 16, 0, 15))
                strength = max(0.05, 1 - math.hypot(x, y) / vision_range) * confidence
                result[destination + sector] = strength
        return result

    def summary(self):
        return {"enabled": self.config.memory_enabled, "seconds": self.config.memory_seconds,
                "food_confidence": float(self.traces[0, 2]), "water_confidence": float(self.traces[1, 2]),
                "food_age": float(self.traces[0, 3]), "water_age": float(self.traces[1, 3])}
