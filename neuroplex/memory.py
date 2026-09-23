"""Finite sensory memory from observations and body motion, never hidden targets.

This is an engineered working-memory system, not learned recurrent recall.
Its persistent signals also drive association neurons in the spiking network.
"""

import math
import numpy as np
from .cover import SENSES


class SensoryMemory:
    def __init__(self, config):
        self.config = config
        # Food and water: estimated egocentric x, y, confidence, age.
        self.traces = np.zeros((2, 4), dtype=np.float64)
        # Cover actually occupied, and the last observed threat. Body-relative
        # x/y, confidence, age, observed quality/strength. Never world coordinates.
        self.sites = np.zeros((2, 5), dtype=np.float64)

    def clear(self):
        self.traces.fill(0)
        self.sites.fill(0)

    def advance(self, forward, lateral, rotation, dt):
        cosine, sine = math.cos(rotation), math.sin(rotation)
        x, y = self.traces[:, 0].copy(), self.traces[:, 1].copy()
        self.traces[:, 0] = cosine * x + sine * y - forward
        self.traces[:, 1] = -sine * x + cosine * y - lateral
        self.traces[:, 2] *= math.exp(-dt / self.config.memory_seconds)
        self.traces[:, 3] += dt
        expired = self.traces[:, 3] >= self.config.memory_seconds
        self.traces[expired] = 0
        x, y = self.sites[:, 0].copy(), self.sites[:, 1].copy()
        self.sites[:, 0] = cosine * x + sine * y - forward
        self.sites[:, 1] = -sine * x + cosine * y - lateral
        durations = np.array([self.config.cover_memory_seconds, self.config.threat_memory_seconds])
        self.sites[:, 2] *= np.exp(-dt / durations)
        self.sites[:, 3] += dt
        self.sites[self.sites[:, 3] >= durations] = 0

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
        if len(senses) >= SENSES:
            self.observe_cover(senses, result, vision_range)
        return result

    def observe_cover(self, senses, result, vision_range):
        cover = self.sites[0]
        distance = float(np.linalg.norm(cover[:2]))
        # Re-inspect a remembered site on arrival; moved blocks can invalidate it.
        if distance < self.config.creature_radius * 2 and senses[178] < .125:
            cover[:] = 0
        if senses[178] >= .25 and (cover[2] <= 0 or senses[178] >= cover[4] or distance < 2):
            cover[:] = [0, 0, 1, 0, senses[178]]
        threat = senses[96:112]
        strength = float(threat.max())
        if strength > .001:
            sector = int(threat.argmax())
            angle = (sector + .5) * 2 * math.pi / 16 - math.pi
            distance = (1 - strength) * self.config.predator_detection
            self.sites[1] = [distance * math.cos(angle), distance * math.sin(angle), 1, 0, strength]
        for index, start in ((0, 213), (1, 229)):
            x, y, confidence, age, quality = self.sites[index]
            if confidence <= 0:
                continue
            distance = math.hypot(x, y)
            angle = math.atan2(y, x)
            sector = int(np.clip((angle / (2 * math.pi) + .5) * 16, 0, 15))
            limit = vision_range if index == 0 else self.config.predator_detection
            result[start + sector] = confidence * max(.05, 1 - distance / limit)
        result[245] = self.sites[0, 2] * self.sites[0, 4]
        result[246] = float(self.sites[0, 2] > 0 and np.linalg.norm(self.sites[0, :2]) < 2)
        result[247] = self.sites[1, 2] * self.sites[1, 4]

    def summary(self):
        return {"enabled": self.config.memory_enabled, "seconds": self.config.memory_seconds,
                "food_confidence": float(self.traces[0, 2]), "water_confidence": float(self.traces[1, 2]),
                "food_age": float(self.traces[0, 3]), "water_age": float(self.traces[1, 3]),
                "cover_confidence": float(self.sites[0, 2]), "cover_age": float(self.sites[0, 3]),
                "cover_quality": float(self.sites[0, 4]), "cover_seconds": self.config.cover_memory_seconds,
                "threat_confidence": float(self.sites[1, 2])}
