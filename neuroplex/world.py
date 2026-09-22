"""A single body in a bounded 2D food world. No hidden path finder."""

import math

import numpy as np

from .config import Config


class World:
    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.time = 0.0
        self.x = config.world_width / 2
        self.y = config.world_height / 2
        self.heading = float(self.rng.uniform(-math.pi, math.pi))
        self.energy = config.max_energy * 0.8
        self.alive = True
        self.eaten = 0
        self.distance = 0.0
        self.touch = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.food = self.rng.uniform(
            [2, 2], [config.world_width - 2, config.world_height - 2], (config.food_count, 2)
        )
        self.regrow_at = np.zeros(config.food_count)
        self.retina = np.zeros(32, dtype=np.float32)
        self.sense()

    @property
    def hunger(self):
        return 1.0 - self.energy / self.config.max_energy

    def sense(self) -> np.ndarray:
        c = self.config
        self.retina.fill(0)
        active = self.regrow_at <= self.time
        relative = self.food[active] - [self.x, self.y]
        if len(relative):
            distances = np.linalg.norm(relative, axis=1)
            angles = (np.arctan2(relative[:, 1], relative[:, 0]) - self.heading + math.pi) % (2 * math.pi) - math.pi
            visible = (np.abs(angles) < c.vision_fov / 2) & (distances < c.vision_range)
            bins = ((angles[visible] / c.vision_fov + 0.5) * 16).astype(int)
            np.maximum.at(self.retina[:16], bins, 1.0 - distances[visible] / c.vision_range)
        # Second retina channel measures walls along the same 16 angular rays.
        angles = self.heading + (np.arange(16) + 0.5) / 16 * c.vision_fov - c.vision_fov / 2
        dx, dy = np.cos(angles), np.sin(angles)
        tx = np.where(dx >= 0, c.world_width - self.x, -self.x) / np.where(np.abs(dx) > 1e-9, dx, 1e-9)
        ty = np.where(dy >= 0, c.world_height - self.y, -self.y) / np.where(np.abs(dy) > 1e-9, dy, 1e-9)
        self.retina[16:] = np.clip(1 - np.minimum(tx, ty) / c.vision_range, 0, 1)
        senses = np.zeros(80, dtype=np.float32)
        senses[:32] = self.retina
        senses[32:48] = self.hunger
        # 48:64 is deliberately unused: no thirst in this first experiment.
        senses[64:80] = self.touch
        return senses

    def step(self, speed: float, turn: float) -> dict:
        if not self.alive:
            return {"reward": 0.0, "eaten": 0, "died": False}
        c = self.config
        old_hunger = self.hunger
        self.time += c.world_dt
        self.speed = float(np.clip(speed, -1, 1)) * c.max_speed
        self.turn = float(np.clip(turn, -1, 1)) * c.max_turn
        self.heading = (self.heading + self.turn * c.world_dt + math.pi) % (2 * math.pi) - math.pi
        dx = self.speed * math.cos(self.heading) * c.world_dt
        dy = self.speed * math.sin(self.heading) * c.world_dt
        nx = float(np.clip(self.x + dx, c.creature_radius, c.world_width - c.creature_radius))
        ny = float(np.clip(self.y + dy, c.creature_radius, c.world_height - c.creature_radius))
        self.touch = float(abs(nx - self.x - dx) + abs(ny - self.y - dy) > 1e-8)
        self.distance += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        self.energy -= (c.basal_cost + c.movement_cost * abs(self.speed)) * c.world_dt
        distances = np.linalg.norm(self.food - [self.x, self.y], axis=1)
        mouth = ((distances <= c.creature_radius + c.food_radius)
                 & (self.regrow_at <= self.time))
        count = int(mouth.sum())
        # Ingestion is a body reflex. Choosing where to travel is the brain's job.
        if count:
            self.energy += count * c.food_energy
            self.regrow_at[mouth] = self.time + c.food_regrow_seconds
            self.eaten += count
        self.energy = float(np.clip(self.energy, 0, c.max_energy))
        self.alive = self.energy > 0
        return {"reward": old_hunger - self.hunger, "eaten": count, "died": not self.alive}

    def summary(self):
        c = self.config
        return {
            "width": c.world_width, "height": c.world_height,
            "time": self.time, "alive": self.alive, "x": self.x, "y": self.y,
            "heading": self.heading, "radius": c.creature_radius,
            "energy": self.energy, "max_energy": c.max_energy, "hunger": self.hunger,
            "eaten": self.eaten, "distance": self.distance, "speed": self.speed,
            "turn": self.turn, "touch": self.touch,
            "food": self.food[self.regrow_at <= self.time].tolist(),
            "food_radius": c.food_radius,
            "retina": self.retina.tolist(), "vision_range": c.vision_range,
            "vision_fov": c.vision_fov,
        }
