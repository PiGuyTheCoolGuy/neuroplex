"""One learning creature, resources, and explicitly scripted predator NPCs."""

import math

import numpy as np

from .config import Config
from .curriculum import STAGES


class World:
    ARRAY_NAMES = ("food", "regrow_at", "retina", "water", "water_retina", "predators",
                   "predator_headings", "predator_ready_at", "danger_retina")

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.ecology_rng = np.random.default_rng(config.seed + 3000)
        self.stage = config.habitat_stage
        self.time = 0.0
        self.x = config.world_width / 2
        self.y = config.world_height / 2
        self.heading = float(self.rng.uniform(-math.pi, math.pi))
        self.energy = config.max_energy * 0.8
        self.hydration = config.max_hydration * 0.8
        self.health = config.max_health
        self.death_code = 0
        self.drinks = 0
        self.drinking = False
        self.attacks = 0
        self.damage_taken = 0.0
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
        self.water = self.ecology_rng.uniform(
            [3, 3], [config.world_width - 3, config.world_height - 3], (config.water_count, 2))
        self.water_retina = np.zeros(16, dtype=np.float32)
        self.danger_retina = np.zeros(16, dtype=np.float32)
        self.predators = self.ecology_rng.uniform(
            [3, 3], [config.world_width - 3, config.world_height - 3], (config.predator_count, 2))
        # Start outside contact distance even in a small configured habitat.
        for predator in self.predators:
            if np.linalg.norm(predator - [self.x, self.y]) < 12:
                predator[:] = [3, 3]
        self.predator_headings = self.ecology_rng.uniform(-math.pi, math.pi, config.predator_count)
        self.predator_ready_at = np.full(config.predator_count, 15.0)
        self.sense()

    @property
    def water_active(self):
        return self.config.water_enabled and STAGES[self.stage]["water"]

    @property
    def predator_limit(self):
        return min(self.config.predator_count, STAGES[self.stage]["predators"])

    @property
    def food_limit(self):
        return int(math.ceil(self.config.food_count * STAGES[self.stage]["food_fraction"]))

    @property
    def vision_range(self):
        return self.config.vision_range * STAGES[self.stage]["vision_fraction"]

    @property
    def thirst(self):
        return 1.0 - self.hydration / self.config.max_hydration if self.water_active else 0.0

    @property
    def death_reason(self):
        return ("alive", "starvation", "dehydration", "predator injuries")[self.death_code]

    def active_food(self):
        return (self.regrow_at <= self.time) & (np.arange(len(self.food)) < self.food_limit)

    def set_stage(self, stage):
        previous = self.predator_limit
        self.stage = stage
        # Newly introduced predators have a 15-second non-attacking grace period.
        self.predator_ready_at[previous:self.predator_limit] = self.time + 15
        self.sense()

    def _project(self, points, retina, distance_limit, fov):
        retina.fill(0)
        relative = points - [self.x, self.y]
        if not len(relative):
            return
        distances = np.linalg.norm(relative, axis=1)
        angles = (np.arctan2(relative[:, 1], relative[:, 0]) - self.heading + math.pi) % (2 * math.pi) - math.pi
        in_field = np.ones(len(angles), dtype=bool) if fov >= 2 * math.pi else np.abs(angles) < fov / 2
        visible = in_field & (distances < distance_limit)
        bins = np.clip(((angles[visible] / fov + 0.5) * 16).astype(int), 0, 15)
        np.maximum.at(retina, bins, 1.0 - distances[visible] / distance_limit)

    @property
    def hunger(self):
        return 1.0 - self.energy / self.config.max_energy

    def sense(self) -> np.ndarray:
        c = self.config
        self.retina.fill(0)
        self._project(self.food[self.active_food()], self.retina[:16], self.vision_range, c.vision_fov)
        self._project(self.water if self.water_active else self.water[:0], self.water_retina,
                      self.vision_range, c.vision_fov)
        # Omnidirectional threat sensing prevents looking away from a predator
        # from masquerading as having increased physical safety.
        self._project(self.predators[:self.predator_limit], self.danger_retina,
                      c.predator_detection, 2 * math.pi)
        # Second retina channel measures walls along the same 16 angular rays.
        angles = self.heading + (np.arange(16) + 0.5) / 16 * c.vision_fov - c.vision_fov / 2
        dx, dy = np.cos(angles), np.sin(angles)
        tx = np.where(dx >= 0, c.world_width - self.x, -self.x) / np.where(np.abs(dx) > 1e-9, dx, 1e-9)
        ty = np.where(dy >= 0, c.world_height - self.y, -self.y) / np.where(np.abs(dy) > 1e-9, dy, 1e-9)
        self.retina[16:] = np.clip(1 - np.minimum(tx, ty) / self.vision_range, 0, 1)
        senses = np.zeros(146, dtype=np.float32)
        senses[:32] = self.retina
        senses[32:48] = self.hunger
        senses[48:64] = self.thirst
        senses[64:80] = self.touch
        senses[80:96] = self.water_retina
        senses[96:112] = self.danger_retina
        # 112:144 are supplied by the observation-only working-memory module.
        senses[144] = self.water_active
        senses[145] = self.predator_limit > 0
        return senses

    def _move_predators(self):
        c = self.config
        damage = 0.0
        for i in range(self.predator_limit):
            position = self.predators[i]
            delta = np.array([self.x, self.y]) - position
            distance = float(np.linalg.norm(delta))
            hunting = distance < c.predator_detection and self.time >= self.predator_ready_at[i]
            if hunting:
                self.predator_headings[i] = math.atan2(delta[1], delta[0])
            else:
                self.predator_headings[i] += float(self.ecology_rng.normal(0, 0.7 * math.sqrt(c.world_dt)))
            heading = self.predator_headings[i]
            velocity = c.predator_speed * (1 if hunting else 0.4)
            desired = position + velocity * c.world_dt * np.array([math.cos(heading), math.sin(heading)])
            position[:] = np.clip(desired, c.predator_radius,
                                   [c.world_width - c.predator_radius, c.world_height - c.predator_radius])
            if not np.allclose(desired, position):
                self.predator_headings[i] += math.pi / 2
            if hunting and np.linalg.norm(position - [self.x, self.y]) <= c.predator_radius + c.creature_radius:
                damage += c.predator_damage
                self.attacks += 1
                self.predator_ready_at[i] = self.time + c.predator_cooldown
                self.predator_headings[i] += math.pi
        self.damage_taken += damage
        self.health = max(0.0, self.health - damage)
        if not damage and self.energy > 40 and (not self.water_active or self.hydration > 40):
            if not self.predator_limit or np.linalg.norm(self.predators[:self.predator_limit] - [self.x, self.y], axis=1).min() > 10:
                self.health = min(c.max_health, self.health + c.healing_rate * c.world_dt)
        return damage

    def step(self, speed: float, turn: float) -> dict:
        if not self.alive:
            return {"reward": 0.0, "eaten": 0, "died": False}
        c = self.config
        old_hunger = self.hunger
        old_thirst = self.thirst
        self.time += c.world_dt
        self.speed = float(np.clip(speed, -1, 1)) * c.max_speed
        self.turn = float(np.clip(turn, -1, 1)) * c.max_turn
        self.heading = (self.heading + self.turn * c.world_dt + math.pi) % (2 * math.pi) - math.pi
        dx = self.speed * math.cos(self.heading) * c.world_dt
        dy = self.speed * math.sin(self.heading) * c.world_dt
        nx = float(np.clip(self.x + dx, c.creature_radius, c.world_width - c.creature_radius))
        ny = float(np.clip(self.y + dy, c.creature_radius, c.world_height - c.creature_radius))
        self.touch = float(abs(nx - self.x - dx) + abs(ny - self.y - dy) > 1e-8)
        actual_dx, actual_dy = nx - self.x, ny - self.y
        self.distance += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        self.energy -= (c.basal_cost + c.movement_cost * abs(self.speed)) * c.world_dt
        distances = np.linalg.norm(self.food - [self.x, self.y], axis=1)
        mouth = ((distances <= c.creature_radius + c.food_radius)
                 & self.active_food())
        count = int(mouth.sum())
        # Ingestion is a body reflex. Choosing where to travel is the brain's job.
        food_gain = min(count * c.food_energy, c.max_energy - self.energy)
        if count:
            self.energy += count * c.food_energy
            self.regrow_at[mouth] = self.time + c.food_regrow_seconds
            self.eaten += count
        self.energy = float(np.clip(self.energy, 0, c.max_energy))
        water_gain = 0.0
        drank = 0
        if self.water_active:
            self.hydration -= (c.water_cost + c.movement_water_cost * abs(self.speed)) * c.world_dt
            contact = bool(np.any(np.linalg.norm(self.water - [self.x, self.y], axis=1)
                                  <= c.creature_radius + c.water_radius))
            if contact:
                water_gain = min(c.drinking_rate * c.world_dt, c.max_hydration - self.hydration)
                self.hydration += water_gain
                drank = int(not self.drinking)
                self.drinks += drank
            self.drinking = contact
            self.hydration = float(np.clip(self.hydration, 0, c.max_hydration))
        else:
            self.drinking = False
        damage = self._move_predators()
        self.death_code = 3 if self.health <= 0 else 2 if self.water_active and self.hydration <= 0 else 1 if self.energy <= 0 else 0
        self.alive = self.death_code == 0
        return {"reward": old_hunger - self.hunger + old_thirst - self.thirst,
                "eaten": count, "drank": drank, "water_gain": water_gain, "food_gain": food_gain,
                "damage": damage, "died": not self.alive,
                "forward": actual_dx * math.cos(self.heading) + actual_dy * math.sin(self.heading),
                "lateral": -actual_dx * math.sin(self.heading) + actual_dy * math.cos(self.heading),
                "rotation": self.turn * c.world_dt}

    def render_state(self):
        """Geometry only: cheap enough to stream at the world's 20 Hz tick rate."""
        c = self.config
        return {
            "width": c.world_width, "height": c.world_height,
            "time": self.time, "alive": self.alive, "x": self.x, "y": self.y,
            "heading": self.heading, "radius": c.creature_radius,
            "stage": self.stage, "death_reason": self.death_reason,
            "water": self.water.tolist() if self.water_active else [], "water_radius": c.water_radius,
            "predators": [{"x": float(p[0]), "y": float(p[1]), "heading": float(self.predator_headings[i]),
                           "hunting": bool(self.time >= self.predator_ready_at[i])} for i, p in enumerate(self.predators[:self.predator_limit])],
            "predator_radius": c.predator_radius,
            "food": self.food[self.active_food()].tolist(), "food_radius": c.food_radius,
            "vision_range": self.vision_range, "vision_fov": c.vision_fov,
        }

    def summary(self):
        c = self.config
        return {
            **self.render_state(),
            "energy": self.energy, "max_energy": c.max_energy, "hunger": self.hunger,
            "eaten": self.eaten, "distance": self.distance, "speed": self.speed,
            "hydration": self.hydration, "max_hydration": c.max_hydration, "thirst": self.thirst,
            "water_active": self.water_active, "drinks": self.drinks, "drinking": self.drinking,
            "health": self.health, "max_health": c.max_health, "attacks": self.attacks,
            "damage_taken": self.damage_taken, "stage_name": STAGES[self.stage]["name"],
            "water_retina": self.water_retina.tolist(),
            "danger_retina": self.danger_retina.tolist(),
            "turn": self.turn, "touch": self.touch,
            "retina": self.retina.tolist(),
        }
