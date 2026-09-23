"""One learning creature, resources, and explicitly scripted predator NPCs."""

import math

import numpy as np

from .config import Config
from .curriculum import STAGES
from .geometry import circle_overlaps, cover_at, occluded, ray_hits
from .cover import SENSES


class World:
    ARRAY_NAMES = ("food", "regrow_at", "retina", "water", "water_retina", "predators",
                   "predator_headings", "predator_ready_at", "danger_retina", "blocks", "block_retina", "obstacle_retina", "block_cover_best")

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
        self.blocks = np.empty((0, 2), dtype=np.float64)
        self.block_cover_best = np.zeros(config.block_count, dtype=np.float64)
        self.block_retina = np.zeros(16, dtype=np.float32)
        self.obstacle_retina = np.zeros(16, dtype=np.float32)
        self.push_distance = 0.0
        self.pushed = False
        self.cover = 0.0
        self.best_shelter = 0.0
        self.construction_reward_total = 0.0
        self.shelter_x = self.x
        self.shelter_y = self.y
        self.protected_seconds = 0.0
        self.next_cover_check = 0.0
        self.construction_dirty = False
        self.place_blocks()
        self.sense()

    @property
    def active_blocks(self):
        return self.blocks if self.stage >= 1 else self.blocks[:0]

    def place_blocks(self):
        # Independent seed: adding materials does not consume a saved ecology RNG.
        rng = np.random.default_rng(self.config.seed + 4000)
        half = self.config.block_size / 2
        placed = []
        for _ in range(self.config.block_count):
            for attempt in range(2000):
                point = rng.uniform([half + 3, half + 3],
                                    [self.config.world_width - half - 3, self.config.world_height - half - 3])
                if np.linalg.norm(point - [self.x, self.y]) < 8:
                    continue
                if placed and np.any(np.all(np.abs(np.asarray(placed) - point) < self.config.block_size + .4, axis=1)):
                    continue
                if circle_overlaps(point, half * 1.42 + self.config.food_radius, self.food, 0).any():
                    continue
                if circle_overlaps(point, half * 1.42 + self.config.water_radius, self.water, 0).any():
                    continue
                if circle_overlaps(point, half * 1.42 + self.config.predator_radius, self.predators, 0).any():
                    continue
                placed.append(point)
                break
            else:
                raise ValueError("Not enough free space for blocks; reduce block_count or enlarge the world")
        self.blocks = np.asarray(placed, dtype=np.float64).reshape(-1, 2)
        self.block_cover_best = self.construction_scores()
        self.best_shelter = float(self.block_cover_best.max(initial=0))

    def construction_scores(self):
        blocks = self.active_blocks
        if len(blocks) < 2:
            return np.zeros(len(self.blocks))
        c = self.config
        angles = np.arange(8) * np.pi / 4
        offsets = np.column_stack((np.cos(angles), np.sin(angles))) * (c.block_size / 2 + c.creature_radius + .4)
        candidates = (blocks[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        scores = cover_at(candidates, blocks, c.block_size / 2, c.creature_radius, c.world_width, c.world_height)
        index = int(scores.argmax())
        self.shelter_x, self.shelter_y = map(float, candidates[index])
        return scores.reshape(len(blocks), 8).max(axis=1)

    def construction_score(self):
        return float(self.construction_scores().max(initial=0))

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
        self.block_cover_best = np.maximum(self.block_cover_best, self.construction_scores())
        self.best_shelter = max(self.best_shelter, float(self.block_cover_best.max(initial=0)))
        self.sense()

    def _project(self, points, retina, distance_limit, fov, materials=False):
        retina.fill(0)
        relative = points - [self.x, self.y]
        if not len(relative):
            return
        distances = np.linalg.norm(relative, axis=1)
        angles = (np.arctan2(relative[:, 1], relative[:, 0]) - self.heading + math.pi) % (2 * math.pi) - math.pi
        in_field = np.ones(len(angles), dtype=bool) if fov >= 2 * math.pi else np.abs(angles) < fov / 2
        visible = in_field & (distances < distance_limit)
        if len(self.active_blocks):
            if materials:
                hits = ray_hits(np.repeat([[self.x, self.y]], len(points), axis=0), relative,
                                self.active_blocks, self.config.block_size / 2)
                np.fill_diagonal(hits, np.inf)  # the target block's own face is visible
                visible &= ~np.any(hits < 1, axis=1)
            else:
                visible &= ~occluded(np.array([self.x, self.y]), points, self.active_blocks, self.config.block_size / 2)
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
        tx = np.where(dx >= 0, c.world_width - self.x, -self.x) / np.where(np.abs(dx) > 1e-9, dx, np.where(dx >= 0, 1e-9, -1e-9))
        ty = np.where(dy >= 0, c.world_height - self.y, -self.y) / np.where(np.abs(dy) > 1e-9, dy, np.where(dy >= 0, 1e-9, -1e-9))
        self.retina[16:] = np.clip(1 - np.minimum(tx, ty) / self.vision_range, 0, 1)
        if len(self.active_blocks):
            rays = np.column_stack((dx, dy))
            hits = ray_hits(np.repeat([[self.x, self.y]], 16, axis=0), rays, self.active_blocks, c.block_size / 2).min(axis=1)
            self.retina[16:] = np.maximum(self.retina[16:], np.clip(1 - hits / self.vision_range, 0, 1))
        angles = self.heading + (np.arange(16) + .5) * 2 * math.pi / 16 - math.pi
        rays = np.column_stack((np.cos(angles), np.sin(angles)))
        safe = np.where(np.abs(rays) > 1e-9, rays, np.where(rays >= 0, 1e-9, -1e-9))
        boundary = np.where(rays >= 0, [c.world_width - self.x, c.world_height - self.y], [-self.x, -self.y]) / safe
        ranges = boundary.min(axis=1) - c.creature_radius
        if len(self.active_blocks):
            ranges = np.minimum(ranges, ray_hits(np.repeat([[self.x, self.y]], 16, axis=0), rays,
                                                self.active_blocks, c.block_size / 2 + c.creature_radius).min(axis=1))
        self.obstacle_retina[:] = np.clip(1 - ranges / c.obstacle_range, 0, 1)
        self._project(self.active_blocks, self.block_retina, self.vision_range, c.vision_fov, materials=True)
        self.cover = float(cover_at([[self.x, self.y]], self.active_blocks, c.block_size / 2,
                                    c.creature_radius, c.world_width, c.world_height)[0])
        senses = np.zeros(SENSES, dtype=np.float32)
        senses[:32] = self.retina
        senses[32:48] = self.hunger
        senses[48:64] = self.thirst
        senses[64:80] = self.touch
        senses[80:96] = self.water_retina
        senses[96:112] = self.danger_retina
        # 112:144 are supplied by the observation-only working-memory module.
        senses[144] = self.water_active
        senses[145] = self.predator_limit > 0
        senses[146:162] = self.block_retina
        senses[162:178] = self.obstacle_retina
        senses[178] = self.cover
        senses[179] = self.pushed
        senses[180] = bool(len(self.active_blocks))
        # Typed short-range rays: report the first visible surface, not walls
        # behind a block or a block behind the nearer world boundary.
        fixed = boundary.min(axis=1) - c.creature_radius
        material = (ray_hits(np.repeat([[self.x, self.y]], 16, axis=0), rays,
                            self.active_blocks, c.block_size / 2 + c.creature_radius).min(axis=1)
                    if len(self.active_blocks) else np.full(16, np.inf))
        senses[181:197] = np.where(fixed <= material, np.clip(1 - fixed / c.obstacle_range, 0, 1), 0)
        senses[197:213] = np.where(material < fixed, np.clip(1 - material / c.obstacle_range, 0, 1), 0)
        return senses

    def _body_move(self, position, delta, radius):
        result = np.array(position, dtype=float)
        c = self.config
        for axis in range(2):
            candidate = result.copy()
            candidate[axis] = np.clip(candidate[axis] + delta[axis], radius,
                                      (c.world_width if axis == 0 else c.world_height) - radius)
            if not circle_overlaps(candidate, radius, self.active_blocks, c.block_size / 2).any():
                result = candidate
        return result

    def _push_move(self, delta):
        c = self.config
        position = np.array([self.x, self.y])
        hits = np.flatnonzero(circle_overlaps(position + delta, c.creature_radius, self.active_blocks, c.block_size / 2))
        pushed = 0.0
        if len(hits):
            delta = delta * c.push_speed_fraction
            if len(hits) == 1:
                index = hits[0]
                candidate = self.blocks[index] + delta
                others = np.delete(self.active_blocks, index, axis=0)
                within = np.all(candidate >= c.block_size / 2) and np.all(candidate <= np.array([c.world_width, c.world_height]) - c.block_size / 2)
                blocked = np.any(np.all(np.abs(others - candidate) < c.block_size - 1e-8, axis=1))
                blocked |= circle_overlaps(candidate, c.food_radius, self.food, c.block_size / 2).any()
                blocked |= circle_overlaps(candidate, c.water_radius, self.water, c.block_size / 2).any()
                blocked |= circle_overlaps(candidate, c.predator_radius, self.predators[:self.predator_limit], c.block_size / 2).any()
                if within and not blocked:
                    self.blocks[index] = candidate
                    pushed = float(np.linalg.norm(delta))
        moved = self._body_move(position, delta, c.creature_radius)
        self.push_distance += pushed
        self.pushed = pushed > 0
        self.construction_dirty |= self.pushed
        return moved, pushed

    def _move_predators(self):
        c = self.config
        damage = 0.0
        for i in range(self.predator_limit):
            position = self.predators[i]
            delta = np.array([self.x, self.y]) - position
            distance = float(np.linalg.norm(delta))
            hidden = bool(occluded(position, np.array([[self.x, self.y]]), self.active_blocks, c.block_size / 2)[0])
            hunting = distance < c.predator_detection and self.time >= self.predator_ready_at[i] and not hidden
            if hidden and distance < c.predator_detection:
                self.protected_seconds += c.world_dt / max(1, self.predator_limit)
            if hunting:
                self.predator_headings[i] = math.atan2(delta[1], delta[0])
            else:
                self.predator_headings[i] += float(self.ecology_rng.normal(0, 0.7 * math.sqrt(c.world_dt)))
            heading = self.predator_headings[i]
            velocity = c.predator_speed * (1 if hunting else 0.4)
            desired = position + velocity * c.world_dt * np.array([math.cos(heading), math.sin(heading)])
            position[:] = self._body_move(position, desired - position, c.predator_radius)
            if not np.allclose(desired, position):
                self.predator_headings[i] += math.pi / 2
            if (hunting and np.linalg.norm(position - [self.x, self.y]) <= c.predator_radius + c.creature_radius
                    and not occluded(position, np.array([[self.x, self.y]]), self.active_blocks, c.block_size / 2)[0]):
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
        (nx, ny), pushed = self._push_move(np.array([dx, dy]))
        nx, ny = float(nx), float(ny)
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
        construction_gain = 0.0
        if self.construction_dirty and self.time >= self.next_cover_check:
            scores = self.construction_scores()
            construction_gain = float(np.maximum(0, scores - self.block_cover_best).sum())
            self.block_cover_best = np.maximum(scores, self.block_cover_best)
            self.best_shelter = max(float(scores.max(initial=0)), self.best_shelter)
            self.next_cover_check = self.time + .5
            self.construction_dirty = False
        self.construction_reward_total += c.construction_reward * construction_gain
        self.death_code = 3 if self.health <= 0 else 2 if self.water_active and self.hydration <= 0 else 1 if self.energy <= 0 else 0
        self.alive = self.death_code == 0
        return {"reward": old_hunger - self.hunger + old_thirst - self.thirst,
                "eaten": count, "drank": drank, "water_gain": water_gain, "food_gain": food_gain,
                "damage": damage, "died": not self.alive,
                "pushed": pushed, "construction_gain": construction_gain,
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
                           "hunting": bool(self.time >= self.predator_ready_at[i]
                                           and np.linalg.norm(p - [self.x, self.y]) < c.predator_detection
                                           and not occluded(p, np.array([[self.x, self.y]]), self.active_blocks, c.block_size / 2)[0])}
                          for i, p in enumerate(self.predators[:self.predator_limit])],
            "predator_radius": c.predator_radius,
            "food": self.food[self.active_food()].tolist(), "food_radius": c.food_radius,
            "vision_range": self.vision_range, "vision_fov": c.vision_fov,
            "blocks": self.active_blocks.tolist(), "block_size": c.block_size,
            "shelter": {"x": self.shelter_x, "y": self.shelter_y, "best": self.best_shelter},
        }

    def summary(self):
        c = self.config
        senses = self.sense()
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
            "block_retina": self.block_retina.tolist(), "obstacle_retina": self.obstacle_retina.tolist(),
            "fixed_retina": senses[181:197].tolist(), "material_retina": senses[197:213].tolist(),
            "cover": self.cover, "push_distance": self.push_distance, "protected_seconds": self.protected_seconds,
            "construction_reward_total": self.construction_reward_total,
        }
