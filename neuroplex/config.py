"""Small, explicit defaults for a CPU-only OptiPlex. Time is in seconds."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    seed: int = 7
    world_width: float = 144.0
    world_height: float = 90.0
    food_count: int = 48
    food_energy: float = 24.0
    food_regrow_seconds: float = 45.0
    creature_radius: float = 1.1
    food_radius: float = 0.7
    max_energy: float = 100.0
    basal_cost: float = 0.70
    movement_cost: float = 0.12
    max_speed: float = 9.0
    max_turn: float = 2.4
    vision_range: float = 28.0
    vision_fov: float = 4.1887902047863905  # 240 degrees
    world_dt: float = 0.05
    brain_dt: float = 0.005
    neurons: int = 500
    fan_out: int = 32
    eligibility_tau: float = 5.0
    learning_rate: float = 0.001
    weight_max: float = 0.12
    action_seconds: float = 0.25
    policy_learning_rate: float = 0.30
    policy_discount: float = 0.96
    policy_trace_decay: float = 0.40
    exploration_start: float = 0.35
    exploration_floor: float = 0.05
    exploration_decay_decisions: float = 1500.0
    shaping_scale: float = 4.0
    eating_reward: float = 6.0
    pretrained_policy: bool = True
    memory_enabled: bool = True
    memory_seconds: float = 10.0
    curriculum_enabled: bool = True
    habitat_stage: int = 0
    stage_seconds: float = 180.0
    water_enabled: bool = True
    water_count: int = 8
    water_radius: float = 1.7
    max_hydration: float = 100.0
    water_cost: float = 0.35
    movement_water_cost: float = 0.035
    drinking_rate: float = 24.0
    drinking_reward: float = 6.0
    predator_count: int = 2
    predator_speed: float = 4.0
    predator_radius: float = 1.3
    predator_detection: float = 18.0
    predator_damage: float = 15.0
    predator_cooldown: float = 5.0
    max_health: float = 100.0
    healing_rate: float = 0.7
    block_count: int = 24
    block_size: float = 3.2
    push_speed_fraction: float = 0.4
    obstacle_range: float = 12.0
    construction_reward: float = 8.0
    goal_switch_margin: float = 0.4

    def __post_init__(self):
        if self.neurons != 500 or not 1 <= self.fan_out < 420:
            raise ValueError("v1 uses the documented 500-neuron layout; fan_out must be 1..419")
        booleans = {"pretrained_policy", "memory_enabled", "curriculum_enabled", "water_enabled"}
        nonnegative = {"seed", "food_count", "shaping_scale", "exploration_floor", "habitat_stage",
                       "water_count", "predator_count", "block_count", "goal_switch_margin", "construction_reward"}
        for name, value in asdict(self).items():
            if name in booleans:
                if type(value) is not bool:
                    raise ValueError(f"{name} must be boolean")
                continue
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name not in nonnegative and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.shaping_scale < 0 or type(self.pretrained_policy) is not bool:
            raise ValueError("shaping_scale must be nonnegative and pretrained_policy must be boolean")
        if self.food_count < 0:
            raise ValueError("food_count must be non-negative")
        for name in ("seed", "food_count", "water_count", "predator_count", "habitat_stage", "block_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.habitat_stage <= 4 or self.predator_count > 8:
            raise ValueError("habitat stage must be 0..4; at most 8 predators")
        if self.block_count > 64 or not 0 < self.push_speed_fraction <= 1:
            raise ValueError("at most 64 blocks; push speed fraction must be in (0, 1]")
        if self.goal_switch_margin < 0 or self.construction_reward < 0:
            raise ValueError("goal margin and construction reward must be nonnegative")
        if abs(round(self.world_dt / self.brain_dt) * self.brain_dt - self.world_dt) > 1e-9:
            raise ValueError("world_dt must be a multiple of brain_dt")
        if min(self.world_width, self.world_height) <= 8 * self.creature_radius:
            raise ValueError("world is too small")
        if not 0 < self.policy_discount < 1 or not 0 < self.policy_trace_decay <= 1:
            raise ValueError("policy discount and trace decay must be in (0, 1]")
        if not 0 <= self.exploration_floor <= self.exploration_start <= 1:
            raise ValueError("invalid exploration schedule")
        if abs(round(self.action_seconds / self.world_dt) * self.world_dt - self.action_seconds) > 1e-9:
            raise ValueError("action_seconds must be a multiple of world_dt")


GROUPS = [
    ("Vision", 0, 32),
    ("Hunger", 32, 48),
    ("Thirst", 48, 64),
    ("Touch", 64, 80),
    ("Association", 80, 400),
    ("Forward", 400, 425),
    ("Backward", 425, 450),
    ("Left", 450, 475),
    ("Right", 475, 500),
]
