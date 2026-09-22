"""Small, explicit defaults for a CPU-only OptiPlex. Time is in seconds."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 7
    world_width: float = 96.0
    world_height: float = 60.0
    food_count: int = 64
    food_energy: float = 24.0
    food_regrow_seconds: float = 25.0
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

    def __post_init__(self):
        if self.neurons != 500 or not 1 <= self.fan_out < 420:
            raise ValueError("v1 uses the documented 500-neuron layout; fan_out must be 1..419")
        for name, value in asdict(self).items():
            if name not in ("seed", "food_count", "pretrained_policy", "shaping_scale", "exploration_floor") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.shaping_scale < 0 or type(self.pretrained_policy) is not bool:
            raise ValueError("shaping_scale must be nonnegative and pretrained_policy must be boolean")
        if self.food_count < 0:
            raise ValueError("food_count must be non-negative")
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
    ("Reserved", 48, 64),
    ("Touch", 64, 80),
    ("Association", 80, 400),
    ("Forward", 400, 425),
    ("Backward", 425, 450),
    ("Left", 450, 475),
    ("Right", 475, 500),
]
