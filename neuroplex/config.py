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
    learning_rate: float = 0.025
    weight_max: float = 0.12

    def __post_init__(self):
        if self.neurons != 500 or not 1 <= self.fan_out < 420:
            raise ValueError("v1 uses the documented 500-neuron layout; fan_out must be 1..419")
        for name, value in asdict(self).items():
            if name not in ("seed", "food_count") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.food_count < 0:
            raise ValueError("food_count must be non-negative")
        if abs(round(self.world_dt / self.brain_dt) * self.brain_dt - self.world_dt) > 1e-9:
            raise ValueError("world_dt must be a multiple of brain_dt")
        if min(self.world_width, self.world_height) <= 8 * self.creature_radius:
            raise ValueError("world is too small")


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
