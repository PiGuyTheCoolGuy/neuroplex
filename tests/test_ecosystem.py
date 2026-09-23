import math
import numpy as np
import pytest

from neuroplex.config import Config
from neuroplex.world import World
from neuroplex.memory import SensoryMemory
from neuroplex.policy import MotorPolicy
from neuroplex.simulation import Simulation
from neuroplex.curriculum import ready_to_advance


def test_water_only_matters_when_enabled_and_drinking_has_no_full_tank_bonus():
    w = World(Config(habitat_stage=0, water_count=1))
    w.hydration = 0.001
    w.step(0, 0)
    assert w.alive and w.hydration == 0.001
    w.set_stage(2)
    w.water[0] = [w.x, w.y]
    w.hydration = 50
    first = w.step(0, 0)
    assert first["drank"] == 1 and w.hydration > 50
    assert w.step(0, 0)["drank"] == 0
    w.hydration = 100
    assert w.step(0, 0)["water_gain"] == pytest.approx(w.config.water_cost * w.config.world_dt)
    w.water[0] = [2, 2]
    w.hydration = 0.001
    w.step(0, 0)
    assert not w.alive and w.death_reason == "dehydration"


def test_predators_sense_behind_damage_on_contact_and_have_attack_cooldown():
    w = World(Config(habitat_stage=3, predator_count=1))
    w.heading = 0
    w.predators[0] = [w.x - 5, w.y]
    assert w.sense()[96:112].max() > 0
    w.predators[0] = [w.x + 1, w.y]
    w.predator_ready_at[0] = 0
    outcome = w.step(0, 0)
    assert outcome["damage"] == w.config.predator_damage
    assert w.health == 85 and w.attacks == 1
    assert w.step(0, 0)["damage"] == 0
    w.predators[0] = [w.x, w.y]
    w.predator_ready_at[0] = 0
    w.health = 1
    w.step(0, 0)
    assert not w.alive and w.death_reason == "predator injuries"


def test_stage_changes_food_visibility_without_deleting_or_resetting_resources():
    w = World(Config())
    food = w.food.copy()
    w.regrow_at[0] = 25
    energy = w.energy
    w.set_stage(4)
    assert w.food_limit == 20 and w.predator_limit == 2 and w.water_active
    assert w.vision_range < w.config.vision_range
    np.testing.assert_array_equal(food, w.food)
    assert w.regrow_at[0] == 25 and w.energy == energy


def test_memory_uses_old_sight_and_body_motion_then_expires():
    config = Config(memory_seconds=2)
    memory = SensoryMemory(config)
    senses = np.zeros(146, dtype=np.float32)
    senses[8] = 0.6
    memory.observe(senses, 28)
    original = memory.traces[0, :2].copy()
    senses[8] = 0
    memory.advance(1, 0, math.pi / 2, 0.5)
    remembered = memory.observe(senses, 28)
    assert remembered[112:128].any() and not remembered[:16].any()
    assert memory.traces[0, 0] == pytest.approx(original[1] - 1)
    assert memory.traces[0, 1] == pytest.approx(-original[0])
    memory.advance(0, 0, 0, 2)
    assert not memory.observe(senses, 28)[112:144].any()
    assert not SensoryMemory(config).observe(senses, 28)[112:144].any()


def test_goal_choice_is_learned_not_a_fixed_thirst_or_escape_reflex():
    policy = MotorPolicy(Config(exploration_floor=0))
    senses = np.zeros(146, dtype=np.float32)
    senses[144] = 1
    senses[48:64] = 0.9
    senses[8] = senses[88] = 0.8
    senses[100] = 0.9
    state = policy.encode_goal(senses)
    policy.goal_values[state, :3] = [30, 20, 10]
    policy.begin(senses, False)
    assert policy.goal == 0  # even while very thirsty and threatened
    policy.reset_activity()
    policy.goal_values[state, :3] = [10, 30, 20]
    policy.begin(senses, False)
    assert policy.goal == 1
    policy.reset_activity()
    policy.goal_values[state, :3] = [10, 20, 30]
    policy.begin(senses, False)
    assert policy.goal == 2


def test_curriculum_requires_a_full_healthy_window_and_drinking():
    w = World(Config(habitat_stage=2))
    w.time = 200
    samples = [{"stage": 2, "age": 145 + i * 5, "energy": 80, "hydration": 80,
                "health": 100, "food_per_minute": 12, "drinks": int(i > 3)} for i in range(12)]
    assert ready_to_advance(w, samples, 0, 180)
    samples[-1]["hydration"] = 20
    assert not ready_to_advance(w, samples, 0, 180)
    samples[-1]["hydration"] = 80
    assert not ready_to_advance(w, samples, 100, 180)
    assert not ready_to_advance(w, samples[-3:], 0, 180)


def test_metrics_and_completed_lives_persist_when_a_new_life_begins(tmp_path):
    sim = Simulation()
    for _ in range(101):
        sim.tick()
    sim.world.energy = 0.0001
    sim.world.food[:] = [2, 2]
    sim.tick()
    assert not sim.world.alive and sim.life_records[-1]["cause"] == "starvation"
    elapsed, count = sim.elapsed, len(sim.metrics)
    sim.new_life()
    assert sim.elapsed == elapsed and len(sim.metrics) == count
    for _ in range(100):
        sim.tick()
    assert sim.metrics[-1]["life"] == 2 and sim.metrics[-1]["time"] > elapsed
    sim.save(tmp_path / "state.npz")
    restored = Simulation.load(tmp_path / "state.npz")
    assert restored.metrics == sim.metrics and restored.life_records == sim.life_records


@pytest.mark.parametrize("setting", [{"memory_seconds": float("nan")}, {"habitat_stage": True}, {"predator_count": 9}])
def test_invalid_ecosystem_settings_are_rejected(setting):
    with pytest.raises(ValueError):
        Config(**setting)
