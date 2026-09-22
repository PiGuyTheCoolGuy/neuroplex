import numpy as np

from neuroplex.config import Config
from neuroplex.world import World


def test_starvation_is_terminal_and_reward_is_drive_reduction():
    world = World(Config(food_count=0))
    initial_hunger, total_reward = world.hunger, 0
    while world.alive:
        total_reward += world.step(0, 0)["reward"]
    assert world.energy == 0
    assert abs(total_reward - (initial_hunger - 1)) < 1e-10
    time = world.time
    world.step(1, 1)
    assert world.time == time


def test_food_is_consumed_once_and_regrows_after_delay():
    world = World(Config(food_count=1))
    world.food[0] = [world.x, world.y]
    world.energy = 40
    assert world.step(0, 0)["eaten"] == 1
    assert world.energy > 63
    assert world.step(0, 0)["eaten"] == 0
    world.time = float(world.regrow_at[0])
    assert world.step(0, 0)["eaten"] == 1


def test_retina_is_egocentric_and_hides_distant_and_rear_food():
    world = World(Config(food_count=1))
    world.heading = 0
    world.food[0] = [world.x + 10, world.y]
    senses = world.sense()
    assert senses[8] > 0
    assert np.all(senses[32:48] == world.hunger)
    assert np.all(senses[48:64] == 0)
    world.food[0] = [world.x - 10, world.y]
    assert not world.sense()[:16].any()
    world.food[0] = [world.x + 40, world.y]
    assert not world.sense()[:16].any()


def test_collision_bounds_body_without_homing_or_teleportation():
    world = World(Config(food_count=0))
    world.x = world.config.world_width - world.config.creature_radius
    world.heading = 0
    world.step(1, 0)
    assert world.touch == 1
    assert world.x == world.config.world_width - world.config.creature_radius
    assert world.heading == 0  # wall contact never supplies an automatic turn
