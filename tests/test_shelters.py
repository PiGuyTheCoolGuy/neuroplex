import json
import math

import numpy as np
import pytest

from neuroplex.config import Config
from neuroplex.experiments import escape_practice, fingerprint
from neuroplex.geometry import circle_overlaps, cover_at, occluded, ray_hits
from neuroplex.lab import ExperimentRequest
from neuroplex.policy import BUILD_START, ESCAPE_START, MotorPolicy
from neuroplex.runner import Runner
from neuroplex.simulation import Simulation
from neuroplex.world import World


def empty_world(**kwargs):
    return World(Config(habitat_stage=4, food_count=0, water_count=0, predator_count=0, **kwargs))


def test_new_world_has_less_resource_density_and_blocks_after_warmup():
    c = Config()
    assert (c.world_width, c.world_height, c.food_count, c.water_count) == (144, 90, 48, 8)
    assert c.food_regrow_seconds == 45
    assert c.food_count / (c.world_width * c.world_height) == pytest.approx((64 / (96 * 60)) / 3)
    w = World(c)
    assert w.render_state()["blocks"] == []
    w.set_stage(1)
    assert len(w.active_blocks) == 24
    assert not circle_overlaps([w.x, w.y], c.creature_radius, w.blocks, c.block_size / 2).any()
    for i, block in enumerate(w.blocks):
        assert not np.any(np.all(np.abs(w.blocks[:i] - block) < c.block_size, axis=1))


def test_ray_occlusion_parallel_and_behind_origins():
    blocks = np.array([[10., 10.]])
    origin = np.array([5., 10.])
    assert occluded(origin, np.array([[15., 10.], [5., 15.]]), blocks, 1.6).tolist() == [True, False]
    hits = ray_hits([[5, 10], [5, 15], [15, 10]], [[1, 0], [1, 0], [1, 0]], blocks, 1.6)
    assert hits[0, 0] == pytest.approx(3.4)
    assert np.isinf(hits[1:, 0]).all()
    w = empty_world(block_count=0)
    w.heading = math.pi / 16  # exact axis-aligned rays must not invent nearby walls
    assert not w.sense()[162:178].any()


def test_creature_can_push_blocks_but_not_overlap_or_push_them_through_each_other():
    w = empty_world(block_count=2)
    w.x, w.y, w.heading = 10, 10, 0
    w.blocks[:] = [[12.73, 10], [30, 30]]
    before = w.blocks.copy()
    outcome = w.step(1, 0)
    assert outcome["pushed"] > 0 and w.blocks[0, 0] > before[0, 0] and w.x > 10
    assert not circle_overlaps([w.x, w.y], w.config.creature_radius, w.blocks, w.config.block_size / 2).any()
    w.blocks[1] = w.blocks[0] + [w.config.block_size, 0]
    before = w.blocks.copy()
    assert w.step(1, 0)["pushed"] == 0
    np.testing.assert_array_equal(w.blocks, before)
    assert w.touch == 1


def test_blocks_cannot_leave_world_or_bury_resources():
    w = World(Config(habitat_stage=1, block_count=1, food_count=1, water_count=0, predator_count=0))
    half = w.config.block_size / 2
    w.x, w.y, w.heading = w.config.world_width - 2 * half - w.config.creature_radius - .03, 10, 0
    w.blocks[0] = [w.config.world_width - half, 10]
    before = w.blocks.copy()
    assert w.step(1, 0)["pushed"] == 0
    np.testing.assert_array_equal(w.blocks, before)
    w.x, w.y = 10, 10
    w.blocks[0] = [12.73, 10]
    w.food[0] = [12.73 + half + w.config.food_radius, 10]
    before = w.blocks.copy()
    assert w.step(1, 0)["pushed"] == 0
    np.testing.assert_array_equal(w.blocks, before)


def test_blocks_hide_food_and_predators_and_stop_predator_motion():
    w = World(Config(habitat_stage=4, block_count=1, predator_count=1, food_count=1, water_count=0))
    w.x, w.y, w.heading = 10, 10, 0
    w.blocks[0] = [14, 10]
    w.food[0] = [18, 10]
    w.predators[0] = [18, 10]
    w.predator_ready_at[0] = 0
    senses = w.sense()
    assert not senses[:16].any() and not senses[96:112].any()
    assert senses[146:162].any() and senses[162:178].any()
    assert not w.render_state()["predators"][0]["hunting"]
    moved = w._body_move(np.array([18., 10.]), np.array([-2., 0.]), w.config.predator_radius)
    assert moved[0] == 18
    assert w.step(0, 0)["damage"] == 0
    assert w.protected_seconds > 0
    w.blocks[0] = [40, 40]
    assert w.sense()[96:112].any()


def test_cover_requires_multiple_blocks_and_an_exit_not_a_world_corner():
    blocks = np.array([[20., 20.], [23.2, 20.], [20., 23.2]])
    args = (1.6, 1.1, 144, 90)
    assert cover_at([[23.2, 23.2]], blocks, *args)[0] >= .5
    assert cover_at([[23.2, 23.2]], blocks[:1], *args)[0] == 0
    assert cover_at([[1.1, 1.1]], blocks, *args)[0] == 0
    sealed = np.array([[30., 26.8], [30., 33.2], [26.8, 30.], [33.2, 30.]])
    assert cover_at([[30., 30.]], sealed, *args)[0] == 0


def test_only_improved_structure_is_rewarded_not_repeated_touch_or_body_movement():
    w = empty_world(block_count=3)
    w.blocks[:] = [[20, 20], [23.2, 20], [20, 23.2]]
    w.best_shelter = 0
    w.block_cover_best.fill(0)
    w.construction_dirty = True  # pending measurement after a physical rearrangement
    first = w.step(0, 0)
    assert first["construction_gain"] > 0
    assert w.step(0, 0)["construction_gain"] == 0
    w.construction_dirty = True
    w.time += .6
    assert w.step(0, 0)["construction_gain"] == 0
    w.x, w.y = w.shelter_x, w.shelter_y
    assert w.step(0, 0)["construction_gain"] == 0


def test_escape_distinguishes_corner_from_open_space_early():
    policy = MotorPolicy(Config())
    senses = np.zeros(181, dtype=np.float32)
    senses[100] = .8
    open_state = policy.encode_motor(senses, 2)
    senses[169] = .5  # obstacle still six units away, before touch
    wall_state = policy.encode_motor(senses, 2)
    senses[166] = .5
    corner_state = policy.encode_motor(senses, 2)
    assert len({open_state, wall_state, corner_state}) == 3
    assert ESCAPE_START <= min(open_state, wall_state, corner_state) < BUILD_START


def test_escape_remains_learned_and_reduces_exploration_near_predators():
    policy = MotorPolicy(Config(exploration_floor=0))
    senses = np.zeros(181, dtype=np.float32)
    senses[100] = .9
    manager = policy.encode_goal(senses)
    policy.goal_values[manager] = [0, 0, 20, 0]
    motor = policy.encode_motor(senses, 2)
    policy.values[motor, 5] = 50  # deliberately choose backward, no hidden steering overrides it
    policy.begin(senses, False)
    assert policy.goal == 2 and policy.action == 5 and policy.risk_scale == .2
    policy.reset_activity()
    policy.goal_values[manager] = [50, 0, 0, 0]
    policy.begin(senses, False)
    assert policy.goal == 0  # no hardcoded emergency goal switch


def test_build_goal_is_available_only_for_observed_material_and_is_learned():
    policy = MotorPolicy(Config(exploration_floor=0))
    senses = np.zeros(181, dtype=np.float32)
    senses[180] = 1
    assert 3 not in policy.available_goals(senses)
    senses[154] = .8
    policy.goal_values[policy.encode_goal(senses), 3] = 30
    policy.begin(senses, False)
    assert policy.goal == 3 and policy.state >= BUILD_START


def test_block_checkpoint_resumes_exactly_mid_push(tmp_path):
    sim = Simulation(Config(habitat_stage=4, block_count=2))
    sim.world.blocks[0] = [sim.world.x + 2.73, sim.world.y]
    sim.world.heading = 0
    sim.world.step(1, 0)
    sim.save(tmp_path / "checkpoint.npz")
    loaded = Simulation.load(tmp_path / "checkpoint.npz")
    for _ in range(20):
        sim.tick()
        loaded.tick()
    assert sim.snapshot() == loaded.snapshot()
    np.testing.assert_array_equal(sim.world.blocks, loaded.world.blocks)


def test_real_v3_schema_upgrades_once_preserving_learning_and_archiving_original(tmp_path):
    sim = Simulation(Config(world_width=96, world_height=60, food_count=64, water_count=12,
                            food_regrow_seconds=25, habitat_stage=4))
    for _ in range(10):
        sim.tick()
    sim.paused = True
    sim.brain.learning = False
    path = tmp_path / "checkpoint.npz"
    sim.save(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(data["metadata"]))
    metadata["version"] = 3
    new_fields = {"block_count", "block_size", "push_speed_fraction", "obstacle_range", "construction_reward", "goal_switch_margin"}
    for key in ("config", "world_config"):
        metadata[key] = {k: v for k, v in metadata[key].items() if k not in new_fields}
    for key in ("values", "visits", "eligibility"):
        data["policy_" + key] = data["policy_" + key][:459]
    for key in ("goal_values", "goal_visits", "goal_eligibility"):
        data["policy_" + key] = data["policy_" + key][:, :3]
    data["policy_skill_updates"] = data["policy_skill_updates"][:3]
    metadata["policy"].pop("risk_scale")
    for key in ("blocks", "block_retina", "obstacle_retina", "block_cover_best"):
        data.pop("world_" + key)
    for key in ("push_distance", "pushed", "cover", "best_shelter", "shelter_x", "shelter_y",
                "protected_seconds", "next_cover_check", "construction_dirty", "construction_reward_total"):
        metadata["world"].pop(key)
    # v3 held a legacy escape row, not the new extended observation state.
    metadata["policy"]["state"] = 306
    metadata["policy"]["goal"] = 2
    data["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
    original = path.read_bytes()
    runner = Runner(tmp_path)
    try:
        upgraded = runner.sim
        assert (upgraded.config.world_width, upgraded.config.world_height, upgraded.config.food_count) == (144, 90, 48)
        assert upgraded.world.stage == 4 and upgraded.world.time == sim.world.time
        assert upgraded.world.energy == sim.world.energy and upgraded.world.health == sim.world.health
        assert upgraded.paused and not upgraded.brain.learning
        np.testing.assert_array_equal(upgraded.brain.weights, sim.brain.weights)
        np.testing.assert_array_equal(upgraded.brain.policy.values[:459], sim.brain.policy.values[:459])
        np.testing.assert_array_equal(upgraded.brain.policy.goal_values[:, :3], sim.brain.policy.goal_values[:, :3])
        assert (tmp_path / "checkpoint.v3.npz").read_bytes() == original
    finally:
        runner.close()
    runner = Runner(tmp_path)
    try:
        assert runner.sim.config.world_width == 144 and runner.sim.migrated_from is None
        assert (tmp_path / "checkpoint.v3.npz").read_bytes() == original
    finally:
        runner.close()


def test_escape_practice_preserves_other_skills_and_uses_unseen_frozen_audit(tmp_path):
    source = Simulation()
    before = fingerprint(source)
    request = {"seed": 331, "seconds": .5, "episodes": 2, "trials": 2}
    result = escape_practice(source, request, tmp_path, lambda *args, **kwargs: None)
    candidate = Simulation.load(tmp_path / "champion.npz")
    assert fingerprint(source) == before
    assert result["summary"]["all_weights_frozen"]
    assert result["audit"]["seeds"] == result["baseline_audit"]["seeds"]
    assert not set(result["audit"]["seeds"]) & {331, 432}
    for section in (slice(0, ESCAPE_START), slice(BUILD_START, None)):
        np.testing.assert_array_equal(source.brain.policy.values[section], candidate.brain.policy.values[section])
    np.testing.assert_array_equal(source.brain.weights, candidate.brain.weights)
    assert candidate.brain.policy.goal_updates > source.brain.policy.goal_updates


def test_practice_and_block_configuration_limits():
    assert ExperimentRequest(kind="escape").stage == 3
    for settings in ({"kind": "escape", "seconds": 61}, {"episodes": 201}):
        with pytest.raises(ValueError):
            ExperimentRequest(**settings)
    for settings in ({"block_count": 65}, {"block_count": -1}, {"push_speed_fraction": 2}):
        with pytest.raises(ValueError):
            Config(**settings)


def test_adopting_escape_practice_preserves_learning_since_the_copy_was_taken(tmp_path, monkeypatch):
    runner = Runner(tmp_path / "live")
    directory = tmp_path / "practice"
    directory.mkdir()
    try:
        runner.sim.save(directory / "source.npz")
        escape_practice(runner.sim, {"seed": 88, "seconds": .5, "episodes": 2, "trials": 1},
                        directory, lambda *args, **kwargs: None)
        runner.sim.brain.weights[0] += .001
        runner.sim.brain.policy.values[:306] += 2
        runner.sim.brain.policy.values[BUILD_START:] += 3
        runner.sim.brain.policy.goal_values[:, 3] += 4
        food_water = runner.sim.brain.policy.values[:306].copy()
        building = runner.sim.brain.policy.values[BUILD_START:].copy()
        build_goals = runner.sim.brain.policy.goal_values[:, 3].copy()
        weights = runner.sim.brain.weights.copy()
        runner.sim.world.alive = False
        monkeypatch.setattr(runner.lab, "champion", lambda: directory / "champion.npz")
        runner.adopt_champion()
        assert runner.sim.life == 2
        np.testing.assert_array_equal(runner.sim.brain.weights, weights)
        np.testing.assert_array_equal(runner.sim.brain.policy.values[:306], food_water)
        np.testing.assert_array_equal(runner.sim.brain.policy.values[BUILD_START:], building)
        np.testing.assert_array_equal(runner.sim.brain.policy.goal_values[:, 3], build_goals)
    finally:
        runner.close()
