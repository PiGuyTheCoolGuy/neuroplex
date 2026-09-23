import json
import math

import numpy as np
import pytest

from neuroplex.config import Config
from neuroplex.brain import Brain
from neuroplex.cover import CAPACITY, CoverLearner, SENSES, cover_features
from neuroplex.experiments import fingerprint, shelter_practice, shelter_trial
from neuroplex.lab import ExperimentRequest
from neuroplex.memory import SensoryMemory
from neuroplex.policy import BUILD_START, COVER_START, MotorPolicy, MOTOR_CURRENTS
from neuroplex.runner import Runner
from neuroplex.simulation import Simulation
from neuroplex.world import World


def test_typed_rays_distinguish_fixed_and_movable_first_surfaces():
    w = World(Config(habitat_stage=1, block_count=1, food_count=0, water_count=0, predator_count=0))
    w.x, w.y, w.heading = w.config.world_width - 8, 20, 0
    w.blocks[0] = [w.x + 4, w.y]
    sensed = w.sense()
    assert sensed[197 + 8] > 0 and sensed[181 + 8] == 0  # nearer block hides wall
    w.blocks[0] = [30, 30]
    sensed = w.sense()
    assert sensed[181 + 8] > 0 and sensed[197 + 8] == 0
    w.x = 30
    w.y = 10
    assert not w.sense()[197:213].any()  # the remote block is outside the local sensor


def test_cover_memory_records_only_visited_cover_and_moves_with_body():
    memory = SensoryMemory(Config())
    senses = np.zeros(SENSES, dtype=np.float32)
    senses[146:162] = .6  # visible materials are not an already-known safe site
    assert memory.observe(senses, 28)[245] == 0
    senses[178] = .5
    observed = memory.observe(senses, 28)
    assert observed[245] == .5 and observed[246] == 1
    memory.advance(5, 0, 0, 1)
    senses[178] = 0
    recalled = memory.observe(senses, 28)
    assert recalled[245] > 0 and recalled[246] == 0
    np.testing.assert_allclose(memory.sites[0, :2], [-5, 0])
    memory.advance(0, 0, math.pi / 2, 1)
    np.testing.assert_allclose(memory.sites[0, :2], [0, 5], atol=1e-10)
    memory.advance(0, 5, 0, 1)
    assert memory.observe(senses, 28)[245] == 0  # returning finds the cover removed
    senses[178] = .5
    memory.observe(senses, 28)
    memory.advance(5, 0, 0, 181)
    senses[178] = 0
    assert memory.observe(senses, 28)[245] == 0  # time limit, even if never revisited


def test_threat_memory_does_not_track_hidden_movement_and_disable_removes_hints():
    config = Config()
    memory = SensoryMemory(config)
    senses = np.zeros(SENSES, dtype=np.float32)
    senses[104] = .75
    memory.observe(senses, 28)
    location = memory.sites[1, :2].copy()
    senses[104] = 0
    memory.advance(0, 0, 0, 2)
    recalled = memory.observe(senses, 28)
    assert recalled[247] > 0
    policy = MotorPolicy(config)
    assert 2 in policy.available_goals(recalled)
    assert policy.encode_goal(recalled) != policy.encode_goal(senses)
    np.testing.assert_array_equal(memory.sites[1, :2], location)
    memory.advance(0, 0, 0, 5)
    assert memory.observe(senses, 28)[247] == 0
    off = SensoryMemory(Config(memory_enabled=False))
    senses[178] = .6
    assert not off.observe(senses, 28)[213:].any()


def test_use_cover_does_not_override_learned_goal_or_action():
    policy = MotorPolicy(Config(exploration_floor=0, pretrained_policy=False))
    senses = np.zeros(SENSES, dtype=np.float32)
    senses[245], senses[221] = .5, .8  # remembered site straight ahead
    state = policy.encode_goal(senses)
    policy.goal_values[state, 4] = 20
    motor = policy.encode_motor(senses, 4)
    policy.values[motor, 5] = 50  # deliberately move away from the site
    np.testing.assert_array_equal(policy.begin(senses, False), MOTOR_CURRENTS[5])
    assert policy.goal == 4 and policy.action == 5
    policy.reset_activity()
    policy.goal_values[state, 0] = 100
    policy.begin(senses, False)
    assert policy.goal == 0


def test_own_experience_generalizes_and_replay_never_edits_food_values():
    policy = MotorPolicy(Config(pretrained_policy=False))
    senses = np.zeros(SENSES, dtype=np.float32)
    senses[153], senses[155], senses[180] = .75, .4, 1
    state = policy.encode_motor(senses, 3)
    learner = policy.cover_learner
    learner.current_features[:] = cover_features(senses)
    before = policy.values[:306].copy()
    learner.observe(3, 4, state, state, 5, .96, False, cover_features(senses), policy.values, policy.config)
    similar = senses.copy()
    similar[153] = .7
    assert learner.scores(3, cover_features(similar))[4] > 0
    assert np.count_nonzero(learner.weights[1].any(axis=1)) == 1
    assert not learner.weights[0].any() and not learner.weights[2].any()
    np.testing.assert_array_equal(policy.values[:306], before)
    assert learner.count == 1 and learner.updates[1] == 3


def test_cover_shaping_cannot_pay_for_a_repeated_peekaboo_cycle_or_camping():
    policy = MotorPolicy(Config(pretrained_policy=False))
    gamma = policy.config.policy_discount ** (policy.config.world_dt / policy.config.action_seconds)
    states = []
    for cover in (.5, 0, .5):
        senses = np.zeros(SENSES, dtype=np.float32)
        senses[178], senses[247] = cover, .8
        states.append(senses)
    total = sum(gamma**i * (gamma * policy.potential(states[i + 1]) - policy.potential(states[i])) for i in range(2))
    assert total == pytest.approx((gamma**2 - 1) * policy.potential(states[0]))
    assert total < 0
    assert (gamma - 1) * policy.potential(states[0]) < 0


def test_cover_replay_and_memory_resume_exactly_and_freeze(tmp_path):
    sim = shelter_trial(Simulation(), 211, True, True)
    sim.brain.policy.goal_values[:, 3] = 50
    for _ in range(19):
        sim.tick()
    assert sim.brain.policy.cover_learner.count > 0
    sim.save(tmp_path / "brain.npz")
    resumed = Simulation.load(tmp_path / "brain.npz")
    for _ in range(21):
        sim.tick()
        resumed.tick()
    assert sim.snapshot() == resumed.snapshot()
    for name in CoverLearner.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(sim.brain.policy.cover_learner, name), getattr(resumed.brain.policy.cover_learner, name))
    np.testing.assert_array_equal(sim.memory.sites, resumed.memory.sites)
    sim.brain.learning = False
    original = fingerprint(sim)
    data = sim.brain.policy.cover_learner.replay_info.copy()
    for _ in range(25):
        sim.tick()
    assert fingerprint(sim) == original
    np.testing.assert_array_equal(data, sim.brain.policy.cover_learner.replay_info)


def test_v4_upgrade_preserves_material_layout_and_all_learned_values_without_expansion(tmp_path):
    sim = Simulation(Config(habitat_stage=4))
    sim.world.blocks[0] += [1, 1]
    sim.brain.policy.values[BUILD_START:COVER_START] += 3
    sim.world.time = 13
    sim.world.construction_reward_total = 9
    sim.paused = True
    path = tmp_path / "checkpoint.npz"
    sim.save(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files if not k.startswith("cover_") and k != "memory_sites"}
    metadata = json.loads(str(data["metadata"]))
    metadata["version"] = 4
    metadata.pop("cover_learner")
    for key in ("config", "world_config"):
        for field in ("cover_memory_seconds", "threat_memory_seconds", "cover_learning_rate", "cover_replay_steps", "cover_shaping"):
            metadata[key].pop(field)
    for field in ("values", "visits", "eligibility"):
        data["policy_" + field] = data["policy_" + field][:COVER_START, :6]
    for field in ("goal_values", "goal_visits", "goal_eligibility"):
        data["policy_" + field] = data["policy_" + field][:108, :4]
    data["policy_skill_updates"] = data["policy_skill_updates"][:4]
    data["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
    original = path.read_bytes()
    runner = Runner(tmp_path)
    try:
        loaded = runner.sim
        assert loaded.config.world_width == 144 and loaded.config.food_count == 48
        assert loaded.world.time == 13 and loaded.world.construction_reward_total == 9 and loaded.paused
        np.testing.assert_array_equal(sim.world.blocks, loaded.world.blocks)
        np.testing.assert_array_equal(sim.brain.weights, loaded.brain.weights)
        np.testing.assert_array_equal(sim.brain.policy.values[:COVER_START, :6], loaded.brain.policy.values[:COVER_START, :6])
        assert not loaded.brain.policy.values[:COVER_START, 6].any()
        np.testing.assert_array_equal(sim.brain.policy.goal_values[:108, :4], loaded.brain.policy.goal_values[:108, :4])
        assert (tmp_path / "checkpoint.v4.npz").read_bytes() == original
    finally:
        runner.close()


def test_shelter_practice_isolated_with_unseen_audits_and_selective_adoption(tmp_path):
    source = Simulation()
    original = fingerprint(source)
    request = ExperimentRequest(kind="shelter", episodes=2, seconds=5.0, trials=1).model_dump()
    source.save(tmp_path / "source.npz")
    result = shelter_practice(source, request, tmp_path, lambda *args, **kwargs: None)
    assert fingerprint(source) == original
    candidate = Simulation.load(tmp_path / "champion.npz")
    np.testing.assert_array_equal(candidate.brain.weights, source.brain.weights)
    np.testing.assert_array_equal(candidate.brain.policy.values[:BUILD_START], source.brain.policy.values[:BUILD_START])
    np.testing.assert_array_equal(candidate.brain.policy.cover_learner.weights[0], source.brain.policy.cover_learner.weights[0])
    assert result["audit"]["seeds"] == result["baseline_audit"]["seeds"]
    assert result["audit"]["seeds"] != result["habitat_audit"]["seeds"]
    assert result["summary"]["all_weights_frozen"] and result["habitat_audit"]["summary"]["all_weights_frozen"]
    runner = Runner(tmp_path / "live")
    try:
        runner.lab.champion = lambda: tmp_path / "champion.npz"
        current = runner.sim.brain.policy
        current.values[:BUILD_START] += 7
        current.cover_learner.weights[0] += 2
        preserved = current.values[:BUILD_START].copy()
        escaped = current.cover_learner.weights[0].copy()
        runner.sim.world.alive = False
        runner.adopt_champion()
        np.testing.assert_array_equal(current.values[:BUILD_START], preserved)
        np.testing.assert_array_equal(current.cover_learner.weights[0], escaped)
        np.testing.assert_array_equal(current.values[BUILD_START:], candidate.brain.policy.values[BUILD_START:])
    finally:
        runner.close()


def test_replay_and_practice_limits():
    for kwargs in ({"cover_replay_steps": 9}, {"cover_replay_steps": -1}, {"cover_learning_rate": .8}, {"cover_memory_seconds": 0}):
        with pytest.raises(ValueError):
            Config(**kwargs)
    with pytest.raises(ValueError):
        ExperimentRequest(kind="shelter", seconds=61.0)
    assert CoverLearner(1).replay_features.nbytes < 1_000_000


def test_dedicated_material_inputs_drive_neurons_and_rest_settles_body_motion():
    baseline, observed = Brain(Config()), Brain(Config())
    empty, material = np.zeros(SENSES), np.zeros(SENSES)
    material[197:213] = 1
    for _ in range(10):
        baseline.step(empty)
        observed.step(material)
    assert observed.rates[200:216].mean() > baseline.rates[200:216].mean()
    brain = Brain(Config(pretrained_policy=False, exploration_floor=0))
    brain.learning = False
    brain.policy.values[:, 6] = 50
    brain.rates[400:425] = 50  # start with forward momentum in the rate decoder
    for _ in range(20):
        speed, turn = brain.advance(empty)
        brain.observe(empty, 0, 0, True)
    assert brain.policy.action == 6
    assert abs(speed) < .01 and abs(turn) < .01
