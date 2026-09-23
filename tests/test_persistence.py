import numpy as np
import pytest
import json
from dataclasses import fields

from neuroplex.brain import Brain
from neuroplex.runner import Runner
from neuroplex.simulation import Simulation
from neuroplex.policy import MotorPolicy
from neuroplex.config import Config


def test_checkpoint_resumes_identically_including_random_streams(tmp_path):
    sim = Simulation()
    for _ in range(33):  # save during a held action, not just at its boundary
        sim.tick()
    path = tmp_path / "checkpoint.npz"
    sim.save(path)
    resumed = Simulation.load(path)
    for _ in range(40):
        sim.tick()
        resumed.tick()
    for name in Brain.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(sim.brain, name), getattr(resumed.brain, name))
    for name in MotorPolicy.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(sim.brain.policy, name), getattr(resumed.brain.policy, name))
    assert sim.world.summary() == resumed.world.summary()
    assert sim.snapshot() == resumed.snapshot()
    resumed.save(path)
    assert path.with_suffix(".previous.npz").is_file()


def test_pause_and_death_do_not_reset_the_brain():
    sim = Simulation()
    sim.tick()
    v = sim.brain.v.copy()
    weights = sim.brain.weights.copy()
    sim.paused = True
    sim.tick()
    np.testing.assert_array_equal(sim.brain.v, v)
    sim.paused = False
    sim.world.alive = False
    sim.tick()
    np.testing.assert_array_equal(sim.brain.v, v)
    sim.new_life()
    assert sim.world.alive and sim.life == 2
    np.testing.assert_array_equal(sim.brain.weights, weights)


def test_corrupt_file_is_rejected_and_not_overwritten(tmp_path):
    path = tmp_path / "checkpoint.npz"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="not overwritten"):
        Runner(tmp_path)
    assert path.read_bytes() == b"not a checkpoint"


def test_second_process_cannot_use_the_same_world(tmp_path):
    runner = Runner(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already using"):
            Runner(tmp_path)
    finally:
        runner.close()


@pytest.mark.parametrize("alive", [True, False])
def test_v1_upgrade_keeps_world_memories_and_an_original_backup(tmp_path, alive):
    sim = Simulation()
    sim.tick()
    sim.paused = True
    sim.world.alive = alive
    if not alive:
        sim.world.energy = 0
        sim.world.death_code = 1
    sim.brain.threshold[400:] = 1.7
    original_world = sim.world.summary()
    original_weights = sim.brain.weights.copy()
    path = tmp_path / "checkpoint.npz"
    sim.save(path)
    # Synthesize the exact old schema, without relying on a developer's save.
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files
                if not k.startswith("policy_") and k != "brain_motor_current"}
    metadata = json.loads(str(data["metadata"]))
    metadata["version"] = 1
    metadata.pop("policy")
    metadata.pop("policy_rng")
    metadata["simulation"]["total_reward"] = metadata["simulation"].pop("total_drive_reward")
    field_names = [field.name for field in fields(Config)]
    old_fields = set(field_names[:field_names.index("pretrained_policy") + 1])
    old_fields -= {"action_seconds", "policy_learning_rate", "policy_discount", "policy_trace_decay",
                   "exploration_start", "exploration_floor", "exploration_decay_decisions",
                   "shaping_scale", "eating_reward", "pretrained_policy"}
    for key in ("config", "world_config"):
        metadata[key] = {k: v for k, v in metadata[key].items() if k in old_fields}
        metadata[key]["learning_rate"] = 0.025
    metadata["world"] = {k: v for k, v in metadata["world"].items()
                         if k in ("time", "x", "y", "heading", "energy", "alive", "eaten", "distance", "touch", "speed", "turn")}
    metadata["simulation"] = {k: v for k, v in metadata["simulation"].items()
                              if k in ("paused", "speed", "life", "ticks", "total_reward", "history", "events", "saved_at")}
    data["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
    original_bytes = path.read_bytes()
    runner = Runner(tmp_path)
    try:
        assert runner.sim.world.x == original_world["x"] * 1.5
        assert runner.sim.world.time == original_world["time"]
        assert runner.sim.world.energy == original_world["energy"]
        assert runner.sim.world.alive == original_world["alive"]
        assert runner.sim.world.config.world_width == original_world["width"] * 1.5
        np.testing.assert_array_equal(runner.sim.brain.weights, original_weights)
        assert runner.sim.paused
        assert runner.sim.brain.policy.bootstrap_updates == 2400
        assert np.all(runner.sim.brain.threshold[400:] == 1)
        assert (tmp_path / "checkpoint.v1.npz").read_bytes() == original_bytes
    finally:
        runner.close()
    runner = Runner(tmp_path)
    try:
        assert runner.sim.migrated_from is None
        assert (tmp_path / "checkpoint.v1.npz").read_bytes() == original_bytes
    finally:
        runner.close()


def test_v2_upgrade_keeps_food_values_mid_action_and_archives_original(tmp_path):
    sim = Simulation(Config(memory_enabled=False))
    for _ in range(13):
        sim.tick()
    sim.paused = True
    sim.brain.learning = False
    original_values = sim.brain.policy.values[:153].copy()
    original_weights = sim.brain.weights.copy()
    path = tmp_path / "checkpoint.npz"
    sim.save(path)
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    metadata = json.loads(str(data["metadata"]))
    metadata["version"] = 2
    for key in ("config", "world_config"):
        names = list(metadata[key])
        allowed = set(names[:names.index("pretrained_policy") + 1])
        metadata[key] = {k: v for k, v in metadata[key].items() if k in allowed}
    for key in ("values", "visits", "eligibility"):
        data["policy_" + key] = data["policy_" + key][:153]
    metadata["policy"] = {k: v for k, v in metadata["policy"].items() if k not in ("goal", "goal_state", "goal_updates", "risk_scale")}
    metadata["world"] = {k: v for k, v in metadata["world"].items()
                         if k in ("time", "x", "y", "heading", "energy", "alive", "eaten", "distance", "touch", "speed", "turn")}
    metadata["simulation"] = {k: v for k, v in metadata["simulation"].items()
                              if k in ("paused", "speed", "life", "ticks", "total_reward", "total_drive_reward", "history", "events", "saved_at")}
    data["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
    before = path.read_bytes()
    runner = Runner(tmp_path)
    try:
        resumed = runner.sim
        np.testing.assert_array_equal(resumed.brain.policy.values[:153], original_values)
        np.testing.assert_array_equal(resumed.brain.policy.values[153:306], original_values)
        np.testing.assert_array_equal(resumed.brain.weights, original_weights)
        assert not resumed.brain.policy.pending  # geometry/observation change starts fresh action credit
        assert resumed.world.energy == sim.world.energy and resumed.world.time == sim.world.time
        assert resumed.paused and not resumed.brain.learning
        assert resumed.stage_started == sim.world.time
        assert (tmp_path / "checkpoint.v2.npz").read_bytes() == before
    finally:
        runner.close()


def test_ecosystem_checkpoint_resumes_memory_predators_and_goal_learning_exactly(tmp_path):
    sim = Simulation(Config(habitat_stage=4))
    for _ in range(113):
        sim.tick()
    sim.save(tmp_path / "checkpoint.npz")
    resumed = Simulation.load(tmp_path / "checkpoint.npz")
    for _ in range(37):
        sim.tick()
        resumed.tick()
    assert sim.snapshot() == resumed.snapshot()
    np.testing.assert_array_equal(sim.memory.traces, resumed.memory.traces)
    for name in MotorPolicy.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(sim.brain.policy, name), getattr(resumed.brain.policy, name))
