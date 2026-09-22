import json
from types import SimpleNamespace

import numpy as np
import pytest

from neuroplex import runner as runner_module
from neuroplex.config import Config
from neuroplex.experiments import fingerprint, run_episode
from neuroplex.runner import Runner
from neuroplex.simulation import Simulation


@pytest.fixture
def habitat(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    runner = Runner(tmp_path)
    try:
        yield runner, clock
    finally:
        runner.close()


def die(sim):
    sim.world.regrow_at.fill(1e9)
    sim.world.energy = .001
    sim.tick()
    assert not sim.world.alive and sim.world.death_reason == "starvation"


@pytest.mark.parametrize("speed", [1, 10])
def test_auto_life_waits_real_seconds_and_preserves_all_learning(habitat, speed):
    runner, clock = habitat
    sim = runner.sim
    sim.set_stage(4)
    for _ in range(25):
        sim.tick()
    sim.speed = speed
    die(sim)
    sim.brain.learning = False
    before = fingerprint(sim)
    counters = (sim.brain.policy.decisions, sim.brain.policy.updates, sim.brain.policy.goal_updates)
    metrics, lives, elapsed = list(sim.metrics), list(sim.life_records), sim.elapsed
    runner.update_auto_life()
    assert runner.auto_life_status()["remaining"] == 10
    clock.now += 9.9
    runner.update_auto_life()
    assert sim.life == 1 and not sim.world.alive
    clock.now += .2
    runner.update_auto_life()
    assert sim.life == 2 and sim.world.alive and sim.world.time == 0
    assert sim.world.stage == 4 and sim.speed == speed and sim.curriculum
    assert not sim.brain.learning and fingerprint(sim) == before
    assert (sim.brain.policy.decisions, sim.brain.policy.updates, sim.brain.policy.goal_updates) == counters
    assert (list(sim.metrics), list(sim.life_records), sim.elapsed) == (metrics, lives, elapsed)
    assert not np.any(sim.memory.traces)
    assert not runner.auto_life_status()["waiting"]
    runner.update_auto_life()
    assert sim.life == 2  # exactly one body
    restored = Simulation.load(runner.checkpoint)
    assert restored.life == 2 and fingerprint(restored) == before
    previous = Simulation.load(runner.checkpoint.with_suffix(".previous.npz"))
    assert not previous.world.alive and fingerprint(previous) == before


def test_pause_freezes_countdown_disable_cancels_and_delay_change_restarts(habitat):
    runner, clock = habitat
    sim = runner.sim
    die(sim)
    runner.update_auto_life()
    clock.now += 3
    sim.paused = True
    runner.update_auto_life(allow_restart=False)
    assert runner.auto_life_status()["remaining"] == 7
    clock.now += 1000
    runner.update_auto_life()
    assert sim.life == 1 and runner.auto_life_status()["remaining"] == 7
    sim.paused = False
    runner.update_auto_life(allow_restart=False)
    clock.now += 2
    runner.update_auto_life()
    assert runner.auto_life_status()["remaining"] == 5
    sim.auto_life = False
    runner.update_auto_life(allow_restart=False)
    clock.now += 1000
    runner.update_auto_life()
    assert sim.life == 1 and runner.auto_life_status()["remaining"] is None
    sim.auto_life = True
    runner.update_auto_life(allow_restart=False)
    assert runner.auto_life_status()["remaining"] == 10
    clock.now += 8
    sim.auto_life_delay = 3
    runner.update_auto_life(allow_restart=False)
    assert runner.auto_life_status()["remaining"] == 3
    clock.now += 3
    runner.update_auto_life()
    assert sim.life == 2


def test_manual_new_life_clears_pending_restart(habitat):
    runner, clock = habitat
    die(runner.sim)
    runner.update_auto_life()
    runner.sim.new_life()
    runner.update_auto_life(allow_restart=False)
    clock.now += 100
    runner.update_auto_life()
    assert runner.sim.life == 2 and not runner.auto_life_status()["waiting"]


def test_service_restart_gives_dead_checkpoint_a_fresh_delay(tmp_path):
    sim = Simulation()
    die(sim)
    sim.auto_life_delay = 27
    sim.paused = True
    sim.save(tmp_path / "checkpoint.npz")
    runner = Runner(tmp_path)
    try:
        assert runner.sim.life == 1
        assert runner.auto_life_status() == {"remaining": 27, "paused": True, "waiting": True}
    finally:
        runner.close()


def test_settings_roundtrip_and_old_v3_save_gets_safe_defaults(tmp_path):
    sim = Simulation()
    sim.auto_life = False
    sim.auto_life_delay = 42
    path = tmp_path / "checkpoint.npz"
    sim.save(path)
    loaded = Simulation.load(path)
    assert not loaded.auto_life and loaded.auto_life_delay == 42
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    metadata = json.loads(str(data["metadata"]))
    metadata["simulation"].pop("auto_life")
    metadata["simulation"].pop("auto_life_delay")
    data["metadata"] = np.array(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
    loaded = Simulation.load(path)
    assert loaded.auto_life and loaded.auto_life_delay == 10
    assert fingerprint(loaded) == fingerprint(sim)


def test_evaluation_still_ends_at_death_even_with_auto_life_on():
    sim = Simulation(Config(curriculum_enabled=False))
    sim.brain.learning = False
    sim.world.regrow_at.fill(1e9)
    sim.world.energy = .001
    result = run_episode(sim, 60)
    assert sim.auto_life and sim.life == 1
    assert not result["alive"] and result["survival_seconds"] == sim.config.world_dt
    assert result["weights_frozen"]


def test_runtime_error_blocks_auto_restart(habitat):
    runner, clock = habitat
    die(runner.sim)
    runner.update_auto_life()
    clock.now += 50
    runner.error = "Test failure"
    runner.update_auto_life()
    assert runner.sim.life == 1 and runner.auto_life_status()["paused"]


def test_stream_caches_are_shared_compact_and_dont_change_physics(habitat):
    runner, clock = habitat
    runner.sim.record_metrics()
    before = runner.sim.snapshot()
    packets = runner.stream_messages({})
    assert {kind for kind, _, _ in packets} == {"frame", "telemetry", "history", "charts"}
    frames = {kind: json.loads(payload) for kind, _, payload in packets}
    assert "brain" not in frames["frame"] and "metrics" not in frames["frame"]
    assert "retina" not in frames["frame"]["world"]
    assert "metrics" not in frames["telemetry"] and "history" not in frames["telemetry"]
    assert "history" not in frames["charts"]
    assert frames["frame"]["world"]["food"] == before["world"]["food"]
    assert len(packets[0][2]) < len(json.dumps(before)) / 2
    assert runner.stream_messages({}) == packets  # a second viewer reuses serialized objects
    seen = {kind: stamp for kind, stamp, _ in packets}
    assert runner.stream_messages(seen) == []
    clock.now += .051
    updates = runner.stream_messages(seen)
    assert [kind for kind, _, _ in updates] == ["frame"]
    assert runner.sim.snapshot() == before
    clock.now += 5
    assert "charts" not in [kind for kind, _, _ in runner.stream_messages(seen)]
    runner.sim.record_metrics()
    assert "charts" in [kind for kind, _, _ in runner.stream_messages(seen)]
