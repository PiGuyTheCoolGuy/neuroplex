import numpy as np
import pytest

from neuroplex.brain import Brain
from neuroplex.runner import Runner
from neuroplex.simulation import Simulation


def test_checkpoint_resumes_identically_including_random_streams(tmp_path):
    sim = Simulation()
    for _ in range(30):
        sim.tick()
    path = tmp_path / "checkpoint.npz"
    sim.save(path)
    resumed = Simulation.load(path)
    for _ in range(40):
        sim.tick()
        resumed.tick()
    for name in Brain.ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(sim.brain, name), getattr(resumed.brain, name))
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
