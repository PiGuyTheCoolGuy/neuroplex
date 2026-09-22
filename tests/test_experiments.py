import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuroplex.config import Config
from neuroplex.simulation import Simulation
from neuroplex.experiments import evaluate, evolve, fingerprint, fresh_trial, mutate
from neuroplex.lab import Laboratory, ExperimentRequest
from neuroplex.runner import Runner
from neuroplex.server import create_app


def test_frozen_trials_preserve_all_learned_arrays_and_source_state():
    source = Simulation(Config(habitat_stage=4))
    for _ in range(8):
        source.tick()
    snapshot = source.snapshot()
    before = fingerprint(source)
    result = evaluate(source, 1, [7001, 7002], 4)
    assert result["summary"]["all_weights_frozen"]
    assert all(not r["shaping_enabled"] and not r["learning"] for r in result["trials"])
    assert fingerprint(source) == before and source.snapshot() == snapshot
    trial = fresh_trial(source, 7011, 4)
    np.testing.assert_array_equal(source.brain.pre, trial.brain.pre)
    np.testing.assert_array_equal(source.brain.post, trial.brain.post)
    assert trial.world.time == 0 and trial.world.alive


def test_mutation_changes_only_child_and_respects_synaptic_signs():
    source = Simulation()
    before = fingerprint(source)
    child = fresh_trial(source, 333, 0, learning=True)
    genes = mutate(child, np.random.default_rng(123))
    assert fingerprint(source) == before and fingerprint(child) != before
    assert genes["memory_seconds"] != source.config.memory_seconds
    assert np.all(child.brain.weights[child.brain.plastic] >= 0)
    np.testing.assert_array_equal(source.brain.weights[~source.brain.plastic], child.brain.weights[~child.brain.plastic])


def test_evolution_keeps_an_elite_and_audits_on_separate_worlds(tmp_path):
    source = Simulation()
    request = {"seed": 555, "stage": 2, "generations": 2, "population": 2,
               "seconds": 0.5, "evaluation_seconds": 0.5}
    result = evolve(source, request, tmp_path, lambda *_args, **_kwargs: None)
    assert len(result["generations"]) == 2
    assert result["generations"][1]["best_fitness"] >= result["generations"][0]["best_fitness"]
    assert not set(result["audit"]["seeds"]) & set(result["selection_seeds"])
    assert result["audit"]["seeds"] == result["baseline_audit"]["seeds"]
    assert result["summary"]["all_weights_frozen"]
    champion = Simulation.load(tmp_path / "champion.npz")
    assert champion.brain.policy.source == "evolution:generation-2"


def test_real_worker_finishes_and_cancels_without_modifying_main_creature(tmp_path):
    source = Simulation()
    before = source.snapshot()
    lab = Laboratory(tmp_path)
    try:
        lab.start(source, {"kind": "evaluate", "seconds": 5.0, "trials": 1, "stage": 3})
        with pytest.raises(ValueError, match="already running"):
            lab.start(source, {"kind": "evaluate"})
        lab.process.wait(timeout=40)
        lab.refresh(force=True)
        assert lab.status["state"] == "completed", lab.status
        assert lab.status["summary"]["all_weights_frozen"]
        assert len(lab.history) == 1
        assert source.snapshot() == before
        lab.start(source, {"kind": "evolve", "seconds": 600.0, "generations": 10})
        lab.cancel()
        lab.close()
        assert lab.status["state"] == "cancelled"
        assert source.snapshot() == before
    finally:
        lab.close()
    reloaded = Laboratory(tmp_path)
    assert len(reloaded.history) == 1 and reloaded.status["state"] == "cancelled"


def test_champion_adoption_requires_death_and_preserves_a_backup(tmp_path):
    runner = Runner(tmp_path)
    try:
        with pytest.raises(ValueError, match="after"):
            runner.adopt_champion()
        job = runner.lab.directory / ("a" * 32)
        job.mkdir()
        champion = Simulation()
        champion.brain.policy.values[0, 0] = 123
        champion.save(job / "champion.npz")
        runner.lab.champion_id = job.name
        runner.sim.world.energy = 0
        runner.sim.world.alive = False
        runner.sim.world.death_code = 1
        runner.adopt_champion()
        assert runner.sim.world.alive and runner.sim.life == 2
        assert runner.sim.brain.policy.values[0, 0] == 123
        assert (tmp_path / "checkpoint.before-evolution-life-1.npz").exists()
    finally:
        runner.close()


def test_api_ecosystem_controls_export_and_job_limits(tmp_path):
    headers = {"X-Neuroplex-Client": "dashboard"}
    with TestClient(create_app(tmp_path)) as client:
        for action, value in (("pause", None), ("stage", 3), ("curriculum", False), ("memory", False), ("auto_evaluate", False)):
            result = client.post("/api/control", headers=headers, json={"action": action, "value": value})
            assert result.status_code == 200
        state = result.json()
        assert state["world"]["water_active"] and len(state["world"]["predators"]) == 1
        assert not state["memory"]["enabled"] and not state["curriculum"]["enabled"]
        assert client.get("/api/metrics.csv").status_code == 200
        assert client.post("/api/experiments", json={}).status_code == 403
        assert client.post("/api/experiments", headers=headers, json={"population": 100}).status_code == 422
        assert client.post("/api/control", headers=headers, json={"action": "stage", "value": 5}).status_code == 422
        assert client.post("/api/control", headers=headers, json={"action": "adopt_champion"}).status_code == 409
