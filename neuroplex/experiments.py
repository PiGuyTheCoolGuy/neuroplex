"""Bounded, sequential evaluation and evolution; invoked by the dashboard or CLI.

Evolution inherits learned weights (Lamarckian inheritance) and mutates weights
and learning parameters. Selection worlds are separate from training worlds;
the final audit also uses worlds never used for selection.
"""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from .simulation import Simulation
from .policy import ESCAPE_START, BUILD_START

STRUCTURE = ("pre", "post", "inhibitory", "plastic", "weights", "initial_weights")
LEARNED = ("values", "visits", "goal_values", "goal_visits", "skill_updates")
GENES = {"policy_learning_rate": (0.05, 0.5), "policy_trace_decay": (0.1, 0.8),
         "exploration_decay_decisions": (500.0, 5000.0), "memory_seconds": (3.0, 20.0)}
CANCELLED = threading.Event()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fingerprint(sim):
    digest = hashlib.sha256()
    for name in STRUCTURE:
        digest.update(getattr(sim.brain, name).tobytes())
    for name in LEARNED:
        digest.update(getattr(sim.brain.policy, name).tobytes())
    return digest.hexdigest()


def copy_model(source, destination):
    for name in STRUCTURE:
        getattr(destination.brain, name)[:] = getattr(source.brain, name)
    for name in LEARNED:
        getattr(destination.brain.policy, name)[:] = getattr(source.brain.policy, name)
    for name in ("updates", "bootstrap_updates", "source", "goal_updates"):
        setattr(destination.brain.policy, name, getattr(source.brain.policy, name))
    destination.brain.policy.reset_activity()


def fresh_trial(source, seed, stage, learning=False):
    config = replace(source.config, seed=seed, habitat_stage=stage, curriculum_enabled=False,
                     shaping_scale=source.config.shaping_scale if learning else 0.0)
    trial = Simulation(config)
    copy_model(source, trial)
    trial.brain.learning = learning
    trial.curriculum = False
    trial.auto_evaluate = False
    return trial


def run_episode(sim, seconds, progress=None):
    before = fingerprint(sim)
    started = time.monotonic()
    totals = np.zeros(3)
    for step in range(round(seconds / sim.config.world_dt)):
        if CANCELLED.is_set():
            raise InterruptedError("Experiment cancelled")
        sim.tick()
        totals += [sim.world.energy, sim.world.hydration, sim.world.health]
        if progress and step % 100 == 0:
            progress(sim.world.time)
        if not sim.world.alive:
            break
    frozen = before == fingerprint(sim)
    if not sim.brain.learning and not frozen:
        raise AssertionError("Frozen evaluation modified learned parameters")
    w = sim.world
    means = totals / max(sim.ticks, 1)
    return {"seed": sim.config.seed, "stage": w.stage, "requested_seconds": seconds,
            "survival_seconds": w.time, "alive": w.alive, "cause": w.death_reason,
            "food": w.eaten, "drinks": w.drinks, "attacks": w.attacks, "damage": w.damage_taken,
            "food_per_minute": 60 * w.eaten / max(w.time, sim.config.world_dt),
            "energy": w.energy, "hydration": w.hydration, "health": w.health,
            "mean_energy": float(means[0]), "mean_hydration": float(means[1]), "mean_health": float(means[2]),
            "learning": sim.brain.learning, "weights_frozen": frozen,
            "shaping_enabled": sim.config.shaping_scale > 0,
            "exploration_floor": sim.config.exploration_floor,
            "wall_seconds": time.monotonic() - started}


def escape_trial(source, seed, learning):
    """Curriculum scenarios, not demonstrations: wall/corner starts, random heading."""
    config = replace(source.config, seed=seed, food_count=0, water_enabled=False,
                     predator_count=1, block_count=0, habitat_stage=3, curriculum_enabled=False,
                     shaping_scale=source.config.shaping_scale if learning else 0.0)
    trial = Simulation(config)
    copy_model(source, trial)
    trial.brain.learning = learning
    trial.auto_life = trial.auto_evaluate = False
    rng = np.random.default_rng(seed + 6000)
    w = trial.world
    margin = float(rng.uniform(3.5, 6))
    positions = [(margin, margin), (config.world_width - margin, margin),
                 (margin, config.world_height - margin), (config.world_width - margin, config.world_height - margin),
                 (margin, config.world_height / 2), (config.world_width / 2, margin),
                 (config.world_width - margin, config.world_height / 2), (config.world_width / 2, config.world_height - margin)]
    w.x, w.y = positions[int(rng.integers(len(positions)))]
    w.heading = float(rng.uniform(-np.pi, np.pi))
    inward = np.array([config.world_width / 2 - w.x, config.world_height / 2 - w.y])
    inward /= np.linalg.norm(inward)
    w.predators[0] = np.array([w.x, w.y]) + inward * rng.uniform(7, 12)
    w.predator_ready_at[0] = 0
    w.sense()
    return trial


def escape_practice(source, request, directory, report):
    """Train copies; retain food/water/build tables and recurrent weights exactly.

    Only obstacle-aware escape values and goal selection transfer back to the
    candidate. No correct actions, path planner, or steering rule supplies labels.
    Independent frozen audit seeds are never used to choose a candidate.
    """
    learner = fresh_trial(source, request["seed"], source.world.stage, learning=True)
    seconds = min(60.0, request["seconds"])
    for episode in range(request["episodes"]):
        if CANCELLED.is_set():
            raise InterruptedError("Experiment cancelled")
        report({"phase": "wall and corner escape practice", "episode": episode + 1}, force=True)
        trial = escape_trial(learner, request["seed"] + episode * 101, learning=True)
        run_episode(trial, seconds, lambda age: report({"episode_seconds": age}))
        previous_updates = int(learner.brain.policy.skill_updates[2])
        for name in ("values", "visits"):
            getattr(learner.brain.policy, name)[ESCAPE_START:BUILD_START] = getattr(trial.brain.policy, name)[ESCAPE_START:BUILD_START]
        for name in ("goal_values", "goal_visits"):
            getattr(learner.brain.policy, name)[:] = getattr(trial.brain.policy, name)
        learner.brain.policy.skill_updates[2] = trial.brain.policy.skill_updates[2]
        learner.brain.policy.updates += int(trial.brain.policy.skill_updates[2]) - previous_updates
        learner.brain.policy.goal_updates = trial.brain.policy.goal_updates
    learner.brain.policy.source = f"escape-practice:{request['episodes']}"
    seeds = [request["seed"] + 3_000_001 + i * 103 for i in range(request["trials"])]
    reports = []
    for model in (source, learner):
        records = []
        for i, seed in enumerate(seeds):
            report({"phase": "frozen escape audit", "trial": i + 1}, force=True)
            records.append(run_episode(escape_trial(model, seed, learning=False), seconds))
        reports.append({"seeds": seeds, "trials": records, "summary": aggregate(records)})
    learner.save(directory / "champion.npz")
    return {"summary": reports[1]["summary"], "audit": reports[1], "baseline_audit": reports[0],
            "champion_available": True, "episodes": request["episodes"], "seconds_per_episode": seconds,
            "method": "Online escape and goal learning in wall/corner scenarios; food, water, construction and SNN weights retained; no action labels"}


def aggregate(records):
    return {"trials": len(records), "survived": sum(r["alive"] for r in records),
            "mean_survival_seconds": float(np.mean([r["survival_seconds"] for r in records])),
            "mean_food_per_minute": float(np.mean([r["food_per_minute"] for r in records])),
            "mean_drinks": float(np.mean([r["drinks"] for r in records])),
            "mean_damage": float(np.mean([r["damage"] for r in records])),
            "all_weights_frozen": all(r["weights_frozen"] for r in records)}


def evaluate(source, seconds, seeds, stage, progress=None):
    original = fingerprint(source)
    records = []
    for i, seed in enumerate(seeds):
        trial = fresh_trial(source, seed, stage)
        callback = (lambda age, i=i: progress(i, age)) if progress else None
        records.append(run_episode(trial, seconds, callback))
    assert fingerprint(source) == original, "Evaluation changed its source model"
    return {"summary": aggregate(records), "trials": records, "source_fingerprint": original,
            "seeds": seeds, "stage": stage, "seconds": seconds}


def mutate(sim, rng):
    changes = {name: float(np.clip(getattr(sim.config, name) * np.exp(rng.normal(0, 0.15)), low, high))
               for name, (low, high) in GENES.items()}
    sim.config = replace(sim.config, **changes)
    sim.brain.config = sim.brain.policy.config = sim.memory.config = sim.config
    sim.world.config = replace(sim.world.config, **changes)
    for name in ("values", "goal_values"):
        values = getattr(sim.brain.policy, name)
        values += rng.normal(0, 0.20, values.shape)
        np.clip(values, -100, 200, out=values)
    weights = sim.brain.weights
    weights[sim.brain.plastic] += rng.normal(0, 0.002, int(sim.brain.plastic.sum()))
    weights[sim.brain.plastic] = np.clip(weights[sim.brain.plastic], 0, sim.config.weight_max)
    return changes


def fitness(records):
    # Survival dominates; normalized body condition breaks survival ties.
    return float(np.mean([100 * r["survival_seconds"] / r["requested_seconds"]
                           + (r["energy"] + r["hydration"] + r["health"]) / 30
                           + 0.05 * r["food_per_minute"] - 0.02 * r["damage"] for r in records]))


def evolve(source, request, directory, report):
    rng = np.random.default_rng(request["seed"])
    stage = request["stage"]
    parents = [source]
    generations = []
    selection_seeds = [request["seed"] + 1_000_003, request["seed"] + 1_000_033]
    champion = source
    for generation in range(1, request["generations"] + 1):
        candidates = []
        for candidate in range(request["population"]):
            # Keep the previous champion unchanged as a true elitist candidate.
            child = fresh_trial(parents[candidate % len(parents)], request["seed"] + generation * 100,
                                stage, learning=True)
            genes = {}
            if candidate:
                genes = mutate(child, rng)
                report({"phase": "training", "generation": generation, "candidate": candidate + 1})
                run_episode(child, request["seconds"], lambda age: report({"episode_seconds": age}))
            report({"phase": "selection", "generation": generation, "candidate": candidate + 1})
            measured = evaluate(child, request["evaluation_seconds"], selection_seeds, stage,
                                lambda trial, age: report({"trial": trial + 1, "episode_seconds": age}))
            score = fitness(measured["trials"])
            record = {"generation": generation, "candidate": candidate + 1, "fitness": score,
                      "genes": genes, "selection": measured["summary"]}
            candidates.append((score, child, record))
            with (directory / "results.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        candidates.sort(key=lambda item: item[0], reverse=True)
        parents = [entry[1] for entry in candidates[:2]]
        champion = parents[0]
        champion.brain.policy.source = f"evolution:generation-{generation}"
        champion.save(directory / "champion.npz")
        generations.append({"generation": generation, "best_fitness": candidates[0][0],
                            "mean_fitness": float(np.mean([entry[0] for entry in candidates])),
                            "selection": candidates[0][2]["selection"]})
        report({"generations": generations, "champion_available": True}, force=True)
    report({"phase": "unseen audit"}, force=True)
    audit = evaluate(champion, request["evaluation_seconds"],
                     [request["seed"] + 2_000_003, request["seed"] + 2_000_033], stage,
                     lambda trial, age: report({"trial": trial + 1, "episode_seconds": age}))
    baseline = evaluate(source, request["evaluation_seconds"], audit["seeds"], stage)
    return {"generations": generations, "audit": audit, "baseline_audit": baseline,
            "summary": audit["summary"], "selection_seeds": selection_seeds,
            "champion_available": True,
            "method": "Sequential elitist evolution; inherited learned weights; mutated learning parameters and weights"}


def execute(request, source_path, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "kind": request["kind"], "phase": "starting", "started_at": time.time(),
              "stage": request["stage"], "request": request}
    last_write = 0.0

    def report(update, force=False):
        nonlocal last_write
        status.update(update)
        now = time.monotonic()
        if force or now - last_write >= 0.5:
            atomic_json(directory / "status.json", status)
            last_write = now

    report({}, force=True)
    try:
        source = Simulation.load(source_path)
        if request["kind"] == "evaluate":
            seeds = [request["seed"] + 10_007 + i * 101 for i in range(request["trials"])]
            result = evaluate(source, request["seconds"], seeds, request["stage"],
                              lambda trial, age: report({"phase": "frozen evaluation", "trial": trial + 1,
                                                          "episode_seconds": age}))
        elif request["kind"] == "escape":
            result = escape_practice(source, request, directory, report)
        else:
            result = evolve(source, request, directory, report)
        atomic_json(directory / "result.json", result)
        report({**result, "state": "completed", "phase": "finished", "finished_at": time.time()}, force=True)
    except InterruptedError:
        report({"state": "cancelled", "phase": "cancelled", "finished_at": time.time()}, force=True)
    except Exception as exc:
        report({"state": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": time.time()}, force=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="JSON experiment settings")
    parser.add_argument("--source", type=Path, required=True, help="snapshot to copy, never overwrite")
    parser.add_argument("--output", type=Path, required=True, help="new experiment output directory")
    args = parser.parse_args()
    if hasattr(os, "nice"):
        os.nice(10)
    signal.signal(signal.SIGTERM, lambda *_: CANCELLED.set())
    request = json.loads(args.request.read_text())
    # The same bounds apply to CLI and dashboard jobs.
    from .lab import ExperimentRequest
    request = ExperimentRequest(**request).model_dump()
    if any((args.output / name).exists() for name in ("source.npz", "result.json", "champion.npz")) and args.source.parent != args.output:
        parser.error("output must be a new experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    # A crashed foreground server can leave a worker alive. Do not let a new
    # server start a second CPU experiment against the same experiment directory.
    with (args.output.parent / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another experiment worker is still running in this directory")
        execute(request, args.source, args.output)


if __name__ == "__main__":
    main()
