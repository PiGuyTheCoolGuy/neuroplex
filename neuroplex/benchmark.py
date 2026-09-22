"""Train without a teacher; then test learned values on fresh, frozen worlds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .config import Config
from .simulation import Simulation


def train(seed: int, seconds: float, food_count: int = 32, report_progress: bool = False):
    sim = Simulation(Config(seed=seed, food_count=food_count, pretrained_policy=False,
                            curriculum_enabled=False, memory_enabled=False))
    started = time.perf_counter()
    for _ in range(round(seconds / sim.config.world_dt)):
        sim.tick()
        if not sim.world.alive:
            raise RuntimeError(f"Training creature starved at {sim.world.time:.2f}s; no reset or rescue was performed")
        if report_progress and sim.ticks % round(60 / sim.config.world_dt) == 0:
            print(json.dumps({"type": "training_progress", "seconds": round(sim.world.time, 2),
                              "food_eaten": sim.world.eaten, "energy": round(sim.world.energy, 2)}), flush=True)
    metadata = {
        "seed": seed, "food_count": food_count, "simulated_seconds": round(sim.world.time, 6),
        "updates": sim.brain.policy.updates, "food_eaten": sim.world.eaten,
        "alive": sim.world.alive,
        "method": "online TD motor policy; complete LIF simulation; no teacher or resets",
    }
    artifact = {"format": 1, "training": metadata,
                "values": sim.brain.policy.values[:153].tolist(), "visits": sim.brain.policy.visits[:153].tolist()}
    return artifact, {"type": "training", **metadata, "wall_seconds": round(time.perf_counter() - started, 3)}


def evaluate(seed: int, seconds: float, values: np.ndarray | None, food_count: int = 32):
    # Both conditions get identical physics, initial SNN weights, RNG seeds, and
    # food. Only learned action values differ. Reward shaping is disabled here.
    sim = Simulation(Config(seed=seed, food_count=food_count, pretrained_policy=False, shaping_scale=0,
                            curriculum_enabled=False, memory_enabled=False))
    sim.brain.learning = False
    if values is not None:
        sim.brain.policy.values[:153] = values
    initial_values = sim.brain.policy.values.copy()
    initial_weights = sim.brain.weights.copy()
    started = time.perf_counter()
    total_energy = 0.0
    first_food = None
    for _ in range(round(seconds / sim.config.world_dt)):
        sim.tick()
        total_energy += sim.world.energy
        if first_food is None and sim.world.eaten:
            first_food = round(sim.world.time, 2)
        if not sim.world.alive:
            break
    frozen = (np.array_equal(initial_values, sim.brain.policy.values)
              and np.array_equal(initial_weights, sim.brain.weights))
    if not frozen:
        raise AssertionError("Evaluation changed supposedly frozen weights")
    return {
        "type": "evaluation", "seed": seed,
        "condition": "trained_frozen" if values is not None else "untrained_frozen",
        "food_count": food_count, "seconds_requested": seconds,
        "seconds_survived": round(sim.world.time, 2), "alive_at_limit": sim.world.alive,
        "food_eaten": sim.world.eaten, "energy": round(sim.world.energy, 2),
        "mean_energy": round(total_energy / max(sim.ticks, 1), 2),
        "first_food_seconds": first_food, "weights_frozen": frozen, "shaping_enabled": False,
        "exploration_rate": float(sim.brain.policy.exploration(False)),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=180, help="seconds per evaluation world")
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503, 504, 505])
    parser.add_argument("--food-count", type=int, default=32)
    parser.add_argument("--train-seconds", type=float, default=0,
                        help="0: evaluate bundled policy; positive: train a fresh policy first")
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--export-policy", type=Path, help="save the trained policy as JSON")
    parser.add_argument("--require-improvement", action="store_true",
                        help="exit nonzero unless all trained agents survive and mean food exceeds 2x control")
    args = parser.parse_args()
    if args.seconds <= 0 or args.train_seconds < 0 or args.food_count < 0:
        parser.error("seconds must be positive, training seconds and food count nonnegative")
    if args.export_policy and not args.train_seconds:
        parser.error("--export-policy requires --train-seconds")
    if args.train_seconds:
        artifact, record = train(args.train_seed, args.train_seconds, args.food_count, report_progress=True)
        print(json.dumps(record), flush=True)
        if args.export_policy:
            args.export_policy.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.export_policy.with_suffix(".tmp")
            temporary.write_text(json.dumps(artifact, indent=2) + "\n")
            temporary.replace(args.export_policy)
    else:
        artifact = json.loads((Path(__file__).parent / "assets" / "foraging-v0.2.json").read_text())
    if artifact["training"]["seed"] in args.seeds:
        parser.error("evaluation seeds must differ from the training seed")
    values = np.asarray(artifact["values"], dtype=np.float64)
    results = []
    for seed in args.seeds:
        for policy in (None, values):
            record = evaluate(seed, args.seconds, policy, args.food_count)
            results.append(record)
            print(json.dumps(record), flush=True)
    controls, trained = results[::2], results[1::2]
    baseline_mean = float(np.mean([r["food_eaten"] for r in controls]))
    trained_mean = float(np.mean([r["food_eaten"] for r in trained]))
    passed = all(r["alive_at_limit"] for r in trained) and trained_mean > 2 * baseline_mean
    print(json.dumps({"type": "summary", "training_seed": artifact["training"]["seed"],
                      "untrained_mean_food": baseline_mean, "trained_mean_food": trained_mean,
                      "trained_survived": sum(r["alive_at_limit"] for r in trained),
                      "trials": len(trained), "gate_passed": passed}), flush=True)
    if args.require_improvement and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
