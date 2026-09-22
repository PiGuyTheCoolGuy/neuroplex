"""Reproducible learning-vs-frozen comparisons; never touches saved worlds."""

import argparse
import json
import time

from .config import Config
from .simulation import Simulation


def run(seed: int, seconds: float, learning: bool):
    sim = Simulation(Config(seed=seed))
    sim.brain.learning = learning
    started = time.perf_counter()
    for _ in range(round(seconds / sim.config.world_dt)):
        sim.tick()
        if not sim.world.alive:
            break
    elapsed = time.perf_counter() - started
    return {
        "seed": seed, "learning": learning, "seconds_requested": seconds,
        "seconds_survived": round(sim.world.time, 2), "alive_at_limit": sim.world.alive,
        "food_eaten": sim.world.eaten, "energy": round(sim.world.energy, 2),
        "weight_change": sim.brain.summary()["weight_change"],
        "wall_seconds": round(elapsed, 3),
        "simulated_seconds_per_wall_second": round(sim.world.time / max(elapsed, 1e-9), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    for seed in args.seeds:
        for learning in (False, True):
            print(json.dumps(run(seed, args.seconds, learning)), flush=True)


if __name__ == "__main__":
    main()
