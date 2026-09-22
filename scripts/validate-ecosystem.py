"""Reproducible bounded behavior checks, not a claim of general survival skill.

Run from the repo: .venv/bin/python scripts/validate-ecosystem.py
"""

import argparse
import json
from pathlib import Path
import resource
import time

from neuroplex.config import Config
from neuroplex.experiments import run_episode, evaluate, atomic_json
from neuroplex.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/ecosystem-smoke-v0.3.json"))
    args = parser.parse_args()
    results = []
    for name, config, seconds in (
        ("water_online", Config(seed=601, habitat_stage=2, curriculum_enabled=False), 300),
        ("predators_online", Config(seed=602, habitat_stage=4, curriculum_enabled=False), 300),
        ("automatic_curriculum", Config(seed=7), 600),
    ):
        sim = Simulation(config)
        last_report = -60

        def progress(age):
            nonlocal last_report
            if age - last_report >= 59:
                last_report = age
                print(json.dumps({"case": name, "seconds": round(age, 1), "stage": sim.world.stage,
                                  "food": sim.world.eaten, "drinks": sim.world.drinks,
                                  "health": sim.world.health}), flush=True)

        record = {"case": name, **run_episode(sim, seconds, progress)}
        record["goal_updates"] = sim.brain.policy.goal_updates
        record["skill_updates"] = sim.brain.policy.skill_updates.tolist()
        record["metrics_samples"] = len(sim.metrics)
        results.append(record)
        print(json.dumps(record), flush=True)
    baseline = Simulation(Config(seed=7))
    frozen = evaluate(baseline, 60, [801, 802], 0)
    artifact = {"software": "0.3.0", "recorded_at": time.time(), "online_runs": results,
                "frozen_food_smoke": frozen, "peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "limits": "One online seed per scenario and two short food trials. New water, predator, memory, and evolution improvements are not established by these checks. Not run on the user's OptiPlex."}
    atomic_json(args.output, artifact)
    print(json.dumps({"output": str(args.output), "peak_rss_mib_linux": artifact["peak_rss_mib_linux"]}), flush=True)


if __name__ == "__main__":
    main()
