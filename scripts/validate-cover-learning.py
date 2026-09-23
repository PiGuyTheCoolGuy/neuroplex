"""Bounded, reproducible copy-only training and held-out audits. No live data writes."""

import argparse
import json
from pathlib import Path
import tempfile
import time

from neuroplex.experiments import fingerprint, shelter_practice
from neuroplex.lab import ExperimentRequest
from neuroplex.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = ExperimentRequest(kind="shelter", seed=args.seed, episodes=args.episodes,
                                seconds=args.seconds, trials=args.trials).model_dump()
    source = Simulation()
    before = fingerprint(source)
    started = time.monotonic()

    def progress(update, **kwargs):
        if "phase" in update:
            print(json.dumps(update), flush=True)

    with tempfile.TemporaryDirectory(prefix="neuroplex-cover-") as directory:
        result = shelter_practice(source, request, Path(directory), progress)
        candidate = Simulation.load(Path(directory) / "champion.npz")
        learner = candidate.brain.policy.cover_learner
        result["learned_feature_updates"] = learner.updates.tolist()
        result["skill_updates"] = candidate.brain.policy.skill_updates.tolist()
    assert fingerprint(source) == before
    result.update(source_unchanged=True, wall_seconds=time.monotonic() - started,
                  request=request, limits="One training seed; short separate frozen audits, not proof of planned shelters or long-term survival. Development CPU, not the OptiPlex. No candidate bundled or automatically adopted.")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"practice_before": result["baseline_audit"]["summary"],
                      "practice_after": result["summary"],
                      "habitat_before": result["baseline_habitat_audit"]["summary"],
                      "habitat_after": result["habitat_audit"]["summary"],
                      "feature_updates": result["learned_feature_updates"],
                      "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
