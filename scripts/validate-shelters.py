"""Reproducible CPU-only escape audit and a short construction/foraging smoke run.

Uses temporary copies, never the live data directory. No teacher actions.
"""

import argparse
import json
from pathlib import Path
import tempfile
import time

from neuroplex.config import Config
from neuroplex.experiments import escape_practice, fingerprint
from neuroplex.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 200 or not 5 <= args.seconds <= 60:
        parser.error("episodes must be 1–200 and seconds 5–60")
    source = Simulation()
    original = fingerprint(source)
    last_episode = 0

    def progress(update, **kwargs):
        nonlocal last_episode
        episode = update.get("episode", 0)
        if episode and (episode == 1 or episode % 10 == 0) and episode != last_episode:
            print(json.dumps({"phase": "practice", "episode": episode}), flush=True)
            last_episode = episode

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="neuroplex-shelter-validation-") as directory:
        request = {"seed": args.seed, "seconds": args.seconds, "episodes": args.episodes, "trials": 5}
        result = escape_practice(source, request, Path(directory), progress)
    assert fingerprint(source) == original
    smoke = Simulation(Config(seed=71, habitat_stage=1, curriculum_enabled=False))
    for _ in range(round(120 / smoke.config.world_dt)):
        smoke.tick()
        if not smoke.world.alive:
            break
    w = smoke.world
    report = {"escape_practice": result, "source_unchanged": True,
              "scarce_blocks_120s": {"alive": w.alive, "seconds": w.time, "food": w.eaten,
                                      "energy": w.energy, "blocks": len(w.active_blocks),
                                      "distance_pushed": w.push_distance, "best_cover": w.best_shelter,
                                      "construction_reward_earned": w.construction_reward_total,
                                      "build_updates": int(smoke.brain.policy.skill_updates[3])},
              "wall_seconds": time.monotonic() - started,
              "limits": "Development CPU, not OptiPlex. Short escape trials are not long-run survival evidence. Cover mechanics do not prove learned shelter construction."}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
