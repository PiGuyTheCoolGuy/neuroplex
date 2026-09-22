# v0.1 validation

These are observations from the development environment, not performance claims
about the user's OptiPlex. Default configuration; Python 3.12; dependencies pinned
in `requirements.txt` and `requirements-dev.txt`.

## Automated behavior checks

`python -m pytest -q`: **12 tests passed**.

The tests cover:

- Metabolic starvation with no food, terminal death, and exact hunger-change reward.
- Food consumed once, energy restored, and regrowth delayed correctly.
- Egocentric retina excludes out-of-range and out-of-view food.
- Collision constrains the body without supplying an automatic turn.
- Causal versus reverse spike order produces the expected eligibility signs.
- Reward direction changes eligible weights; zero-eligibility synapses are untouched.
- Frozen learning keeps weights fixed; plasticity preserves inhibitory signs and bounds.
- Checkpoint restart reproduces subsequent brain arrays, world, and metrics **exactly**.
- Pause/death retain state; an explicit new life preserves learned weights.
- Invalid checkpoint rejection and exclusion of two processes sharing one world.
- Dashboard assets, HTTP controls, validation, WebSocket snapshots, and shutdown save.

`bash setup.sh` and `pip check` also succeeded. Shell scripts passed `bash -n`, and
dashboard JavaScript passed `node --check`. The systemd service generator was
exercised with a temporary output path, and the generated unit passed
`systemd-analyze verify`. An actual systemd login/boot session is
not available here, so enabling the service on the OptiPlex remains a deployment
step rather than a tested remote action.

## Learning versus frozen controls

Command:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m neuroplex.benchmark --seconds 180 --seeds 7 8 9
```

For each seed, both conditions start with the same world and random brain. Frozen
means synaptic weights are held fixed; neural activity, intrinsic threshold
adaptation, and exploration continue. Each trial is a fresh isolated simulation
and never reads or writes `data/checkpoint.npz`.

| Seed | Learning | Food eaten | Energy at 180 s | Alive at 180 s | Mean absolute weight change |
| ---: | --- | ---: | ---: | --- | ---: |
| 7 | Frozen | 22 | 89.84 | Yes | 0 |
| 7 | On | 35 | 94.84 | Yes | 0.01267 |
| 8 | Frozen | 21 | 87.63 | Yes | 0 |
| 8 | On | 11 | 33.02 | Yes | 0.01364 |
| 9 | Frozen | 18 | 97.27 | Yes | 0 |
| 9 | On | 18 | 89.97 | Yes | 0.01757 |

The result is mixed: **the experiment has functioning online plasticity, but these
trials do not demonstrate a consistent learning advantage**. All trials are
censored at 180 seconds, so they do not establish complete lifespans. Dense food
and intrinsic exploration can keep even frozen networks alive. Trajectories diverge
when weights change, so one favorable trajectory is insufficient evidence of a
learned food-seeking strategy. Raw results are in `benchmark-v0.1.jsonl`.

A preliminary seed-7 run lasted 60 simulated seconds, ate 11 food items, and ended
with 71.73 energy. It is a functionality check, not additional independent evidence
of learning. We did not tune defaults on the three-seed comparison above.

## Resource observations

The six headless 180-second trials took approximately **23–37 wall-clock seconds**
each in the development environment (about 4.9–7.9× simulated speed). Concurrent
work and CPU differences affect these timings. A separate short headless run had
about **32.4 MiB peak resident memory**, excluding a browser and the web server.
The default arrays are small; high playback speed, not memory capacity, is the
likely constraint on an older OptiPlex. Begin at 1× and use the actual-speed readout.

No GPU was used. No experiment was run directly on the user's machine.

## Browser check

The dashboard passed a real Chromium browser check at desktop (1440 px) and
phone (390 px) widths. The smoke check exercised the live stream, pause, learning
freeze, speed selection, and save controls. There were no JavaScript errors or
mobile horizontal overflow; both screenshots were visually inspected. Browser
testing dependencies are development tools only; Ubuntu deployment requires no
Node.js or browser installation on the server.
