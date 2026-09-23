# v0.4 validation: materials, scarcity and escape practice

Run on the development Linux CPU, **not the user's OptiPlex**. No claim of reliable
autonomous shelter planning or universal predator survival follows from these checks.
Raw results: [shelter-smoke-v0.4.json](shelter-smoke-v0.4.json).

## Automated checks

- **64 Python tests pass**: previous learning, ecology, evaluation, auto-life and
  streaming checks plus block/visibility physics, geometry credit, policy context,
  practice isolation, selective adoption and checkpoint migration.
- **7 JavaScript tests pass**: pose/angle interpolation, bounded buffering,
  disconnect behavior, pushed-block animation and resetting on habitat resizing.
- Format-4 checkpoints resume exactly, including material positions, credit and
  action state. A fixture with the actual v3 shapes upgrades once, preserves all old
  learned values/synapses and needs, scales the world, and archives the original bytes.
- Physics tests reject overlap, block chains, out-of-bounds pushes and burying food;
  blocks hide targets and stop predators. Cover credit requires multiple blocks
  and an exit; a boundary corner and a closed box do not qualify. Repeated touches
  or undo/redo cannot repeatedly earn the same per-material improvement credit.
- Escape contexts distinguish open space, wall and corner before contact. A
  deliberately chosen learned action/goal is still obeyed: there is no hidden
  emergency steering controller. Material availability and building choices are tested.
- Escape-practice tests preserve the source, freeze audit parameters, separate seeds,
  and retain current food/water/build/SNN learning when adopting after the main
  creature has continued learning since the source snapshot was taken.

The two existing test dependency deprecation warnings remain; no new dependency
is required by this release.
`bash setup.sh` also successfully installed the editable v0.4.0 package.

## Bounded learning measurement

Reproduction:

```bash
.venv/bin/python scripts/validate-shelters.py
```

Training seed 907, 40 episodes × 12 simulated seconds. Wall/corner scenarios have
one approaching predator, random headings, no food/water distraction and no blocks.
The full spiking body runs; no correct actions are supplied. Only escape and goal
learning accumulates into the candidate. No training result is installed in the
live habitat or bundled as new pretrained weights.

Candidate and original then run on the same five unseen seeds, all learned
parameters frozen and progress shaping disabled:

| Measure | Starting model | After practice |
| --- | ---: | ---: |
| Survived 12 seconds | 5/5 | 5/5 |
| Mean damage | 27 | 15 |
| Total attacks across trials | 9 | 5 |

Seeds: 3000908, 3001011, 3001114, 3001217, 3001320. Damage improves on three worlds
and ties on two. The audit did not select a candidate or feed further training.
This is a small, short comparison using a starting food-trained brain, not the
user's saved brain. It does not isolate each escape-code change or prove long-term
survival, success among blocks, or generalization to every predator arrangement.

## Larger-world material smoke check

Separate seed 71, displayed stage 2 (CLI stage 1), 120 simulated seconds, learning
enabled, 144×90 world, 24 blocks and stage-reduced food:

| Measure | Result |
| --- | ---: |
| Alive at 120 seconds | Yes |
| Food eaten | 11 |
| Final energy | 87.59 |
| Distance blocks pushed | 19.59 units |
| Building-policy updates | 189 |
| Construction reward earned | 8.00 |
| Highest available cover score | 0.625 |

These show actual movement of material and increases in the geometric cover
objective. They do **not** show a learned house, deliberate U-shaped construction,
or use of a completed shelter. Initial natural cover is included in the highest
available score and earns no bonus. The reward counter was checked in a repeat
of this same deterministic smoke run, with food and push-distance results identical.

Practice, its audit, and the original smoke run together took approximately
58.4 wall-clock seconds here. OptiPlex timing can differ substantially.

## Browser checks

A real Chromium session exercised a live server at desktop 1440 px and mobile
390 px, including death/restart, pause, stage changes, block rendering, world
dimensions, the escape-practice form and worker, result comparison and protection
against adopting while alive. No JavaScript errors or horizontal overflow.

During a short moving-view sample: **19.4 geometry updates/s**, **59.7 rendered
frames/s**, approximately **2,815 bytes per geometry message**. Pushed-block
interpolation also has a deterministic unit test. Performance still depends on
CPU/network/browser; a partially paused measurement window will show fewer draws.

## Remaining work

Measure longer, paired survival trials in the actual scarce/material habitat
before concluding the organism handles predators well. Construction needs stronger
site memory, temporal planning, and a decision to occupy useful cover. Weather,
food storage, repairs and lasting structures across lives are future capabilities.
Physical cover scoring and improved escape representations are foundations for
those behaviors, not substitutes for demonstrating them.
