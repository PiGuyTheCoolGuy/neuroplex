# Learning validation

Current block physics, escape practice, migration and browser checks are in
[VALIDATION-v0.5.md](VALIDATION-v0.5.md). Older habitat measurements below use older
geometry/resource defaults and are not directly comparable to the expanded world.

Automatic-life and streaming verification is in [VALIDATION-v0.3.1.md](VALIDATION-v0.3.1.md).
Ecosystem verification is in [VALIDATION-v0.3.md](VALIDATION-v0.3.md).
The historical v0.2 measurements below apply to food-only foraging, not to the new
water, predator, memory or evolution features. The food asset itself is unchanged.

## Historical v0.2 learning validation

## What was measured

The trained **motor action values** produce useful foraging on unseen layouts.
This is a hybrid TD-policy / spiking-controller result, not proof that recurrent
STDP alone learned navigation. All runs used the complete LIF simulation and CPU
only; none ran on the user's OptiPlex.

The bundled policy was learned from zero action values in **one continuous
600-second simulation**, seed 7, with 32 food patches. It received 2,400 online
updates, ate 169 food items, and stayed alive. No teacher, demonstrations, automatic
reset, food teleport, energy rescue, or hard-coded food-steering command was used.
Progress reward does encode prior knowledge about facing and approaching food.

The first minute of that training run found 6 food items; the fifth found 24.
This single training trajectory is descriptive, not the main evidence of learning.
The exported values are in `neuroplex/assets/foraging-v0.2.json`.

## Final evaluation

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m neuroplex.benchmark \
  --seconds 300 --seeds 501 502 503 504 505 --require-improvement
```

These five seeds were reserved for the final check after development on other
seeds. Each condition used matching initial food, body, SNN weights, and random
seeds. The **only trained parameters transferred were the 918 action values**.
All policy and recurrent weights were frozen and checked for exact equality at
the end. Progress reward was disabled. Both conditions retained the same 5%
exploration floor; random trajectories can diverge after different choices.
There was no food-coordinate input or action teacher.

Each world had 32 regrowing food patches (the live default has 64), with a
300-simulated-second limit. Death ended that trial; no respawn extended it.

| World seed | Untrained food | Trained food | Untrained alive at 300 s | Trained alive at 300 s | Trained final energy |
| ---: | ---: | ---: | --- | --- | ---: |
| 501 | 18 | 126 | Yes | Yes | 95.06 |
| 502 | 3 | 131 | No | Yes | 95.84 |
| 503 | 5 | 147 | No | Yes | 99.09 |
| 504 | 11 | 126 | No | Yes | 99.21 |
| 505 | 2 | 126 | No | Yes | 97.69 |

Mean food: **7.8 untrained versus 131.2 trained**, about **16.8×** as much in these
trials. Survival at the time limit: **1/5 versus 5/5**. The preset command's
pass condition (all trained agents survive; mean food exceeds 2× control) passed.
This comparison is against untrained values in the *same v0.2 architecture*, not
a matched replay of v0.1. Raw records: [benchmark-v0.2.jsonl](benchmark-v0.2.jsonl).

## A failure found during development

An earlier completely greedy evaluation on seeds 201–205 ate much more food than
its control but survived only 4/5 trials. Seed 202 ate 47 food items, then got stuck
and starved at 171.95 s. That candidate failed the survival gate. Its results are
preserved in [development-v0.2-greedy.jsonl](development-v0.2-greedy.jsonl).

Keeping a 5% exploration floor even with frozen weights resolved that development
case: a 300-second rerun on seed 202 ate 132 food items and remained alive. The final
evaluation above then used different seeds (501–505). Exploration is independent
of food direction, and both final conditions receive the same exploration rate.

## Continued learning in the live default world

A separate default run (64 food patches, seed 7, bundled starting values) continued
learning for **600 simulated seconds**. It ate **331 food items**, stayed alive,
ended at **99.24 energy**, and performed **2,400 new updates**. It did not freeze
learning or reset after the starting policy was loaded. See
[live-run-v0.2.jsonl](live-run-v0.2.jsonl).

This run has a denser world than the training and comparison worlds; its food
count should not be directly interpreted as a learning gain over those runs.

## Software verification

**20 automated tests passed**, covering world/energy rules, causal spike traces,
weight bounds, chosen-action credit, terminal TD handling, shaping-cycle accounting,
frozen weights and counters, motor spikes being necessary for movement, exact
mid-action checkpoint continuation, version-1 migration for living and dead bodies,
permanent original-checkpoint backup, process exclusion, API controls, and live
WebSocket snapshots.

A real Chromium browser check passed at desktop (1440 px) and phone (390 px)
widths: no JavaScript errors or horizontal overflow. It exercised the learned-policy
panel, live update counters, pause, freezing, the 5% frozen exploration indicator,
and saving. Screenshots were visually inspected. Editable package installation and
CLI parsing passed. The existing Ubuntu/systemd workflow needs no new dependency.

## Limits and reproduction

Five worlds and a ten-minute live run are finite tests. They demonstrate learned
food seeking on this simple environment; they do not guarantee indefinite survival,
reliable behavior on every seed, or generalization to changed sensory/motor geometry,
sparse-food worlds, predators, or obstacles. Exploration can still make mistakes.
The Q-table is a deliberate change from relying solely on recurrent STDP.

To reproduce training (separate from the running creature):

```bash
.venv/bin/python -m neuroplex.benchmark --train-seconds 600 --train-seed 7 \
  --food-count 32 --seconds 300 --seeds 501 502 503 504 505 \
  --export-policy data/retrained-policy.json --require-improvement
```

Training prints progress every simulated minute. The export is not automatically
installed over an existing live policy. To watch a new creature learn from zero:

```bash
bash start.sh --port 8001 --data-dir data/from-scratch --untrained --seed 7
```

Use a new data directory. Neither command erases a saved world. Default startup
uses the bundled learned motor values, then continues adapting online. Historical
v0.1 measurements remain in [VALIDATION-v0.1.md](VALIDATION-v0.1.md).
