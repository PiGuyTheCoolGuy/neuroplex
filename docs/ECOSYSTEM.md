# v0.3 design and experimental limits

This describes the v0.3 foundation. v0.4 changes dimensions, resource density,
observations, policy tables and checkpoints; see [SHELTERS.md](SHELTERS.md).

## What is programmed, transferred, and learned

The recurrent LIF network keeps the original 500 neurons and 16,000 synapses.
Food's 153 × 6 table is preserved, including the user's own updated values on
save migration. Water gets a copy of those motor values as explicit transfer
learning. Escape gets a zero-initialized 153 × 6 table. The goal selector is a
zero-initialized 108 × 3 table. Total action/goal parameters: 3,078.

Goal context discretizes hunger, thirst, threat proximity, and food/water visibility.
The learned goal chooses which resource/threat channel the motor table attends to.
Water is available only in water-enabled stages; escape is available when threat
sensing detects a predator. Among available choices there is no programmed need
priority or turn-to-resource/turn-away-from-predator command. Tests deliberately
set contrary goal values and verify that decisions follow those learned values.

Actor updates use the current skill's next state. The goal table bootstraps over
available next goals. Both use TD returns with bounded eligibility traces. Switching
goals clears motor credit traces. Exploration decays separately for each skill and
for goal selection. Frozen trials retain the same observation-independent 5% motor
and goal exploration floor; they freeze learned values and synapses, not neural
activity or intrinsic homeostasis.

The body primitives, sensing, reward, memory mechanism, stage definitions, and
predators are engineered. This is not a claim that STDP alone has produced general
intelligence, emergent memory, or a fully biological ecosystem.

## Observation layout and memory

| Observation slots | Signal |
| --- | --- |
| 0–15 | Food retina, 240° |
| 16–31 | Wall rays |
| 32–47 | Hunger |
| 48–63 | Thirst (zero before water stages) |
| 64–79 | Wall touch |
| 80–95 | Water retina, 240° |
| 96–111 | Threat proximity/bearing, 360° |
| 112–127 | Remembered food, when food is not currently visible |
| 128–143 | Remembered water, when water is not currently visible |
| 144–145 | Water/predator availability |

Working memory reconstructs an approximate relative vector from the strongest
observed bin and brightness. Body displacement and rotation update that vector.
Confidence decays exponentially and expires after 10 seconds. It cannot see moved,
hidden or regrown resources; it may be wrong. Eating clears the old food trace.
Visible resources take priority over memory. Memory and ecology signals also excite
association neurons 80–143; the motor output still requires LIF spikes. This memory
is an explicit sensory mechanism, not learned episodic recall.

## Reward and ecology

The original food-only reward is retained before water stages. With water active,
food reward scales with energy actually restored; drinking rewards actual hydration
gain. A full water tank only earns compensation for the tiny amount it consumes.
Injuries cost 0.5 reward per health point. Time, wall and terminal penalties remain.
The shaping potential combines observed/remembered resource approach, current
needs, and negative threat proximity. It does not depend on the selected goal,
so changing goals alone cannot create a potential jump. Every transition uses
`gamma_tick * next_potential - previous_potential`, with terminal potential zero.
Evaluation sets shaping scale to zero.

Water pools are fixed and refill hydration at 24 units/s on contact. Basal thirst
cost is 0.35/s plus 0.035 per movement unit. A visit is counted on contact entry,
not on every simulation tick. Health starts at 100. Predators move at 4 units/s
while hunting, roam at 40% of that speed, and detect a creature within 18 units.
An attack takes 15 health and imposes a five-second cooldown. Newly introduced
predators have a 15-second grace period. Health recovers slowly while adequately
fed/hydrated and away from predators. Predators' pursuit is scripted and has access
to the creature's location within detection range; the creature gets sensors only.

Stages keep resource arrays and regrowth timers, varying how many patches are
active rather than deleting them. Stages use food fractions 1, .65, .65, .65, .4;
vision fractions 1, .85, .85, .85, .65; and predator counts 0, 0, 0, 1, 2.
Advancement waits 180 seconds and requires a 60-second window with at least 12
samples: minimum energy 45, current food rate at least 6/min, minimum health 75,
and, in water stages, minimum hydration 35 plus a water visit. No automatic demotion,
healing, refilling, teleporting or main-creature respawning is performed.

## Evaluation and evolution

Evaluation constructs a fresh body/world and copies the complete recurrent
topology with its weights and all learned value tables. Transient neural activity,
credit traces and resource memory reset. Worlds have new seeds; curriculum is
disabled, learned weights are frozen and shaping is off. Parameter hashes are
checked before and after every frozen episode, including visitation counters.
The source model is checked too. Only a copy is evaluated: the main creature is
free to continue its usual online learning.

Evolution uses sequential candidates with the same topology. It inherits learned
weights (a Lamarckian experiment), adds small Gaussian mutations to action/goal
values and excitatory weights, and mutates bounded learning rate, trace decay,
exploration decay and memory lifetime. Inhibitory signs and body physics do not
mutate. The unchanged champion is always a candidate; other descendants receive
one learning episode. All candidates share training conditions within a generation
and use the same two separate frozen selection seeds. The top two become parents.
The final champion and starting model are both audited on two additional seeds
that were not used for training or selection. Selection can overfit, and the audit
may show no improvement or a regression; the dashboard reports the actual outcome.

Fitness rewards fraction of the trial survived most strongly, then body condition
and food intake, with an injury penalty. The server permits one low-priority worker
at a time, at most eight candidates, ten generations, and bounded episode lengths.
No parallel population is running in RAM. Threaded math libraries are restricted
to one thread for the worker. A long job can still consume one CPU core.

## Persistence and measurement

Checkpoint format 3 includes ecology RNG, memory, goal learning, all previous
brain/world state, curriculum timers, six-hour metrics and completed lifetimes.
Version 1/2 upgrades archive the original file before writing format 3. Food weights,
world and pause/freeze settings are preserved. Existing living bodies are not
respawned. New features start at stage 0 with a fresh advancement window.
Since v0.3.1, dead live-world checkpoints can start another life after the configured
wall-clock delay (default 10 seconds, paused when the simulation is paused).
The automatic-life toggle and delay are additive format-3 fields; old saves get the
defaults without losing learned values. Automatic respawn is a Runner feature,
not part of Simulation.tick(), so experiment episodes still terminate at death.

The dashboard graphs up to six simulated hours, sampled every five seconds.
Food/drink rates and reward rates use the preceding approximately 60 seconds of
the same lifetime. Stage transitions and different trial durations confound simple
comparisons: use fixed-stage, matching-duration frozen trials to measure a change.
CSV gives the full retained samples; on-disk JSONL logs rotate to bound storage.
Evolution history shows selection fitness; its separate audit tests generalization.
