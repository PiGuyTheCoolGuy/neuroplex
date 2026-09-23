# v0.5: learning from blocks and remembered cover

The objective is a better learner on the existing CPU budget. This release does
not supply correct pushes, a house layout, a route to safety, or a finished shelter
policy. Existing 500-neuron topology and 16,000 synapses stay intact. The motor
controller remains a hybrid of tabular TD learning and spiking actuators.

## What the creature can sense

The original 181 inputs retain their positions. New inputs total 248:

| Slots | Observation |
| --- | --- |
| 181–196 | Fixed boundary, first visible surface on 16 nearby 360° rays |
| 197–212 | Movable block surfaces on those same rays |
| 213–228 | Body-relative bearing/proximity to one remembered, previously occupied cover site |
| 229–244 | Body-relative estimate of the last observed threat |
| 245 | Remembered cover quality × confidence |
| 246 | Whether the body has returned close to the remembered site |
| 247 | Last observed threat strength × fading confidence |

Surface sensing is limited to 12 units and reports the nearer surface. A boundary
behind a block is not revealed. It detects material type, **not whether a push will
succeed**; an apparently movable block can be jammed by other objects.

Named input banks in the existing neural population are visible blocks (144–159),
obstacles (160–175), current cover (176–183), fixed boundaries (184–199), block
surfaces (200–215), cover memory (216–231), and threat memory (232–247). They feed
real neuron currents; their observations also enter the action-value learner.
More input cells alone would not demonstrate improved decisions, so no claim of
purely emergent SNN shelter planning is made.

Cover memory records **only the body's current location** when actual multi-block
cover is at least 0.25 and has the existing local exit proxy. It remembers one
site, integrates observed body motion, fades, and expires at 180 simulated seconds.
Returning to a site without cover clears it. Better/current nearby cover can
replace the old memory. It has no map, global coordinates or hidden shelter target.
Threat memory uses the observed bearing and range, fades for six seconds, and
does not track unseen movement. Both action learning and the goal selector consult
that fading estimate; occlusion does not instantly label a threat-free situation.
Memory resets for a new life and when disabled;
checkpoint restart preserves it exactly. It is an engineered memory mechanism.

## Learning and actions

The 4,335 original motor rows keep their six learned action columns. A seventh
**Rest** action starts at zero and sends zero motor currents; actual motion still
comes from spiking motor rates, which settle naturally. No rule selects Rest.
The new **Use cover** goal adds 153 rows, seeded from existing food navigation,
just as block approach already transferred food experience. No route or shelter
layout is included in that transfer.

The manager expands from 108 × 4 to 432 × 5: hunger, thirst, danger and resource
visibility, plus whether cover is remembered and currently occupied. The old
108 × 4 values remain exact; copies seed the new contexts. Goal and action choices
remain learned. A test deliberately selecting backward while the remembered site
is ahead verifies that there is no return-to-cover controller overriding values.

For escape, building and using cover, a small **linear residual action-value model**
adds to the existing table's values. It has 102 normalized features: the local
sensory arrays, needs, cover/push feedback, two observed block bearings/proximities,
their relationship, and interactions of needs/threats with cover memory.
Weights start at zero. Nearby situations can share experience without filling
every coarse table entry independently. This is not a deep neural network.

Each completed action stores its actual observations, chosen action, return,
terminal flag, and table context in a 512-entry ring. One direct update and two
uniformly sampled replay updates train only that action's skill. Replay never
chooses a world action, adds food/health, creates geometric reward, or updates
food/water motor tables. No demonstrations, imagined outcomes, hindsight target
labels or remote service are used. It runs in the existing process; replay arrays
use under 1 MB. New lives keep the buffer and learned weights, while clearing
transient action state. All buffer contents, RNG and mid-action features are saved.
Freezing learning disables both direct and replay weight updates.

Totals: 4,488 × 7 motor values + 432 × 5 manager values = **33,576 table values**;
3 × 7 × 102 = **2,142 feature weights**. Recurrent STDP remains online and is
modulated by actual action TD error, not replayed reward. There is no claim of
convergence under these partially observed, changing environments.

## Incentives and limits

The existing physical construction reward remains: 8 × a new per-material cover
improvement, requiring multiple blocks, body clearance and a local exit. Initial
natural arrangements establish the baseline; touching/pushing alone earns no
bonus. Previous best per-material scores prevent repeating the same improvement
forever. This is an engineered geometric objective, not the creature inventing
the concept of a house. It is scalar environmental feedback; the policy receives
no coordinates of the world's best cover location.

The existing potential gains a bounded term:

`cover_shaping × recent_observed_threat × current_local_cover × (1 - strongest_need)`

It uses observations and fading memory, never hidden predator positions or the
world's protected-time metric. As with existing shaping, reward adds
`gamma_tick × Phi(next) - Phi(current)`, with zero terminal potential. Staying at
the same positive potential or completing a repeated cycle does not yield a
positive discounted shaping bonus. Food/water costs, attack penalties, movement
costs, predator physics and resource scarcity remain in force. This supplies a
prior that cover during observed danger can be useful, **not a chosen action**.
It is a local proxy; it cannot guarantee an exit route or survival.

## Focused practice and honest measurement

Shelter practice trains a copy on random loose blocks near the body in a 64 × 48
area. Initial blocks are sampled individually, without a target arrangement.
The first half of episodes has no active predator; the second includes one.
Ordinary goal selection, needs, push physics and rewards remain active. It can
choose food or escape and fail to build. Only building/use-cover motor learning
and manager learning accumulate; other motor skills and recurrent weights stay
at the source values. No generated experience is installed automatically.

After training, candidate and baseline run frozen on identical unseen layouts,
then on a separate set of full-size scarce habitat seeds. Audit shaping is off.
These seeds do not select a winner or feed training. Reports include construction
reward actually earned, pushed distance, protected time, damage and food intake.
Natural cover and incidental pushes can affect these measures; none proves
deliberate multi-step construction. Results may regress. Adoption is manual after
death and preserves newer live food/water/escape motor learning and synapses.

The v0.4 checkpoint is archived before migration to format 5. Existing block
positions, resources, needs, time and learned arrays remain; habitat expansion
applies only to older v1–v3 saves. New skills begin untrained apart from disclosed
navigation transfer. The v0.2 food asset is unchanged. There is still no permanent
home across lives, object carrying, blueprint, learned model of future pushes, or
multi-step planning system.
