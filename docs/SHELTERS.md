# v0.4 shelter foundations and obstacle-aware escape

For current sensory memory, shared feature learning, replay and shelter practice,
see [COVER-LEARNING.md](COVER-LEARNING.md). The physics below remain in use; the
observation counts and missing-memory limitations describe v0.4.

## Scope

The creature can move materials by pushing, arrange barriers, and receive learning
credit for improving useful cover. This is not a preprogrammed shelter builder or
a demonstrated long-horizon construction planner. There is still one learning
creature, and predators are explicitly scripted NPCs.

Default habitat: 144×90, 48 food patches, 8 water sources and 45-second food regrowth.
That is 2.25× the old area and one-third the old base food density. Stage fractions
remain unchanged: stage 2/3/4 have 32 food patches, stage 5 has 20. There are 24
3.2-unit square blocks, active from displayed stage 2 (CLI stage 1). Food-only
stage 1 remains free of material obstacles. Body sizes/speeds and need costs stay
unchanged; scarcity is not secretly offset by replenishing the creature.

## Physics and cover

Circle-versus-square collision prevents overlap. Bodies slide along blocked axes;
this does not choose an escape heading. A creature contacting one movable block
can translate it, at 40% normal speed. Chain pushing is not implemented. Blocks
cannot overlap each other, leave world bounds, cover resources, or move through
predators. Predators cannot push blocks. At the default 50 ms timestep, movements
are smaller than block dimensions.

Square slab ray tests occlude food, water, and 360° threat sight. Predators pursue
only within their existing detection range when an unblocked sight line exists;
otherwise they roam and collide with blocks. Attacks also check the sight line.
They do not get an omniscient path planner around a shelter. A single block can
temporarily screen the creature; a multi-block structure can provide more cover.
Neither effect guarantees safety, and closed structures can strand a creature.

The construction reward is an engineered geometry objective:

- Sample eight possible body positions around each material block.
- Reject positions overlapping solids or world bounds.
- Count the fraction of 16 sight rays blocked within eight units, requiring at
  least two separate blocks. World edges never count as construction.
- Require adjacent open body-clearance rays over six units, inside world bounds,
  as a conservative exit check. This is a local proxy, not global reachability.
- After a physical push, measure at most twice per second and reward only increases
  above each material's previous best local score, scaled by 8. Every material has
  bounded lifetime credit. Undoing and redoing an arrangement cannot farm it forever.
- Natural initial cover establishes the baseline and earns no construction bonus.

Successful pushing still consumes energy; it earns no touch/distance bonus. The
ordinary failed-movement penalty is omitted for a successful push. A push against
an immovable barrier still incurs contact cost. Long-term metrics and completed-life
records include push distance and building reward earned. Best cover is a geometry
measure that includes initial cover; it is not proof the creature constructed it.

Layouts and credit high-water marks persist in checkpoints. Each new life still
creates a new resource/material layout, matching the existing new-life semantics.
Permanent homes across lives will require a separate persistent-world design.

## Observations and learned decisions

The original 0–145 observation slots remain. New slots:

| Indices | Observation |
| --- | --- |
| 146–161 | Visible material bearings/proximity, 240° |
| 162–177 | Nearby walls/solid obstacles, 360°, 12-unit range with body clearance |
| 178 | Local multi-block cover fraction |
| 179 | Whether the previous movement pushed a block |
| 180 | Whether materials are active |

These also drive association neurons. There is no global shelter coordinate,
target location, route, or action label in the creature's observations.

The first 459×6 policy entries retain v3 food, water and legacy escape learning.
An additional 816×6 escape table distinguishes 17 threat sectors, three threat
distance levels, and 16 combinations of front/left/right/back blockage. Blockage
activates roughly seven units ahead rather than waiting for contact. Existing
escape values seed these contexts; they are not replaced by teacher commands.

The 3060×6 construction table represents block bearing/proximity, legacy local
wall context, cover/pushing state, and four relative-bearing classes for another
visible block (plus an absent class). Approaching material explicitly transfers
food-navigation values; pushing sequences and layouts are not supplied.

Goal selection expands from 108×3 to 108×4: food, water, escape and build cover.
The new goal is available only when material is observed. Total stored action and
goal values: 26,442. Learned values still select all actions and goals; a test
deliberately preferring food under threat is not overridden by a hidden reflex.

Near predators, exploratory probabilities are multiplied by 0.2, including the
frozen policy's exploration floor. A 0.4 value margin favors retaining the current
goal over nearly tied alternatives. These are disclosed behavioral priors, not
learned abilities. Distance, facing an open escape direction and facing obstacles
contribute to discounted potential shaping; proximity and danger-related failed
movement incur transition costs. The facing-away term is engineered knowledge,
just as the earlier food-progress reward was. No expert selects a left/right action.

## Escape practice

A single low-priority worker trains a copy in bounded wall/corner scenarios with
random headings and an approaching predator. There is no food/water distraction,
no material obstruction, and no expert steering. Both goal choice and escape motor
values learn online through the complete spiking-body simulation. Every episode
has fresh neural/activity state and ends on death or its time limit.

Only escape and goal changes accumulate into the candidate; its source's food,
water, construction and recurrent weights stay exact. Candidate and baseline then
run on the same separate seeds with all learned weights frozen and shaping off.
The audit does not select the candidate or supply further training. A candidate
may be worse; the dashboard shows both damage and short-trial survival.

Adoption is manual, after death. Escape adoption replaces escape values and merges
only goal entries changed by practice, while preserving the live brain's latest
other skills and recurrent weights. This avoids rolling back learning performed
while the copy was training. Evolution adoption retains its older whole-model
behavior. Both archive the previous checkpoint.

## Upgrade and next capabilities

Format 4 stores the added observations, blocks, construction credit, and expanded
policies. Loading an old file retains the old arrays as prefixes/columns, seeds new
contexts, and clears obsolete transient action credit. Starting the live Runner
archives the original v1/v2/v3 file and expands the saved geometry once. Needs,
age, learned weights, pause/freeze, difficulty (v3), metrics and life history remain.
Spatial traces clear after resizing. New saves resume exactly, including mid-push
geometry and pending actions. Checkpoints are local data, never Git content.

The next useful steps are learning when to occupy/leave cover, remembering safe
sites longer than ten seconds, carrying and storing food, then multi-step placement
and repair. Weather/warmth could make shelter useful beyond avoiding predators.
More creatures, cooperation and reproduction should follow reliable individual
survival, rather than hiding individual learning failures in a larger population.
