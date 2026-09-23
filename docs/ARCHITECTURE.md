# Neuroplex architecture

Current v0.4 mechanics and learning extensions are in [SHELTERS.md](SHELTERS.md):
solid pushable blocks, occlusion, construction credit, obstacle-aware escape,
isolated escape practice and format-4 migration. The following sections document
the earlier foundation; their historical table sizes are not current totals.

The v0.3 extension is documented in [ECOSYSTEM.md](ECOSYSTEM.md): water,
predators, goal learning, sensory memory, curriculum, evaluation and evolution.
The food-controller foundation below describes v0.2. Its 918-value table is now
one of three motor tables, alongside a 324-value goal selector. The bundled food
asset remains unchanged. See the extension for the current checkpoint format,
observation layout, rewards and retention policy.

v0.3.1 adds a Runner-owned, pausable wall-clock next-life timer, shared 20 Hz
geometry caches, and browser-side pose interpolation. Experiments still end at
death. See [release verification](VALIDATION-v0.3.1.md) for the protocol cadence,
state-preservation checks and measured rendering limits.

## v0.2 food-controller foundation

v0.2 keeps the persistent **500-neuron / 16,000-synapse recurrent LIF network** and
adds a small **tabular TD motor policy**. This is explicitly a hybrid architecture.
The action-value table is not a spiking layer, and the demonstrated improvement
should not be attributed to recurrent STDP alone. All learning is local and online;
there is no backpropagation, autodiff, replay buffer, minibatch training, action
teacher, or hard-coded rule that selects a turn toward food.

## Why v0.1 needed a change

The original network broadcast hunger changes to all eligible excitatory synapses.
That made weights change without reliably assigning credit to useful movement.
It also left the motor interface drifting with recurrent activity. v0.2 learns
values for specific motor actions, gives progress feedback before the next meal,
and keeps the spiking actuators stable enough to associate actions with outcomes.

The old architecture and mixed initial results remain in
[ARCHITECTURE-v0.1.md](ARCHITECTURE-v0.1.md) and
[VALIDATION-v0.1.md](VALIDATION-v0.1.md).

## Continuous loop

The world advances by 50 ms; the LIF network takes ten 5 ms steps per world step.
Every 250 ms the motor policy selects an intent and holds it while the spiking
motor populations execute it. State, traces, weights, values, and random streams
continue across decisions. A live creature is never automatically reset or rescued.

1. Encode food/wall retina, hunger, and touch.
2. Choose a motor intent from learned values, with occasional exploration.
3. Apply that intent as currents to the four motor populations.
4. Advance the LIF network; decode firing rates into body movement.
5. Move, spend energy, eat food within reach, and observe the result.
6. Accumulate reward; at the action boundary update action values and eligibility.
7. Use a clipped TD error to modulate the recurrent STDP traces.

The browser only observes and controls this server-side loop. Simulation speed
changes wall-clock pacing, never the neural or physical integration step.

## Observations and action values

The policy sees only the existing **egocentric sensory vector**, never world food
coordinates, desired headings, future events, or ground-truth action labels.
Its intentionally compact representation has 153 possible contexts:

- 17 food sectors: the brightest of sixteen angular food bins, or no visible food.
- Three proximity bands, using food brightness thresholds 0.45 and 0.80.
- Three wall contexts: clear, nearby wall stronger on the left, or stronger on the
  right. This uses central wall brightness and touch.

There are six possible motor intents: forward, curve left, curve right, turn left,
turn right, and backward. The **153 × 6 = 918 action values** start at zero in an
untrained experiment. Encoding where food appears does not specify which action
is correct. The policy must learn that association from its action outcomes.

A greedy intent maximizes the context's action value; exact ties are random.
Exploration probability during learning is:

```text
epsilon = 0.05 + (0.35 - 0.05) * exp(-learning_updates / 1500)
```

When learning is frozen, weights stay fixed but a 5% exploration floor remains.
This is independent of food bearing. It helps escape repeated poor actions in a
coarse or unfamiliar context. The frozen comparison gives both trained and
untrained conditions the same floor. Setting `exploration_floor=0` is supported
for experiments, but a completely greedy policy can get stuck.

## Reward and credit assignment

Living still costs 0.70 energy/s plus 0.12 per commanded movement unit. Food restores
up to 24 energy; at zero the creature dies. Ingestion remains a body reflex.
The old hunger-change reward is retained separately as `total_drive_reward`.

The new learning reward, at each world step, is:

```text
base = 6 * food_eaten - 0.2 * dt - 2.4 * dt * wall_contact
if dead: base -= 6
reward = base + gamma_tick * Phi(next_observation) - Phi(observation)
```

The progress potential uses only visible food:

```text
Phi = 4 * max_over_visual_bins(
    food_brightness * (0.3 + 0.7 * max(0, cos(bin_angle)))
)
```

This deliberately supplies prior knowledge: approaching and facing visible food
is useful. It supplies **scalar feedback, not a steering command**. It is reward
shaping, not an action teacher. Eating gets a separate positive reward even at
full energy, because this experiment's objective is continual food seeking.
Time and wall-contact penalties discourage stalling and pushing against walls.

`gamma_tick = 0.96 ** (world_dt / action_seconds)` matches the policy's discount.
A held action accumulates discounted rewards `R` and a cumulative discount `G`:

```text
delta = R + G * max_a Q(next_context, a) - Q(context, chosen_action)
trace *= G * 0.40
trace[context, chosen_action] = 1
Q += 0.30 * clip(delta, -10, 10) * trace
```

At death, both the bootstrap value and next potential are zero. A nongreedy action
cuts earlier eligibility, following Watkins-style Q(lambda) credit assignment.
Q is bounded to [-100, 200]. There are no updates when learning is frozen.

Discounted potential differences telescope, so repeatedly rotating between the
same visual states cannot accumulate a free positive shaping return. The test
suite checks this and terminal handling. This property does not imply convergence:
observations are partial, contexts are coarse, and exploratory behavior can fail.

## Spiking network and motor interface

The layout is unchanged:

| Neurons (inclusive) | Role |
| --- | --- |
| 0–31 | Food and wall retina |
| 32–47 | Hunger |
| 48–63 | Reserved, unused thirst input |
| 64–79 | Touch |
| 80–399 | Recurrent association/memory |
| 400–424 | Forward motor population |
| 425–449 | Backward motor population |
| 450–474 | Left motor population |
| 475–499 | Right motor population |

Sixty-four association neurons remain inhibitory, with fixed negative outgoing
weights. The 13,952 excitatory synapses retain bounded reward-modulated STDP.
Pre/post traces decay over 20 ms and eligibility over 5 s. A completed action's
TD error, clipped to [-1, 1], modulates them at learning rate 0.001. Adaptive
thresholds now regulate association neurons only. Very small rates/traces are
zeroed to avoid CPU denormal arithmetic.

Motor intents specify currents, not movement coordinates. Recurrent input to
motor neurons is scaled by 0.05, motor thresholds stay at 1, and their rate filter
uses 40 ms. This keeps the action interface stable while allowing actual spikes
to determine speed and turning:

```text
speed_fraction = clip((forward_rate - backward_rate) / 50, -1, 1)
turn_fraction = clip((right_rate - left_rate) / 50, -1, 1)
```

The association network remains plastic, but the **tabular motor policy is the
main navigation learner** in this version. It is not accurate to describe the
measured foraging skill as emerging solely from an unstructured recurrent SNN.

## Trained starting values and persistence

`assets/foraging-v0.2.json` contains values learned in a complete, continuous
600-second simulation: seed 7, 32 food patches, zero initial action values,
2,400 updates, 169 food items, no death, rescue, teacher, or reset. The data was
exported from training, not hand-authored into a state/action rule.

New worlds use these values by default and keep learning. `--untrained` uses zero
values for a **new** data directory. It never erases an existing checkpoint.
New lives retain both recurrent weights and action values; only transient activity
and action traces reset.

Checkpoint format 2 stores all original neural/world arrays plus policy values,
visits, eligibility, action-in-progress, discounted return, counters, and the
policy's independent RNG. Mid-action restore is tested for exact continuation.

A format-1 checkpoint upgrades automatically. Its body, food, time, preferences,
and recurrent weights are preserved. The new learned policy is added; motor
thresholds are set to the new fixed value. The original file is retained as
`checkpoint.v1.npz`, independently of the rotating `checkpoint.previous.npz`.
A dead creature stays dead until the user requests a new life.

## References and scope

- Ng, Harada & Russell (1999), potential-based shaping:
  <https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf>
- Chung & Kozma (2020), action feedback and TD-modulated spiking plasticity:
  <https://arxiv.org/abs/2008.13044>
- The original recurrent rule is inspired by Florian (2007):
  <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>

This implementation is not a reproduction of those papers. The compact motor
policy is an engineering choice for learnable, CPU-cheap food seeking. It does not
provide general intelligence, learned vision, planning, or a guarantee of survival
under arbitrary world settings. See [VALIDATION.md](VALIDATION.md) for controlled
measurements and their limits.
