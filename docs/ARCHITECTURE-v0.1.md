# Neuroplex v0.1 architecture

The lifetime-learning design follows the earlier Neuroplex discussion: a small,
continuously running, recurrent leaky integrate-and-fire (LIF) spiking neural
network with local eligibility traces and a global homeostatic reward signal.
The recurrent excitatory network is plastic; this is not a fixed reservoir with
only a trained readout. It is also not a CNN or an LLM.

## The loop

Every 50 ms of simulated time:

1. Encode the current egocentric food/wall retina, hunger, and collision touch.
2. Advance the same brain through ten 5 ms LIF steps.
3. Decode four motor populations' smoothed firing rates into speed and turning.
4. Move the body, spend energy, and consume any food within mouth reach.
5. Compute hunger reduction and apply it to the existing synaptic eligibility.
6. Carry **all** state forward into the next tick.

The Python worker is independent of the browser. WebSocket snapshots are sent at
5 Hz. Multiple clients watch the same one creature. A fixed physical step means
changing playback speed cannot change integration accuracy. If the CPU is slow,
simulated time slows rather than skipping neural steps.

## Neural layout

| Neuron indices (inclusive) | Count | Role |
| --- | ---: | --- |
| 0–15 | 16 | Food retina: brightest visible food per angular bin |
| 16–31 | 16 | Walls: ray distance transformed into brightness |
| 32–47 | 16 | Hunger intensity |
| 48–63 | 16 | Reserved thirst channels, zero in v0.1 |
| 64–79 | 16 | Body collision touch |
| 80–399 | 320 | Recurrent association/memory |
| 400–424 | 25 | Forward motor population |
| 425–449 | 25 | Backward motor population |
| 450–474 | 25 | Left motor population |
| 475–499 | 25 | Right motor population |

Each neuron targets 32 distinct association/motor neurons: **16,000 directed
synapses**, with no self-connections. Incoming recurrent synapses do not target
sensory neurons; this keeps sensory encoding clean. Sensory and motor projections
are otherwise randomly initialized. We do not preassign associations such as
“green food on left → turn left”.

Sixty-four association neurons are inhibitory (20% of that group), with fixed
negative outgoing weights. The other 436 neurons are excitatory: **13,952 plastic
synapses** remain nonnegative and are clipped at 0.12. Neurons cannot switch between
inhibitory and excitatory. Synapses are sparse pre/post index arrays, avoiding a
dense matrix multiplication or a Python object per connection.

## LIF dynamics and exploration

Voltage uses normalized units with reset 0 and initial threshold 1:

```text
v += (dt / 20ms) * (external_current - v)
v += sum(weight * previous_presynaptic_spike)
v += small independent voltage noise
if v >= threshold and not refractory:
    spike = 1
    v = 0
    refractory = 5ms
```

Sensory current is `0.05 + 2.6 * sensory_intensity`. Previous-bin spikes supply
recurrent input, giving synapses a one-step delay. Rates use a 250 ms exponential
filter. Bounded adaptive thresholds in nonsensory neurons target roughly 12 Hz
over a slower time scale; this reduces runaway recurrent activity, without
claiming precise homeostatic control. Thresholds remain in [0.7, 1.8].

Exploration is **explicitly innate**: nonsensory tonic drive, a small forward
motor bias, voltage noise, and three temporally correlated motor-current noise
signals. None can access food position, bearing, or the best action. These signals
keep the initial network from being motionless. Their effects flow through the
same motor neurons as sensory/recurrent influences.

```text
speed_fraction = tanh((forward_rate - backward_rate) / 12)
turn_fraction  = tanh((right_rate - left_rate) / 8)
speed = speed_fraction * 9 world_units/second
turn  = turn_fraction * 2.4 radians/second
```

Touch signals a wall collision. There is no hard-coded bounce, turn-away reflex,
food homing, target position lookup in the brain, or movement teleport.

## Plasticity and reward

Each neuron has exponentially decaying pre/post spike traces (20 ms). Each synapse
has eligibility (5 s), so a delayed food reward can affect recent spike pairings.
Using decayed traces **before** adding current spikes:

```text
pre_trace  *= exp(-dt / 20ms)
post_trace *= exp(-dt / 20ms)
eligibility *= exp(-dt / 5s)
eligibility += post_spike * pre_trace[pre]
eligibility -= 1.05 * pre_spike * post_trace[post]
eligibility = clip(eligibility, -8, +8)
pre_trace += spike
post_trace += spike
```

Only excitatory connections receive eligibility updates. Causal pre-before-post
pairing contributes positively; the reverse contributes negatively. Same-bin
spikes do not impose an arbitrary ordering.

At the end of each world step:

```text
hunger = 1 - energy / 100
reward = hunger_before - hunger_after
weight += 0.025 * reward * eligibility
weight = clip(weight, 0, 0.12)       # excitatory only
```

This scalar is a *reward modulator*, not a learned dopamine model, TD critic, or
reward prediction error. Negative reward reverses the sign of eligible updates.
It does not necessarily weaken every connection. Learning freeze stops synaptic
updates but not intrinsic threshold adaptation or ongoing state dynamics.

The reward is applied once per world step as a discrete energy change; it is not
multiplied by dt again. Basal energy cost is 0.70/s, movement cost 0.12 per commanded
distance unit, and each food gives up to 24 energy (capped at 100). A blocked body
still spends energy attempting to move. There is no extra reward for approaching
food, staying alive, touching walls, or dying. At zero energy the body and neural
loop stop. There is no automatic reset.

Because reward is a difference of a bounded drive, cumulative reward telescopes
to initial hunger minus current hunger. **Do not use cumulative reward as evidence
of improving lifetime behavior.** Measure food eaten, survival, and sustained
energy, and compare frozen controls over multiple initial conditions.

## Memory and limitations

- **Immediate state:** membrane voltage, refractory status, and previous spikes.
- **Short-lived context:** recurrent activity, 250 ms rates, 20 ms spike traces,
  5 s eligibility, and motor noise state.
- **Longer-lived memory:** reward-dependent synaptic weights. They persist across
  saves and, if requested, across a manually started new life.
- **No explicit episodic recall:** no symbolic memory, remembered food map,
  hippocampal module, planning algorithm, replay, or supervised teacher.

The simple topology is an experimental substrate. Sparse rewards, anti-causal
updates, exploration bias, credit assignment, and homeostatic dynamics can make
learning unstable or counterproductive. An individual 180-second comparison is
too short to establish convergence. Food density and random exploration can
produce plenty of eating without learned navigation. Future changes should be
justified with multi-seed controlled experiments, not an attractive trajectory.

## Persistence and extension boundaries

Checkpoints include the full dynamic state and both NumPy random generators.
Resuming within the same pinned software stack reproduces subsequent simulation
steps exactly in the checkpoint test. They use a version field; incompatible
versions fail explicitly. Runtime speed, pause, learning setting, and recent
statistics are saved too. No offline time is simulated.

Start additional behavioral features in `world.py` and a corresponding documented
sensory/motor mapping. The existing thirst block is deliberately reserved. More
neurons require changing the explicit layout and checkpoint version, not simply
editing a count. The current implementation deliberately handles only one body.

## Background

This is a simplified engineering implementation inspired by three-factor learning,
not a numerical reproduction of a particular paper:

- R. V. Florian (2007), *Reinforcement learning through modulation of spike-timing-
  dependent synaptic plasticity*: <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>
- Gerstner et al. (2018), *Eligibility Traces and Plasticity on Behavioral Time Scales*:
  <https://arxiv.org/abs/1801.05219>
- FastAPI lifespan (one runner, clean shutdown):
  <https://fastapi.tiangolo.com/advanced/events/>
- FastAPI WebSockets: <https://fastapi.tiangolo.com/advanced/websockets/>
