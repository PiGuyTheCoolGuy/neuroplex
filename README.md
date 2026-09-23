# Neuroplex

**One creature learning to forage, evade predators, and arrange blocks into cover.**

Neuroplex v0.5 is a CPU-only artificial-life experiment for Ubuntu 24.04. A Python
process runs the world continuously; your laptop's browser displays it. Closing the
browser does not stop the simulation. There is no GPU, cloud model, PyTorch, Node.js
build, or paid API to configure.

The brain has **500 leaky integrate-and-fire neurons and 16,000 sparse synapses**,
plus **33,576 stored action/goal values and 2,142 shared feature weights**.
Named sensory banks distinguish fixed boundaries, movable block surfaces and
remembered cover. Learned values choose food, water, escape, building or using cover;
spiking neurons produce movement. Recurrent STDP and value learning continue during its life.
The small cover learner replays up to **512 of its own experienced transitions**.
There is no action teacher, shelter blueprint, route planner or hidden safe-site input. After death, the live
habitat can automatically start a fresh body while retaining the learned brain.

**It still starts with the v0.2 learned food-seeking values**, obtained from ten
simulated minutes of training. Water navigation transfers that motor experience;
balancing needs and escaping predators are new learning problems. Predators use
programmed hunting rules. Working memory is an engineered, fading sensory trace,
not a demonstrated emergent neural memory system.
Use `--untrained` in a new data directory to watch learning from zero. Controlled
tests freeze all weights and disable progress rewards in unfamiliar worlds;
[validation](docs/VALIDATION.md) records results and limits.

## Upgrading from v0.1–v0.4

Run this in your Ubuntu / VS Code SSH terminal:

```bash
cd ~/neuroplex &&
systemctl --user stop neuroplex &&
git pull --ff-only &&
bash setup.sh &&
systemctl --user start neuroplex
```

If running in the foreground, use Ctrl+C before updating, then `bash start.sh`.
Refresh the browser (Ctrl+F5). **Upgrading from v0.4 keeps the existing habitat,
block positions, age, needs, pause/freeze settings and all learned values.**
The original is archived as `data/checkpoint.v4.npz`. New sensor memory, feature
weights and a Rest action start without shelter training. Pending action credit
resets for the new observation/action schema; learned memories are retained.

**For v0.1–v0.3 only, the live habitat expands once: 1.5× each dimension,
25% fewer food patches, one-third fewer water sources, and at least 45-second food
regrowth.** The standard 96×60 world becomes 144×90. Coordinates scale proportionally;
the creature keeps its age, energy, hydration, health, learned values, recurrent
weights and pause/freeze settings. New materials are added. Old checkpoints are
archived as `data/checkpoint.v1.npz`, `.v2.npz`, or `.v3.npz` before writing format 5.
Old sensory/action credit traces restart because the geometry changed. Existing
v3 worlds keep their habitat stage; v1/v2 start at the gentle food stage.
This expansion does not repeat on later restarts. **Auto-start next life is on by
default, with a 10-real-second
delay**, including older saves with no setting yet. A saved pause holds that countdown;
press Resume to continue. A saved learning-freeze setting also stays frozen until
you turn learning back on. Neither update nor respawn discards learned weights.

The packaging fix explicitly excludes runtime `data/` from package discovery;
you never need to remove saves to run setup. If you manually applied that fix,
Git may ask you to commit or stash the edit before pulling. Keep your changes.

## What is new

| Feature | Behavior |
| --- | --- |
| Larger, scarcer habitat | 144×90 by default; 48 food patches, 8 water sources, 45-second regrowth; stage modifiers make resources scarcer still |
| Building materials | 24 pushable solid blocks from stage 2; blocks stop bodies and sight lines; learned building goal and rewards for improved cover with exits |
| Cover learning | Separate fixed/movable surface inputs, shared geometry features and replay of 512 own experiences; no supplied actions |
| Remembering and using cover | Remembers a visited cover site for up to 180 simulated seconds, corrects it on return, and learns a Use cover goal; Rest is a learned action |
| Shelter practice | Random loose materials, then threats, on a copy; separate frozen tests in new practice layouts and full habitats; manual adoption |
| Escape decisions | Earlier 360° obstacle sensing, joint wall/corner/threat states, reduced near-danger exploration and a margin before switching goals |
| Escape practice | Bounded wall/corner drills on a copy, independent frozen audit and optional selective adoption after death |
| Automatic next life | On by default; 10 real seconds after death, adjustable 1–300; keeps learned synapses, skill/goal values, difficulty, and lifetime records |
| Smooth live view | 20 Hz geometry stream, browser animation targeting 60 fps, 100 ms interpolation buffer; no video encoder or extra dependencies |
| Learning history | Five-second samples; six simulated hours of food/drink rates, energy, hydration, health, reward and learning updates; CSV export |
| Curriculum | Foraging → scarce food → water → one predator → two predators and shorter vision |
| Sensory memory | Remembers estimated food/water direction for 10 seconds, integrates body motion, expires; can be disabled |
| Water | Drinks on contact; moving and living consume hydration; dehydration can end a life |
| Predators | Scripted roaming and hunting, limited speed and sensing range, damage, cooldowns, health and recovery |
| Frozen evaluation | New worlds, frozen learned weights, no shaping reward; manual or every 30 simulated minutes |
| Evolution | Sequential population, inherited learned weights, bounded mutations, unchanged elite, held-out selection and a separate final audit |

The main habitat still contains **one learning creature**. Evolution runs copies
in separate worlds, one at a time. Predators are NPCs, not an evolving second species.
There is no mating or uncontrolled population growth in the live habitat.

Start at **1×** on the OptiPlex. In **Habitat curriculum**, leave *Advance when ready*
on for gradual introduction, or select a stage manually. Each stage needs at least
three simulated minutes, then a full healthy one-minute performance window.
For water stages, it must also have visited water during that window. Freezing
learning pauses automatic advancement. Higher difficulty does not refill its needs.

Use **Evaluation & evolution** to test a snapshot. Choose the stage, duration and
seed, then run. The default small jobs are quick checks; use **180–300 seconds per
world** when judging thirst and predator survival. Compare matching stages and
durations, not raw results from different difficulties. A short perfect trial is
not proof of long-term skill. Jobs can be cancelled; the main creature keeps running.

For evolution, start with population 4 and 3 generations. The seconds field is the
training time for each mutated descendant; the separate selection/audit field sets
the frozen trial length. This can take several minutes on an older CPU. The result
compares the evolved winner and starting model on the same final audit worlds.
Improvement is not guaranteed. After the main creature dies, **Start next life with
trained candidate** imports the evolution winner and its learning parameters, archiving the old
creature as `checkpoint.before-evolution-life-N.npz`. Ordinary **Start new life** keeps
the main creature's own learning instead. **Pause or turn off Auto-start next life**
if you want time to inspect a death or adopt a champion. Champion adoption is always
manual; automatic next lives never import an evolution result.

## Blocks, shelters, and escape practice

Select **2 · Scarce food** or a later habitat stage to see the brown blocks.
The creature can push them with its body, at 40% normal movement speed. Blocks cannot
pass through each other, leave the map, or bury food/water. Predators cannot push
them, see through them or attack through them. A wall/corner made of blocks can
therefore provide real cover. The creature still needs a way out to forage and drink.

The **Build cover** goal learns motor choices using material bearings, another
visible block's relative direction, contact and local cover. Approaching a block
transfers food-navigation experience; no layout, U-shaped shelter, or correct push
sequence is programmed. An explicit geometric reward measures added cover from
multiple blocks with room to exit. It rewards new improvements per material; merely
touching blocks or repeatedly restoring the same arrangement does not earn more.
The dashboard distinguishes distance pushed from **building reward actually earned**.

The **Use cover** goal gets a fading, body-relative memory of cover the creature
has actually occupied. It cannot discover a distant shelter through walls. The
learner can choose to return, leave, or rest; none of those actions is forced.
A six-second memory of an observed threat prevents immediate forgetting behind a
block. It estimates the last observed location, never a hidden predator's movements.
Safety shaping uses only recent threat observations, local cover, and current needs,
as a discounted potential difference; there is no recurring payment for camping.

**Reliable multi-step shelter planning is still unproven.** There are no roofs,
doors, object grasping, permanent home memory or construction plans.
Blocks and their arrangement survive service restarts. As before, a
new life creates a fresh world, so it does not inherit the previous life's buildings.

To practice building and using cover:

1. Leave **Learning on** and **Sensory memory** enabled. Building works from **2 · Scarce food** onward.
2. Under **Evaluation & evolution**, choose **Shelter practice · blocks & cover**.
   Start with **40 episodes × 30 seconds** and **5 evaluation worlds**.
3. Compare construction reward, protected seconds, damage and food intake. The
   dashboard also shows transfer to the normal full-size scarce habitat; a copy
   that improves a practice score can still be worse at survival. Our short v0.5
   check increased protected time in dense layouts but reduced building reward
   and full-habitat food intake; do not assume practice is an automatic upgrade.
4. Adoption is optional and only after death. Turn off automatic lives or pause
   the death countdown, then **Start next life with trained candidate** if useful.
   This imports building/use-cover motor learning and changed manager entries;
   live food/water/escape motor learning and synapses are retained. The previous
   checkpoint is archived. No trained shelter policy is bundled or auto-installed.

To work on cornering and hesitation:

1. In **Evaluation & evolution**, choose **Escape practice · walls & corners**.
2. Start with **40 episodes**, **12 seconds per world**, and **3–5 evaluation worlds**.
   Episodes are limited to 5–60 seconds, with 1–200 episodes and one low-priority CPU worker.
3. Compare the candidate's **mean damage** against the starting model on the same
   unseen frozen worlds. Both may survive a short test while one suffers more attacks.
4. If the result is useful, pause or disable automatic lives before the next death,
   then choose **Start next life with trained candidate**. Escape-practice adoption
   merges the new escape values and changed goal entries while preserving the live
   creature's latest food, water, building and recurrent synaptic learning, including
   learning that happened while practice was running. The old checkpoint is archived.

Practice supplies challenging scenarios, not teacher actions. More detailed wall
states, reward shaping and reduced random exploration help learning, but the policy
can still make bad decisions. In the earlier v0.4 five-world check, 40 practice episodes reduced
mean damage from **27 to 15**. That is limited evidence, not a survival guarantee.
[Current learning design](docs/COVER-LEARNING.md) and [v0.5 validation](docs/VALIDATION-v0.5.md)
give details; [SHELTERS.md](docs/SHELTERS.md) describes the underlying v0.4 physics.

## Automatic lives and smooth viewing

Under the habitat, **Auto-start next life** lets it continue unattended, even with
no browser connected. Its death and final learning are recorded before respawn.
The fresh body/world has a new seed; learned synapses, food/water/escape values,
goal values, exploration experience, learning on/frozen state, current habitat stage,
and long-term metrics survive, including construction values. Short-lived sensory traces and neural activity reset.
This does not make it invincible or guarantee that it learns predator avoidance.

The delay is **real seconds**, unaffected by 1×–10× simulation speed. Pause freezes
the countdown; turning the toggle off cancels it. Changing the delay or enabling
the toggle again starts a full new countdown. These two settings save immediately.
Restarting the server while dead also gives a fresh full delay; powered-off time
does not count. Manual **Start new life** still works immediately. Frozen evaluation
and evolution trials continue to end at death; they do not respawn within a trial.

The browser receives small motion frames over the existing WebSocket and draws
intermediate positions locally. This is live **state streaming**, not video. Creature
and predator movement, and pushed blocks, interpolate; food does not slide around, and the view never
predicts movement beyond the latest received position. Death, pause, new lives,
stage changes and reconnects reset the interpolation buffer. Brain/vital readouts
refresh at 5 Hz; graphs at most once a second, with long-term data only when changed.
All viewers share cached serialized frames. Slow connections skip old frames instead
of building a playback queue. Browser visibility throttles drawing, not learning.

The line below the habitat shows measured network-update and rendering rates.
20 updates/s and about 60 rendered frames/s are targets, not OptiPlex guarantees;
start at 1× if actual simulation speed falls behind. No new port, GPU, video service,
Node.js installation, or change to the SSH tunnel is required.

## 1. Install on the OptiPlex (Ubuntu 24.04)

Run these in an SSH terminal or a **VS Code terminal connected to Ubuntu**:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip
cd ~
git clone https://github.com/PiGuyTheCoolGuy/neuroplex.git
cd neuroplex
bash setup.sh
bash start.sh
```

If you already cloned the empty repo, `cd ~/neuroplex` and `git pull --ff-only`
instead of cloning again. `setup.sh` creates `.venv` and installs the tested pinned
dependencies. Keep this terminal running for your first trial. Ctrl+C stops the
server cleanly and saves its state. Python 3.12 is included with Ubuntu 24.04.

## 2. Watch from Windows

The default address is **127.0.0.1:8000 on the Ubuntu machine**. Reach it with either:

**VS Code Remote SSH:** connect to the OptiPlex, open `~/neuroplex`, open the **Ports**
tab beside Terminal, choose **Forward a Port**, enter `8000`, and open the forwarded
address. Normally that is <http://localhost:8000>. If VS Code chooses another local
port, use the address it displays. Install Microsoft's **Remote - SSH** extension
on Windows and the **Python** extension in the SSH window. VS Code settings in the
repo select `.venv/bin/python`.

**Or Windows PowerShell:** leave this running (substitute your current IP if needed):

```powershell
ssh -N -L 8000:127.0.0.1:8000 treic@192.168.1.20
```

Open <http://localhost:8000> in your Windows browser. Use only one forwarding method
at a time for local port 8000. If occupied, forward local port 8001 instead:
`ssh -N -L 8001:127.0.0.1:8000 treic@192.168.1.20`, then open `localhost:8001`.

No Ubuntu firewall change is needed for SSH forwarding. The dashboard has control
buttons and no login system; keep its default loopback binding. An explicit
`bash start.sh --host 0.0.0.0` exposes it on the LAN to anyone who can reach that port.

## 3. Keep it running after disconnecting, and start at boot

First press **Ctrl+C** in the terminal running `start.sh`, then run on Ubuntu:

```bash
cd ~/neuroplex
bash scripts/install-service.sh
sudo loginctl enable-linger "$(whoami)"
systemctl --user status neuroplex
```

This installs a user-level systemd service. Linger allows that service to start at
boot and remain running after logout. It uses the same `data/` directory and port
8000. The computer must remain powered on and awake.

```bash
systemctl --user stop neuroplex       # save and stop
systemctl --user start neuroplex      # resume saved world
systemctl --user restart neuroplex    # restart after editing Python
journalctl --user -u neuroplex -f      # follow logs; Ctrl+C exits log view
```

If `systemctl --user` reports it cannot connect to the user bus, run these commands
in a normal SSH login as `treic`, not in a `sudo` or `su` shell. After enabling
linger, reconnect if necessary. To uninstall the service, stop/disable it with
`systemctl --user disable --now neuroplex`, remove
`~/.config/systemd/user/neuroplex.service`, and run `systemctl --user daemon-reload`.

## Controls and what you are looking at

- **Pause / Resume:** stops or resumes simulated time, neural activity, and any next-life countdown.
- **1× / 2× / 5× / 10×:** requests a wall-clock speed. The 5 ms neural and 50 ms
  world steps stay fixed. Actual speed depends on your CPU; begin at 1×.
- **Learning on / frozen:** toggles both TD value updates and recurrent STDP.
  Neurons and intrinsic regulation continue when frozen, with a 5% exploration
  floor to help escape repeated actions. This does not erase learned values.
- **Save checkpoint:** saves now. Automatic saves occur every 30 real seconds
  and on normal shutdown.
- **Auto-start next life:** enabled by default, with an adjustable delay after death.
  Disable it to wait for a manual restart. Pausing holds the countdown.
- **Start new life:** immediately starts one new body/world after death, retaining
  synaptic weights and motor/goal values while resetting transient activity.
  Neither manual nor automatic respawning evolves a population.
- **Habitat:** creature, food, blue water sources, coral predators, recent trail, and a 240° field of vision. Food
  regrows at the same position 45 simulated seconds after consumption by default.
- **Through its eyes:** separate food, wall, and water channels plus omnidirectional
  threat sensing, material sight and close obstacle sensing. Blocks occlude food,
  water and threat sight. The creature has no access to global target coordinates.
- **Inside the brain:** all 500 neurons, shaded by firing rate; motor populations;
  eligibility strength and distance of current weights from their initial values.
- **Learning to survive:** starting experience, learned goal and motor intent,
  new learning updates, exploration rate, TD prediction error, and reward components.

Eating within mouth reach is an automatic body reflex, not a learned fifth motor
command. The motor policy chooses among seven body intents (including Rest); the four neural motor
populations produce forward/backward movement and left/right turning. Eating gets
a reward; time, wall contact, and death incur penalties. **Potential-based progress
feedback** rewards getting closer to and facing visible food. This is explicit
reward shaping; it supplies no desired motor action. The learner only sees the
retina and body sensations, not global food coordinates.

With water active, rewards use energy/hydration actually restored, preventing a
full creature from collecting unlimited rewards by camping on a resource. Predator
injuries carry a penalty. The learner chooses its response; no built-in rule says
which turn to take or when to switch from food to water. Reward design, sensor
encoding, motor primitives, body reflexes and predator behavior are programmed.

## Editing and updating from VS Code

Open the repo **inside the Remote SSH window** so the code and interpreter run on
Ubuntu. Python changes take effect after restarting Neuroplex. Browser files take
effect on browser refresh. There is no automatic Python reload, because duplicate
workers or restarts could interfere with the saved world.

```bash
cd ~/neuroplex &&
systemctl --user stop neuroplex &&
git pull --ff-only &&
bash setup.sh &&
systemctl --user start neuroplex
```

Commit or stash your own code edits before pulling updates if Git reports a
conflict; do not discard them. For debugging while the main service runs, the
included VS Code launch configuration uses **port 8001 and `data/debug/`**, a
separate world. Forward 8001 for that debug session.

Key files:

| File | Purpose |
| --- | --- |
| `neuroplex/config.py` | World, energy, timing, and plasticity defaults |
| `neuroplex/brain.py` | Sparse LIF neurons, STDP traces, weight updates, motor decoding |
| `neuroplex/policy.py` | Local TD action-value learning, credit traces, exploration and shaping |
| `neuroplex/assets/foraging-v0.2.json` | Actual values exported from a continuous training run |
| `neuroplex/world.py` | Body, food, retina, energy and reward |
| `neuroplex/memory.py` | Fading resource observations and motion integration |
| `neuroplex/curriculum.py` | Habitat stages and performance gates |
| `neuroplex/experiments.py` | Frozen trials, inheritance, mutation, selection and audit |
| `neuroplex/lab.py` | Bounded experiment worker, cancellation and result retention |
| `neuroplex/simulation.py` | Sense → brain → move → reward loop and checkpoints |
| `neuroplex/runner.py` | Simulation worker, pacing, process lock, autosaves |
| `neuroplex/server.py` | FastAPI routes and live WebSocket stream |
| `neuroplex/static/` | Plain HTML/CSS/JavaScript dashboard |

See [architecture](docs/ARCHITECTURE.md) for equations, the neuron map, and extension
points. Config defaults apply to **new worlds**; a resumed checkpoint carries its
original config to preserve consistent dynamics. Use a separate data directory
for a fresh experiment:

```bash
bash start.sh --port 8001 --data-dir data/experiment-2 --seed 8
```

To open a separate predator habitat immediately, add `--stage 3`; `--stage 4`
starts the hardest habitat. These flags only configure new data directories.

To start with **zero motor values** instead of the learned starting values:

```bash
bash start.sh --port 8001 --data-dir data/from-scratch --untrained --seed 7
```

Use a new directory; this flag never discards an existing checkpoint. Learning from
scratch is still an experiment and may fail on some seeds. Default pretrained
worlds can seek food immediately and continue adapting.

## Saving and recovery

`data/checkpoint.npz` contains weights, voltages, spike state, eligibility and spike
traces, thresholds, all four RNG states, motor and goal values and eligibility, the held
action and accumulated return, sensory/site memory, the cover replay buffer, resources, predators, blocks and construction credit, creature, settings, metrics, and
events. It is a numeric NumPy archive with JSON
metadata, loaded with `allow_pickle=False`. The previous save is kept as
`data/checkpoint.previous.npz`. Both are local to your OptiPlex and ignored by Git.

Graceful restarts resume the saved simulation state; powered-off time is not simulated.
The wall-clock next-life countdown starts afresh if the saved creature is dead.
After a power loss, up to the last 30 seconds of wall-clock progress may be lost.
A corrupt checkpoint stops startup with an error instead of silently resetting
your brain. To restore the previous save, stop the service, preserve the corrupt
file with a new name, then copy `checkpoint.previous.npz` to `checkpoint.npz` and
restart. Back up the whole `data/` directory while the service is stopped.

Only one process may use a data directory. If you see “already using”, stop the
other instance or select another `--data-dir`; never remove a lock to bypass it.

Metrics are also written to `data/metrics.jsonl`, rotating at 5 MB with three backups
(about 20 MB total). The checkpoint retains the last six simulated hours and the
last 256 completed lifetimes. Experiments live in `data/experiments/`: the latest 20
job directories plus the last champion are retained; older generated job directories
are removed automatically. The index keeps 100 compact results. Download a result
you want to keep before retention removes its detailed job files. Cancelling keeps
any champion from a completed generation. Server shutdown stops the worker too.

## Tests and a learning comparison

```bash
cd ~/neuroplex
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate-ecosystem.py
.venv/bin/python -m neuroplex.benchmark --seconds 300 --seeds 501 502 503 504 505 --require-improvement
```

The Python suite includes automatic-life timing/persistence and streaming checks.
Optional browser-interpolation unit tests need Node.js only on your development
machine: `node --test tests/test_motion.cjs` (no npm dependencies). Node.js is not
needed to run Neuroplex. [v0.5 validation](docs/VALIDATION-v0.5.md) describes the
live-browser checks and remaining performance limits.

Reproduce the bounded escape-practice and larger-world smoke check without touching
your saved creature: `.venv/bin/python scripts/validate-shelters.py`.

The original food benchmark fixes stage 0, turns sensory memory and curriculum off,
and compares trained and zero motor values on matched fresh worlds,
with all weights frozen, the same exploration rate, and progress rewards disabled.
It prints JSON lines and leaves your saved creature untouched. The pass gate
requires every trained trial to survive and mean food to exceed 2x the control.
This is a bounded regression check, not a universal survival guarantee.

To reproduce training from scratch and then evaluate the result:

```bash
.venv/bin/python -m neuroplex.benchmark --train-seconds 600 --train-seed 7 \
  --food-count 32 --seconds 300 --seeds 501 502 503 504 505 \
  --export-policy data/retrained-policy.json --require-improvement
```

The exported JSON is an experimental artifact; this command does not overwrite
your live world's policy. [Validation notes](docs/VALIDATION.md) include measured
results and limitations.

Six gigabytes of system RAM is ample for this small default brain. The process is
primarily single-core work; high speed settings may fully use that core. Actual
OptiPlex performance should be checked using the dashboard's actual-speed readout.
