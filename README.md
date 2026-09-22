# Neuroplex

**One creature, one spiking brain, one goal: eat to stay alive.**

Neuroplex v0.2 is a CPU-only artificial-life experiment for Ubuntu 24.04. A Python
process runs the world continuously; your laptop's browser displays it. Closing the
browser does not stop the simulation. There is no GPU, cloud model, PyTorch, Node.js
build, or paid API to configure.

The brain has **500 leaky integrate-and-fire neurons and 16,000 sparse synapses**,
plus a **918-value TD motor policy** that learns which movement helps in each visual
context. This is a hybrid: action values choose motor intent, and spiking neurons
produce movement. Recurrent STDP and motor learning both continue during its life.
There is no backpropagation, action teacher, replay buffer, or automatic life reset.

**It now starts with learned food-seeking values**, obtained from ten simulated
minutes of continuous training. Food rewards, progress feedback, action-specific
credit, and decreasing exploration replace the old undirected learning behavior.
Use `--untrained` in a new data directory to watch learning from zero. Controlled
tests freeze all weights and disable progress rewards in unfamiliar worlds;
[validation](docs/VALIDATION.md) records results and limits.

## Upgrading from v0.1

Run this in your Ubuntu / VS Code SSH terminal:

```bash
cd ~/neuroplex
systemctl --user stop neuroplex
git pull --ff-only
bash setup.sh
systemctl --user start neuroplex
```

If running in the foreground, use Ctrl+C before updating, then `bash start.sh`.
Refresh the browser. Your existing world and recurrent weights are preserved;
the learned motor policy is added automatically. The original checkpoint is kept
as `data/checkpoint.v1.npz` for rollback. If the creature was already dead, click
**Start new life · keep memory**. A saved pause or learning-freeze setting is also
preserved, so press Resume / Learning on if needed.

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

- **Pause / Resume:** stops or resumes simulated time, including neural activity.
- **1× / 2× / 5× / 10×:** requests a wall-clock speed. The 5 ms neural and 50 ms
  world steps stay fixed. Actual speed depends on your CPU; begin at 1×.
- **Learning on / frozen:** toggles both TD value updates and recurrent STDP.
  Neurons and intrinsic regulation continue when frozen, with a 5% exploration
  floor to help escape repeated actions. This does not erase learned values.
- **Save checkpoint:** saves now. Automatic saves occur every 30 real seconds
  and on normal shutdown.
- **Start new life:** available only after starvation. Creates one new body/world,
  retains synaptic weights and motor action values, and resets transient activity. It
  does not happen automatically and does not evolve a population.
- **Habitat:** creature, food, recent trail, and a 240° field of vision. Food
  regrows at the same position 25 simulated seconds after consumption.
- **Through its eyes:** sixteen food brightness bins and sixteen wall-distance
  bins. The brain has no access to global food coordinates.
- **Inside the brain:** all 500 neurons, shaded by firing rate; motor populations;
  eligibility strength and distance of current weights from their initial values.
- **Learning to forage:** pretrained/from-scratch source, chosen motor intent,
  new learning updates, exploration rate, TD prediction error, and reward components.

Eating within mouth reach is an automatic body reflex, not a learned fifth motor
command. The motor policy chooses among six body intents; the four neural motor
populations produce forward/backward movement and left/right turning. Eating gets
a reward; time, wall contact, and death incur penalties. **Potential-based progress
feedback** rewards getting closer to and facing visible food. This is explicit
reward shaping; it supplies no desired motor action. The learner only sees the
retina and body sensations, not global food coordinates.

There is no thirst, predator, mating, reproduction, evolution, object manipulation,
language model, or episodic-memory database in v0.2. The thirst input block is
reserved but unused. The current scope is deliberately one creature eating food.

## Editing and updating from VS Code

Open the repo **inside the Remote SSH window** so the code and interpreter run on
Ubuntu. Python changes take effect after restarting Neuroplex. Browser files take
effect on browser refresh. There is no automatic Python reload, because duplicate
workers or restarts could interfere with the saved world.

```bash
cd ~/neuroplex
systemctl --user stop neuroplex
git pull --ff-only
bash setup.sh
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

To start with **zero motor values** instead of the learned starting values:

```bash
bash start.sh --port 8001 --data-dir data/from-scratch --untrained --seed 7
```

Use a new directory; this flag never discards an existing checkpoint. Learning from
scratch is still an experiment and may fail on some seeds. Default pretrained
worlds can seek food immediately and continue adapting.

## Saving and recovery

`data/checkpoint.npz` contains weights, voltages, spike state, eligibility and spike
traces, thresholds, all three RNG states, motor values and eligibility, the held
action and accumulated return, food/regrowth, creature, settings, metrics, and
events. It is a numeric NumPy archive with JSON
metadata, loaded with `allow_pickle=False`. The previous save is kept as
`data/checkpoint.previous.npz`. Both are local to your OptiPlex and ignored by Git.

Graceful restarts resume the exact saved state; powered-off time is not simulated.
After a power loss, up to the last 30 seconds of wall-clock progress may be lost.
A corrupt checkpoint stops startup with an error instead of silently resetting
your brain. To restore the previous save, stop the service, preserve the corrupt
file with a new name, then copy `checkpoint.previous.npz` to `checkpoint.npz` and
restart. Back up the whole `data/` directory while the service is stopped.

Only one process may use a data directory. If you see “already using”, stop the
other instance or select another `--data-dir`; never remove a lock to bypass it.

## Tests and a learning comparison

```bash
cd ~/neuroplex
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m neuroplex.benchmark --seconds 300 --seeds 501 502 503 504 505 --require-improvement
```

The benchmark compares trained and zero motor values on matched fresh worlds,
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
