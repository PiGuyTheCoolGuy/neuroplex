# Neuroplex

**One creature, one spiking brain, one goal: eat to stay alive.**

Neuroplex v0.1 is a CPU-only artificial-life experiment for Ubuntu 24.04. A Python
process runs the world continuously; your laptop's browser displays it. Closing the
browser does not stop the simulation. There is no GPU, cloud model, PyTorch, Node.js
build, or paid API to configure.

The brain has **500 leaky integrate-and-fire neurons and 16,000 sparse synapses**.
It changes excitatory connections online using reward-modulated STDP and five-second
eligibility traces. Recurrent activity provides short-lived state; synaptic changes
provide longer-lived memory. No backpropagation, training batches, or automatic
episode resets are involved.

This is an experimental learning system, **not a pretrained food-finding agent**.
It starts with random connections and an innate tendency to explore. It can eat,
starve, get stuck, or learn unhelpful behavior. Weight changes alone do not prove
that it learned better navigation. See [validation](docs/VALIDATION.md).

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
- **Learning on / frozen:** toggles reward-driven weight updates. Neurons, memory
  traces, exploration, and intrinsic activity regulation still run when frozen.
- **Save checkpoint:** saves now. Automatic saves occur every 30 real seconds
  and on normal shutdown.
- **Start new life:** available only after starvation. Creates one new body/world,
  retains synaptic weights, and explicitly resets transient neural activity. It
  does not happen automatically and does not evolve a population.
- **Habitat:** creature, food, recent trail, and a 240° field of vision. Food
  regrows at the same position 25 simulated seconds after consumption.
- **Through its eyes:** sixteen food brightness bins and sixteen wall-distance
  bins. The brain has no access to global food coordinates.
- **Inside the brain:** all 500 neurons, shaded by firing rate; motor populations;
  eligibility strength and distance of current weights from their initial values.

Eating within mouth reach is an automatic body reflex, not a learned fifth motor
command. The four neural motor populations determine forward/backward movement
and left/right turning. Food, metabolism, and movement change hunger. The reward
is exactly **hunger_before − hunger_after**; there is no hidden food-direction
reward or steering controller.

There is no thirst, predator, mating, reproduction, evolution, object manipulation,
language model, or episodic-memory database in v0.1. The thirst input block is
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

## Saving and recovery

`data/checkpoint.npz` contains weights, voltages, spike state, eligibility and spike
traces, adaptive thresholds, motor noise, both RNG states, food/regrowth, creature,
settings, recent metrics, and events. It is a numeric NumPy archive with JSON
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
.venv/bin/python -m neuroplex.benchmark --seconds 180 --seeds 7 8 9
```

The benchmark runs separate, deterministic worlds with learning on and frozen,
prints JSON lines, and leaves your saved creature untouched. Compare food eaten,
survival time, and energy across seeds. It does **not** certify intelligent behavior
or long-term learning. [Validation notes](docs/VALIDATION.md) include measured
results and their limits.

Six gigabytes of system RAM is ample for this small default brain. The process is
primarily single-core work; high speed settings may fully use that core. Actual
OptiPlex performance should be checked using the dashboard's actual-speed readout.
