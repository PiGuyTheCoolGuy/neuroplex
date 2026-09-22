# v0.3 verification

These are bounded software and behavior checks, run in the development environment,
not on the user's OptiPlex. The new ecology is an experiment. Food pretraining is
unchanged, but reliable predator avoidance and improved learning from evolution
have not been established across a broad set of worlds.

## Automated checks

38 pytest tests passed. Coverage includes the original food and spiking-neuron
checks plus water consumption/dehydration, omnidirectional predator sensing,
contact damage and attack cooldowns, staged resource availability, sensory-memory
motion integration and expiration, learned goal-choice causality, curriculum gates,
long-term metrics and completed lives, exact ecosystem checkpoint continuation,
v1/v2 save migration and original-file backups, frozen evaluation/source integrity,
bounded mutation and inhibitory signs, elitist selection and separate audit worlds,
a real worker process, cancellation, champion adoption, and API bounds/exports.

A real Chromium check passed at 1440-pixel desktop and 390-pixel phone widths,
with no JavaScript errors or horizontal overflow. It exercised stage changes,
memory toggling, a completed frozen evaluation, a completed evolution job with
audit, cancellation, learning freeze, CSV/result downloads and checkpoint saving.
Both screenshots were visually inspected. Adoption remained disabled while the
main creature was alive. The main creature was not replaced by the experiments.

The goal-choice test deliberately makes the learned food value highest while
the creature is very thirsty and threatened. It chooses food; changing the
learned values changes the chosen goal. This checks that needs and threats do
not secretly select a fixed action or a fixed priority behind the learned table.

The evolution test checks that an unchanged elite is retained and selection
fitness cannot decrease on the fixed selection seeds. This property is not a
guarantee that performance improves on the separate audit seeds.

## Longer online runs

Reproduce with `.venv/bin/python scripts/validate-ecosystem.py`. The full records
are in [ecosystem-smoke-v0.3.json](ecosystem-smoke-v0.3.json). Each scenario starts
with the same bundled food experience; goal selection and escape are new. Learning
is enabled. No resets or energy rescues occur within an episode.

| Scenario | Seed | Simulated time | Food | Water visits | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Food and water, fixed stage | 601 | 300 s | 59 | 32 | Alive; energy 85.02, hydration 100, health 100 |
| Two predators, fixed wild stage | 602 | 220.6 s of 300 | 31 | 17 | Died from predator injuries after 11 attacks |
| Automatic curriculum | 7 | 600 s | 208 | 31 | Alive in first-predator stage; energy 94.65, hydration 96.05, health 57.50 |

The curriculum advanced after approximately 180, 360 and 540 simulated seconds.
The final body had taken four attacks. Its reduced health prevents immediately
advancing to the two-predator stage under the configured performance gate.
The failure at the hardest stage is retained in the results: predator escape
must still be learned and may fail. These online trajectories are not a matched
learning-versus-control study, and water visits count entries, not volume consumed.

Two additional 60-second frozen food trials on seeds 801 and 802 both survived,
averaging 36 food/minute, with learned values and synapses unchanged and shaping
off. These short trials check that basic food navigation still operates; they do
not replace the historical v0.2 five-world, 300-second comparison.

## Runtime and operation

The standalone simulation validation peaked at approximately 36.4 MiB RSS on Linux.
That excludes a running web server, browser, and evolution worker. The measured
300-second water run took about 41.8 wall seconds; the 600-second curriculum took
67.5 wall seconds in this environment. Do not treat those speeds as OptiPlex timings.
The live server and a low-priority experiment can each use a CPU core. Start at 1×
and observe the actual-speed readout.

All learning remains CPU/NumPy based and no new runtime package dependency was
introduced. Package discovery stays restricted to `neuroplex`, so saved runtime
data is not a second Python package. Setup and wheel-content checks are performed
with a real saved world present; the save is verified byte-for-byte unchanged.

## How to measure a real improvement

Use matching habitat stages, episode lengths and reserved seeds. Run frozen
evaluations before and after a training interval. For water/predators, choose at
least 180–300 simulated seconds per world and multiple seeds. Read survival and
injury results alongside food rate. Do not count success on evolution's repeatedly
used selection seeds as new evidence of generalization; use its separate audit.
Repeat on additional seeds before concluding that a new memory setting, evolved
model or training interval is better. There is no guaranteed learning deadline.
