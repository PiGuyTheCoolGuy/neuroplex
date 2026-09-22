# v0.3.1 automatic lives and smooth state streaming

Verified on the development Linux environment, **not the user's OptiPlex**.
The learning rules, bundled policy, world physics, and predator rules are unchanged.
The [v0.3 ecological measurements](VALIDATION-v0.3.md) remain the relevant bounded
learning checks; this release does not establish better predator survival.

## Automated regression checks

`python -m pytest -q`: **49 passed**. This includes the 38 previous checks and:

- Automatic respawn after real-time delay at both 1× and 10× simulation speed.
- Exact learned-parameter fingerprint, visits/update counters, frozen setting,
  habitat stage, lifetime records, elapsed time, and metrics preserved across death.
- Exactly one new body; fresh sensory/activity traces; checkpoint and previous-save
  backup retain the new and completed life respectively at the transition.
- Pause freezes the countdown, disable cancels it, delay change restarts it,
  manual new-life cancels the pending transition, and runtime errors block respawn.
- Persistent settings, compatibility with v3 saves lacking the new optional fields,
  a fresh delay on server restart, and strict HTTP control bounds/types.
- A real running server starts the next life with no WebSocket client connected.
- Frozen trial death still ends the trial instead of spawning within evaluation.
- Shared serialized motion caches, compact frame payloads, unchanged physics when
  read, and no repeated transmission of unchanged long-term chart data.

`node --test tests/test_motion.cjs`: **6 passed**, with no npm dependencies:

- Body/predator interpolation between received poses; no source-state mutation.
- Shortest-angle rotation across the -π/π boundary.
- Discrete food disappearance, not interpolation between unrelated resources.
- No extrapolation beyond the last received position during a disconnect.
- Buffer reset for new life, death, pause, speed/stage change, clock rewind and
  a long reception gap; bounded memory while the tab is not drawing.

Node.js is only needed for these optional developer tests, not to run Neuroplex.
Two existing dependency deprecation warnings remain in the Python test output.
`bash setup.sh` also completed successfully and installed the v0.3.1 editable
package without adding dependencies or requiring the runtime data to be removed.

## Live Chromium check

Started a real server with a temporary, dead, paused stage-4 checkpoint and known
learned-parameter fingerprint. Through the dashboard:

1. Confirmed the default enabled 10-second timer was held by Pause.
2. Changed the delay to 2 seconds, disabled automatic lives and resumed; the dead
   creature stayed dead beyond the delay.
3. Re-enabled automatic lives; Life 2 began, death overlay closed, and the learned
   fingerprint remained exact. The learning-frozen flag stayed frozen.
4. Observed running motion for approximately 3.2 seconds. Measured **19.0 geometry
   updates/s**, **58.4 canvas draws/s**, and **2,137 bytes per geometry packet** on
   average. 187 distinct predator positions were rendered, demonstrating actual
   interpolation between the fewer received poses, not just repeated redraws.
5. Confirmed pause held the displayed pose, a deliberately stale packet could not
   reverse a control result, and habitat stage changes continued to work.

Inspected desktop (1440 px) and mobile (390 px) screenshots, including the mobile
death overlay and next-life controls. No horizontal overflow or JavaScript errors.
These are short developer-machine checks, not a long-run/slow-network benchmark.

## Performance design and limits

- Target geometry stream: 20 Hz. Small JSON state messages, not encoded video.
- Browser target: approximately 60 fps using a 100 ms interpolation buffer.
- Brain/vital telemetry: 5 Hz, without the growing chart arrays.
- Vital history: at most 1 Hz and only when changed. Long-term graphs: only on a
  new sample/lifetime, at most 1 Hz. All clients reuse the same serialized caches.
- Clients request only current cached messages; the application keeps no playback
  queue. A blocked socket write times out after 5 seconds. Ordinary network buffers
  can still introduce latency; this is not a hard real-time delivery guarantee.
- Frame and rendering rates are shown below the habitat. Paused scenes draw less;
  background tabs are throttled by the browser. The server keeps learning either way.

Rendering does not increase the neural or physics timestep or alter rewards.
At higher simulation speeds, each displayed interval can span more simulated time.
Actual throughput and smoothness depend on CPU, browser, network and screen refresh
rate. Start at 1× on the OptiPlex; no GPU, codec, extra service, or new tunnel is needed.

Automatic lives preserve learning but do not guarantee improvement or immortality.
Pause or disable them to inspect a death or manually adopt an evolved champion.
