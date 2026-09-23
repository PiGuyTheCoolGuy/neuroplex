"use strict";
const $ = (id) => document.getElementById(id);
let state = null,
  socket = null,
  connected = false,
  trail = [],
  lastLife = null;
const motion = new MotionBuffer(100);
const streamTimes = {frame: -1, telemetry: -1, history: -1, charts: -1};
let worldDirty = false, lastPaint = 0, lastTrail = -Infinity;
let frameCount = 0, paintCount = 0, rateStarted = performance.now();
let eventSignature = "",
  noticeTimer = null,
  evaluationSignature = "",
  experimentStageInitialized = false;
const retinaCells = Array.from({ length: 32 }, (_, i) => {
  const cell = document.createElement("span");
  cell.title = `${i < 16 ? "Food" : "Wall"} sensor ${(i % 16) + 1}`;
  $("retina").append(cell);
  return cell;
});
const ecologyCells = Array.from({ length: 32 }, (_, i) => {
  const cell = document.createElement("span");
  $("ecology-retina").append(cell);
  return cell;
});
const motorRows = ["Forward", "Backward", "Left", "Right"].map((name) => {
  const row = document.createElement("div");
  row.className = "motor-row";
  const label = document.createElement("span");
  label.textContent = name;
  const track = document.createElement("div");
  track.className = "motor-track";
  const fill = document.createElement("div");
  fill.className = "motor-fill";
  track.append(fill);
  const value = document.createElement("strong");
  value.textContent = "0.0";
  row.append(label, track, value);
  $("motors").append(row);
  return { fill, value };
});
const clock = (seconds) => {
  const total = Math.floor(seconds),
    minutes = Math.floor(total / 60),
    rest = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
};
function notice(text, transient = false) {
  clearTimeout(noticeTimer);
  $("notice").textContent = text;
  $("notice").hidden = !text;
  if (transient)
    noticeTimer = setTimeout(() => {
      $("notice").hidden = true;
    }, 3500);
}
function connection(ok) {
  connected = ok;
  $("connection").classList.toggle("offline", !ok);
  $("connection").replaceChildren();
  const dot = document.createElement("span");
  dot.className = "dot";
  $("connection").append(
    dot,
    document.createTextNode(ok ? "Connected to habitat" : "Reconnecting…"),
  );
  ["pause", "save", "learning", "new-life", "stage", "curriculum", "memory-enabled", "auto-evaluate", "auto-life", "auto-life-delay"].forEach(
    (id) => ($(id).disabled = !ok),
  );
  document
    .querySelectorAll("[data-speed]")
    .forEach((button) => (button.disabled = !ok));
  ["run-experiment", "cancel-experiment", "adopt-champion"].forEach((id) => {
    if (!ok) $(id).disabled = true;
  });
}
function connect() {
  socket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
  );
  socket.onopen = () => {
    // The server's monotonic clock can change across a restart.
    Object.keys(streamTimes).forEach((key) => (streamTimes[key] = -1));
    motion.frames = [];
    connection(true);
    notice("");
  };
  socket.onmessage = (event) => {
    receive(JSON.parse(event.data));
  };
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    connection(false);
    notice(
      "Connection lost. The server continues running; reconnecting automatically.",
    );
    setTimeout(connect, 1500);
  };
}
async function control(action, value) {
  if (!connected) return;
  try {
    const response = await fetch("/api/control", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Neuroplex-Client": "dashboard",
      },
      body: JSON.stringify({ action, value }),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "Invalid control request",
      );
    receive(result);
    if (action === "save")
      notice(
        "Checkpoint saved. The world and brain will resume from here after a restart.",
        true,
      );
  } catch (error) {
    notice(error.message);
  }
}
$("pause").onclick = () => control(state?.paused ? "resume" : "pause");
$("learning").onclick = () => control("learning", !state?.brain.learning);
$("save").onclick = () => control("save");
$("new-life").onclick = () => control("new_life");
$("stage").onchange = () => control("stage", Number($("stage").value));
$("curriculum").onchange = () => control("curriculum", $("curriculum").checked);
$("memory-enabled").onchange = () => control("memory", $("memory-enabled").checked);
$("auto-evaluate").onchange = () => control("auto_evaluate", $("auto-evaluate").checked);
$("auto-life").onchange = () => control("auto_life", $("auto-life").checked);
$("auto-life-delay").onchange = () => {
  if ($("auto-life-delay").reportValidity()) control("auto_life_delay", Number($("auto-life-delay").value));
};
$("cancel-experiment").onclick = () => control("cancel_experiment");
$("adopt-champion").onclick = () => control("adopt_champion");
$("trend-metric").onchange = () => state && drawTrends();
$("experiment-kind").onchange = () => {
  const evolving = $("experiment-kind").value === "evolve";
  const escape = $("experiment-kind").value === "escape";
  document.querySelectorAll(".evolution-field").forEach((field) => (field.hidden = !evolving));
  document.querySelectorAll(".evaluation-field").forEach((field) => (field.hidden = evolving));
  document.querySelectorAll(".escape-field").forEach((field) => (field.hidden = !escape));
  $("escape-help").hidden = !escape;
  $("experiment-stage").disabled = escape;
  if (escape) { $("experiment-stage").value = "3"; $("experiment-seconds").value = "12"; }
  $("experiment-seconds").max = escape ? "60" : "600";
};
$("experiment-form").onsubmit = async (event) => {
  event.preventDefault();
  if (!connected) return;
  $("run-experiment").disabled = true;
  const body = {
    kind: $("experiment-kind").value, stage: Number($("experiment-stage").value),
    seconds: Number($("experiment-seconds").value), seed: Number($("experiment-seed").value),
    trials: Number($("experiment-trials").value), population: Number($("experiment-population").value),
    generations: Number($("experiment-generations").value), evaluation_seconds: Number($("experiment-eval-seconds").value),
    episodes: Number($("experiment-episodes").value),
  };
  try {
    const response = await fetch("/api/experiments", {method: "POST", headers: {
      "Content-Type": "application/json", "X-Neuroplex-Client": "dashboard",
    }, body: JSON.stringify(body)});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Check the experiment settings.");
    receive(result);
  } catch (error) {
    notice(error.message);
    $("run-experiment").disabled = !connected;
  }
};
document
  .querySelectorAll("[data-speed]")
  .forEach(
    (button) =>
      (button.onclick = () => control("speed", Number(button.dataset.speed))),
  );
$("show-vision").onchange = () => { worldDirty = true; };

function receive(packet) {
  const kind = packet.kind || "state", stamp = packet.stream_time;
  // Separate watermarks prevent a delayed HTTP response or cached frame from
  // rewinding newer live motion, controls, or charts.
  if ((kind === "state" || kind === "telemetry") && stamp > streamTimes.telemetry) {
    streamTimes.telemetry = stamp;
    const {history, metrics, latest_metric, life_records, ...details} = packet;
    state = {...state, ...details};
    const latest = motion.frames.at(-1)?.frame;
    if (latest && stamp <= streamTimes.frame) applyFrameDetails(latest);
    if (kind === "state" && stamp > streamTimes.charts) {
      Object.assign(state, {metrics, latest_metric, life_records});
      streamTimes.charts = stamp;
      drawTrends();
    }
    if (kind === "state" && stamp > streamTimes.history) {
      state.history = history;
      streamTimes.history = stamp;
      drawHistory();
    }
    update();
  }
  if (["frame", "state", "telemetry"].includes(kind) && stamp > streamTimes.frame) {
    streamTimes.frame = stamp;
    if (kind === "frame") frameCount++;
    if (lastLife !== packet.life) { trail = []; lastTrail = -Infinity; }
    lastLife = packet.life;
    motion.push(packet, performance.now());
    worldDirty = true;
    if (state) {
      applyFrameDetails(packet);
      updateLifecycle();
    }
  }
  if (kind === "charts" && state && stamp > streamTimes.charts) {
    streamTimes.charts = stamp;
    Object.assign(state, {metrics: packet.metrics,
      latest_metric: packet.latest_metric, life_records: packet.life_records});
    drawTrends();
  }
  if (kind === "history" && state && stamp > streamTimes.history) {
    streamTimes.history = stamp;
    state.history = packet.history;
    drawHistory();
  }
}

function applyFrameDetails(packet) {
  for (const key of ["life", "paused", "speed", "auto_life", "auto_life_delay", "auto_life_status"]) state[key] = packet[key];
  state.world = {...state.world, ...packet.world};
}

function updateLifecycle() {
  const w = state.world;
  document.title = `Neuroplex · Life ${String(state.life).padStart(3, "0")}`;
  $("lifetime").textContent = clock(w.time);
  $("life-number").textContent = `LIFE ${String(state.life).padStart(2, "0")}`;
  $("pause").textContent = state.paused ? "▶ Resume" : "Ⅱ Pause";
  $("pause").disabled = !connected;
  $("world-status").textContent = !w.alive ? "LIFE ENDED" : state.paused ? "PAUSED" : "LIVE SIMULATION";
  $("death-overlay").hidden = w.alive;
  $("death-reason").textContent = `Life ended: ${w.death_reason}.`;
  $("new-life").disabled = !connected || w.alive;
  $("auto-life").checked = state.auto_life;
  if (document.activeElement !== $("auto-life-delay")) $("auto-life-delay").value = state.auto_life_delay;
  const timer = state.auto_life_status;
  const text = !state.auto_life ? "Automatic next life is off." : timer.paused
    ? `Countdown paused · ${Math.ceil(timer.remaining ?? state.auto_life_delay)} seconds left. Resume to continue.`
    : `Next life in ${Math.ceil(timer.remaining ?? state.auto_life_delay)} seconds · learning preserved.`;
  if ($("auto-life-countdown").textContent !== text) $("auto-life-countdown").textContent = text;
}

function update() {
  const w = state.world,
    b = state.brain;
  $("world-dimensions").textContent = `${w.width} × ${w.height}`;
  $("construction-status").textContent = w.blocks.length
    ? `${w.blocks.length} movable blocks · ${w.push_distance.toFixed(1)} units pushed · ${w.construction_reward_total.toFixed(2)} building reward earned · ${w.protected_seconds.toFixed(1)}s screened from predators`
    : "Food-only warm-up. Blocks appear from stage 2 (Scarce food).";
  updateLifecycle();
  document
    .querySelectorAll("[data-speed]")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(Number(button.dataset.speed) === state.speed),
      ),
    );
  $("actual-speed").textContent =
    `${state.runtime.actual_speed.toFixed(1)}× actual`;
  $("learning").setAttribute("aria-pressed", String(b.learning));
  $("learning").lastChild.textContent = b.learning
    ? " Learning on"
    : " Learning frozen";
  $("energy").textContent = w.energy.toFixed(1);
  $("energy-bar").style.width = `${(100 * w.energy) / w.max_energy}%`;
  $("energy-bar").style.background = w.energy < 25 ? "#e2a76d" : "#86e9c5";
  $("hunger").textContent = `${Math.round(w.hunger * 100)}%`;
  $("vital-status").textContent = !w.alive
    ? "DEAD"
    : w.energy < 25
      ? "HUNGRY"
      : "ALIVE";
  $("movement").textContent = `${Math.abs(w.speed).toFixed(1)} u/s`;
  $("hydration").textContent = w.water_active ? `${w.hydration.toFixed(1)} / ${w.max_hydration}` : "Not needed at this stage";
  $("hydration-bar").style.width = `${w.water_active ? 100 * w.hydration / w.max_hydration : 0}%`;
  $("health").textContent = `${w.health.toFixed(1)} / ${w.max_health}`;
  $("health-bar").style.width = `${100 * w.health / w.max_health}%`;
  $("drinks-attacks").textContent = `${w.drinks} / ${w.attacks}`;
  $("stage").value = String(w.stage);
  $("stage-name").textContent = w.stage_name.toUpperCase();
  $("curriculum").checked = state.curriculum.enabled;
  $("memory-enabled").checked = state.memory.enabled;
  $("auto-evaluate").checked = state.auto_evaluate;
  $("curriculum-status").textContent = state.curriculum.enabled
    ? `${clock(state.curriculum.stage_seconds)} in this stage · minimum ${clock(state.curriculum.minimum_seconds)} plus a healthy 60-second performance window. ${b.learning ? "" : "Advancement pauses while learning is frozen."}`
    : "Manual habitat: the stage stays fixed until you change it.";
  if (!experimentStageInitialized) {
    $("experiment-stage").value = String(w.stage);
    experimentStageInitialized = true;
  }
  $("eaten").textContent = w.eaten;
  $("distance").replaceChildren(
    document.createTextNode(`${w.distance.toFixed(1)} `),
  );
  const unit = document.createElement("em");
  unit.textContent = "u";
  $("distance").append(unit);
  $("reward").textContent = `${b.reward >= 0 ? "+" : ""}${b.reward.toFixed(3)}`;
  $("network-size").textContent =
    `${b.neurons} / ${b.synapses.toLocaleString()}`;
  $("firing-rate").textContent = `${b.mean_rate.toFixed(1)} Hz`;
  $("weight-change").textContent = b.weight_change.toFixed(5);
  $("eligibility").textContent = b.eligibility_mean.toFixed(3);
  $("policy-source").textContent = b.policy.source.startsWith("escape-practice:") ? "Escape practice" : b.policy.source.startsWith("evolution:") ? "Evolved" : b.policy.source.startsWith("bundled:") ? "Pretrained food skill" : "From scratch";
  $("policy-action").textContent = b.policy.action;
  $("policy-goal").textContent = b.policy.goal;
  $("goal-updates").textContent = b.policy.goal_updates.toLocaleString();
  $("policy-updates").textContent = b.policy.live_updates.toLocaleString();
  $("policy-exploration").textContent =
    `${(100 * b.policy.exploration).toFixed(1)}%`;
  $("policy-td").textContent = b.policy.td_error.toFixed(3);
  $("policy-base").textContent = b.policy.base_reward.toFixed(3);
  $("policy-shaping").textContent = b.policy.shaping_reward.toFixed(3);
  retinaCells.forEach((cell, i) => {
    const intensity = w.retina[i];
    cell.style.background =
      i < 16
        ? `rgba(134,233,197,${0.04 + intensity * 0.96})`
        : `rgba(112,181,223,${0.04 + intensity * 0.96})`;
    cell.title = `${i < 16 ? "Food" : "Wall"} sensor ${(i % 16) + 1}: ${intensity.toFixed(2)}`;
  });
  ecologyCells.forEach((cell, i) => {
    const intensity = i < 16 ? w.water_retina[i] : w.danger_retina[i - 16];
    cell.style.background = `rgba(${i < 16 ? "112,181,223" : "237,143,137"},${0.04 + intensity * 0.96})`;
    cell.title = `${i < 16 ? "Water" : "Threat"} ${(i % 16) + 1}: ${intensity.toFixed(2)}`;
  });
  $("memory-status").textContent = state.memory.enabled
    ? `Sensory traces · food ${Math.round(state.memory.food_confidence * 100)}% (${state.memory.food_age.toFixed(1)}s) · water ${Math.round(state.memory.water_confidence * 100)}% · expire after ${state.memory.seconds.toFixed(1)}s.`
    : "Sensory memory is off.";
  const lives = state.life_records.slice(-5);
  $("life-summary").textContent = lives.length
    ? lives.map((life) => `Life ${life.life}: ${clock(life.survival_seconds)} · ${life.cause} · ${life.food} food`).join(" | ")
    : "No completed lifetimes yet. Survival times and causes of death will appear here.";
  updateLab();
  motorRows.forEach(({ fill, value }, i) => {
    fill.style.width = `${Math.min(100, (b.motor_rates[i] / 50) * 100)}%`;
    value.textContent = b.motor_rates[i].toFixed(1);
  });
  const signature = JSON.stringify(state.events);
  if (signature !== eventSignature) {
    $("events").replaceChildren();
    state.events.forEach((event) => {
      const li = document.createElement("li"),
        time = document.createElement("time"),
        msg = document.createElement("span");
      time.textContent = clock(event.time);
      msg.textContent = event.message;
      li.append(time, msg);
      $("events").append(li);
    });
    eventSignature = signature;
  }
  $("saved").textContent = state.saved_at
    ? `SAVED ${new Date(state.saved_at).toLocaleTimeString()}`
    : "Autosaves every 30 seconds";
  if (state.runtime.error) notice(`Simulation stopped: ${state.runtime.error}`);
  drawNeurons();
}
function updateLab() {
  const lab = state.lab, current = lab.current;
  $("lab-status").textContent = current.state.toUpperCase();
  $("run-experiment").disabled = !connected || lab.running;
  $("cancel-experiment").disabled = !connected || !lab.running;
  $("adopt-champion").disabled = !connected || lab.running || state.world.alive || !lab.champion_available;
  const parts = [current.phase || "One low-priority CPU worker"];
  if (current.generation) parts.push(`generation ${current.generation}/${current.request.generations}`);
  if (current.candidate) parts.push(`candidate ${current.candidate}/${current.request.population}`);
  if (current.trial) parts.push(`world ${current.trial}`);
  if (current.episode) parts.push(`episode ${current.episode}/${current.request.episodes}`);
  if (current.episode_seconds != null) parts.push(`${current.episode_seconds.toFixed(1)} simulated seconds`);
  $("lab-progress").textContent = current.error || parts.join(" · ");
  const result = current.summary;
  $("lab-result").textContent = result
    ? `${current.kind === "escape" ? "Escape audit" : current.kind === "evolve" ? "Unseen audit" : "Frozen evaluation"}: ${result.survived}/${result.trials} survived the full trial · ${result.mean_food_per_minute.toFixed(1)} food/min · ${result.mean_survival_seconds.toFixed(1)}s mean survival · ${result.mean_damage.toFixed(1)} mean damage.`
    : "";
  if (current.baseline_audit) $("lab-result").textContent += ` Starting model: ${current.baseline_audit.summary.survived}/${current.baseline_audit.summary.trials} survived those same audit worlds, ${current.baseline_audit.summary.mean_damage.toFixed(1)} mean damage.`;
  $("download-result").hidden = current.state !== "completed";
  $("evolution-chart").hidden = !current.generations?.length;
  if (current.generations?.length) drawChart("evolution-chart", current.generations, [{key: "best_fitness", color: "#b6a1e9"}], "generation");
  const signature = JSON.stringify(lab.history);
  if (signature !== evaluationSignature) {
    $("evaluation-history").replaceChildren();
    lab.history.slice(-8).reverse().forEach((record) => {
      const row = document.createElement("p");
      row.textContent = `${new Date(record.finished_at * 1000).toLocaleString()} · ${record.kind} · ${state.curriculum.stages[record.stage]} · ${record.summary.survived}/${record.summary.trials} survived · ${record.summary.mean_food_per_minute.toFixed(1)} food/min`;
      $("evaluation-history").append(row);
    });
    evaluationSignature = signature;
  }
}
function context(id) {
  const canvas = $(id),
    ratio = window.devicePixelRatio || 1,
    box = canvas.getBoundingClientRect();
  const width = Math.round(box.width * ratio),
    height = Math.round(box.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);
  return [ctx, box.width, box.height];
}
function drawWorld(w) {
  const [ctx, width, height] = context("world"),
    scale = width / w.width;
  ctx.save();
  ctx.scale(scale, height / w.height);
  ctx.strokeStyle = "#1c2b31";
  ctx.lineWidth = 0.055;
  ctx.beginPath();
  for (let x = 0; x <= w.width; x += 4) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, w.height);
  }
  for (let y = 0; y <= w.height; y += 4) {
    ctx.moveTo(0, y);
    ctx.lineTo(w.width, y);
  }
  ctx.stroke();
  if (trail.length > 1) {
    ctx.beginPath();
    trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.strokeStyle = "#857bad48";
    ctx.lineWidth = 0.2;
    ctx.stroke();
  }
  if ($("show-vision").checked && w.alive) {
    const gradient = ctx.createRadialGradient(
      w.x,
      w.y,
      0,
      w.x,
      w.y,
      w.vision_range,
    );
    gradient.addColorStop(0, "#86e9c514");
    gradient.addColorStop(1, "#86e9c501");
    ctx.beginPath();
    ctx.moveTo(w.x, w.y);
    ctx.arc(
      w.x,
      w.y,
      w.vision_range,
      w.heading - w.vision_fov / 2,
      w.heading + w.vision_fov / 2,
    );
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.strokeStyle = "#86e9c51b";
    ctx.lineWidth = 0.08;
    ctx.stroke();
  }
  for (const [x, y] of w.water) {
    ctx.beginPath();
    ctx.arc(x, y, w.water_radius, 0, Math.PI * 2);
    ctx.fillStyle = "#70b5df35";
    ctx.fill();
    ctx.strokeStyle = "#70b5dfb0";
    ctx.lineWidth = 0.12;
    ctx.stroke();
  }
  for (const [x, y] of w.blocks || []) {
    const half = w.block_size / 2;
    ctx.fillStyle = "#8e7354";
    ctx.fillRect(x - half, y - half, w.block_size, w.block_size);
    ctx.strokeStyle = "#d3b583";
    ctx.lineWidth = .15;
    ctx.strokeRect(x - half, y - half, w.block_size, w.block_size);
    ctx.beginPath(); ctx.moveTo(x - half + .3, y - half + .4); ctx.lineTo(x + half - .3, y - half + .4); ctx.stroke();
  }
  for (const predator of w.predators) {
    ctx.save();
    ctx.translate(predator.x, predator.y);
    ctx.rotate(predator.heading);
    ctx.beginPath();
    ctx.moveTo(w.predator_radius * 1.5, 0);
    ctx.lineTo(-w.predator_radius, -w.predator_radius);
    ctx.lineTo(-w.predator_radius * 0.5, 0);
    ctx.lineTo(-w.predator_radius, w.predator_radius);
    ctx.closePath();
    ctx.fillStyle = predator.hunting ? "#ed8f89" : "#b77675";
    ctx.fill();
    ctx.restore();
  }
  for (const [x, y] of w.food) {
    ctx.beginPath();
    ctx.arc(x, y, w.food_radius + 0.45, 0, Math.PI * 2);
    ctx.fillStyle = "#86e9c50b";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, w.food_radius * 0.55, 0, Math.PI * 2);
    ctx.fillStyle = "#83caaa";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x - 0.1, y - 0.1, 0.09, 0, Math.PI * 2);
    ctx.fillStyle = "#d3ffe9";
    ctx.fill();
  }
  ctx.translate(w.x, w.y);
  ctx.rotate(w.heading);
  ctx.beginPath();
  ctx.arc(0, 0, w.radius + 0.6, 0, Math.PI * 2);
  ctx.strokeStyle = "#b49af340";
  ctx.lineWidth = 0.1;
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(0, 0, w.radius * 1.3, w.radius, 0, 0, Math.PI * 2);
  ctx.fillStyle = w.alive ? "#b6a1e9" : "#68707c";
  ctx.fill();
  ctx.fillStyle = "#252239";
  for (const y of [-0.35, 0.35]) {
    ctx.beginPath();
    ctx.arc(0.7, y, 0.13, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.beginPath();
  ctx.moveTo(w.radius * 1.8, 0);
  ctx.lineTo(w.radius * 2.35, 0);
  ctx.strokeStyle = "#e2d7ff";
  ctx.lineWidth = 0.12;
  ctx.stroke();
  ctx.restore();
}
function drawNeurons() {
  const [ctx, width, height] = context("neurons"),
    cols = 25,
    rows = 20,
    cw = width / cols,
    ch = height / rows;
  state.brain.rates.forEach((rate, i) => {
    const color =
      i < 80 ? "112,181,223" : i < 400 ? "134,233,197" : "180,154,243";
    ctx.fillStyle = `rgba(${color},${0.12 + Math.min(1, rate / 45) * 0.88})`;
    ctx.fillRect(
      (i % cols) * cw + 1,
      Math.floor(i / cols) * ch + 1,
      cw - 2,
      ch - 2,
    );
  });
}
function drawHistory() {
  drawChart("history", state.history, [
    {key: "energy", color: "#86e9c5"}, {key: "hydration", color: "#70b5df"}, {key: "health", color: "#ed8f89"},
  ], "time", 100);
  if (state.history.length) $("history-range").textContent = `${clock(state.history[0].time)} — ${clock(state.history.at(-1).time)}`;
}
function drawTrends() {
  const key = $("trend-metric").value;
  drawChart("trends", state.metrics, [{key, color: "#86e9c5"}]);
  const last = state.latest_metric;
  $("trend-current").textContent = Number.isFinite(last?.[key]) ? last[key].toFixed(key === "weight_change" ? 5 : 2) : "Waiting for samples";
  if (state.metrics.length) $("trend-caption").textContent = `${clock(state.metrics[0].time)} — ${clock(last.time)} total simulated time · up to 6 hours`;
}
function drawChart(id, data, series, xKey = "time", fixedMax = null) {
  const [ctx, width, height] = context(id);
  if (width < 1 || height < 1) return;
  const left = 38, right = width - 8, top = 12, bottom = height - 18;
  const values = data.flatMap((point) => series.map((s) => point[s.key]).filter(Number.isFinite));
  const minimum = Math.min(0, ...values), maximum = fixedMax ?? Math.max(1, ...values);
  const range = Math.max(1e-8, maximum - minimum);
  ctx.font = "9px ui-monospace, monospace";
  [minimum, minimum + range / 2, maximum].forEach((value) => {
    const y = bottom - ((value - minimum) / range) * (bottom - top);
    ctx.fillStyle = "#69808b";
    ctx.fillText(value.toFixed(range < 1 ? 3 : 0), 0, y + 3);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y);
    ctx.strokeStyle = "#24313a"; ctx.lineWidth = 0.6; ctx.stroke();
  });
  if (!data.length) return;
  const start = data[0][xKey], span = Math.max(1, data.at(-1)[xKey] - start);
  series.forEach(({key, color}) => {
    ctx.beginPath();
    let previousLife = null, drawing = false;
    data.forEach((point) => {
      if (!Number.isFinite(point[key])) { drawing = false; return; }
      const x = left + ((point[xKey] - start) / span) * (right - left);
      const y = bottom - ((point[key] - minimum) / range) * (bottom - top);
      if (!drawing || (previousLife !== null && point.life !== previousLife)) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      drawing = true; previousLife = point.life ?? null;
    });
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
    if (data.length === 1) {
      ctx.beginPath(); ctx.arc(left, bottom - ((data[0][key] - minimum) / range) * (bottom - top), 2, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
    }
  });
  ctx.fillStyle = "#69808b";
  ctx.fillText(xKey === "generation" ? `generation ${start}` : clock(start), left, height - 2);
  const endLabel = xKey === "generation" ? String(data.at(-1)[xKey]) : clock(data.at(-1)[xKey]);
  ctx.fillText(endLabel, right - ctx.measureText(endLabel).width, height - 2);
}
window.addEventListener("resize", () => {
  if (state) {
    worldDirty = true;
    drawNeurons();
    drawHistory();
    drawTrends();
    updateLab();
  }
});
document.addEventListener("visibilitychange", () => {
  const latest = motion.frames.at(-1);
  motion.frames = [];
  if (latest) motion.push(latest.frame, performance.now());
  worldDirty = true;
});
function animate(now) {
  const latest = motion.frames.at(-1);
  const moving = connected && latest && !latest.frame.paused && latest.frame.world.alive && now - latest.at < 250;
  if (state && !document.hidden && (worldDirty || moving) && now - lastPaint >= 1000 / 65) {
    const world = motion.sample(now);
    if (world) {
      if (moving && now - lastTrail >= 200) {
        trail.push([world.x, world.y]);
        if (trail.length > 300) trail.shift();
        lastTrail = now;
      }
      drawWorld(world);
      lastPaint = now;
      paintCount++;
      worldDirty = false;
    }
  }
  if (now - rateStarted >= 2000) {
    const seconds = (now - rateStarted) / 1000;
    $("stream-status").textContent = connected
      ? `${Math.round(frameCount / seconds)} state updates/s · ${Math.round(paintCount / seconds)} rendered frames/s · 100 ms smoothing buffer`
      : "Disconnected · last known frame";
    frameCount = paintCount = 0;
    rateStarted = now;
  }
  requestAnimationFrame(animate);
}
connection(false);
connect();
requestAnimationFrame(animate);
