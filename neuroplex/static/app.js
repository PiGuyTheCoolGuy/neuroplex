"use strict";
const $ = (id) => document.getElementById(id);
let state = null,
  socket = null,
  connected = false,
  trail = [],
  lastTime = -1,
  lastLife = null;
let eventSignature = "",
  noticeTimer = null;
const retinaCells = Array.from({ length: 32 }, (_, i) => {
  const cell = document.createElement("span");
  cell.title = `${i < 16 ? "Food" : "Wall"} sensor ${(i % 16) + 1}`;
  $("retina").append(cell);
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
  ["pause", "save", "learning", "new-life"].forEach(
    (id) => ($(id).disabled = !ok),
  );
  document
    .querySelectorAll("[data-speed]")
    .forEach((button) => (button.disabled = !ok));
}
function connect() {
  socket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
  );
  socket.onopen = () => {
    connection(true);
    notice("");
  };
  socket.onmessage = (event) => {
    state = JSON.parse(event.data);
    update();
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
    state = result;
    update();
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
document
  .querySelectorAll("[data-speed]")
  .forEach(
    (button) =>
      (button.onclick = () => control("speed", Number(button.dataset.speed))),
  );
$("show-vision").onchange = () => state && drawWorld();

function update() {
  const w = state.world,
    b = state.brain;
  if (lastLife !== state.life || w.time < lastTime) trail = [];
  if (w.time !== lastTime) {
    trail.push([w.x, w.y]);
    if (trail.length > 300) trail.shift();
  }
  lastTime = w.time;
  lastLife = state.life;
  document.title = `Neuroplex · Life ${String(state.life).padStart(3, "0")}`;
  $("lifetime").textContent = clock(w.time);
  $("life-number").textContent = `LIFE ${String(state.life).padStart(2, "0")}`;
  $("pause").textContent = state.paused ? "▶ Resume" : "Ⅱ Pause";
  $("pause").disabled = !connected || !w.alive;
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
  $("world-status").textContent = !w.alive
    ? "LIFE ENDED"
    : state.paused
      ? "PAUSED"
      : "LIVE SIMULATION";
  $("death-overlay").hidden = w.alive;
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
  retinaCells.forEach((cell, i) => {
    const intensity = w.retina[i];
    cell.style.background =
      i < 16
        ? `rgba(134,233,197,${0.04 + intensity * 0.96})`
        : `rgba(112,181,223,${0.04 + intensity * 0.96})`;
    cell.title = `${i < 16 ? "Food" : "Wall"} sensor ${(i % 16) + 1}: ${intensity.toFixed(2)}`;
  });
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
  drawWorld();
  drawNeurons();
  drawHistory();
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
function drawWorld() {
  const [ctx, width, height] = context("world"),
    w = state.world,
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
  const [ctx, width, height] = context("history"),
    data = state.history,
    left = 27,
    right = width - 8,
    top = 10,
    bottom = height - 15;
  ctx.font = "9px ui-monospace, monospace";
  [0, 50, 100].forEach((value) => {
    const y = bottom - (value / 100) * (bottom - top);
    ctx.fillStyle = "#69808b";
    ctx.fillText(String(value), 0, y + 3);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.strokeStyle = "#24313a";
    ctx.lineWidth = 0.6;
    ctx.stroke();
  });
  if (data.length < 2) return;
  const start = data[0].time,
    end = data[data.length - 1].time,
    span = Math.max(1, end - start);
  const point = (p) => [
    left + ((p.time - start) / span) * (right - left),
    bottom - (p.energy / 100) * (bottom - top),
  ];
  ctx.beginPath();
  data.forEach((p, i) => {
    const [x, y] = point(p);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = "#86e9c5";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.closePath();
  const gradient = ctx.createLinearGradient(0, top, 0, bottom);
  gradient.addColorStop(0, "#86e9c524");
  gradient.addColorStop(1, "#86e9c500");
  ctx.fillStyle = gradient;
  ctx.fill();
  $("history-range").textContent = `${clock(start)} — ${clock(end)}`;
}
window.addEventListener("resize", () => {
  if (state) {
    drawWorld();
    drawNeurons();
    drawHistory();
  }
});
connection(false);
connect();
