"use strict";

// Render a short buffer of authoritative frames; never predict future motion.
// This changes only the picture, not the physics, learning, or simulation clock.
class MotionBuffer {
  constructor(delay = 100) {
    this.delay = delay;
    this.frames = [];
  }

  push(frame, at) {
    const previous = this.frames.at(-1);
    const reset = !previous || previous.frame.life !== frame.life ||
      previous.frame.world.stage !== frame.world.stage ||
      previous.frame.paused !== frame.paused || previous.frame.speed !== frame.speed ||
      previous.frame.world.alive !== frame.world.alive ||
      frame.world.time < previous.frame.world.time || at - previous.at > 1000;
    if (reset) this.frames = [];
    this.frames.push({frame, at});
    if (this.frames.length > 12) this.frames.shift();
    return reset;
  }

  sample(now) {
    const latest = this.frames.at(-1);
    if (!latest) return null;
    if (latest.frame.paused || !latest.frame.world.alive) return latest.frame.world;
    const target = now - this.delay;
    while (this.frames.length > 2 && this.frames[1].at <= target) this.frames.shift();
    const left = this.frames[0], right = this.frames[1];
    if (!right || target >= latest.at) return latest.frame.world;
    const t = Math.max(0, Math.min(1, (target - left.at) / Math.max(1, right.at - left.at)));
    const a = left.frame.world, b = right.frame.world;
    const mix = (x, y) => x + (y - x) * t;
    const angle = (x, y) => x + Math.atan2(Math.sin(y - x), Math.cos(y - x)) * t;
    const pose = (x, y) => ({x: mix(x.x, y.x), y: mix(x.y, y.y), heading: angle(x.heading, y.heading)});
    return {...(t < 1 ? a : b), ...pose(a, b), time: mix(a.time, b.time),
      // Resources do not slide when eaten or regrown; only bodies interpolate.
      food: t < 1 ? a.food : b.food, water: t < 1 ? a.water : b.water,
      predators: a.predators.length === b.predators.length
        ? a.predators.map((p, i) => ({...(t < 1 ? p : b.predators[i]), ...pose(p, b.predators[i])})) : b.predators};
  }
}

if (typeof module !== "undefined") module.exports = {MotionBuffer};
else globalThis.MotionBuffer = MotionBuffer;
