// Optional developer tests: node --test tests/test_motion.cjs (no npm install).
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {MotionBuffer} = require('../neuroplex/static/motion.js');

function frame(x, changes = {}) {
  return {life: 1, paused: false, speed: 1,
    world: {x, y: 5, time: x, heading: 0, alive: true, stage: 4,
      food: [[1, 2]], water: [[3, 4]], predators: [{x: x + 10, y: 10, heading: 0}], ...changes}};
}

test('interpolates creature and predators between authoritative frames', () => {
  const buffer = new MotionBuffer();
  buffer.push(frame(0), 0);
  buffer.push(frame(10), 50);
  const picture = buffer.sample(125);
  assert.equal(picture.x, 5);
  assert.equal(picture.predators[0].x, 15);
  assert.equal(picture.time, 5);
  assert.equal(buffer.frames[0].frame.world.x, 0); // input not mutated
});

test('uses shortest rotation across the -pi/pi boundary', () => {
  const buffer = new MotionBuffer();
  buffer.push(frame(0, {heading: Math.PI - .1}), 0);
  buffer.push(frame(1, {heading: -Math.PI + .1}), 50);
  assert.ok(Math.abs(buffer.sample(125).heading - Math.PI) < 1e-10);
});

test('food disappears discretely, never slides to a different resource', () => {
  const buffer = new MotionBuffer();
  buffer.push(frame(0), 0);
  buffer.push(frame(1, {food: []}), 50);
  assert.deepEqual(buffer.sample(125).food, [[1, 2]]);
  assert.deepEqual(buffer.sample(150).food, []);
});

test('freezes at the last received pose instead of extrapolating on disconnect', () => {
  const buffer = new MotionBuffer();
  buffer.push(frame(0), 0);
  buffer.push(frame(10), 50);
  assert.equal(buffer.sample(10000).x, 10);
});

test('new life, stage, pause, death, speed and reconnect discard old interpolation', () => {
  for (const change of [
    f => {f.life = 2;}, f => {f.world.stage = 2;}, f => {f.paused = true;},
    f => {f.world.alive = false;}, f => {f.speed = 10;}, f => {f.world.time = -1;},
  ]) {
    const buffer = new MotionBuffer();
    buffer.push(frame(0), 0);
    const next = frame(20);
    change(next);
    assert.equal(buffer.push(next, 50), true);
    assert.equal(buffer.sample(60).x, 20);
  }
  const buffer = new MotionBuffer();
  buffer.push(frame(0), 0);
  assert.equal(buffer.push(frame(25), 2000), true);
  assert.equal(buffer.sample(2010).x, 25);
});

test('bounds the frame buffer even when the tab is not drawing', () => {
  const buffer = new MotionBuffer();
  for (let i = 0; i < 100; i++) buffer.push(frame(i), i * 50);
  assert.ok(buffer.frames.length <= 12);
  assert.equal(buffer.sample(10000).x, 99);
});
