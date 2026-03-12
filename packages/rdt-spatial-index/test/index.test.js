"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { RDTIndex, RDTFastIndex, rdtGridSize } = require("../src/index.cjs");

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rand() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function bruteCounts(points, queries, radius) {
  const r2 = radius * radius;
  const out = new Int32Array(queries.length);
  for (let i = 0; i < queries.length; i++) {
    const qx = queries[i][0];
    const qy = queries[i][1];
    let hits = 0;
    for (let j = 0; j < points.length; j++) {
      const dx = points[j][0] - qx;
      const dy = points[j][1] - qy;
      if (dx * dx + dy * dy <= r2) {
        hits += 1;
      }
    }
    out[i] = hits;
  }
  return out;
}

test("rdtGridSize basic behavior", () => {
  assert.equal(rdtGridSize(0), 2);
  assert.equal(rdtGridSize(1), 2);
  assert.equal(rdtGridSize(10, 1.5, 32), 3);
  assert.equal(rdtGridSize(1_000_000, 2.0, 16), 16);
});

test("RDTIndex matches brute force counts", () => {
  const randP = mulberry32(11);
  const randQ = mulberry32(12);
  const points = new Array(800);
  const queries = new Array(50);
  for (let i = 0; i < points.length; i++) {
    points[i] = [randP() * 1000, randP() * 1000];
  }
  for (let i = 0; i < queries.length; i++) {
    queries[i] = [randQ() * 1000, randQ() * 1000];
  }

  const radius = 35.0;
  const idx = new RDTIndex({ alpha: 1.5, maxLeaf: 64 });
  idx.build(points);

  const got = idx.query(queries, radius);
  const want = bruteCounts(points, queries, radius);
  assert.deepEqual(Array.from(got), Array.from(want));
});

test("RDTFastIndex matches reference index", () => {
  const randP = mulberry32(21);
  const randQ = mulberry32(22);
  const points = new Array(2000);
  const queries = new Array(80);
  for (let i = 0; i < points.length; i++) {
    points[i] = [randP() * 500, randP() * 500];
  }
  for (let i = 0; i < queries.length; i++) {
    queries[i] = [randQ() * 500, randQ() * 500];
  }

  const radius = 20.0;
  const ref = new RDTIndex({ alpha: 1.5, maxLeaf: 96 });
  const fast = new RDTFastIndex({ alpha: 1.5, maxLeaf: 96 });
  ref.build(points);
  fast.build(points);

  const refOut = ref.query(queries, radius);
  const fastOut = fast.query(queries, radius);
  assert.deepEqual(Array.from(fastOut), Array.from(refOut));
});

test("empty index query returns empty counts", () => {
  const idx = new RDTFastIndex();
  idx.build([]);
  const out = idx.query([[0, 0], [1, 1]], 2);
  assert.deepEqual(Array.from(out), [0, 0]);
});
