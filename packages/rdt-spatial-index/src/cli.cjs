#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { RDTIndex, RDTFastIndex } = require("./index.cjs");

function usage() {
  process.stdout.write(
    [
      "rdt-spatial-index CLI",
      "",
      "Usage:",
      "  rdt-spatial-index smoke",
      "  rdt-spatial-index query --points points.json --queries queries.json --radius 30 [--variant fast|ref] [--out counts.json]",
      "",
      "Notes:",
      "  - points/queries JSON files must be arrays of [x, y] pairs.",
      "  - variant defaults to fast."
    ].join("\n") + "\n"
  );
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const value = argv[i + 1];
    if (value == null || value.startsWith("--")) {
      out[key] = true;
      continue;
    }
    out[key] = value;
    i += 1;
  }
  return out;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rand() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function runSmoke() {
  const randPoints = mulberry32(1);
  const randQueries = mulberry32(2);

  const n = 10_000;
  const qn = 64;
  const points = new Array(n);
  const queries = new Array(qn);
  for (let i = 0; i < n; i++) {
    points[i] = [randPoints() * 1000, randPoints() * 1000];
  }
  for (let i = 0; i < qn; i++) {
    queries[i] = [randQueries() * 1000, randQueries() * 1000];
  }

  const idx = new RDTFastIndex({ alpha: 1.5, maxLeaf: 96 });
  idx.build(points);
  const out = idx.query(queries, 30.0);

  process.stdout.write(
    JSON.stringify(
      {
        ok: true,
        variant: "fast",
        points: points.length,
        queries: queries.length,
        firstCounts: Array.from(out.slice(0, 5))
      },
      null,
      2
    ) + "\n"
  );
}

function readJsonArray(filePath, label) {
  const abs = path.resolve(process.cwd(), filePath);
  const raw = fs.readFileSync(abs, "utf8");
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON array`);
  }
  return parsed;
}

function runQuery(args) {
  const pointsPath = args.points;
  const queriesPath = args.queries;
  const radiusRaw = args.radius;
  if (!pointsPath || !queriesPath || radiusRaw == null) {
    throw new Error("query requires --points, --queries, and --radius");
  }

  const radius = Number(radiusRaw);
  if (!Number.isFinite(radius) || radius < 0) {
    throw new Error("--radius must be a non-negative number");
  }

  const variant = String(args.variant || "fast").toLowerCase();
  const points = readJsonArray(pointsPath, "points");
  const queries = readJsonArray(queriesPath, "queries");

  const IndexClass = variant === "ref" ? RDTIndex : RDTFastIndex;
  const idx = new IndexClass();
  idx.build(points);
  const out = Array.from(idx.query(queries, radius));

  if (args.out) {
    const outAbs = path.resolve(process.cwd(), args.out);
    fs.writeFileSync(outAbs, JSON.stringify(out, null, 2) + "\n", "utf8");
  } else {
    process.stdout.write(JSON.stringify(out) + "\n");
  }
}

function main() {
  const argv = process.argv.slice(2);
  const cmd = argv[0];

  if (!cmd || cmd === "-h" || cmd === "--help") {
    usage();
    process.exit(0);
  }

  try {
    if (cmd === "smoke") {
      runSmoke();
      return;
    }
    if (cmd === "query") {
      runQuery(parseArgs(argv.slice(1)));
      return;
    }
    usage();
    process.exitCode = 2;
  } catch (err) {
    process.stderr.write(`error: ${err.message}\n`);
    process.exitCode = 1;
  }
}

main();
