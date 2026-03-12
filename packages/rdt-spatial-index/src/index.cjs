"use strict";

function assertFiniteNumber(value, name) {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
}

function clampInt(value, lo, hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

function normalizePoints(points, label) {
  if (!Array.isArray(points)) {
    throw new TypeError(`${label} must be an array of [x, y] points`);
  }
  const n = points.length;
  const px = new Float64Array(n);
  const py = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = points[i];
    if (!Array.isArray(p) || p.length !== 2) {
      throw new TypeError(`${label}[${i}] must be a [x, y] pair`);
    }
    const x = Number(p[0]);
    const y = Number(p[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new TypeError(`${label}[${i}] must contain finite numbers`);
    }
    px[i] = x;
    py[i] = y;
  }
  return { px, py, n };
}

function circleBox(cx, cy, r2, x0, y0, x1, y1) {
  const px = cx < x0 ? x0 : cx > x1 ? x1 : cx;
  const py = cy < y0 ? y0 : cy > y1 ? y1 : cy;
  const dx = cx - px;
  const dy = cy - py;
  return dx * dx + dy * dy <= r2;
}

function rdtGridSize(nLocal, alpha = 1.5, maxGrid = 32) {
  if (!Number.isFinite(nLocal) || nLocal < 0) {
    throw new TypeError("nLocal must be a non-negative finite number");
  }
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new TypeError("alpha must be > 0");
  }
  if (!Number.isFinite(maxGrid) || maxGrid < 2) {
    throw new TypeError("maxGrid must be >= 2");
  }
  if (nLocal <= 1) {
    return 2;
  }
  const g = Math.max(2, Math.floor(Math.log(nLocal + 1.0) ** alpha));
  return Math.min(maxGrid, g);
}

class RDTIndex {
  constructor(options = {}) {
    const {
      x0 = 0.0,
      y0 = 0.0,
      x1 = 1000.0,
      y1 = 1000.0,
      alpha = 1.5,
      maxLeaf = 128,
      maxDepth = 20,
      maxGrid = 32,
      verbose = false
    } = options;

    [x0, y0, x1, y1, alpha, maxLeaf, maxDepth, maxGrid].forEach((v, i) => {
      const names = ["x0", "y0", "x1", "y1", "alpha", "maxLeaf", "maxDepth", "maxGrid"];
      assertFiniteNumber(Number(v), names[i]);
    });
    if (!(x1 > x0 && y1 > y0)) {
      throw new RangeError("Invalid bounding box");
    }
    if (maxLeaf < 1) {
      throw new RangeError("maxLeaf must be >= 1");
    }
    if (maxDepth < 1) {
      throw new RangeError("maxDepth must be >= 1");
    }
    if (maxGrid < 2) {
      throw new RangeError("maxGrid must be >= 2");
    }

    this.bounds = [Number(x0), Number(y0), Number(x1), Number(y1)];
    this.alpha = Number(alpha);
    this.maxLeaf = Math.floor(maxLeaf);
    this.maxDepth = Math.floor(maxDepth);
    this.maxGrid = Math.floor(maxGrid);
    this.verbose = Boolean(verbose);

    this._px = new Float64Array(0);
    this._py = new Float64Array(0);
    this._order = new Int32Array(0);
    this._nodes = [];
    this._built = false;
  }

  get built() {
    return this._built;
  }

  get count() {
    return this._px.length;
  }

  build(points) {
    const t0 = this.verbose ? Date.now() : 0;
    const normalized = normalizePoints(points, "points");

    if (normalized.n === 0) {
      this._px = new Float64Array(0);
      this._py = new Float64Array(0);
      this._order = new Int32Array(0);
      this._nodes = [];
      this._built = true;
      return;
    }

    this._px = normalized.px;
    this._py = normalized.py;
    this._order = new Int32Array(normalized.n);
    for (let i = 0; i < normalized.n; i++) {
      this._order[i] = i;
    }

    const [x0, y0, x1, y1] = this.bounds;
    this._nodes = [
      {
        x0,
        y0,
        x1,
        y1,
        depth: 0,
        start: 0,
        end: normalized.n,
        leaf: true,
        grid: 0,
        children: []
      }
    ];

    const stack = [0];
    while (stack.length > 0) {
      const nid = stack.pop();
      const node = this._nodes[nid];
      const cnt = node.end - node.start;

      if (cnt <= this.maxLeaf || node.depth >= this.maxDepth) {
        node.leaf = true;
        continue;
      }

      const g = rdtGridSize(cnt, this.alpha, this.maxGrid);
      const w = node.x1 - node.x0;
      const h = node.y1 - node.y0;
      if (w <= 0 || h <= 0) {
        node.leaf = true;
        continue;
      }

      const cw = w / g;
      const ch = h / g;
      if (cw <= 0 || ch <= 0) {
        node.leaf = true;
        continue;
      }

      const size = cnt;
      const counts = new Int32Array(g * g);
      const pairs = new Array(size);

      for (let k = 0; k < size; k++) {
        const id = this._order[node.start + k];
        const lx = this._px[id];
        const ly = this._py[id];
        const ix = clampInt(Math.floor((lx - node.x0) / cw), 0, g - 1);
        const iy = clampInt(Math.floor((ly - node.y0) / ch), 0, g - 1);
        const cid = iy * g + ix;
        counts[cid] += 1;
        pairs[k] = { cid, id };
      }

      const nonzero = [];
      for (let c = 0; c < counts.length; c++) {
        if (counts[c] > 0) {
          nonzero.push(c);
        }
      }

      if (nonzero.length <= 1) {
        node.leaf = true;
        continue;
      }

      pairs.sort((a, b) => a.cid - b.cid);
      for (let k = 0; k < size; k++) {
        this._order[node.start + k] = pairs[k].id;
      }

      node.leaf = false;
      node.grid = g;
      node.children = [];

      let cursor = node.start;
      for (let i = 0; i < nonzero.length; i++) {
        const cellId = nonzero[i];
        const c = counts[cellId];
        const childStart = cursor;
        const childEnd = cursor + c;
        cursor = childEnd;

        const cx = cellId % g;
        const cy = Math.floor(cellId / g);
        const cx0 = node.x0 + cx * cw;
        const cy0 = node.y0 + cy * ch;
        const cx1 = cx0 + cw;
        const cy1 = cy0 + ch;

        const child = {
          x0: cx0,
          y0: cy0,
          x1: cx1,
          y1: cy1,
          depth: node.depth + 1,
          start: childStart,
          end: childEnd,
          leaf: true,
          grid: 0,
          children: []
        };
        this._nodes.push(child);
        const childId = this._nodes.length - 1;
        node.children.push(childId);
        stack.push(childId);
      }
    }

    this._built = true;

    if (this.verbose) {
      const ms = Date.now() - t0;
      const s = this.summary();
      process.stderr.write(
        `RDT build: n=${this.count}, nodes=${s.nodes}, leaves=${s.leaves}, maxDepth=${s.maxDepth}, ${ms.toFixed(1)} ms\n`
      );
    }
  }

  query(queries, radius) {
    if (!this._built) {
      throw new Error("Index not built");
    }
    assertFiniteNumber(radius, "radius");
    if (radius < 0) {
      throw new RangeError("radius must be >= 0");
    }

    const normalized = normalizePoints(queries, "queries");
    const out = new Int32Array(normalized.n);
    if (this._nodes.length === 0) {
      return out;
    }
    const r2 = radius * radius;

    for (let i = 0; i < normalized.n; i++) {
      const qx = normalized.px[i];
      const qy = normalized.py[i];
      let hits = 0;
      const stack = [0];

      while (stack.length > 0) {
        const nid = stack.pop();
        const node = this._nodes[nid];
        if (!circleBox(qx, qy, r2, node.x0, node.y0, node.x1, node.y1)) {
          continue;
        }
        if (node.leaf) {
          for (let p = node.start; p < node.end; p++) {
            const id = this._order[p];
            const dx = this._px[id] - qx;
            const dy = this._py[id] - qy;
            if (dx * dx + dy * dy <= r2) {
              hits += 1;
            }
          }
        } else {
          for (let c = 0; c < node.children.length; c++) {
            stack.push(node.children[c]);
          }
        }
      }
      out[i] = hits;
    }

    return out;
  }

  summary() {
    const nodes = this._nodes.length;
    let leaves = 0;
    let maxDepth = 0;
    for (let i = 0; i < nodes; i++) {
      const node = this._nodes[i];
      if (node.leaf) {
        leaves += 1;
      }
      if (node.depth > maxDepth) {
        maxDepth = node.depth;
      }
    }
    return {
      built: this._built,
      points: this.count,
      nodes,
      leaves,
      maxDepth,
      alpha: this.alpha,
      maxLeaf: this.maxLeaf,
      maxGrid: this.maxGrid,
      bounds: [...this.bounds]
    };
  }
}

class RDTFastIndex extends RDTIndex {
  constructor(options = {}) {
    super(options);
    this._leafX0 = new Float64Array(0);
    this._leafY0 = new Float64Array(0);
    this._leafX1 = new Float64Array(0);
    this._leafY1 = new Float64Array(0);
    this._leafStart = new Int32Array(0);
    this._leafEnd = new Int32Array(0);
  }

  build(points) {
    super.build(points);
    this._extractLeafArrays();
  }

  _extractLeafArrays() {
    const leaves = [];
    for (let i = 0; i < this._nodes.length; i++) {
      if (this._nodes[i].leaf) {
        leaves.push(this._nodes[i]);
      }
    }
    const L = leaves.length;
    this._leafX0 = new Float64Array(L);
    this._leafY0 = new Float64Array(L);
    this._leafX1 = new Float64Array(L);
    this._leafY1 = new Float64Array(L);
    this._leafStart = new Int32Array(L);
    this._leafEnd = new Int32Array(L);

    for (let i = 0; i < L; i++) {
      const n = leaves[i];
      this._leafX0[i] = n.x0;
      this._leafY0[i] = n.y0;
      this._leafX1[i] = n.x1;
      this._leafY1[i] = n.y1;
      this._leafStart[i] = n.start;
      this._leafEnd[i] = n.end;
    }
  }

  query(queries, radius) {
    if (!this._built) {
      throw new Error("Index not built");
    }
    assertFiniteNumber(radius, "radius");
    if (radius < 0) {
      throw new RangeError("radius must be >= 0");
    }

    const normalized = normalizePoints(queries, "queries");
    const out = new Int32Array(normalized.n);
    const L = this._leafX0.length;
    if (L === 0) {
      return out;
    }
    const r2 = radius * radius;

    for (let i = 0; i < normalized.n; i++) {
      const qx = normalized.px[i];
      const qy = normalized.py[i];
      let hits = 0;

      for (let li = 0; li < L; li++) {
        if (
          !circleBox(
            qx,
            qy,
            r2,
            this._leafX0[li],
            this._leafY0[li],
            this._leafX1[li],
            this._leafY1[li]
          )
        ) {
          continue;
        }

        const start = this._leafStart[li];
        const end = this._leafEnd[li];
        for (let p = start; p < end; p++) {
          const id = this._order[p];
          const dx = this._px[id] - qx;
          const dy = this._py[id] - qy;
          if (dx * dx + dy * dy <= r2) {
            hits += 1;
          }
        }
      }
      out[i] = hits;
    }

    return out;
  }

  summary() {
    const s = super.summary();
    return {
      ...s,
      indexVariant: "RDTFastIndex",
      cachedLeaves: this._leafX0.length
    };
  }
}

module.exports = {
  RDTIndex,
  RDTFastIndex,
  rdtGridSize
};
