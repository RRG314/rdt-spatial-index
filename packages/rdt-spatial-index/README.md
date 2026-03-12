# @rrg314/rdt-spatial-index

Node.js package for RDT Spatial Index (2D radius queries).

This package provides:
- `RDTIndex`: readable reference implementation.
- `RDTFastIndex`: practical default variant with cached-leaf query path.
- `rdtGridSize`: RDT local subdivision rule utility.
- `rdt-spatial-index` CLI for smoke/query workflows.

## Install

```bash
npm install @rrg314/rdt-spatial-index
```

## API Quick Start

```js
const { RDTFastIndex } = require("@rrg314/rdt-spatial-index");

const points = [
  [0, 0],
  [10, 0],
  [0, 10],
  [10, 10]
];

const queries = [[5, 5], [100, 100]];

const idx = new RDTFastIndex({ alpha: 1.5, maxLeaf: 96 });
idx.build(points);
const counts = idx.query(queries, 8.0);

console.log(Array.from(counts));
```

## CLI

Smoke check:

```bash
npx @rrg314/rdt-spatial-index smoke
```

Run query from JSON files:

```bash
npx @rrg314/rdt-spatial-index query \
  --points points.json \
  --queries queries.json \
  --radius 30 \
  --variant fast \
  --out counts.json
```

## Notes

- Returns exact counts for radius queries.
- This package is a JavaScript implementation path for Node users.
- Python compiled extensions in the main repository are separate and not part
  of this npm package.
