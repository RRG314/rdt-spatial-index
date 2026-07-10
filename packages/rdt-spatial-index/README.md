# RDT Spatial Index Node.js Package Source

Node.js package source for RDT Spatial Index (2D radius queries).

Registry status checked 2026-07-10: `@sreid90/rdt-spatial-index` is published
on npm at `0.1.0`. This checkout targets draft source version `0.1.1`; those
changes are not available from npm until a separate publish step is performed.

This package provides:
- `RDTIndex`: readable reference implementation.
- `RDTFastIndex`: practical default variant with cached-leaf query path.
- `rdtGridSize`: RDT local subdivision rule utility.
- `rdt-spatial-index` CLI for smoke/query workflows.

## Validate Current Source

```bash
npm install
npm test
```

## Published npm Package

Use the npm package only when you want the previous registry release:

```bash
npm install @sreid90/rdt-spatial-index
```

## API Quick Start

```js
const { RDTFastIndex } = require("./src/index.cjs");

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
node ./src/cli.cjs smoke
```

Run query from JSON files:

```bash
node ./src/cli.cjs query \
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
