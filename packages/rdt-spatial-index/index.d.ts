export type Point2D = readonly [number, number];

export interface RDTOptions {
  x0?: number;
  y0?: number;
  x1?: number;
  y1?: number;
  alpha?: number;
  maxLeaf?: number;
  maxDepth?: number;
  maxGrid?: number;
  verbose?: boolean;
}

export interface RDTSummary {
  built: boolean;
  points: number;
  nodes: number;
  leaves: number;
  maxDepth: number;
  alpha: number;
  maxLeaf: number;
  maxGrid: number;
  bounds: [number, number, number, number];
  indexVariant?: string;
  cachedLeaves?: number;
}

export declare function rdtGridSize(
  nLocal: number,
  alpha?: number,
  maxGrid?: number
): number;

export declare class RDTIndex {
  constructor(options?: RDTOptions);
  readonly built: boolean;
  readonly count: number;
  readonly bounds: [number, number, number, number];
  readonly alpha: number;
  readonly maxLeaf: number;
  readonly maxDepth: number;
  readonly maxGrid: number;
  readonly verbose: boolean;

  build(points: ReadonlyArray<Point2D>): void;
  query(queries: ReadonlyArray<Point2D>, radius: number): Int32Array;
  summary(): RDTSummary;
}

export declare class RDTFastIndex extends RDTIndex {
  constructor(options?: RDTOptions);
  build(points: ReadonlyArray<Point2D>): void;
  query(queries: ReadonlyArray<Point2D>, radius: number): Int32Array;
  summary(): RDTSummary;
}
