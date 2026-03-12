# Full Benchmark Results Summary

All methods: 100% exact match vs. brute force across all configurations.

| Dataset | N | Method | Build mean(ms) | Build std | Query mean(ms) | Query std | Memory(MB) |
|---------|---|--------|---------------|-----------|---------------|-----------|------------|
| Uniform Random | 50,000 | RDT (base) | 5.17 | 0.33 | 241.76 | 0.51 | 3.83 |
| Uniform Random | 50,000 | RDT-Fast | 5.16 | 0.30 | 29.64 | 0.44 | 3.83 |
| Uniform Random | 50,000 | RDT-Opt | 861.67 | 3.25 | 50.31 | 0.27 | 9.55 |
| Uniform Random | 50,000 | Uniform Grid | 12.97 | 0.32 | 14.89 | 0.10 | 3.93 |
| Uniform Random | 50,000 | KD-Tree (custom) | 35.17 | 0.31 | 53.26 | 0.09 | 5.59 |
| Uniform Random | 50,000 | Quadtree | 7.47 | 0.28 | 31.63 | 0.24 | 3.57 |
| Uniform Random | 50,000 | rdt_cython | 5.22 | 0.32 | 2.20 | 0.05 | 3.83 |
| Uniform Random | 50,000 | rdt_c | 5.08 | 0.32 | 1.14 | 0.13 | 3.83 |
| Clustered (8 clusters) | 50,000 | RDT (base) | 28.23 | 1.16 | 477.83 | 1.32 | 7.92 |
| Clustered (8 clusters) | 50,000 | RDT-Fast | 32.39 | 1.08 | 357.92 | 1.90 | 9.02 |
| Clustered (8 clusters) | 50,000 | RDT-Opt | 857.64 | 1.66 | 50.95 | 0.13 | 11.73 |
| Clustered (8 clusters) | 50,000 | Uniform Grid | 12.40 | 0.32 | 12.13 | 0.15 | 3.87 |
| Clustered (8 clusters) | 50,000 | KD-Tree (custom) | 30.38 | 0.09 | 61.57 | 0.58 | 5.59 |
| Clustered (8 clusters) | 50,000 | Quadtree | 9.29 | 0.29 | 48.58 | 0.17 | 4.02 |
| Clustered (8 clusters) | 50,000 | rdt_cython | 33.39 | 0.78 | 14.01 | 0.24 | 9.02 |
| Clustered (8 clusters) | 50,000 | rdt_c | 31.76 | 1.65 | 10.73 | 0.30 | 9.02 |
| Sparse+Dense | 50,000 | RDT (base) | 12.28 | 0.30 | 346.48 | 1.74 | 3.84 |
| Sparse+Dense | 50,000 | RDT-Fast | 12.88 | 0.28 | 122.62 | 0.04 | 3.84 |
| Sparse+Dense | 50,000 | RDT-Opt | 608.39 | 0.76 | 40.69 | 0.07 | 7.15 |
| Sparse+Dense | 50,000 | Uniform Grid | 12.72 | 0.38 | 15.40 | 0.03 | 3.93 |
| Sparse+Dense | 50,000 | KD-Tree (custom) | 34.39 | 1.13 | 56.02 | 0.41 | 5.59 |
| Sparse+Dense | 50,000 | Quadtree | 8.75 | 0.29 | 35.58 | 0.28 | 4.22 |
| Sparse+Dense | 50,000 | rdt_cython | 13.44 | 0.64 | 5.21 | 0.23 | 3.84 |
| Sparse+Dense | 50,000 | rdt_c | 13.86 | 0.26 | 3.74 | 0.09 | 3.84 |
| Adversarial Line | 50,000 | RDT (base) | 11.38 | 0.47 | 29.30 | 0.07 | 3.83 |
| Adversarial Line | 50,000 | RDT-Fast | 11.32 | 0.32 | 19.88 | 0.35 | 3.83 |
| Adversarial Line | 50,000 | RDT-Opt | 644.61 | 0.53 | 9.32 | 0.31 | 9.06 |
| Adversarial Line | 50,000 | Uniform Grid | 13.05 | 0.57 | 2.43 | 0.05 | 3.84 |
| Adversarial Line | 50,000 | KD-Tree (custom) | 22.82 | 0.19 | 53.29 | 0.25 | 5.59 |
| Adversarial Line | 50,000 | Quadtree | 15.01 | 0.05 | 12.23 | 0.32 | 4.74 |
| Adversarial Line | 50,000 | rdt_cython | 11.63 | 0.24 | 2.58 | 0.16 | 3.83 |
| Adversarial Line | 50,000 | rdt_c | 12.12 | 0.24 | 1.82 | 0.00 | 3.83 |
| Adversarial Hotspot | 50,000 | RDT (base) | 11.60 | 2.80 | 238.73 | 2.14 | 3.91 |
| Adversarial Hotspot | 50,000 | RDT-Fast | 10.53 | 0.32 | 26.77 | 0.15 | 3.91 |
| Adversarial Hotspot | 50,000 | RDT-Opt | 642.41 | 0.75 | 19.12 | 0.23 | 7.76 |
| Adversarial Hotspot | 50,000 | Uniform Grid | 12.76 | 0.20 | 7.14 | 0.04 | 3.93 |
| Adversarial Hotspot | 50,000 | KD-Tree (custom) | 33.90 | 0.30 | 15.51 | 0.10 | 5.59 |
| Adversarial Hotspot | 50,000 | Quadtree | 10.45 | 0.45 | 11.43 | 0.12 | 4.94 |
| Adversarial Hotspot | 50,000 | rdt_cython | 10.77 | 0.38 | 2.69 | 0.12 | 3.91 |
| Adversarial Hotspot | 50,000 | rdt_c | 10.83 | 0.37 | 1.97 | 0.03 | 3.91 |
| Fractal (Cantor) | 50,000 | RDT (base) | 4.98 | 0.42 | 235.73 | 0.02 | 3.83 |
| Fractal (Cantor) | 50,000 | RDT-Fast | 5.11 | 0.30 | 22.56 | 0.33 | 3.83 |
| Fractal (Cantor) | 50,000 | RDT-Opt | 861.21 | 0.97 | 41.10 | 2.61 | 9.55 |
| Fractal (Cantor) | 50,000 | Uniform Grid | 13.27 | 0.20 | 11.30 | 0.29 | 3.93 |
| Fractal (Cantor) | 50,000 | KD-Tree (custom) | 35.36 | 0.34 | 40.53 | 0.25 | 5.59 |
| Fractal (Cantor) | 50,000 | Quadtree | 7.45 | 0.30 | 24.44 | 0.30 | 3.57 |
| Fractal (Cantor) | 50,000 | rdt_cython | 5.16 | 0.19 | 1.90 | 0.12 | 3.83 |
| Fractal (Cantor) | 50,000 | rdt_c | 5.10 | 0.33 | 0.93 | 0.03 | 3.83 |
| Regular Grid | 50,000 | RDT (base) | 3.44 | 0.04 | 229.43 | 0.26 | 3.83 |
| Regular Grid | 50,000 | RDT-Fast | 3.84 | 0.28 | 17.02 | 0.05 | 3.83 |
| Regular Grid | 50,000 | RDT-Opt | 967.99 | 14.14 | 32.63 | 0.16 | 19.80 |
| Regular Grid | 50,000 | Uniform Grid | 12.49 | 0.28 | 8.20 | 0.27 | 3.93 |
| Regular Grid | 50,000 | KD-Tree (custom) | 17.49 | 0.20 | 29.47 | 0.16 | 5.59 |
| Regular Grid | 50,000 | Quadtree | 4.31 | 0.35 | 17.80 | 0.53 | 3.54 |
| Regular Grid | 50,000 | rdt_cython | 3.60 | 0.06 | 1.19 | 0.30 | 3.83 |
| Regular Grid | 50,000 | rdt_c | 3.79 | 0.31 | 0.73 | 0.04 | 3.83 |
| Taxi-like (real-world) | 50,000 | RDT (base) | 17.68 | 1.97 | 213.39 | 1.40 | 4.55 |
| Taxi-like (real-world) | 50,000 | RDT-Fast | 19.25 | 0.75 | 115.88 | 0.16 | 5.04 |
| Taxi-like (real-world) | 50,000 | RDT-Opt | 849.56 | 10.29 | 37.08 | 0.51 | 8.32 |
| Taxi-like (real-world) | 50,000 | Uniform Grid | 12.37 | 0.37 | 8.54 | 0.38 | 3.89 |
| Taxi-like (real-world) | 50,000 | KD-Tree (custom) | 30.91 | 0.22 | 38.55 | 0.33 | 5.59 |
| Taxi-like (real-world) | 50,000 | Quadtree | 11.29 | 0.46 | 30.87 | 0.09 | 4.31 |
| Taxi-like (real-world) | 50,000 | rdt_cython | 19.83 | 1.17 | 7.00 | 0.07 | 5.04 |
| Taxi-like (real-world) | 50,000 | rdt_c | 19.55 | 0.75 | 5.33 | 0.14 | 5.04 |
| OSM-like (real-world) | 50,000 | RDT (base) | 4.82 | 0.12 | 230.71 | 0.59 | 3.83 |
| OSM-like (real-world) | 50,000 | RDT-Fast | 5.22 | 0.20 | 17.26 | 0.06 | 3.83 |
| OSM-like (real-world) | 50,000 | RDT-Opt | 876.53 | 3.70 | 30.16 | 0.25 | 9.84 |
| OSM-like (real-world) | 50,000 | Uniform Grid | 13.59 | 0.37 | 8.77 | 0.27 | 3.93 |
| OSM-like (real-world) | 50,000 | KD-Tree (custom) | 35.45 | 0.34 | 29.50 | 0.28 | 5.59 |
| OSM-like (real-world) | 50,000 | Quadtree | 7.69 | 0.34 | 18.45 | 0.48 | 3.55 |
| OSM-like (real-world) | 50,000 | rdt_cython | 5.45 | 0.38 | 1.61 | 0.01 | 3.83 |
| OSM-like (real-world) | 50,000 | rdt_c | 5.13 | 0.38 | 0.83 | 0.03 | 3.83 |