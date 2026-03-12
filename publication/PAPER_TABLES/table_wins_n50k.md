# Wins/Losses Summary — N=50,000

| Dataset | Query Winner | Build Winner | Grid q(ms) | KD q(ms) | RDT-Fast q(ms) | RDT-Opt q(ms) |
|---------|-------------|--------------|------------|----------|---------------|--------------|
| Uniform Random | **Uniform Grid** | RDT-Fast |   14.9 |   53.3 |   29.6 |   50.3 |
| Clustered (8 clusters) | **Uniform Grid** | Uniform Grid |   12.1 |   61.6 |  357.9 |   50.9 |
| Sparse+Dense | **Uniform Grid** | Uniform Grid |   15.4 |   56.0 |  122.6 |   40.7 |
| Adversarial Line | **Uniform Grid** | RDT-Fast |    2.4 |   53.3 |   19.9 |    9.3 |
| Adversarial Hotspot | **Uniform Grid** | RDT-Fast |    7.1 |   15.5 |   26.8 |   19.1 |
| Fractal (Cantor) | **Uniform Grid** | RDT-Fast |   11.3 |   40.5 |   22.6 |   41.1 |
| Regular Grid | **Uniform Grid** | RDT-Fast |    8.2 |   29.5 |   17.0 |   32.6 |
| Taxi-like (real-world) | **Uniform Grid** | Uniform Grid |    8.5 |   38.6 |  115.9 |   37.1 |
| OSM-like (real-world) | **Uniform Grid** | RDT-Fast |    8.8 |   29.5 |   17.3 |   30.2 |

## Query Win Counts (N=50K)
- Uniform Grid: 9 / 9 datasets

## Build Win Counts (N=50K)
- RDT-Fast: 6 / 9 datasets
- Uniform Grid: 3 / 9 datasets