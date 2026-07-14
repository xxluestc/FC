# Simplified random-load allocation comparison

This is the first-pass chain-completion experiment. The degradation block is
kept as an online engineering proxy. It is not presented as a precise physical
rate calibration.

## Setup

- Three stacks, exactly two online, FC-only power interface.
- Random loads: empirical 1 s Markov, Zuo slow 30 s, Zuo fast 30 s.
- Strategies: Average, Zuo-style DC-average, Rotating, Instant without health
  terms, and Instant with health terms.
- All strategies execute the same health state transition. The ablation changes
  only what the planner reads.

## Closed-loop validation

All 150 runs contain 180 steps. Every step updates health with exactly two
stacks online. There are no tracking failures, constraint violations, clipped
points, or safety overrides.

## Health-objective ablation

- empirical_random_1s: health-aware vs no-health max-stack damage -0.91%, damage imbalance -0.91%, hydrogen intensity -0.10%.
- zuo_slow_random_30s: health-aware vs no-health max-stack damage -2.21%, damage imbalance -2.21%, hydrogen intensity -0.10%.
- zuo_fast_random_30s: health-aware vs no-health max-stack damage -2.22%, damage imbalance -2.22%, hydrogen intensity -0.12%.

See `aggregate_metrics.csv`, `paired_vs_average.csv`, and
`paired_health_ablation.csv` for paired results and uncertainty.
