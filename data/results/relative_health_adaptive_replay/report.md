# Relative health-adaptive real-load replay

## Decision

`LIMITED_GO_MECHANISM_ONLY` for an internal, coefficient-free mechanism demonstration only.

The experiment fixes the total fleet health-progress budget for every policy
and allocates that budget by executed stack-energy share.  The health-aware
policy may prefer a less-progressed stack only among actions within
0.50 kW of the best available tracking error.  It therefore
cannot improve the health result by withholding demanded power.

This is not independent method-effectiveness evidence, a physical
degradation-rate model, an action-causal ageing model, or evidence of
real-vehicle lifetime extension.

## Primary paired effects

Primary normalized fleet budget: `0.030`.  Primary
health-loading weight: `0.20`.  The statistical unit is
one later-time development block after averaging all six assignments of the
same three initial health levels.

| Metric, health reserve minus performance-adaptive IV | Mean difference | Block bootstrap 95% interval |
|---|---:|---:|
| Final maximum `h` | -0.012217 | [-0.014793, -0.009643] |
| Final `h` range | -0.022112 | [-0.026271, -0.017540] |
| Energy-weighted loaded `h` | -0.044099 | [-0.052481, -0.034891] |
| Useful electrical efficiency (percentage points) | +0.0006 | [-0.0003, +0.0017] |
| Tracking MAE (kW) | +0.0211 | [+0.0159, +0.0267] |

Maximum fleet-budget conservation error across all episodes and policies:
`2.220e-16`.

## Evidence boundary

- LZW supplies only `h -> theta(h) -> IV/power`.
- The 21UBE0022-derived archive supplies `target_power_kw` demand replay only.
- The total health budget is identical across policies; only its allocation can change.
- Start/stop, ramp, and load-amplitude degradation coefficients are absent.
- The three stacks are virtual health positions on one measured theta manifold.
- February--June 2026 are nested later-time development replay, not an untouched final holdout.
- The policy directly minimizes health-weighted loading, so part of the health
  improvement is mechanical and only verifies the intended redistribution.
- A weight of 0.10 is worse than health-blind control on final maximum `h`;
  weak health weights are not uniformly robust.

The allowed claim is that online health degree can change stack selection and
redistribute a fixed relative health budget while preserving hard constraints.
The prohibited claim is that this proves an independently effective algorithm,
measured efficiency improvement, or reduced physical PEMFC degradation on the
real vehicle.
