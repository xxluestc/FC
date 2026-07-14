# LZW health-progress gate H0

## Decision

`GO` for a descriptive `theta -> h -> theta(h)` manifold.  `NO-GO` for an
action-resolved degradation transition or controller integration.

The fit uses only the ordered LZW UKF-PF theta trajectory.  It does not read
the cumulative damage proxy `D`, Gamma parameters, or Ghaderi/Pei action
coefficients.  `h=1` denotes only the endpoint of this recorded trajectory; it
does not denote SOH=0, EOL, failure, or a known RUL boundary.

## Diagnostics

| Component | Degradation-aligned Spearman vs row | In-sample normalized RMSE |
|---|---:|---:|
| `i0_A_per_cm2` | 0.999995 | 0.000987 |
| `ih_A_per_cm2` | 0.998603 | 0.003764 |
| `R_ohm_reported_ohm_cm2` | 0.999850 | 0.000883 |

The reconstruction error is descriptive and in-sample because `h` is built
from the same three theta components.  It verifies numerical consistency of
the manifold, not predictive validity.  The older MAT identity remains
separate from vehicle `21UBE0022`.
