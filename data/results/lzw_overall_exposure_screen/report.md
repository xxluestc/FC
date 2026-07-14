# LZW overall-exposure stop-gate screen

## Decision

`rejected_for_action_resolved_degradation`. The physical exposure trend can be retained only as an
endpoint/trend sensitivity. It is not accepted as an action-resolved online
degradation law, and no controller code was changed.

## Primary chronological test

- Validation-selected time model: `stack_on_clock`
- Validation-selected load model: `stack_on_plus_charge`
- Validation-selected physical model: `stack_on_clock`
- Best physical vs zero RMSE improvement: 19.517%
  (moving-block 95% interval 15.858% to
  23.678%)
- Load vs time RMSE improvement: 0.000%
  (moving-block 95% interval 0.000% to
  0.000%)
- Stack-on/charge correlation in the first 85%: 0.964552
- Load beats time in 0/4
  expanding-window checks.

## Interpretation boundary

The LZW theta/IV chain supplies the health target. Raw LZW current and voltage
samples supply exposure. Zuo supplies neither a coefficient nor a health
target in this screen. A high cumulative fit is not accepted when event-count
controls are stronger, load adds no stable value beyond time, exposure terms
are collinear, or the raw sampling period is unverified.
