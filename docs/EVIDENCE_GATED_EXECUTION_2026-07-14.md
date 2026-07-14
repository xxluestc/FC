# Evidence-gated execution reset

Date: 2026-07-14
Branch: `codex/gamma-health-foundation`

## 1. Why this reset is required

The previous random-load comparison is not accepted as a completed scientific
chain. It copied Zuo's four load-state values into a different stack system,
produced no state transition in some 180 s runs, used a single literature rate
through 25--270 A, and let fixed event constants dominate short simulations.
The software path ran, but the numerical result did not establish a valid
degradation-aware allocation method.

This document supersedes any earlier status statement that calls the Zuo-real
load calibration, action-resolved degradation model, or policy-effect chain
scientifically complete. Historical code may remain as a labelled baseline or
negative-control implementation, but its former improvement numbers are not
evidence for the new route.

## 2. Frozen research scope

- Three virtual PEMFC stacks in an N+1 architecture.
- Zero, one, or two stacks may produce positive power; never more than two.
- The current layer selects operating stacks and allocates their power.
- No battery decision and no learned future-demand predictor in this stage.
- G-drive vehicle data supply load statistics and held-out replay only.
- The older LZW MAT chain supplies a cross-dataset health/performance prior
  only; it is not a row-wise posterior for vehicle `21UBE0022`.

## 3. Evidence roles

| Source | Accepted role | Prohibited interpretation |
|---|---|---|
| `21UBE0022` DCDC input voltage/current | Single-stack operating-power process, event statistics, held-out replay | Nameplate rating or multi-stack degradation trajectory |
| LZW UKF-PF `theta=[i0, ih, R]` | Normalized trajectory coordinate `h` and `theta(h)` performance manifold | SOH, EOL, RUL, or action-causal degradation coefficients |
| Zuo 2024 | N+1 topology, event-based random-load idea, comparison baseline | Reuse of `[2.9, 4.1, 5.8, 7.0]` or its fitted deterioration parameters |
| Ghaderi/Pei coefficients | Literature sensitivity bounds or negative control | Coefficients identified from LZW or `21UBE0022` |

## 4. Independent review roles

Three read-only reviews are required before a method layer is promoted:

1. Data reviewer: source fields, exclusions, segment boundaries, temporal
   split, and source hashes.
2. Health reviewer: identity, alignment, monotonicity, identifiability, and
   prohibited claims.
3. Method reviewer: objective/constraint consistency, baseline strength,
   leakage, statistical unit, and solver validation.

An implementation agent may edit only its assigned file set. The lead agent
reviews every patch, reruns tests, inspects generated data, and owns the final
go/no-go decision. Agent output is advice or code input, never evidence by
itself.

## 5. Stage gates

### L0: empirical event load

Required before any controller experiment:

- Zuo load-state values do not enter fitting or generation.
- The configured interval `[2026-01-30, 2026-02-07)` contributes zero derived
  rows and creates an explicit segment break.
- No transition crosses a source gap, off boundary, or exclusion interval.
- Interpolated telemetry cannot create an event boundary.
- State count is selected from the calibration data under predeclared
  occupancy and event-count requirements, rather than fixed for visual appeal.
- Held-out power distribution, event dwell, transition rows, event rate, and
  zero-transition probability are reported together.
- The empirical service profile preserves the held-out probability of a flat
  180 s window; a flat service realization is therefore not an automatic
  failure. A separate, explicitly labelled engineering-stress profile may
  compress vehicle-derived dwell times, but must leave fitted power states and
  transition rows unchanged and must contain multiple events in 180 s.

### H0: LZW health-progress manifold

Required before online health updates:

- `h` is constructed directly from the LZW theta trajectory without `D`,
  Gamma, or Ghaderi/Pei coefficients.
- Event keys are unique and order-consistent.
- The three oriented parameter components and the aggregate `h` are monotone
  after an explicitly reported projection.
- `theta(h)` reconstruction errors and endpoint definitions are saved.
- `h=1` is labelled only as the endpoint of the observed LZW trajectory.
- No action-level transition law is inferred at this gate.

### A0: action-health transition identifiability

Current status: **NO-GO**.

The existing screen found that load exposure adds no stable held-out value over
time exposure, cumulative channels are highly collinear, and event counts can
fit as well as the physical exposure model. A future transition model must use
interval exposures and pass rank, condition-number, VIF, temporal holdout,
block-bootstrap, sign-stability, and placebo-clock checks. Otherwise its
coefficients remain an uncertainty set for sensitivity analysis only.

### C0: robust mixed-integer control

Current status: **PHYSICAL-DEGRADATION OBJECTIVE BLOCKED; RELATIVE-HEALTH ADAPTATION ALLOWED**.

MIQP/MISOCP integration starts only after the preceding gates. Its discrete
mode and continuous power solution must be checked against an exact small-scale
enumeration oracle, with zero hard-constraint leakage and a recorded optimality
gap. A more sophisticated solver cannot validate an unidentified degradation
law.

### A0-R: coefficient-free relative-health adaptation

Current status: **LIMITED GO FOR MECHANISM FEASIBILITY ONLY**.

This fallback fixes the same dimensionless fleet health-progress budget for
every policy, allocates each step's budget by executed stack-power share, and
updates `h -> theta(h) -> IV/power` online.  It contains no start/stop, ramp,
load-amplitude, or calendar-ageing coefficient.  On 29 later-time development
blocks and all six initial-health assignments, the primary health-reserve
policy reduced final maximum `h` by `0.012217` relative to an online
performance-adaptive controller (block-bootstrap 95% interval
`[-0.014793,-0.009643]`) with a `0.0211 kW` tracking-MAE increase.  Useful
electrical efficiency changed by only `+0.0006` percentage points and its
interval crossed zero.  The policy directly optimizes health-weighted loading,
so part of the result is mechanical.  This establishes controller-state
coupling and budget redistribution only.  It does not establish independent
method effectiveness, physical ageing reduction, or real-vehicle lifetime
extension.

## 6. Statistical and comparison rules

- Average, Daisy Chain/Rotating, a Zuo-style event baseline, and an exact
  one-event oracle remain in the comparison set.
- The controller and evaluator must not rely on the same unvalidated
  degradation coefficients as independent proof.
- Segments, calendar blocks, or months are the statistical units. Per-second
  rows and Monte Carlo seeds are not independent real-world samples.
- Primary metrics include demand tracking, hydrogen, start/stop count,
  switching/load-change exposure, maximum and second-largest health progress,
  health imbalance, constraint violations, and computation time.
- A method is not promoted when improvement appears only under one borrowed
  coefficient vector, one initial-health ordering, or a weakened baseline.

## 7. Immediate execution order

1. Implement and validate the empirical event-load model.
2. Implement and validate `theta -> h -> theta(h)`.
3. Review both result packages and record go/no-go decisions.
4. Revisit action-health transition identifiability using interval data.
5. Only then implement the event-triggered robust mixed-integer controller.

No policy-effect figure is generated in steps 1--3. The only allowed figures
are load-model validation and health-manifold diagnostics.
