# Synthetic health-observer interface validation

This experiment validates software timing and uncertainty propagation only. The observation is a noisy direct degradation proxy generated from a deterministic conditional-mean truth model with a 1.15 heterogeneity factor. Gamma uncertainty is propagated in the observer belief rather than sampled into the synthetic truth. This is not a 21UBE0022 measurement and does not validate an online SOH posterior.

- Horizon: 720 h; observation interval: 24 h.
- Open-loop RMSE: 0.122200 %-point.
- Corrected-belief RMSE: 0.022866 %-point.
- Synthetic RMSE reduction: 81.29%.
- Posterior 95% interval coverage: 100.00%.
- Monotonic projections: 16.

The only admissible conclusion is that the explicit `predict -> execute -> correct -> next decision` interface can reduce an injected model drift under its synthetic direct-observation assumption. Real correction remains blocked on a vehicle/stack-linked MAT observation chain and a validated voltage/current-to-health measurement model.
