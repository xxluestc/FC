"""State-aware tree ensembles for speed, power, and physics residuals."""

from sklearn.ensemble import ExtraTreesRegressor


def build_model(seed: int = 2026) -> ExtraTreesRegressor:
    """Build a deterministic multi-output estimator.

    ExtraTrees is used because it handles nonlinear operating-state boundaries,
    needs no feature scaling, and is substantially faster than training one
    neural network per horizon on this data size.
    """

    return ExtraTreesRegressor(
        n_estimators=50,
        max_depth=24,
        min_samples_leaf=2,
        max_features=0.85,
        bootstrap=True,
        max_samples=0.65,
        n_jobs=-1,
        random_state=seed,
    )
