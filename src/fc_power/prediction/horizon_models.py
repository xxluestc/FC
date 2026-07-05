"""Horizon-specific direct-power models with an optional brake-aware branch."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def build_regressor(family: str, seed: int):
    """Create a deterministic multi-output regressor for one forecast horizon."""

    if family == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=50,
            max_depth=26,
            min_samples_leaf=2,
            max_features=0.9,
            bootstrap=True,
            max_samples=0.7,
            n_jobs=-1,
            random_state=seed,
        )
    if family == "hist_gradient_boosting":
        base = HistGradientBoostingRegressor(
            max_iter=70,
            learning_rate=0.07,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=seed,
        )
        return MultiOutputRegressor(base, n_jobs=1)
    if family == "xgboost":
        return XGBRegressor(
            n_estimators=180,
            max_depth=8,
            learning_rate=0.045,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model family: {family}")


class BrakeAwareExtraTrees:
    """Predict a brake event, then blend brake/non-brake regression experts."""

    def __init__(self, seed: int = 2026):
        self.classifier = ExtraTreesClassifier(
            n_estimators=60,
            max_depth=22,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        self.general = build_regressor("extratrees", seed)
        self.brake = build_regressor("extratrees", seed + 1)
        self.non_brake = build_regressor("extratrees", seed + 2)

    def fit(self, features: np.ndarray, targets: np.ndarray):
        targets = np.asarray(targets).reshape(len(targets), -1)
        brake_event = targets.min(axis=1) < -5.0
        self.classifier.fit(features, brake_event)
        self.general.fit(features, targets)
        if brake_event.sum() >= 100:
            self.brake.fit(features[brake_event], targets[brake_event])
            self.has_brake_expert = True
        else:
            self.has_brake_expert = False
        if (~brake_event).sum() >= 100:
            self.non_brake.fit(features[~brake_event], targets[~brake_event])
            self.has_non_brake_expert = True
        else:
            self.has_non_brake_expert = False
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        general = np.asarray(self.general.predict(features)).reshape(len(features), -1)
        probability = self.classifier.predict_proba(features)[:, 1]
        brake = (
            np.asarray(self.brake.predict(features)).reshape(len(features), -1)
            if self.has_brake_expert
            else general
        )
        non_brake = (
            np.asarray(self.non_brake.predict(features)).reshape(len(features), -1)
            if self.has_non_brake_expert
            else general
        )
        # Soft routing is less brittle than a hard 0.5 branch at brake transitions.
        routed = probability[:, None] * brake + (1 - probability[:, None]) * non_brake
        return 0.75 * routed + 0.25 * general

    def predict_brake_probability(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(features)[:, 1]
