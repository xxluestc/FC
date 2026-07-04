"""Compact state-aware multi-output random-forest speed predictor."""
from sklearn.ensemble import RandomForestRegressor

def build_model(seed=2026):
    return RandomForestRegressor(n_estimators=80,max_depth=22,min_samples_leaf=3,max_features=.8,max_samples=.7,n_jobs=-1,random_state=seed)

