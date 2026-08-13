"""Model factory aligned with Green Time JEJU planning (RE = LightGBM HPO)."""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb


def build_model(target: str):
    if target == "smp":
        return ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        )
    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        max_depth=-1,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
