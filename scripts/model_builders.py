"""Green Time JEJU용 회귀 모델 후보 팩토리."""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
import lightgbm as lgb


MODEL_CANDIDATES = (
    "lightgbm_balanced",
    "lightgbm_regularized",
    "hist_gradient_boosting",
    "extra_trees",
)


def build_model(target: str, candidate: str | None = None):
    candidate = candidate or "lightgbm_balanced"
    if candidate == "lightgbm_balanced":
        return lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.95,
            reg_lambda=0.2,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
    if candidate == "lightgbm_regularized":
        return lgb.LGBMRegressor(
            n_estimators=550,
            learning_rate=0.03,
            num_leaves=23,
            min_child_samples=45,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
    if candidate == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42,
        )
    if candidate == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=42,
        )
    raise ValueError(f"알 수 없는 모델 후보: {candidate}")
