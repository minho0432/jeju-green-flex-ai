"""제주 SMP·재생에너지 예측 모델을 공정하게 검증하고 데모 예측을 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from model_utils import (
    FEATURE_COLUMNS,
    LAG_COLUMNS,
    MARKET_WEIGHT,
    RENEWABLE_WEIGHT,
    WEATHER_COLUMNS,
    add_lag_features,
    make_features,
    make_live_features,
    score_against_history,
)
from live_forecast import build_forecast_only_model
from model_builders import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DEMO_PATH = ROOT / "data" / "demo" / "demo_forecast.csv"
DEMO_PREDICTION_PATH = OUTPUT_DIR / "demo_predictions.csv"
BACKTEST_PATH = OUTPUT_DIR / "backtest_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
TARGETS = {
    "smp": "predicted_smp",
    "renewable_mwh": "predicted_renewable_mwh",
}
N_SPLITS = 5
TEST_DAYS_PER_SPLIT = 30
SMP_CORRECTION_WEIGHT = 0.75


def average_lag_baseline(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "smp":
        return (frame["smp_lag_24h"] + frame["smp_lag_168h"]) / 2
    return (
        frame["renewable_lag_24h"] + frame["renewable_lag_168h"]
    ) / 2


def fit_model(model, train_x, train_frame: pd.DataFrame, target: str):
    """SMP는 강한 과거 기준의 오차만 AI가 보정하고, 재생에너지는 직접 예측한다."""
    if target == "smp":
        baseline = average_lag_baseline(train_frame, target)
        model.fit(train_x, train_frame[target] - baseline)
    else:
        model.fit(train_x, train_frame[target])
    return model


def predict_model(model, test_x, test_frame: pd.DataFrame, target: str) -> np.ndarray:
    prediction = model.predict(test_x)
    if target == "smp":
        baseline = average_lag_baseline(test_frame, target).to_numpy()
        prediction = baseline + SMP_CORRECTION_WEIGHT * prediction
    else:
        prediction = np.maximum(prediction, 0)
    return prediction


def metric_dict(actual: pd.Series, prediction: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "rmse": round(float(mean_squared_error(actual, prediction) ** 0.5), 4),
        "r2": round(float(r2_score(actual, prediction)), 4),
    }


def baseline_columns(frame: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    """미래를 보지 않고 만들 수 있는 현실적인 단순 예측 세 가지."""
    if target == "smp":
        lag24 = frame["smp_lag_24h"]
        lag168 = frame["smp_lag_168h"]
    else:
        lag24 = frame["renewable_lag_24h"]
        lag168 = frame["renewable_lag_168h"]
    return {
        "24_hours_ago": lag24,
        "168_hours_ago": lag168,
        "average_24_168": (lag24 + lag168) / 2,
    }
