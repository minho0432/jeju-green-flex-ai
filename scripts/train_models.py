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
from model_builders import build_model  # LightGBM RE (HPO); SMP는 점수 미사용


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
