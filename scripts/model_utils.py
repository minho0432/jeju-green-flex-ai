"""AI 학습과 예측에서 공통으로 사용하는 시간·날씨 특성 생성 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    *WEATHER_COLUMNS,
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간과 날씨를 AI가 읽을 수 있는 숫자 표로 바꾼다."""
    timestamp = pd.to_datetime(df["timestamp"])
    features = pd.DataFrame(index=df.index)
    features["hour"] = timestamp.dt.hour
    features["day_of_week"] = timestamp.dt.dayofweek
    features["month"] = timestamp.dt.month
    features["day_of_year"] = timestamp.dt.dayofyear
    features["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    features["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    features["day_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofyear / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofyear / 365.25)
    for column in WEATHER_COLUMNS:
        features[column] = pd.to_numeric(df[column], errors="coerce")
    return features[FEATURE_COLUMNS]


def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """숫자를 0~100 점수로 바꾼다. 같은 값은 같은 점수를 받는다."""
    score = series.rank(method="average", pct=True) * 100
    if not higher_is_better:
        score = 100 - score + (100 / max(len(series), 1))
    return score.clip(0, 100)
