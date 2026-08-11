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

# 팀 제품정의: 친환경 기회를 핵심으로 두고 SMP는 보조 신호로만 사용한다.
RENEWABLE_WEIGHT = 0.80
MARKET_WEIGHT = 0.20

LAG_COLUMNS = [
    "smp_lag_24h",
    "smp_lag_168h",
    "renewable_lag_24h",
    "renewable_lag_168h",
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
    *LAG_COLUMNS,
]

LIVE_FEATURE_COLUMNS = [
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


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """하루 전·일주일 전 같은 시각의 값을 입력 열로 추가한다."""
    result = df.sort_values("timestamp").copy()
    result["smp_lag_24h"] = result["smp"].shift(24)
    result["smp_lag_168h"] = result["smp"].shift(168)
    result["renewable_lag_24h"] = result["renewable_mwh"].shift(24)
    result["renewable_lag_168h"] = result["renewable_mwh"].shift(168)
    return result


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
    for column in LAG_COLUMNS:
        features[column] = pd.to_numeric(df[column], errors="coerce")
    return features[FEATURE_COLUMNS]


def make_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """오늘 예보 모드에서 미리 알 수 있는 시간·날씨 정보만 숫자로 바꾼다."""
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
    return features[LIVE_FEATURE_COLUMNS]


def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """숫자를 0~100 점수로 바꾼다. 같은 값은 같은 점수를 받는다."""
    score = series.rank(method="average", pct=True) * 100
    if not higher_is_better:
        score = 100 - score + (100 / max(len(series), 1))
    return score.clip(0, 100)


def score_against_history(
    values: pd.Series, history: pd.Series, higher_is_better: bool
) -> pd.Series:
    """하루 안의 순위가 아니라 과거 전체 분포를 기준으로 0~100점을 만든다."""
    reference = np.sort(pd.to_numeric(history, errors="coerce").dropna().to_numpy())
    if len(reference) == 0:
        raise ValueError("점수 기준으로 사용할 과거 데이터가 없습니다.")
    positions = np.searchsorted(reference, values.to_numpy(), side="right")
    scores = positions / len(reference) * 100
    if not higher_is_better:
        scores = 100 - scores
    return pd.Series(np.clip(scores, 0, 100), index=values.index)
