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

RENEWABLE_WEIGHT = 1.00
MARKET_WEIGHT = 0.00
DEFAULT_RENEWABLE_AI_ALPHA = 1.0  # improve_renewable 검증 결과

LAG_COLUMNS = [
    "smp_lag_24h",
    "smp_lag_48h",
    "smp_lag_168h",
    "renewable_lag_24h",
    "renewable_lag_48h",
    "renewable_lag_168h",
]

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_daytime",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    *WEATHER_COLUMNS,
    "radiation_x_wind",
    "radiation_sq",
    "wind_sq",
    *LAG_COLUMNS,
]

LIVE_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_daytime",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    *WEATHER_COLUMNS,
    "radiation_x_wind",
    "radiation_sq",
    "wind_sq",
]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """하루·이틀·일주일 전 같은 시각 값을 입력 열로 추가한다."""
    result = df.sort_values("timestamp").copy()
    if "smp" in result.columns:
        result["smp_lag_24h"] = result["smp"].shift(24)
        result["smp_lag_48h"] = result["smp"].shift(48)
        result["smp_lag_168h"] = result["smp"].shift(168)
    if "renewable_mwh" in result.columns:
        result["renewable_lag_24h"] = result["renewable_mwh"].shift(24)
        result["renewable_lag_48h"] = result["renewable_mwh"].shift(48)
        result["renewable_lag_168h"] = result["renewable_mwh"].shift(168)
    return result


def _calendar_block(timestamp: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=timestamp.index)
    hour = timestamp.dt.hour
    features["hour"] = hour
    features["day_of_week"] = timestamp.dt.dayofweek
    features["month"] = timestamp.dt.month
    features["day_of_year"] = timestamp.dt.dayofyear
    features["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    features["is_daytime"] = ((hour >= 7) & (hour <= 19)).astype(int)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofyear / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofyear / 365.25)
    features["month_sin"] = np.sin(2 * np.pi * timestamp.dt.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * timestamp.dt.month / 12)
    return features


def _weather_block(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    for column in WEATHER_COLUMNS:
        features[column] = pd.to_numeric(df[column], errors="coerce")
    rad = features["shortwave_radiation"]
    wind = features["wind_speed_10m"]
    features["radiation_x_wind"] = rad * wind
    features["radiation_sq"] = rad ** 2
    features["wind_sq"] = wind ** 2
    return features


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(df["timestamp"])
    features = _calendar_block(timestamp)
    features = _weather_block(df, features)
    for column in LAG_COLUMNS:
        if column in df.columns:
            features[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            features[column] = np.nan
    return features[FEATURE_COLUMNS]


def make_live_features(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(df["timestamp"])
    features = _calendar_block(timestamp)
    features = _weather_block(df, features)
    return features[LIVE_FEATURE_COLUMNS]


def month_hour_baseline(train: pd.DataFrame, target: str, keys: pd.DataFrame) -> pd.Series:
    t = train.copy()
    t["month"] = pd.to_datetime(t["timestamp"]).dt.month
    t["hour"] = pd.to_datetime(t["timestamp"]).dt.hour
    table = t.groupby(["month", "hour"])[target].mean()
    k = keys.copy()
    k["month"] = pd.to_datetime(k["timestamp"]).dt.month
    k["hour"] = pd.to_datetime(k["timestamp"]).dt.hour
    pred = k.set_index(["month", "hour"]).index.map(table)
    pred = pd.Series(pred, index=keys.index).astype(float)
    pred = pred.fillna(t[target].mean())
    return pred


def hybrid_blend(ai_pred, baseline, alpha: float):
    a = np.asarray(ai_pred, dtype=float)
    b = np.asarray(baseline, dtype=float)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * a + (1.0 - alpha) * b


def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    score = series.rank(method="average", pct=True) * 100
    if not higher_is_better:
        score = 100 - score + (100 / max(len(series), 1))
    return score.clip(0, 100)


def score_against_history(
    values: pd.Series, history: pd.Series, higher_is_better: bool
) -> pd.Series:
    reference = np.sort(pd.to_numeric(history, errors="coerce").dropna().to_numpy())
    if len(reference) == 0:
        raise ValueError("점수 기준으로 사용할 과거 데이터가 없습니다.")
    positions = np.searchsorted(reference, values.to_numpy(), side="right")
    scores = positions / len(reference) * 100
    if not higher_is_better:
        scores = 100 - scores
    return pd.Series(np.clip(scores, 0, 100), index=values.index)
