"""AI 학습과 예측에서 공통으로 사용하는 시간·날씨 특성·공급여력 점수."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

# Green Score: 공급여력(재생/수요) 100%. SMP 미사용.
RENEWABLE_WEIGHT = 1.00
MARKET_WEIGHT = 0.00
DEFAULT_RENEWABLE_AI_ALPHA = 1.0
DEFAULT_DEMAND_AI_ALPHA = 1.0
DEMAND_FLOOR_MWH = 50.0

LAG_COLUMNS = [
    "renewable_lag_24h",
    "renewable_lag_48h",
    "renewable_lag_168h",
    "demand_lag_24h",
    "demand_lag_48h",
    "demand_lag_168h",
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


def ensure_demand_column(df: pd.DataFrame) -> pd.DataFrame:
    """수요(MWh): 공식 열 우선, 없으면 solar+wind+lng+bio proxy. 실시간은 KPX currPwrTot."""
    out = df.copy()
    if "demand_mwh" in out.columns and out["demand_mwh"].notna().any():
        return out
    parts = [c for c in ("solar_mwh", "wind_mwh", "lng_mwh", "bio_mwh") if c in out.columns]
    if not parts:
        raise ValueError("demand_mwh를 만들 발전량 열이 없습니다.")
    out["demand_mwh"] = out[parts].sum(axis=1)
    return out


def supply_margin(
    renewable: pd.Series | np.ndarray,
    demand: pd.Series | np.ndarray,
    floor: float = DEMAND_FLOOR_MWH,
) -> pd.Series:
    """공급여력 ≈ 재생 / 수요 (수요 하한 적용)."""
    r = pd.Series(np.asarray(renewable, dtype=float))
    d = pd.Series(np.asarray(demand, dtype=float)).clip(lower=float(floor))
    return (r / d).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values("timestamp").copy()
    timestamp = pd.to_datetime(result["timestamp"])
    for source, prefix in (
        ("renewable_mwh", "renewable"),
        ("demand_mwh", "demand"),
    ):
        if source not in result.columns:
            continue
        for hours in (24, 48, 168):
            shifted = result[source].shift(hours)
            exact_lag = timestamp - timestamp.shift(hours) == pd.Timedelta(hours=hours)
            result[f"{prefix}_lag_{hours}h"] = shifted.where(exact_lag)
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
    frame = ensure_demand_column(df)
    frame = add_lag_features(frame)
    timestamp = pd.to_datetime(frame["timestamp"])
    features = _calendar_block(timestamp)
    features = _weather_block(frame, features)
    for column in LAG_COLUMNS:
        if column in frame.columns:
            features[column] = frame[column]
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


def attach_supply_margin_scores(
    result: pd.DataFrame,
    history: pd.DataFrame,
    renewable_col: str = "predicted_renewable_mwh",
    demand_col: str = "predicted_demand_mwh",
    renewable_lower_col: str = "predicted_renewable_lower",
    demand_upper_col: str = "predicted_demand_upper",
) -> pd.DataFrame:
    """재생·수요 예측으로 공급여력 Green Score를 붙인다."""
    out = result.copy()
    hist = ensure_demand_column(history)
    hist_margin = supply_margin(hist["renewable_mwh"], hist["demand_mwh"])

    out["predicted_supply_margin"] = supply_margin(out[renewable_col], out[demand_col])
    out["supply_margin_score"] = score_against_history(
        out["predicted_supply_margin"], hist_margin, higher_is_better=True
    ).round(1)

    out["renewable_opportunity_score"] = score_against_history(
        out[renewable_col], hist["renewable_mwh"], higher_is_better=True
    ).round(1)
    if "predicted_smp" in out.columns and "smp" in hist.columns:
        out["price_opportunity_score"] = score_against_history(
            out["predicted_smp"], hist["smp"], higher_is_better=False
        ).round(1)
    else:
        out["price_opportunity_score"] = 50.0

    out["green_score"] = out["supply_margin_score"]

    if renewable_lower_col in out.columns and demand_upper_col in out.columns:
        cons_margin = supply_margin(out[renewable_lower_col], out[demand_upper_col])
        out["planning_score"] = score_against_history(
            cons_margin, hist_margin, higher_is_better=True
        ).round(1)
        out["conservative_renewable_score"] = score_against_history(
            out[renewable_lower_col], hist["renewable_mwh"], higher_is_better=True
        ).round(1)
    else:
        out["planning_score"] = out["green_score"]
        out["conservative_renewable_score"] = out["renewable_opportunity_score"]

    out["forecast_risk_points"] = (
        out["green_score"] - out["planning_score"]
    ).clip(lower=0).round(1)
    return out
