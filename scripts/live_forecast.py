"""Open-Meteo 예보로 다음 24시간의 실험용 충전 입력을 만든다."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from model_utils import (
    MARKET_WEIGHT,
    RENEWABLE_WEIGHT,
    WEATHER_COLUMNS,
    make_live_features,
    score_against_history,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
JEJU_LATITUDE = 33.4996
JEJU_LONGITUDE = 126.5312
FORECAST_UNCERTAINTY_MULTIPLIER = 1.25


def build_forecast_only_model(target: str):
    """예보 시점에 알 수 있는 시간·날씨만 사용하는 실험용 모델."""
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )


def train_forecast_only_models() -> tuple[dict[str, object], pd.DataFrame]:
    """배포 서버에서 한 번만 학습할 오늘 예보용 모델 두 개를 만든다."""
    history = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    history = history.dropna(subset=WEATHER_COLUMNS).reset_index(drop=True)
    features = make_live_features(history)
    models: dict[str, object] = {}
    for target in ("smp", "renewable_mwh"):
        model = build_forecast_only_model(target)
        model.fit(features, history[target])
        models[target] = model
    return models, history


def fetch_open_meteo_forecast() -> pd.DataFrame:
    """제주시 기준 내일 0~23시의 시간별 예보를 가져온다."""
    query = urlencode(
        {
            "latitude": JEJU_LATITUDE,
            "longitude": JEJU_LONGITUDE,
            "hourly": ",".join(WEATHER_COLUMNS),
            "timezone": "Asia/Seoul",
            "forecast_days": 3,
        }
    )
    try:
        with urlopen(f"{OPEN_METEO_URL}?{query}", timeout=12) as response:
            payload = json.load(response)
    except Exception as error:  # 네트워크·API 오류를 사용자 메시지로 바꾼다.
        raise RuntimeError(
            "오늘 날씨예보를 가져오지 못했습니다. 잠시 후 다시 시도하거나 검증 모드를 사용하세요."
        ) from error

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError("날씨예보 응답에 시간 정보가 없습니다.")
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for column in WEATHER_COLUMNS:
        if column not in hourly:
            raise RuntimeError(f"날씨예보 응답에 {column} 정보가 없습니다.")
        frame[column] = pd.to_numeric(hourly[column], errors="coerce")

    now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    target_date = (now + pd.Timedelta(days=1)).date()
    frame = frame[frame["timestamp"].dt.date == target_date].reset_index(drop=True)
    if len(frame) != 24 or frame[WEATHER_COLUMNS].isna().any().any():
        raise RuntimeError("다음 24시간의 완전한 날씨예보를 만들지 못했습니다.")
    return frame


def build_live_prediction(
    models: dict[str, object], history: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """오늘 날씨예보를 SMP·재생에너지 예측과 보수적 점수로 변환한다."""
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    features = make_live_features(weather)
    result = weather.copy()
    result["predicted_smp"] = models["smp"].predict(features)
    result["predicted_renewable_mwh"] = np.maximum(
        models["renewable_mwh"].predict(features), 0
    )

    live_metrics = metrics["forecast_only_targets"]
    smp_half_width = (
        live_metrics["smp"]["approx_90_interval_half_width"]
        * FORECAST_UNCERTAINTY_MULTIPLIER
    )
    renewable_half_width = (
        live_metrics["renewable_mwh"]["approx_90_interval_half_width"]
        * FORECAST_UNCERTAINTY_MULTIPLIER
    )
    result["predicted_smp_lower"] = result["predicted_smp"] - smp_half_width
    result["predicted_smp_upper"] = result["predicted_smp"] + smp_half_width
    result["predicted_renewable_lower"] = np.maximum(
        result["predicted_renewable_mwh"] - renewable_half_width, 0
    )
    result["predicted_renewable_upper"] = (
        result["predicted_renewable_mwh"] + renewable_half_width
    )

    result["price_opportunity_score"] = score_against_history(
        result["predicted_smp"], history["smp"], higher_is_better=False
    ).round(1)
    result["renewable_opportunity_score"] = score_against_history(
        result["predicted_renewable_mwh"],
        history["renewable_mwh"],
        higher_is_better=True,
    ).round(1)
    result["green_score"] = (
        MARKET_WEIGHT * result["price_opportunity_score"]
        + RENEWABLE_WEIGHT * result["renewable_opportunity_score"]
    ).round(1)
    result["conservative_price_score"] = score_against_history(
        result["predicted_smp_upper"], history["smp"], higher_is_better=False
    ).round(1)
    result["conservative_renewable_score"] = score_against_history(
        result["predicted_renewable_lower"],
        history["renewable_mwh"],
        higher_is_better=True,
    ).round(1)
    result["planning_score"] = (
        MARKET_WEIGHT * result["conservative_price_score"]
        + RENEWABLE_WEIGHT * result["conservative_renewable_score"]
    ).round(1)
    result["forecast_risk_points"] = (
        result["green_score"] - result["planning_score"]
    ).clip(lower=0).round(1)
    result["source_mode"] = "live_weather_forecast_experiment"
    return result
