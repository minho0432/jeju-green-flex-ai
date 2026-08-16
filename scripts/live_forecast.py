"""Open-Meteo 예보로 다음 24시간의 재생·수요·공급여력 점수를 만든다."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from model_utils import (
    WEATHER_COLUMNS,
    attach_supply_margin_scores,
    ensure_demand_column,
    make_live_features,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
JEJU_LATITUDE = 33.4996
JEJU_LONGITUDE = 126.5312
FORECAST_UNCERTAINTY_MULTIPLIER = 1.25


def build_forecast_only_model(target: str):
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )


def train_forecast_only_models() -> tuple[dict[str, object], pd.DataFrame]:
    """배포 서버: 저장된 LightGBM이 있으면 사용, 없으면 현장 학습."""
    history = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    history = ensure_demand_column(history)
    history = history.dropna(subset=WEATHER_COLUMNS).reset_index(drop=True)
    features = make_live_features(history)
    models: dict[str, object] = {}
    model_dir = ROOT / "models"
    try:
        import joblib

        re_path = model_dir / "renewable_live.joblib"
        dem_path = model_dir / "demand_live.joblib"
        smp_path = model_dir / "smp_live.joblib"
        if re_path.exists():
            models["renewable_mwh"] = joblib.load(re_path)
        if dem_path.exists():
            models["demand_mwh"] = joblib.load(dem_path)
        if smp_path.exists():
            models["smp"] = joblib.load(smp_path)
    except Exception:
        pass
    for target in ("smp", "renewable_mwh", "demand_mwh"):
        if target in models:
            continue
        model = build_forecast_only_model(target)
        model.fit(features, history[target])
        models[target] = model
    return models, history


def fetch_open_meteo_forecast(target_day_offset: int = 1) -> pd.DataFrame:
    if target_day_offset not in {0, 1}:
        raise ValueError("날씨예보 날짜 간격은 오늘 0 또는 내일 1이어야 합니다.")
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
    except Exception as error:
        raise RuntimeError(
            "오늘 날씨예보를 가져오지 못했습니다. 잠시 후 다시 시도하거나 검증 모드를 사용하세요."
        ) from error

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError("날씨예보 응답에 시간 정보가 없습니다.")
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for column in WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(
            hourly.get(column, [None] * len(frame)), errors="coerce"
        )

    now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    target_date = (now + pd.Timedelta(days=target_day_offset)).date()
    frame = frame[frame["timestamp"].dt.date == target_date].reset_index(drop=True)
    if len(frame) != 24 or frame[WEATHER_COLUMNS].isna().any().any():
        raise RuntimeError("다음 24시간의 완전한 날씨예보를 만들지 못했습니다.")
    return frame


def _half_width(metrics: dict, target: str, default: float) -> float:
    try:
        return (
            float(metrics["forecast_only_targets"][target]["approx_90_interval_half_width"])
            * FORECAST_UNCERTAINTY_MULTIPLIER
        )
    except Exception:
        return float(default) * FORECAST_UNCERTAINTY_MULTIPLIER


def build_live_prediction(
    models: dict[str, object], history: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """날씨예보 → 재생·수요 예측 → 공급여력 Green Score."""
    history = ensure_demand_column(history)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

    features = make_live_features(weather)
    result = weather.copy()
    result["predicted_smp"] = models["smp"].predict(features)
    result["predicted_renewable_mwh"] = np.maximum(
        models["renewable_mwh"].predict(features), 0
    )
    if "demand_mwh" in models:
        result["predicted_demand_mwh"] = np.maximum(
            models["demand_mwh"].predict(features), 0
        )
    else:
        from model_utils import month_hour_baseline

        result["predicted_demand_mwh"] = month_hour_baseline(
            history, "demand_mwh", result
        ).to_numpy()

    smp_hw = _half_width(metrics, "smp", 15.0)
    re_hw = _half_width(metrics, "renewable_mwh", 25.0)
    dem_hw = _half_width(metrics, "demand_mwh", 40.0)

    result["predicted_smp_lower"] = result["predicted_smp"] - smp_hw
    result["predicted_smp_upper"] = result["predicted_smp"] + smp_hw
    result["predicted_renewable_lower"] = np.maximum(
        result["predicted_renewable_mwh"] - re_hw, 0
    )
    result["predicted_renewable_upper"] = result["predicted_renewable_mwh"] + re_hw
    result["predicted_demand_lower"] = np.maximum(
        result["predicted_demand_mwh"] - dem_hw, 0
    )
    result["predicted_demand_upper"] = result["predicted_demand_mwh"] + dem_hw

    result = attach_supply_margin_scores(result, history)
    result["source_mode"] = "live_weather_supply_margin"
    return result
