"""Open-Meteo 예보로 다음 24시간의 재생·수요·공급여력 점수를 만든다."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
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


def train_forecast_only_models() -> tuple[dict[str, object], pd.DataFrame]:
    """배포 서버에서는 저장된 예측 모델만 로드하고 현장 재학습하지 않는다."""
    history = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    history = ensure_demand_column(history)
    history = history.dropna(subset=WEATHER_COLUMNS).reset_index(drop=True)

    model_dir = ROOT / "models"
    model_paths = {
        "renewable_mwh": model_dir / "renewable_live.joblib",
        "demand_mwh": model_dir / "demand_live.joblib",
    }
    missing = [path.name for path in model_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "배포용 AI 모델 파일이 없습니다: "
            + ", ".join(missing)
            + ". 학습 후 모델 파일을 저장소에 포함하세요."
        )

    try:
        import joblib

        models = {
            target: joblib.load(path)
            for target, path in model_paths.items()
        }
    except Exception as error:
        raise RuntimeError(
            "저장된 AI 모델을 불러오지 못했습니다 "
            f"({type(error).__name__}). 학습 환경과 배포 환경의 "
            "scikit-learn·LightGBM·joblib 버전을 확인하세요."
        ) from error

    invalid = [target for target, model in models.items() if not hasattr(model, "predict")]
    if invalid:
        raise RuntimeError(
            "예측 기능이 없는 모델 파일입니다: " + ", ".join(invalid)
        )
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

    # 모델 선택 구간에서 정한 혼합비만 사용합니다. 오늘 결과를 보고 임의 조정하지 않습니다.
    blend = metrics.get("green_time", {}).get("deployment_blend", {})
    from model_utils import month_hour_baseline
    renewable_alpha = float(blend.get("renewable_ai_alpha", 1.0))
    demand_alpha = float(blend.get("demand_ai_alpha", 1.0))
    renewable_baseline = month_hour_baseline(history, "renewable_mwh", result).to_numpy()
    demand_baseline = month_hour_baseline(history, "demand_mwh", result).to_numpy()
    result["predicted_renewable_mwh"] = (
        renewable_alpha * result["predicted_renewable_mwh"]
        + (1 - renewable_alpha) * renewable_baseline
    )
    result["predicted_demand_mwh"] = (
        demand_alpha * result["predicted_demand_mwh"]
        + (1 - demand_alpha) * demand_baseline
    )

    re_hw = _half_width(metrics, "renewable_mwh", 25.0)
    dem_hw = _half_width(metrics, "demand_mwh", 40.0)

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
