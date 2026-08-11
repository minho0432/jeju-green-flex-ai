"""여러 기상 시나리오를 AI에 통과시켜 날씨예보 불확실성을 계산한다.

일반 Open-Meteo 예보는 하나의 대표 기상경로를 준다. Ensemble API는 초기
조건을 조금씩 바꾼 여러 예보를 제공한다. 여기서는 각 예보를 기존 AI에
통과시킨 뒤 10~90 백분위 범위를 기존 모델 오차범위에 추가한다.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from model_utils import (
    MARKET_WEIGHT,
    RENEWABLE_WEIGHT,
    WEATHER_COLUMNS,
    make_live_features,
    score_against_history,
)


ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MODEL = "icon_seamless_eps"
MEMBER_PATTERN = re.compile(r"^(?P<variable>.+?)(?:_member(?P<number>\d+))?$")


class WeatherEnsembleError(RuntimeError):
    """앙상블 날씨를 안전하게 사용할 수 없을 때 발생하는 오류."""


def build_ensemble_url() -> str:
    requested = []
    for column in WEATHER_COLUMNS:
        requested.append(column)
    query = urlencode(
        {
            "latitude": 33.4996,
            "longitude": 126.5312,
            "hourly": ",".join(requested),
            "models": ENSEMBLE_MODEL,
            "timezone": "Asia/Seoul",
            "wind_speed_unit": "ms",
            "forecast_days": 3,
        }
    )
    return f"{ENSEMBLE_URL}?{query}"


def _member_suffixes(hourly: dict[str, Any]) -> list[str]:
    member_sets: list[set[str]] = []
    for variable in WEATHER_COLUMNS:
        suffixes: set[str] = set()
        for key in hourly:
            if key == variable:
                suffixes.add("")
            elif key.startswith(f"{variable}_member"):
                suffixes.add(key[len(variable) :])
        member_sets.append(suffixes)
    common = set.intersection(*member_sets) if member_sets else set()
    return sorted(common, key=lambda value: (value != "", value))


def parse_ensemble_response(
    payload: str | bytes | dict[str, Any], target_date: date
) -> pd.DataFrame:
    """Open-Meteo 응답을 시간×앙상블 구성원의 긴 표로 바꾼다."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise WeatherEnsembleError("앙상블 날씨 응답을 읽을 수 없습니다.") from error
    else:
        parsed = payload
    if not isinstance(parsed, dict):
        raise WeatherEnsembleError("앙상블 날씨 응답 구조가 예상과 다릅니다.")
    if parsed.get("error"):
        raise WeatherEnsembleError(
            f"앙상블 날씨 API 오류: {parsed.get('reason', '알 수 없는 오류')}"
        )
    hourly = parsed.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise WeatherEnsembleError("앙상블 날씨 응답에 시간 정보가 없습니다.")
    timestamps = pd.to_datetime(hourly["time"], errors="coerce")
    if timestamps.isna().any():
        raise WeatherEnsembleError("앙상블 날씨의 시각을 읽을 수 없습니다.")

    suffixes = _member_suffixes(hourly)
    if len(suffixes) < 5:
        raise WeatherEnsembleError("사용 가능한 기상 시나리오가 5개 미만입니다.")

    frames: list[pd.DataFrame] = []
    for suffix in suffixes:
        frame = pd.DataFrame({"timestamp": timestamps})
        frame["ensemble_member"] = "control" if not suffix else suffix.lstrip("_")
        for variable in WEATHER_COLUMNS:
            key = f"{variable}{suffix}"
            values = hourly.get(key)
            if not isinstance(values, list) or len(values) != len(timestamps):
                raise WeatherEnsembleError(f"앙상블 날씨의 {key} 길이가 맞지 않습니다.")
            frame[variable] = pd.to_numeric(values, errors="coerce")
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result = result[result["timestamp"].dt.date == target_date].reset_index(drop=True)
    counts = result.groupby("ensemble_member")["timestamp"].count()
    if counts.empty or not (counts == 24).all():
        raise WeatherEnsembleError("선택 날짜의 24시간 앙상블 예보가 완전하지 않습니다.")
    if result[WEATHER_COLUMNS].isna().any().any():
        raise WeatherEnsembleError("앙상블 날씨에 숫자가 아닌 값이 있습니다.")
    return result


def fetch_open_meteo_ensemble(target_day_offset: int = 1) -> pd.DataFrame:
    """제주시 오늘 또는 내일의 다중 날씨 시나리오를 가져온다."""
    if target_day_offset not in {0, 1}:
        raise ValueError("앙상블 날짜 간격은 오늘 0 또는 내일 1이어야 합니다.")
    try:
        with urlopen(build_ensemble_url(), timeout=15) as response:
            payload = json.load(response)
    except Exception as error:
        raise WeatherEnsembleError(
            "여러 날씨 시나리오를 가져오지 못해 기존 예상범위를 사용합니다."
        ) from error
    now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    target_date = (now + pd.Timedelta(days=target_day_offset)).date()
    return parse_ensemble_response(payload, target_date)


def apply_ensemble_uncertainty(
    forecast: pd.DataFrame,
    models: dict[str, object],
    history: pd.DataFrame,
    ensemble_weather: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """여러 기상경로의 AI 결과로 기존 예상범위를 더 보수적으로 넓힌다."""
    required = {"timestamp", "ensemble_member", *WEATHER_COLUMNS}
    missing = required - set(ensemble_weather.columns)
    if missing:
        raise ValueError(f"앙상블 계산에 필요한 열이 없습니다: {sorted(missing)}")
    result = forecast.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    predictions: list[pd.DataFrame] = []
    for member, member_weather in ensemble_weather.groupby("ensemble_member"):
        member_weather = member_weather.sort_values("timestamp").reset_index(drop=True)
        features = make_live_features(member_weather)
        predictions.append(
            pd.DataFrame(
                {
                    "timestamp": member_weather["timestamp"],
                    "ensemble_member": member,
                    "smp": models["smp"].predict(features),
                    "renewable_mwh": np.maximum(
                        models["renewable_mwh"].predict(features), 0
                    ),
                }
            )
        )
    predicted = pd.concat(predictions, ignore_index=True)
    summary = predicted.groupby("timestamp").agg(
        ensemble_smp_p10=("smp", lambda values: values.quantile(0.10)),
        ensemble_smp_p50=("smp", "median"),
        ensemble_smp_p90=("smp", lambda values: values.quantile(0.90)),
        ensemble_renewable_p10=(
            "renewable_mwh",
            lambda values: values.quantile(0.10),
        ),
        ensemble_renewable_p50=("renewable_mwh", "median"),
        ensemble_renewable_p90=(
            "renewable_mwh",
            lambda values: values.quantile(0.90),
        ),
        ensemble_member_count=("ensemble_member", "nunique"),
    ).reset_index()
    result = result.merge(summary, on="timestamp", how="left", validate="one_to_one")
    if result["ensemble_member_count"].isna().any():
        raise WeatherEnsembleError("기본 예보와 앙상블 예보의 시간이 맞지 않습니다.")

    smp_downside = (
        result["predicted_smp"] - result["ensemble_smp_p10"]
    ).clip(lower=0)
    smp_upside = (
        result["ensemble_smp_p90"] - result["predicted_smp"]
    ).clip(lower=0)
    renewable_downside = (
        result["predicted_renewable_mwh"] - result["ensemble_renewable_p10"]
    ).clip(lower=0)
    renewable_upside = (
        result["ensemble_renewable_p90"] - result["predicted_renewable_mwh"]
    ).clip(lower=0)
    result["predicted_smp_lower"] -= smp_downside
    result["predicted_smp_upper"] += smp_upside
    result["predicted_renewable_lower"] = np.maximum(
        result["predicted_renewable_lower"] - renewable_downside, 0
    )
    result["predicted_renewable_upper"] += renewable_upside

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
    renewable_ranges = (
        result["ensemble_renewable_p90"] - result["ensemble_renewable_p10"]
    )
    metadata = {
        "model": ENSEMBLE_MODEL,
        "member_count": int(result["ensemble_member_count"].min()),
        "mean_renewable_p10_p90_range_mwh": float(renewable_ranges.mean()),
        "max_renewable_p10_p90_range_mwh": float(renewable_ranges.max()),
    }
    return result, metadata
