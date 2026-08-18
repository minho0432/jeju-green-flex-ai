"""추천 결과를 자연어로 설명하는 선택적 OpenAI 호환 LLM 연동."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests


DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


def _scheduled_rows(forecast: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    schedule = plan.get("ai_schedule")
    if not isinstance(schedule, pd.DataFrame) or "scheduled_kwh" not in schedule:
        return forecast.iloc[0:0].copy()
    timestamps = schedule.loc[schedule["scheduled_kwh"] > 1e-6, "timestamp"]
    return forecast[forecast["timestamp"].isin(timestamps)]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary(
    forecast: pd.DataFrame,
    plan: dict[str, Any],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = _scheduled_rows(forecast, plan)
    score_column = plan.get("score_column") or "planning_score"
    scores = pd.to_numeric(forecast.get(score_column, pd.Series(dtype=float)), errors="coerce")
    risks = pd.to_numeric(
        forecast.get("forecast_risk_points", pd.Series(dtype=float)), errors="coerce"
    )
    wind = pd.to_numeric(
        forecast.get("wind_speed_10m", pd.Series(dtype=float)), errors="coerce"
    )
    radiation = pd.to_numeric(
        forecast.get("shortwave_radiation", pd.Series(dtype=float)), errors="coerce"
    )

    def values(frame: pd.DataFrame, column: str) -> list[float]:
        if column not in frame:
            return []
        return [round(_number(value), 2) for value in frame[column].dropna().tolist()]

    return {
        "mode": (context or {}).get("mode", "알 수 없음"),
        "has_observed_data": bool((context or {}).get("has_observed", False)),
        "recommended_time_start": (
            selected["timestamp"].min().strftime("%H:%M") if not selected.empty else None
        ),
        "recommended_time_end": (
            (selected["timestamp"].max() + pd.Timedelta(hours=1)).strftime("%H:%M")
            if not selected.empty
            else None
        ),
        "current_soc": _number((context or {}).get("current_soc")),
        "target_soc": _number((context or {}).get("target_soc")),
        "required_grid_kwh": _number(plan.get("required_grid_kwh")),
        "reached_soc": _number(plan.get("reached_soc")),
        "recommended_score_average": round(_number(scores.mean()), 1) if not scores.empty else None,
        "recommended_risk_average": round(_number(risks.mean()), 1) if not risks.empty else None,
        "recommended_wind_speed": values(selected, "wind_speed_10m"),
        "recommended_solar_radiation": values(selected, "shortwave_radiation"),
        "all_forecast_score_min": round(_number(scores.min()), 1) if not scores.empty else None,
        "all_forecast_score_max": round(_number(scores.max()), 1) if not scores.empty else None,
        "all_forecast_risk_max": round(_number(risks.max()), 1) if not risks.empty else None,
        "forecast_wind_average": round(_number(wind.mean()), 2) if not wind.empty else None,
        "forecast_radiation_average": round(_number(radiation.mean()), 2) if not radiation.empty else None,
    }


def _fallback_explanation(summary: dict[str, Any]) -> str:
    start = summary.get("recommended_time_start")
    end = summary.get("recommended_time_end")
    if start and end:
        recommendation = f"추천 충전시간은 {start}~{end}입니다."
    else:
        recommendation = "조건을 만족하는 추천 충전시간이 없습니다."

    reasons: list[str] = []
    risk = summary.get("recommended_risk_average")
    if risk is not None and risk >= 15:
        reasons.append("예측 범위가 넓어 보수적인 점수를 적용했습니다")
    wind = summary.get("forecast_wind_average")
    if wind is not None and wind <= 3:
        reasons.append("평균 바람 예보가 약합니다")
    radiation = summary.get("forecast_radiation_average")
    if radiation is not None and radiation < 100:
        reasons.append("일사량 예보가 낮은 시간대가 포함되어 있습니다")
    if not reasons:
        reasons.append("예측 재생에너지와 제주 전력수요의 공급여력을 비교했습니다")
    return recommendation + " " + " 또한, ".join(reasons) + "."


def explain_recommendation(
    forecast: pd.DataFrame,
    plan: dict[str, Any],
    context: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    timeout: float = 20.0,
) -> str:
    """LLM 설명을 생성하고, 설정·네트워크 오류 시 결정론적 설명을 반환한다."""
    summary = _summary(forecast, plan, context)
    fallback = _fallback_explanation(summary)
    api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return fallback

    payload = {
        "model": model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        "temperature": 0.2,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 제주 EV 충전 추천 설명 도우미다. 제공된 수치만 사용해 한국어로 "
                    "2~3문장으로 설명한다. 수치를 새로 계산하거나 확정적인 표현을 하지 말고, "
                    "예측 오차·날씨 불확실성·Green Score가 낮은 이유·실시간 보정 필요성을 "
                    "해당할 때만 설명한다."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(summary, ensure_ascii=False),
            },
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            endpoint or os.environ.get("OPENAI_BASE_URL", DEFAULT_ENDPOINT),
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM 응답 내용이 비어 있습니다.")
        return content.strip()
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return fallback