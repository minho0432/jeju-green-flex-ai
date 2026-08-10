"""AI 결과 파일과 충전 최적화가 발표 전에 정상인지 한 번에 검사한다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optimizer import make_plan


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATH = ROOT / "outputs" / "demo_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"


def validate() -> None:
    if not PREDICTION_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("python scripts/train_models.py를 먼저 실행하세요.")

    forecast = pd.read_csv(PREDICTION_PATH, parse_dates=["timestamp"])
    required_columns = {
        "predicted_smp",
        "predicted_smp_lower",
        "predicted_smp_upper",
        "predicted_renewable_mwh",
        "predicted_renewable_lower",
        "predicted_renewable_upper",
        "green_score",
        "planning_score",
        "forecast_risk_points",
        "actual_green_score",
    }
    missing = required_columns - set(forecast.columns)
    if missing:
        raise ValueError(f"예측 파일에 필요한 열이 없습니다: {sorted(missing)}")
    if len(forecast) != 24:
        raise ValueError(f"24시간 예측이어야 하지만 {len(forecast)}행입니다.")
    if forecast[list(required_columns)].isna().any().any():
        raise ValueError("예측 결과에 빈칸이 있습니다.")
    if not forecast["green_score"].between(0, 100).all():
        raise ValueError("Green Score가 0~100 범위를 벗어났습니다.")
    if not forecast["planning_score"].between(0, 100).all():
        raise ValueError("보수적 planning_score가 0~100 범위를 벗어났습니다.")
    if (forecast["planning_score"] > forecast["green_score"] + 1e-9).any():
        raise ValueError("보수적 점수가 중심 예측 점수보다 높은 시간이 있습니다.")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for target in ("smp", "renewable_mwh"):
        if target not in metrics["targets"]:
            raise ValueError(f"{target} 평가 결과가 없습니다.")
        if target not in metrics.get("forecast_only_targets", {}):
            raise ValueError(f"{target} 내일 예보용 모델 평가 결과가 없습니다.")
    score_weights = metrics.get("score_weights", {})
    if score_weights.get("renewable") != 0.8 or score_weights.get("market_smp") != 0.2:
        raise ValueError("Green Score 가중치는 재생에너지 80%, SMP 20%여야 합니다.")

    plan = make_plan(
        forecast=forecast,
        current_soc=30,
        target_soc=80,
        battery_kwh=60,
        charger_kw=7,
        efficiency=0.9,
        start_hour=8,
        departure_hour=20,
        retail_price=320,
        base_point_rate=10,
        bonus_point_rate=20,
        reward_threshold=70,
        session_point_cap=1500,
        continuous=True,
        conservative=True,
    )
    if not plan["feasible"] or abs(plan["reached_soc"] - 80) > 1e-6:
        raise ValueError("기본 시나리오가 목표 배터리 80%를 달성하지 못했습니다.")
    if not plan["conservative"]:
        raise ValueError("기본 시나리오가 보수적 점수를 사용하지 않았습니다.")
    if plan["ai"]["guaranteed_points"] <= 0:
        raise ValueError("참여 보장 포인트가 계산되지 않았습니다.")
    if plan["ai"]["settled_total_points"] < plan["ai"]["guaranteed_points"]:
        raise ValueError("정산 포인트가 보장 포인트보다 작습니다.")
    if plan["ai"]["settled_total_points"] > 1500:
        raise ValueError("정산 포인트가 세션 상한을 넘었습니다.")

    print("AI 결과 검사 통과")
    print(f"예측 행 수: {len(forecast)}")
    print(f"추천 충전량: {plan['ai']['energy_kwh']:.2f} kWh")
    print(f"예상 도달 배터리: {plan['reached_soc']:.1f}%")
    print(f"참여 보장 포인트: {plan['ai']['guaranteed_points']:,.0f}P")
    print(f"성과형 보너스: {plan['ai']['settled_bonus_points']:,.0f}P")
    print(f"재현 정산 포인트: {plan['ai']['settled_total_points']:,.0f}P")
    for target in ("smp", "renewable_mwh"):
        values = metrics["targets"][target]
        print(
            f"{target}: AI MAE {values['ai']['mae']}, "
            f"공정한 개선율 {values['mae_improvement_percent']}%"
        )


if __name__ == "__main__":
    validate()
