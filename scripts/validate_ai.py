"""AI 결과 파일과 충전 최적화가 발표 전에 정상인지 한 번에 검사한다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optimizer import derive_point_policy, make_plan
from realtime_adjustment import adjust_forecast_with_observations


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATH = ROOT / "outputs" / "demo_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
HISTORY_PATH = ROOT / "data" / "processed" / "train.csv"


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
    for target in ("smp", "renewable_mwh", "demand_mwh"):
        if target not in metrics["targets"]:
            raise ValueError(f"{target} 평가 결과가 없습니다.")
        if target not in metrics.get("forecast_only_targets", {}):
            raise ValueError(f"{target} 내일 예보용 모델 평가 결과가 없습니다.")
    score_weights = metrics.get("score_weights", {})
    if (
        score_weights.get("renewable_supply_margin") != 1.0
        or score_weights.get("market_smp") != 0.0
    ):
        raise ValueError("Green Score는 재생에너지/전력수요 공급여력 100%여야 합니다.")

    point_policy = derive_point_policy(3_000_000, 100_000)
    if point_policy["maximum_total_rate"] != 30:
        raise ValueError("월 예산 기반 최대 포인트 단가가 30P/kWh가 아닙니다.")

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
        base_point_rate=point_policy["base_point_rate"],
        bonus_point_rate=point_policy["maximum_bonus_rate"],
        partial_reward_threshold=50,
        full_reward_threshold=70,
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

    history = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
    replay_as_of = forecast["timestamp"].dt.normalize().iloc[0] + pd.Timedelta(hours=10)
    adjusted, adjustment_metadata = adjust_forecast_with_observations(
        forecast,
        history,
        as_of=replay_as_of,
    )
    future = adjusted[adjusted["timestamp"] > replay_as_of]
    if future[["actual_smp", "actual_renewable_mwh"]].notna().any().any():
        raise ValueError("실시간 보정 재현에서 미래 실제값이 노출되었습니다.")
    if "actual_green_score" in adjusted.columns:
        raise ValueError("실시간 보정 재현이 미래 실제 점수로 정산할 위험이 있습니다.")
    if adjustment_metadata["observed_hours"] != 11:
        raise ValueError("실시간 보정에 사용한 관측시간 수가 예상과 다릅니다.")
    if pd.Timestamp(adjustment_metadata["score_reference_end"]) > replay_as_of:
        raise ValueError("실시간 보정 점수 기준에 미래 데이터가 포함되었습니다.")

    print("AI 결과 검사 통과")
    print(f"예측 행 수: {len(forecast)}")
    print(f"추천 충전량: {plan['ai']['energy_kwh']:.2f} kWh")
    print(f"예상 도달 배터리: {plan['reached_soc']:.1f}%")
    print(f"참여 보장 포인트: {plan['ai']['guaranteed_points']:,.0f}P")
    print(f"성과형 보너스: {plan['ai']['settled_bonus_points']:,.0f}P")
    print(f"재현 정산 포인트: {plan['ai']['settled_total_points']:,.0f}P")
    print(
        "실시간 보정 재현: "
        f"{adjustment_metadata['observed_hours']}시간 실측 사용, 미래 실제값 차단"
    )
    for target in ("smp", "renewable_mwh", "demand_mwh"):
        values = metrics["forecast_only_targets"][target]
        print(
            f"{target}: 내일 예보형 MAE {values['ai']['mae']}, "
            f"마지막 30일 개선율 {values['mae_improvement_percent']}%"
        )


if __name__ == "__main__":
    validate()
