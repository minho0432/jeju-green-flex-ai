"""사용자 일정 안에서 보수적인 Green Time과 Green Point를 계산한다."""

from __future__ import annotations

import math

import pandas as pd


def _allocate(sorted_slots: pd.DataFrame, required_grid_kwh: float, charger_kw: float):
    """정렬된 시간칸에 필요한 충전량을 앞에서부터 배분한다."""
    schedule = sorted_slots.copy()
    schedule["scheduled_kwh"] = 0.0
    remaining = required_grid_kwh
    for index in schedule.index:
        amount = min(charger_kw, remaining)
        schedule.loc[index, "scheduled_kwh"] = max(amount, 0)
        remaining -= amount
        if remaining <= 1e-9:
            break
    return schedule, max(remaining, 0)


def _summarize(
    schedule: pd.DataFrame,
    retail_price: float,
    base_point_rate: float,
    bonus_point_rate: float,
    reward_threshold: float,
    session_point_cap: float,
) -> dict[str, float | str]:
    """비용과 Green Point를 예측 단계와 사후 정산 단계로 분리한다."""
    used = schedule[schedule["scheduled_kwh"] > 0].copy()
    energy = float(used["scheduled_kwh"].sum())
    gross_cost = energy * retail_price

    if energy == 0:
        return {
            "energy_kwh": 0.0,
            "guaranteed_points": 0.0,
            "expected_bonus_points": 0.0,
            "settled_bonus_points": 0.0,
            "expected_total_points": 0.0,
            "settled_total_points": 0.0,
            "gross_cost_won": 0.0,
            "simulated_minimum_cost_won": 0.0,
            "simulated_expected_cost_won": 0.0,
            "simulated_settled_cost_won": 0.0,
            "market_cost_proxy_won": 0.0,
            "weighted_renewable_mwh": 0.0,
            "weighted_green_score": 0.0,
            "weighted_planning_score": 0.0,
            "settlement_status": "no_charge",
        }

    # 추천시간을 실제로 따른 대가. 예보가 틀려도 회수하지 않는다는 정책 가정이다.
    guaranteed_points = min(energy * base_point_rate, session_point_cap)
    remaining_point_cap = max(session_point_cap - guaranteed_points, 0)
    expected_eligible = float(
        used.loc[used["green_score"] >= reward_threshold, "scheduled_kwh"].sum()
    )
    expected_bonus_points = min(
        expected_eligible * bonus_point_rate, remaining_point_cap
    )

    if "actual_green_score" in used.columns:
        settled_eligible = float(
            used.loc[
                used["actual_green_score"] >= reward_threshold,
                "scheduled_kwh",
            ].sum()
        )
        settled_bonus_points = min(
            settled_eligible * bonus_point_rate, remaining_point_cap
        )
        settlement_status = "historical_replay_settled"
    else:
        settled_bonus_points = 0.0
        settlement_status = "pending_actual_data"

    expected_total_points = guaranteed_points + expected_bonus_points
    settled_total_points = guaranteed_points + settled_bonus_points
    planning_column = "planning_score" if "planning_score" in used.columns else "green_score"

    return {
        "energy_kwh": energy,
        "guaranteed_points": guaranteed_points,
        "expected_bonus_points": expected_bonus_points,
        "settled_bonus_points": settled_bonus_points,
        "expected_total_points": expected_total_points,
        "settled_total_points": settled_total_points,
        "gross_cost_won": gross_cost,
        # 아래 비용은 1P=1원을 가정한 정책 시뮬레이션이며 실제 결제금액이 아니다.
        "simulated_minimum_cost_won": gross_cost - guaranteed_points,
        "simulated_expected_cost_won": gross_cost - expected_total_points,
        "simulated_settled_cost_won": gross_cost - settled_total_points,
        "market_cost_proxy_won": float(
            (used["scheduled_kwh"] * used["predicted_smp"]).sum()
        ),
        "weighted_renewable_mwh": float(
            (used["scheduled_kwh"] * used["predicted_renewable_mwh"]).sum() / energy
        ),
        "weighted_green_score": float(
            (used["scheduled_kwh"] * used["green_score"]).sum() / energy
        ),
        "weighted_planning_score": float(
            (used["scheduled_kwh"] * used[planning_column]).sum() / energy
        ),
        "settlement_status": settlement_status,
    }


def _best_continuous_schedule(
    available: pd.DataFrame,
    required_grid_kwh: float,
    charger_kw: float,
    score_column: str,
    price_column: str,
) -> tuple[pd.DataFrame, float]:
    """여러 번 꽂았다 빼지 않도록 가장 좋은 연속 충전구간을 찾는다."""
    ordered = available.sort_values("timestamp").reset_index(drop=True)
    slots_needed = min(math.ceil(required_grid_kwh / charger_kw), len(ordered))
    best_schedule = None
    best_remaining = required_grid_kwh
    best_key = None

    for start in range(0, len(ordered) - slots_needed + 1):
        candidate = ordered.iloc[start : start + slots_needed].copy()
        schedule, remaining = _allocate(candidate, required_grid_kwh, charger_kw)
        used = schedule[schedule["scheduled_kwh"] > 0]
        score_value = float((used["scheduled_kwh"] * used[score_column]).sum())
        price_value = float((used["scheduled_kwh"] * used[price_column]).sum())
        key = (score_value, -price_value)
        if best_key is None or key > best_key:
            best_schedule = schedule
            best_remaining = remaining
            best_key = key

    if best_schedule is None:
        return _allocate(ordered, required_grid_kwh, charger_kw)
    return best_schedule, best_remaining


def make_plan(
    forecast: pd.DataFrame,
    current_soc: float,
    target_soc: float,
    battery_kwh: float,
    charger_kw: float,
    efficiency: float,
    start_hour: int,
    departure_hour: int,
    retail_price: float,
    base_point_rate: float = 10,
    bonus_point_rate: float = 20,
    reward_threshold: float = 70,
    session_point_cap: float = 1500,
    continuous: bool = True,
    conservative: bool = True,
) -> dict:
    """목표 SOC를 지키면서 가장 좋은 충전 계획을 만든다.

    conservative=True이면 예측 중심값이 아니라 가격 상한·재생에너지 하한으로
    계산한 planning_score를 사용한다. 예보가 빗나갈 때의 위험을 줄이기 위함이다.
    """
    if target_soc <= current_soc:
        raise ValueError("목표 배터리는 현재 배터리보다 높아야 합니다.")
    if not 0 < efficiency <= 1:
        raise ValueError("충전 효율은 0보다 크고 1 이하여야 합니다.")
    if departure_hour <= start_hour:
        raise ValueError("출발 시각은 충전 시작 가능 시각보다 늦어야 합니다.")
    if battery_kwh <= 0 or charger_kw <= 0:
        raise ValueError("배터리 용량과 충전기 출력은 0보다 커야 합니다.")
    if min(retail_price, base_point_rate, bonus_point_rate, session_point_cap) < 0:
        raise ValueError("요금·포인트 단가·포인트 한도는 음수가 될 수 없습니다.")

    data = forecast.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    available = data[
        (data["timestamp"].dt.hour >= start_hour)
        & (data["timestamp"].dt.hour < departure_hour)
    ].copy()
    if available.empty:
        raise ValueError("선택한 시간 안에 충전 가능한 시간이 없습니다.")

    use_conservative = conservative and "planning_score" in available.columns
    score_column = "planning_score" if use_conservative else "green_score"
    price_column = (
        "predicted_smp_upper"
        if use_conservative and "predicted_smp_upper" in available.columns
        else "predicted_smp"
    )

    battery_energy_needed = battery_kwh * (target_soc - current_soc) / 100
    required_grid_kwh = battery_energy_needed / efficiency
    max_grid_kwh = len(available) * charger_kw
    feasible = max_grid_kwh + 1e-9 >= required_grid_kwh
    energy_to_schedule = min(required_grid_kwh, max_grid_kwh)

    if continuous:
        ai_schedule, remaining = _best_continuous_schedule(
            available,
            energy_to_schedule,
            charger_kw,
            score_column,
            price_column,
        )
    else:
        ai_order = available.sort_values(
            [score_column, price_column], ascending=[False, True]
        )
        ai_schedule, remaining = _allocate(ai_order, energy_to_schedule, charger_kw)
    ai_schedule = ai_schedule.sort_values("timestamp")

    baseline_order = available.sort_values("timestamp")
    baseline_schedule, _ = _allocate(baseline_order, energy_to_schedule, charger_kw)

    ai_summary = _summarize(
        ai_schedule,
        retail_price,
        base_point_rate,
        bonus_point_rate,
        reward_threshold,
        session_point_cap,
    )
    baseline_summary = _summarize(
        baseline_schedule,
        retail_price,
        0,
        0,
        reward_threshold,
        0,
    )
    reached_soc = current_soc + energy_to_schedule * efficiency / battery_kwh * 100

    return {
        "feasible": feasible,
        "required_grid_kwh": required_grid_kwh,
        "max_grid_kwh": max_grid_kwh,
        "remaining_kwh": remaining,
        "reached_soc": min(reached_soc, 100),
        "continuous": continuous,
        "conservative": use_conservative,
        "score_column": score_column,
        "ai_schedule": ai_schedule,
        "baseline_schedule": baseline_schedule,
        "ai": ai_summary,
        "baseline": baseline_summary,
    }
