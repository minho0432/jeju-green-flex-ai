"""사용자가 원하는 시간까지 충전량을 채우는 Green Time 계산기."""

from __future__ import annotations

import pandas as pd


def _allocate(sorted_slots: pd.DataFrame, required_grid_kwh: float, charger_kw: float):
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
    reward_rate: float,
    reward_threshold: float,
) -> dict[str, float]:
    used = schedule[schedule["scheduled_kwh"] > 0].copy()
    energy = float(used["scheduled_kwh"].sum())
    eligible_energy = float(
        used.loc[used["green_score"] >= reward_threshold, "scheduled_kwh"].sum()
    )
    reward = eligible_energy * reward_rate
    gross_cost = energy * retail_price
    if energy == 0:
        return {
            "energy_kwh": 0,
            "eligible_kwh": 0,
            "reward_won": 0,
            "gross_cost_won": 0,
            "effective_cost_won": 0,
            "market_cost_proxy_won": 0,
            "weighted_renewable_mwh": 0,
            "weighted_green_score": 0,
        }
    return {
        "energy_kwh": energy,
        "eligible_kwh": eligible_energy,
        "reward_won": reward,
        "gross_cost_won": gross_cost,
        "effective_cost_won": gross_cost - reward,
        "market_cost_proxy_won": float(
            (used["scheduled_kwh"] * used["predicted_smp"]).sum()
        ),
        "weighted_renewable_mwh": float(
            (used["scheduled_kwh"] * used["predicted_renewable_mwh"]).sum()
            / energy
        ),
        "weighted_green_score": float(
            (used["scheduled_kwh"] * used["green_score"]).sum() / energy
        ),
    }


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
    reward_rate: float,
    reward_threshold: float = 70,
) -> dict:
    if target_soc <= current_soc:
        raise ValueError("목표 배터리는 현재 배터리보다 높아야 합니다.")
    if not 0 < efficiency <= 1:
        raise ValueError("충전 효율은 0보다 크고 1 이하여야 합니다.")
    if departure_hour <= start_hour:
        raise ValueError("출발 시각은 충전 시작 가능 시각보다 늦어야 합니다.")

    data = forecast.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    available = data[
        (data["timestamp"].dt.hour >= start_hour)
        & (data["timestamp"].dt.hour < departure_hour)
    ].copy()
    if available.empty:
        raise ValueError("선택한 시간 안에 충전 가능한 시간이 없습니다.")

    battery_energy_needed = battery_kwh * (target_soc - current_soc) / 100
    required_grid_kwh = battery_energy_needed / efficiency
    max_grid_kwh = len(available) * charger_kw
    feasible = max_grid_kwh + 1e-9 >= required_grid_kwh
    energy_to_schedule = min(required_grid_kwh, max_grid_kwh)

    ai_order = available.sort_values(
        ["green_score", "predicted_smp"], ascending=[False, True]
    )
    ai_schedule, remaining = _allocate(ai_order, energy_to_schedule, charger_kw)
    ai_schedule = ai_schedule.sort_values("timestamp")

    baseline_order = available.sort_values("timestamp")
    baseline_schedule, _ = _allocate(
        baseline_order, energy_to_schedule, charger_kw
    )

    ai_summary = _summarize(
        ai_schedule, retail_price, reward_rate, reward_threshold
    )
    baseline_summary = _summarize(
        baseline_schedule, retail_price, 0, reward_threshold
    )
    reached_soc = current_soc + (
        energy_to_schedule * efficiency / battery_kwh * 100
    )

    return {
        "feasible": feasible,
        "required_grid_kwh": required_grid_kwh,
        "max_grid_kwh": max_grid_kwh,
        "remaining_kwh": remaining,
        "reached_soc": min(reached_soc, 100),
        "ai_schedule": ai_schedule,
        "baseline_schedule": baseline_schedule,
        "ai": ai_summary,
        "baseline": baseline_summary,
    }
