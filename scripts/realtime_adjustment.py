"""실측값이 도착할 때 가까운 미래 예측을 안전하게 보정하는 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_utils import score_against_history


REQUIRED_FORECAST_COLUMNS = {
    "timestamp",
    "predicted_smp",
    "predicted_smp_lower",
    "predicted_smp_upper",
    "predicted_renewable_mwh",
    "predicted_renewable_lower",
    "predicted_renewable_upper",
    "actual_smp",
    "actual_renewable_mwh",
}


def _recent_weighted_bias(
    observed: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    lookback_hours: int,
) -> float:
    recent = observed.dropna(subset=[actual_column, prediction_column]).tail(
        lookback_hours
    )
    if recent.empty:
        return 0.0
    residuals = (
        recent[actual_column].to_numpy(dtype=float)
        - recent[prediction_column].to_numpy(dtype=float)
    )
    weights = np.arange(1, len(recent) + 1, dtype=float)
    return float(np.average(residuals, weights=weights))


def _recalculate_scores(result: pd.DataFrame, history: pd.DataFrame) -> None:
    """보정 후 공급여력 Green Score를 다시 계산한다."""
    from model_utils import attach_supply_margin_scores, ensure_demand_column, month_hour_baseline

    history = ensure_demand_column(history)
    if "predicted_demand_mwh" not in result.columns:
        result["predicted_demand_mwh"] = month_hour_baseline(
            history, "demand_mwh", result
        ).to_numpy()
    if "predicted_demand_upper" not in result.columns:
        result["predicted_demand_upper"] = result["predicted_demand_mwh"] * 1.1
    if "predicted_demand_lower" not in result.columns:
        result["predicted_demand_lower"] = result["predicted_demand_mwh"] * 0.9

    scored = attach_supply_margin_scores(result, history)
    for col in scored.columns:
        result[col] = scored[col]


def adjust_forecast_with_observations(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_hours: int = 3,
    decay_hours: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    missing = REQUIRED_FORECAST_COLUMNS - set(forecast.columns)
    if missing:
        raise ValueError(f"실시간 보정에 필요한 열이 없습니다: {sorted(missing)}")
    if lookback_hours < 1:
        raise ValueError("실측 오차 확인시간은 1시간 이상이어야 합니다.")
    if decay_hours <= 0:
        raise ValueError("보정 영향 감소시간은 0보다 커야 합니다.")

    result = forecast.copy().sort_values("timestamp").reset_index(drop=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    as_of = pd.Timestamp(as_of)
    if as_of < result["timestamp"].min() or as_of >= result["timestamp"].max():
        raise ValueError("실측 도착 시각은 예측 하루 안에서 마지막 시간보다 앞이어야 합니다.")

    observed_mask = result["timestamp"] <= as_of
    future_mask = result["timestamp"] > as_of
    observed = result.loc[observed_mask]
    if observed.empty:
        raise ValueError("보정에 사용할 도착 실측값이 없습니다.")

    result["raw_predicted_smp"] = result["predicted_smp"]
    result["raw_predicted_renewable_mwh"] = result["predicted_renewable_mwh"]
    result["observed_actual_smp"] = result["actual_smp"].where(observed_mask)
    result["observed_actual_renewable_mwh"] = result["actual_renewable_mwh"].where(
        observed_mask
    )
    result["actual_smp"] = result["observed_actual_smp"]
    result["actual_renewable_mwh"] = result["observed_actual_renewable_mwh"]
    result = result.drop(columns=["actual_green_score"], errors="ignore")

    smp_bias = _recent_weighted_bias(
        observed, "actual_smp", "predicted_smp", lookback_hours
    )
    renewable_bias = _recent_weighted_bias(
        observed,
        "actual_renewable_mwh",
        "predicted_renewable_mwh",
        lookback_hours,
    )

    lead_hours = (
        (result.loc[future_mask, "timestamp"] - as_of).dt.total_seconds() / 3600
    )
    decay = np.exp(-(lead_hours - 1).clip(lower=0) / decay_hours)
    result["smp_realtime_correction"] = 0.0
    result["renewable_realtime_correction_mwh"] = 0.0
    result.loc[future_mask, "smp_realtime_correction"] = smp_bias * decay.to_numpy()
    result.loc[future_mask, "renewable_realtime_correction_mwh"] = (
        renewable_bias * decay.to_numpy()
    )

    result.loc[future_mask, "predicted_smp"] = (
        result.loc[future_mask, "raw_predicted_smp"]
        + result.loc[future_mask, "smp_realtime_correction"]
    )
    result.loc[future_mask, "predicted_renewable_mwh"] = np.maximum(
        result.loc[future_mask, "raw_predicted_renewable_mwh"]
        + result.loc[future_mask, "renewable_realtime_correction_mwh"],
        0,
    )

    result.loc[future_mask, "predicted_smp_lower"] += result.loc[
        future_mask, "smp_realtime_correction"
    ]
    result.loc[future_mask, "predicted_smp_upper"] += result.loc[
        future_mask, "smp_realtime_correction"
    ]
    result.loc[future_mask, "predicted_renewable_lower"] = np.maximum(
        result.loc[future_mask, "predicted_renewable_lower"]
        + result.loc[future_mask, "renewable_realtime_correction_mwh"],
        0,
    )
    result.loc[future_mask, "predicted_renewable_upper"] = np.maximum(
        result.loc[future_mask, "predicted_renewable_upper"]
        + result.loc[future_mask, "renewable_realtime_correction_mwh"],
        0,
    )

    # 공식 계통 수요 실측이 있으면 수요 예측도 같은 방식으로 보정
    demand_bias = 0.0
    result["demand_realtime_correction_mwh"] = 0.0
    if (
        "actual_demand_mwh" in result.columns
        and "predicted_demand_mwh" in result.columns
    ):
        result["raw_predicted_demand_mwh"] = result["predicted_demand_mwh"]
        result["observed_actual_demand_mwh"] = result["actual_demand_mwh"].where(
            observed_mask
        )
        result["actual_demand_mwh"] = result["observed_actual_demand_mwh"]
        demand_bias = _recent_weighted_bias(
            observed,
            "actual_demand_mwh",
            "predicted_demand_mwh",
            lookback_hours,
        )
        result.loc[future_mask, "demand_realtime_correction_mwh"] = (
            demand_bias * decay.to_numpy()
        )
        result.loc[future_mask, "predicted_demand_mwh"] = np.maximum(
            result.loc[future_mask, "raw_predicted_demand_mwh"]
            + result.loc[future_mask, "demand_realtime_correction_mwh"],
            0,
        )
        if "predicted_demand_lower" in result.columns:
            result.loc[future_mask, "predicted_demand_lower"] = np.maximum(
                result.loc[future_mask, "predicted_demand_lower"]
                + result.loc[future_mask, "demand_realtime_correction_mwh"],
                0,
            )
        if "predicted_demand_upper" in result.columns:
            result.loc[future_mask, "predicted_demand_upper"] = np.maximum(
                result.loc[future_mask, "predicted_demand_upper"]
                + result.loc[future_mask, "demand_realtime_correction_mwh"],
                0,
            )

    score_history = history.copy()
    score_reference_end = "timestamp_not_available"
    if "timestamp" in score_history.columns:
        score_history["timestamp"] = pd.to_datetime(score_history["timestamp"])
        score_history = score_history[score_history["timestamp"] <= as_of]
        if score_history.empty:
            raise ValueError("현재 시각 이전의 Green Score 기준 데이터가 없습니다.")
        score_reference_end = score_history["timestamp"].max().isoformat()
    _recalculate_scores(result, score_history)
    result["source_mode"] = "historical_realtime_adjustment_replay"
    result["observation_as_of"] = as_of

    metadata: dict[str, float | int | str] = {
        "as_of": as_of.isoformat(),
        "observed_hours": int(observed_mask.sum()),
        "future_hours": int(future_mask.sum()),
        "lookback_hours": int(lookback_hours),
        "decay_hours": float(decay_hours),
        "recent_smp_bias": smp_bias,
        "recent_renewable_bias_mwh": renewable_bias,
        "recent_demand_bias_mwh": demand_bias,
        "score_reference_rows": int(len(score_history)),
        "score_reference_end": score_reference_end,
    }
    return result, metadata


def adjust_forecast_with_live_renewables(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    hourly_observations: pd.DataFrame,
    as_of: pd.Timestamp,
    minimum_coverage: float = 0.9,
    lookback_hours: int = 3,
    decay_hours: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    required = {"timestamp", "actual_renewable_mwh", "coverage_ratio"}
    missing = required - set(hourly_observations.columns)
    if missing:
        raise ValueError(f"공식 실측 보정에 필요한 열이 없습니다: {sorted(missing)}")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("실측 완성도 기준은 0보다 크고 1 이하여야 합니다.")

    observations = hourly_observations.copy()
    observations["timestamp"] = pd.to_datetime(observations["timestamp"])
    observations = observations[
        (observations["timestamp"] <= pd.Timestamp(as_of))
        & (observations["coverage_ratio"] >= minimum_coverage)
    ].copy()
    keep_cols = ["timestamp", "actual_renewable_mwh"]
    if "actual_demand_mwh" in observations.columns:
        keep_cols.append("actual_demand_mwh")
    observations = observations[keep_cols]
    if observations.empty:
        raise ValueError("보정에 사용할 완전한 재생에너지 실측시간이 없습니다.")

    prepared = forecast.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared = prepared.drop(
        columns=[
            "actual_smp",
            "actual_renewable_mwh",
            "actual_demand_mwh",
            "actual_green_score",
        ],
        errors="ignore",
    )
    prepared = prepared.merge(observations, on="timestamp", how="left")
    prepared["actual_smp"] = np.nan
    if "actual_demand_mwh" not in prepared.columns:
        prepared["actual_demand_mwh"] = np.nan
    adjusted, metadata = adjust_forecast_with_observations(
        prepared,
        history,
        as_of=as_of,
        lookback_hours=lookback_hours,
        decay_hours=decay_hours,
    )
    adjusted["source_mode"] = "official_jeju_grid_live_adjustment"
    metadata["renewable_observed_hours"] = int(len(observations))
    metadata["demand_observed_hours"] = int(
        observations["actual_demand_mwh"].notna().sum()
        if "actual_demand_mwh" in observations.columns
        else 0
    )
    metadata["api_actual_source"] = "KPX JejuSukub5mToday (renewable + currPwrTot demand)"
    return adjusted, metadata


def five_minute_mw_to_hourly_mwh(
    samples: pd.DataFrame,
    power_columns: tuple[str, ...] = ("solar_mw", "wind_mw"),
) -> pd.DataFrame:
    required = {"timestamp", *power_columns}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"5분 실측 변환에 필요한 열이 없습니다: {sorted(missing)}")

    data = samples.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if data["timestamp"].duplicated().any():
        raise ValueError("같은 시각의 5분 실측값이 중복되어 있습니다.")
    data["hour"] = data["timestamp"].dt.floor("h")
    for column in power_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[list(power_columns)].isna().any().any():
        raise ValueError("5분 실측값에 숫자가 아닌 값이나 빈칸이 있습니다.")

    result = data.groupby("hour", as_index=False)[list(power_columns)].sum()
    result = result.rename(columns={"hour": "timestamp"})
    for column in power_columns:
        result[column.replace("_mw", "_mwh")] = result.pop(column) * (5 / 60)
    counts = data.groupby("hour").size().reindex(result["timestamp"]).to_numpy()
    result["coverage_ratio"] = np.minimum(counts / 12, 1.0)
    mwh_columns = [column.replace("_mw", "_mwh") for column in power_columns]
    result["renewable_mwh"] = result[mwh_columns].sum(axis=1)
    return result
