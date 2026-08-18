"""실측값이 도착할 때 가까운 미래 예측을 안전하게 보정하는 함수.

공식 제주 실시간 태양광·풍력·전력수요 API 실측값을 이용해
가까운 미래 예측을 보정한다.

검증용 과거 재현에서는 실제값을 시간 순서대로 공개해 같은 흐름을 재현한다.
미래 실제값은 보정이나 충전계획에 절대 전달하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_utils import attach_supply_margin_scores


REQUIRED_FORECAST_COLUMNS = {
    "timestamp",
    "predicted_renewable_mwh",
    "predicted_renewable_lower",
    "predicted_renewable_upper",
    "predicted_demand_mwh",
    "predicted_demand_lower",
    "predicted_demand_upper",
    "actual_renewable_mwh",
}


def _recent_weighted_bias(
    observed: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    lookback_hours: int,
) -> float:
    """최근 오차일수록 더 크게 반영한 평균 오차를 계산한다."""
    recent = observed.dropna(
        subset=[actual_column, prediction_column]
    ).tail(lookback_hours)

    if recent.empty:
        return 0.0

    residuals = (
        recent[actual_column].to_numpy(dtype=float)
        - recent[prediction_column].to_numpy(dtype=float)
    )

    weights = np.arange(1, len(recent) + 1, dtype=float)

    return float(np.average(residuals, weights=weights))


def _recalculate_scores(
    result: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    """보정된 재생에너지·수요 예측값으로 Green Score를 다시 계산한다."""
    scored = attach_supply_margin_scores(result, history)

    for column in (
        "predicted_supply_margin",
        "supply_margin_score",
        "renewable_opportunity_score",
        "green_score",
        "planning_score",
        "conservative_renewable_score",
        "forecast_risk_points",
    ):
        result[column] = scored[column]


def adjust_forecast_with_observations(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_hours: int = 3,
    decay_hours: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """현재까지 도착한 실측 오차로 아직 지나지 않은 시간만 보정한다.

    최근 실측 오차의 가중평균을 다음 한 시간에 가장 크게 반영하고,
    먼 시간으로 갈수록 지수적으로 영향력을 줄인다.

    재생에너지 실측은 항상 사용할 수 있고,
    actual_demand_mwh 열이 존재하면 전력수요도 함께 보정한다.

    미래 실제값은 화면·최적화·정산에 전달하지 않는다.
    """

    missing = REQUIRED_FORECAST_COLUMNS - set(forecast.columns)

    if missing:
        raise ValueError(
            f"실시간 보정에 필요한 열이 없습니다: {sorted(missing)}"
        )

    if lookback_hours < 1:
        raise ValueError(
            "실측 오차 확인시간은 1시간 이상이어야 합니다."
        )

    if decay_hours <= 0:
        raise ValueError(
            "보정 영향 감소시간은 0보다 커야 합니다."
        )

    result = (
        forecast.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    result["timestamp"] = pd.to_datetime(result["timestamp"])
    as_of = pd.Timestamp(as_of)

    if (
        as_of < result["timestamp"].min()
        or as_of >= result["timestamp"].max()
    ):
        raise ValueError(
            "실측 도착 시각은 예측 하루 안에서 "
            "마지막 시간보다 앞이어야 합니다."
        )

    observed_mask = result["timestamp"] <= as_of
    future_mask = result["timestamp"] > as_of

    observed = result.loc[observed_mask].copy()

    if observed.empty:
        raise ValueError(
            "보정에 사용할 도착 실측값이 없습니다."
        )

    # ---------------------------------------------------------
    # 보정 전 원본 예측값 저장
    # ---------------------------------------------------------

    result["raw_predicted_renewable_mwh"] = (
        result["predicted_renewable_mwh"]
    )

    result["raw_predicted_demand_mwh"] = (
        result["predicted_demand_mwh"]
    )

    # ---------------------------------------------------------
    # 현재 시각까지 도착한 실제값만 공개
    # ---------------------------------------------------------

    result["observed_actual_renewable_mwh"] = (
        result["actual_renewable_mwh"].where(observed_mask)
    )

    if "actual_demand_mwh" in result.columns:
        result["observed_actual_demand_mwh"] = (
            result["actual_demand_mwh"].where(observed_mask)
        )

    # 미래 실제값이 화면·최적화·정산으로 새지 않도록 숨긴다.
    result["actual_renewable_mwh"] = (
        result["observed_actual_renewable_mwh"]
    )

    if "observed_actual_demand_mwh" in result.columns:
        result["actual_demand_mwh"] = (
            result["observed_actual_demand_mwh"]
        )

    result = result.drop(
        columns=["actual_green_score"],
        errors="ignore",
    )

    # ---------------------------------------------------------
    # 최근 재생에너지 예측 오차 계산
    #
    # bias = 실제 - 예측
    # ---------------------------------------------------------

    renewable_bias = _recent_weighted_bias(
        observed,
        "actual_renewable_mwh",
        "predicted_renewable_mwh",
        lookback_hours,
    )

    # ---------------------------------------------------------
    # 최근 전력수요 예측 오차 계산
    #
    # actual_demand_mwh가 없는 과거 테스트/데이터도 지원한다.
    # ---------------------------------------------------------

    demand_bias = 0.0

    if "actual_demand_mwh" in observed.columns:
        demand_bias = _recent_weighted_bias(
            observed,
            "actual_demand_mwh",
            "predicted_demand_mwh",
            lookback_hours,
        )

    # ---------------------------------------------------------
    # 미래로 갈수록 보정 영향 감소
    # ---------------------------------------------------------

    lead_hours = (
        (
            result.loc[future_mask, "timestamp"]
            - as_of
        ).dt.total_seconds()
        / 3600
    )

    decay = np.exp(
        -(lead_hours - 1).clip(lower=0)
        / decay_hours
    )

    # ---------------------------------------------------------
    # 재생에너지 실시간 보정
    # ---------------------------------------------------------

    result["renewable_realtime_correction_mwh"] = 0.0

    result.loc[
        future_mask,
        "renewable_realtime_correction_mwh",
    ] = renewable_bias * decay.to_numpy()

    result.loc[
        future_mask,
        "predicted_renewable_mwh",
    ] = np.maximum(
        result.loc[
            future_mask,
            "raw_predicted_renewable_mwh",
        ]
        + result.loc[
            future_mask,
            "renewable_realtime_correction_mwh",
        ],
        0,
    )

    # 기존 예상범위의 폭은 유지하고 중심만 이동
    result.loc[
        future_mask,
        "predicted_renewable_lower",
    ] = np.maximum(
        result.loc[
            future_mask,
            "predicted_renewable_lower",
        ]
        + result.loc[
            future_mask,
            "renewable_realtime_correction_mwh",
        ],
        0,
    )

    result.loc[
        future_mask,
        "predicted_renewable_upper",
    ] = np.maximum(
        result.loc[
            future_mask,
            "predicted_renewable_upper",
        ]
        + result.loc[
            future_mask,
            "renewable_realtime_correction_mwh",
        ],
        0,
    )

    # ---------------------------------------------------------
    # 전력수요 실시간 보정
    # ---------------------------------------------------------

    result["demand_realtime_correction_mwh"] = 0.0

    result.loc[
        future_mask,
        "demand_realtime_correction_mwh",
    ] = demand_bias * decay.to_numpy()

    result.loc[
        future_mask,
        "predicted_demand_mwh",
    ] = np.maximum(
        result.loc[
            future_mask,
            "raw_predicted_demand_mwh",
        ]
        + result.loc[
            future_mask,
            "demand_realtime_correction_mwh",
        ],
        0,
    )

    result.loc[
        future_mask,
        "predicted_demand_lower",
    ] = np.maximum(
        result.loc[
            future_mask,
            "predicted_demand_lower",
        ]
        + result.loc[
            future_mask,
            "demand_realtime_correction_mwh",
        ],
        0,
    )

    result.loc[
        future_mask,
        "predicted_demand_upper",
    ] = np.maximum(
        result.loc[
            future_mask,
            "predicted_demand_upper",
        ]
        + result.loc[
            future_mask,
            "demand_realtime_correction_mwh",
        ],
        0,
    )

    # ---------------------------------------------------------
    # 보정된 재생에너지 + 수요로 Green Score 재계산
    # ---------------------------------------------------------

    score_history = history.copy()
    score_reference_end = "timestamp_not_available"

    if "timestamp" in score_history.columns:
        score_history["timestamp"] = pd.to_datetime(
            score_history["timestamp"]
        )

        score_history = score_history[
            score_history["timestamp"] <= as_of
        ]

        if score_history.empty:
            raise ValueError(
                "현재 시각 이전의 Green Score 기준 데이터가 없습니다."
            )

        score_reference_end = (
            score_history["timestamp"]
            .max()
            .isoformat()
        )

    _recalculate_scores(result, score_history)

    result["source_mode"] = (
        "historical_realtime_adjustment_replay"
    )

    result["observation_as_of"] = as_of

    metadata: dict[str, float | int | str] = {
        "as_of": as_of.isoformat(),
        "observed_hours": int(observed_mask.sum()),
        "future_hours": int(future_mask.sum()),
        "lookback_hours": int(lookback_hours),
        "decay_hours": float(decay_hours),
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
    """공식 제주 실측으로 오늘의 남은 예측을 보정한다.

    시간별 재생에너지 실측은 항상 사용한다.

    hourly_observations에 actual_demand_mwh가 존재하면
    전력수요 실측도 함께 전달해 남은 수요 예측을 보정한다.
    """

    required = {
        "timestamp",
        "actual_renewable_mwh",
        "coverage_ratio",
    }

    missing = required - set(hourly_observations.columns)

    if missing:
        raise ValueError(
            f"공식 실측 보정에 필요한 열이 없습니다: "
            f"{sorted(missing)}"
        )

    if not 0 < minimum_coverage <= 1:
        raise ValueError(
            "실측 완성도 기준은 0보다 크고 1 이하여야 합니다."
        )

    observations = hourly_observations.copy()

    observations["timestamp"] = pd.to_datetime(
        observations["timestamp"]
    )

    # ---------------------------------------------------------
    # 기존 재생에너지 실측은 항상 사용.
    # 수요 실측 열이 있으면 함께 사용.
    # ---------------------------------------------------------

    observation_columns = [
        "timestamp",
        "actual_renewable_mwh",
    ]

    if "actual_demand_mwh" in observations.columns:
        observation_columns.append(
            "actual_demand_mwh"
        )

    observations = observations[
        (
            observations["timestamp"]
            <= pd.Timestamp(as_of)
        )
        & (
            observations["coverage_ratio"]
            >= minimum_coverage
        )
    ][observation_columns]

    if observations.empty:
        raise ValueError(
            "보정에 사용할 완전한 실측시간이 없습니다."
        )

    prepared = forecast.copy()

    prepared["timestamp"] = pd.to_datetime(
        prepared["timestamp"]
    )

    # 기존 실제값 제거 후
    # 현재까지 도착한 공식 실측만 다시 merge
    prepared = prepared.drop(
        columns=[
            "actual_renewable_mwh",
            "actual_demand_mwh",
            "actual_green_score",
        ],
        errors="ignore",
    )

    prepared = prepared.merge(
        observations,
        on="timestamp",
        how="left",
    )

    adjusted, metadata = (
        adjust_forecast_with_observations(
            prepared,
            history,
            as_of=as_of,
            lookback_hours=lookback_hours,
            decay_hours=decay_hours,
        )
    )

    adjusted["source_mode"] = (
        "official_jeju_grid_live_adjustment"
    )

    metadata["renewable_observed_hours"] = int(
        observations["actual_renewable_mwh"]
        .notna()
        .sum()
    )

    if "actual_demand_mwh" in observations.columns:
        metadata["demand_observed_hours"] = int(
            observations["actual_demand_mwh"]
            .notna()
            .sum()
        )
    else:
        metadata["demand_observed_hours"] = 0

    metadata["api_actual_source"] = (
        "KPX JejuSukub5mToday"
    )

    return adjusted, metadata


def five_minute_mw_to_hourly_mwh(
    samples: pd.DataFrame,
    power_columns: tuple[str, ...] = (
        "solar_mw",
        "wind_mw",
    ),
) -> pd.DataFrame:
    """5분 MW 실측을 현재 모델과 같은 시간별 MWh로 바꾼다.

    5분 동안 60MW가 유지됐다면 에너지는
    60 × (5 / 60) = 5MWh이다.

    한 시간의 12개 값을 합치면 그 시간의 MWh가 된다.
    누락 여부를 알 수 있도록 coverage_ratio도 함께 반환한다.
    """

    required = {
        "timestamp",
        *power_columns,
    }

    missing = required - set(samples.columns)

    if missing:
        raise ValueError(
            f"5분 실측 변환에 필요한 열이 없습니다: "
            f"{sorted(missing)}"
        )

    data = samples.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    if data["timestamp"].duplicated().any():
        raise ValueError(
            "같은 시각의 5분 실측값이 중복되어 있습니다."
        )

    data["hour"] = data["timestamp"].dt.floor("h")

    for column in power_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data[list(power_columns)].isna().any().any():
        raise ValueError(
            "5분 실측값에 숫자가 아닌 값이나 빈칸이 있습니다."
        )

    result = data.groupby(
        "hour",
        as_index=False,
    )[list(power_columns)].sum()

    result = result.rename(
        columns={"hour": "timestamp"}
    )

    for column in power_columns:
        result[
            column.replace("_mw", "_mwh")
        ] = (
            result.pop(column)
            * (5 / 60)
        )

    counts = (
        data.groupby("hour")
        .size()
        .reindex(result["timestamp"])
        .to_numpy()
    )

    result["coverage_ratio"] = np.minimum(
        counts / 12,
        1.0,
    )

    mwh_columns = [
        column.replace("_mw", "_mwh")
        for column in power_columns
    ]

    result["renewable_mwh"] = result[
        mwh_columns
    ].sum(axis=1)

    return result