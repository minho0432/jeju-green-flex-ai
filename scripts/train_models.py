"""공식 제주 수요를 포함한 예측 모델을 시간순으로 검증하고 저장합니다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from model_builders import build_model
from model_utils import (
    WEATHER_COLUMNS,
    add_lag_features,
    attach_supply_margin_scores,
    make_features,
    make_live_features,
    month_hour_baseline,
    score_against_history,
    supply_margin,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DEMO_PATH = ROOT / "data" / "demo" / "demo_forecast.csv"
DEMO_PREDICTION_PATH = OUTPUT_DIR / "demo_predictions.csv"
BACKTEST_PATH = OUTPUT_DIR / "backtest_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
TARGETS = ("smp", "renewable_mwh", "demand_mwh")
N_SPLITS = 5
TEST_DAYS_PER_SPLIT = 30
DEMO_DATE = pd.Timestamp("2025-12-10")


def metric_dict(actual, prediction) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "rmse": round(float(mean_squared_error(actual, prediction) ** 0.5), 4),
        "r2": round(float(r2_score(actual, prediction)), 4),
    }


def green_time_overlap(actual, prediction, top_fraction: float = 0.30) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    count = max(1, int(len(actual) * top_fraction))
    actual_top = set(np.argsort(actual)[-count:])
    predicted_top = set(np.argsort(prediction)[-count:])
    return round(len(actual_top & predicted_top) / count, 4)


def interval_metrics(actual, prediction, folds) -> tuple[float, float]:
    last_fold = int(np.max(folds))
    calibration = folds < last_fold
    evaluation = folds == last_fold
    half_width = float(np.quantile(np.abs(actual[calibration] - prediction[calibration]), 0.90))
    coverage = float(
        (
            (actual[evaluation] >= prediction[evaluation] - half_width)
            & (actual[evaluation] <= prediction[evaluation] + half_width)
        ).mean()
    )
    return round(half_width, 4), round(coverage, 4)


def train() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    required = {"demand_mwh", "renewable_mwh", "smp", *WEATHER_COLUMNS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"학습 데이터 필수 열 누락: {sorted(missing)}")
    if raw[list(required)].isna().any().any():
        raise ValueError("학습 데이터에 결측값이 있습니다.")

    frame = add_lag_features(raw).dropna().reset_index(drop=True)
    full_x = make_features(frame)
    live_x = make_live_features(frame)
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, test_size=24 * TEST_DAYS_PER_SPLIT)

    parts: list[pd.DataFrame] = []
    fold_info: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(frame), start=1):
        train_frame = frame.iloc[train_index]
        test_frame = frame.iloc[test_index]
        result = test_frame[["timestamp", *TARGETS]].copy()
        result["fold"] = fold
        for target in TARGETS:
            full_model = build_model(target)
            full_model.fit(full_x.iloc[train_index], train_frame[target])
            full_prediction = full_model.predict(full_x.iloc[test_index])
            live_model = build_model(target)
            live_model.fit(live_x.iloc[train_index], train_frame[target])
            live_prediction = live_model.predict(live_x.iloc[test_index])
            baseline = month_hour_baseline(train_frame, target, test_frame)
            if target != "smp":
                full_prediction = np.maximum(full_prediction, 0)
                live_prediction = np.maximum(live_prediction, 0)
            result[f"full_{target}"] = full_prediction
            result[f"live_{target}"] = live_prediction
            result[f"baseline_{target}"] = baseline.to_numpy()

        fold_info.append(
            {
                "fold": fold,
                "train_start": str(train_frame["timestamp"].min()),
                "train_end": str(train_frame["timestamp"].max()),
                "test_start": str(test_frame["timestamp"].min()),
                "test_end": str(test_frame["timestamp"].max()),
                "test_rows": int(len(test_frame)),
            }
        )
        parts.append(result)

    backtest = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    folds = backtest["fold"].to_numpy()
    metrics: dict[str, object] = {
        "data_rows_total": int(len(raw)),
        "data_rows_after_lag": int(len(frame)),
        "official_demand_rows": int(raw["demand_mwh"].notna().sum()),
        "demand_source": "KPX 제주 전력수급현황 계통수요",
        "validation": {
            "method": "5-fold rolling time-series backtest",
            "test_days_per_fold": TEST_DAYS_PER_SPLIT,
            "total_test_rows": int(len(backtest)),
            "folds": fold_info,
        },
        "score_weights": {
            "renewable_supply_margin": 1.0,
            "market_smp": 0.0,
            "note": "Green Score는 예측 재생에너지/예측 수요의 과거 백분위이며 SMP는 사용하지 않음",
        },
        "targets": {},
        "forecast_only_targets": {},
    }

    for target in TARGETS:
        actual = backtest[target].to_numpy()
        full_prediction = backtest[f"full_{target}"].to_numpy()
        live_prediction = backtest[f"live_{target}"].to_numpy()
        baseline = backtest[f"baseline_{target}"].to_numpy()
        full_hw, full_coverage = interval_metrics(actual, full_prediction, folds)
        live_hw, live_coverage = interval_metrics(actual, live_prediction, folds)
        baseline_metrics = metric_dict(actual, baseline)
        full_metrics = metric_dict(actual, full_prediction)
        live_metrics = metric_dict(actual, live_prediction)
        metrics["targets"][target] = {
            "ai": full_metrics,
            "baseline_month_hour": baseline_metrics,
            "mae_improvement_percent": round(
                100 * (baseline_metrics["mae"] - full_metrics["mae"]) / baseline_metrics["mae"], 2
            ),
            "approx_90_interval_half_width": full_hw,
            "last_fold_interval_coverage": full_coverage,
        }
        metrics["forecast_only_targets"][target] = {
            "ai": live_metrics,
            "baseline_month_hour": baseline_metrics,
            "mae_improvement_percent": round(
                100 * (baseline_metrics["mae"] - live_metrics["mae"]) / baseline_metrics["mae"], 2
            ),
            "approx_90_interval_half_width": live_hw,
            "last_fold_interval_coverage": live_coverage,
            "note": "시간·과거 관측 기상만 사용한 검증이며 실제 기상예보 오차는 별도",
        }

    actual_margin = supply_margin(backtest["renewable_mwh"], backtest["demand_mwh"])
    predicted_margin = supply_margin(backtest["live_renewable_mwh"], backtest["live_demand_mwh"])
    baseline_margin = supply_margin(backtest["baseline_renewable_mwh"], backtest["baseline_demand_mwh"])
    metrics["green_time"] = {
        "definition": "percentile(predicted renewable / predicted demand) versus historical distribution",
        "supply_margin_mae_ai": round(float(mean_absolute_error(actual_margin, predicted_margin)), 6),
        "supply_margin_mae_baseline": round(float(mean_absolute_error(actual_margin, baseline_margin)), 6),
        "top_30_percent_overlap_ai": green_time_overlap(actual_margin, predicted_margin),
        "top_30_percent_overlap_baseline": green_time_overlap(actual_margin, baseline_margin),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        name = target.replace("_mwh", "")
        full_model = build_model(target)
        full_model.fit(full_x, frame[target])
        live_model = build_model(target)
        live_model.fit(live_x, frame[target])
        joblib.dump(full_model, MODEL_DIR / f"{name}_full.joblib")
        joblib.dump(live_model, MODEL_DIR / f"{name}_live.joblib")

    demo = backtest[backtest["timestamp"].dt.normalize() == DEMO_DATE].copy()
    if len(demo) != 24:
        raise ValueError("발표용 2025-12-10 예측 24개를 만들지 못했습니다.")
    source = frame[frame["timestamp"].dt.normalize() == DEMO_DATE][["timestamp", *WEATHER_COLUMNS]]
    result = pd.DataFrame(
        {
            "timestamp": demo["timestamp"],
            "actual_smp": demo["smp"],
            "actual_renewable_mwh": demo["renewable_mwh"],
            "actual_demand_mwh": demo["demand_mwh"],
            "predicted_smp": demo["live_smp"],
            "predicted_renewable_mwh": demo["live_renewable_mwh"],
            "predicted_demand_mwh": demo["live_demand_mwh"],
        }
    ).merge(source, on="timestamp")

    interval_columns = (
        ("smp", "predicted_smp", "predicted_smp"),
        ("renewable_mwh", "predicted_renewable_mwh", "predicted_renewable"),
        ("demand_mwh", "predicted_demand_mwh", "predicted_demand"),
    )
    for target, center_column, output_prefix in interval_columns:
        half_width = metrics["forecast_only_targets"][target]["approx_90_interval_half_width"]
        result[f"{output_prefix}_lower"] = result[center_column] - half_width
        result[f"{output_prefix}_upper"] = result[center_column] + half_width
        if target != "smp":
            result[f"{output_prefix}_lower"] = result[f"{output_prefix}_lower"].clip(lower=0)

    history = raw[raw["timestamp"] < DEMO_DATE]
    result = attach_supply_margin_scores(result, history)
    actual_supply_margin = supply_margin(result["actual_renewable_mwh"], result["actual_demand_mwh"])
    historical_supply_margin = supply_margin(history["renewable_mwh"], history["demand_mwh"])
    result["actual_green_score"] = score_against_history(
        actual_supply_margin, historical_supply_margin, higher_is_better=True
    ).round(1)
    result["actual_price_score"] = score_against_history(
        result["actual_smp"], history["smp"], higher_is_better=False
    ).round(1)
    result["actual_renewable_score"] = score_against_history(
        result["actual_renewable_mwh"], history["renewable_mwh"], higher_is_better=True
    ).round(1)

    result.to_csv(DEMO_PREDICTION_PATH, index=False, encoding="utf-8-sig")
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    frame[frame["timestamp"].dt.normalize() == DEMO_DATE].to_csv(
        DEMO_PATH, index=False, encoding="utf-8-sig"
    )
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, metrics


if __name__ == "__main__":
    _, evaluation = train()
    print("모델 학습 및 시간순 검증 완료")
    for target, values in evaluation["forecast_only_targets"].items():
        print(
            f"{target}: live MAE={values['ai']['mae']}, "
            f"baseline MAE={values['baseline_month_hour']['mae']}, "
            f"개선율={values['mae_improvement_percent']}%"
        )
    print("Green Time:", evaluation["green_time"])
