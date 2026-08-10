"""제주 SMP·재생에너지 예측 모델을 공정하게 검증하고 데모 예측을 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from model_utils import (
    FEATURE_COLUMNS,
    LAG_COLUMNS,
    WEATHER_COLUMNS,
    add_lag_features,
    make_features,
    score_against_history,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DEMO_PATH = ROOT / "data" / "demo" / "demo_forecast.csv"
DEMO_PREDICTION_PATH = OUTPUT_DIR / "demo_predictions.csv"
BACKTEST_PATH = OUTPUT_DIR / "backtest_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
TARGETS = {
    "smp": "predicted_smp",
    "renewable_mwh": "predicted_renewable_mwh",
}
N_SPLITS = 5
TEST_DAYS_PER_SPLIT = 30
SMP_CORRECTION_WEIGHT = 0.75


def build_model(target: str):
    if target == "smp":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        )
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )


def average_lag_baseline(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "smp":
        return (frame["smp_lag_24h"] + frame["smp_lag_168h"]) / 2
    return (
        frame["renewable_lag_24h"] + frame["renewable_lag_168h"]
    ) / 2


def fit_model(model, train_x, train_frame: pd.DataFrame, target: str):
    """SMP는 강한 과거 기준의 오차만 AI가 보정하고, 재생에너지는 직접 예측한다."""
    if target == "smp":
        baseline = average_lag_baseline(train_frame, target)
        model.fit(train_x, train_frame[target] - baseline)
    else:
        model.fit(train_x, train_frame[target])
    return model


def predict_model(model, test_x, test_frame: pd.DataFrame, target: str) -> np.ndarray:
    prediction = model.predict(test_x)
    if target == "smp":
        baseline = average_lag_baseline(test_frame, target).to_numpy()
        prediction = baseline + SMP_CORRECTION_WEIGHT * prediction
    else:
        prediction = np.maximum(prediction, 0)
    return prediction


def metric_dict(actual: pd.Series, prediction: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "rmse": round(float(mean_squared_error(actual, prediction) ** 0.5), 4),
        "r2": round(float(r2_score(actual, prediction)), 4),
    }


def baseline_columns(frame: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    """미래를 보지 않고 만들 수 있는 현실적인 단순 예측 세 가지."""
    if target == "smp":
        lag24 = frame["smp_lag_24h"]
        lag168 = frame["smp_lag_168h"]
    else:
        lag24 = frame["renewable_lag_24h"]
        lag168 = frame["renewable_lag_168h"]
    return {
        "24_hours_ago": lag24,
        "168_hours_ago": lag168,
        "average_24_168": (lag24 + lag168) / 2,
    }


def train() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    frame = (
        add_lag_features(raw)
        .dropna(subset=[*WEATHER_COLUMNS, *LAG_COLUMNS])
        .reset_index(drop=True)
    )
    if frame.isna().any().any():
        raise ValueError("학습 데이터에 빈칸이 있습니다. validate_data.py를 먼저 실행하세요.")

    features = make_features(frame)
    test_size = 24 * TEST_DAYS_PER_SPLIT
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, test_size=test_size)
    backtest_parts: list[pd.DataFrame] = []
    fold_boundaries: list[dict[str, object]] = []

    for fold_number, (train_index, test_index) in enumerate(splitter.split(features), start=1):
        train_frame = frame.iloc[train_index]
        test_frame = frame.iloc[test_index]
        fold_result = test_frame[["timestamp", "smp", "renewable_mwh"]].copy()
        fold_result["fold"] = fold_number

        for target, output_column in TARGETS.items():
            model = build_model(target)
            fit_model(model, features.iloc[train_index], train_frame, target)
            prediction = predict_model(
                model, features.iloc[test_index], test_frame, target
            )
            fold_result[output_column] = prediction
            for name, values in baseline_columns(test_frame, target).items():
                fold_result[f"{target}_baseline_{name}"] = values.to_numpy()

        fold_boundaries.append(
            {
                "fold": fold_number,
                "train_start": str(train_frame["timestamp"].min()),
                "train_end": str(train_frame["timestamp"].max()),
                "test_start": str(test_frame["timestamp"].min()),
                "test_end": str(test_frame["timestamp"].max()),
                "test_rows": len(test_frame),
            }
        )
        backtest_parts.append(fold_result)

    backtest = pd.concat(backtest_parts, ignore_index=True).sort_values("timestamp")
    metrics: dict[str, object] = {
        "data_rows_after_lag": len(frame),
        "validation": {
            "method": "5-fold rolling time-series backtest",
            "test_days_per_fold": TEST_DAYS_PER_SPLIT,
            "total_test_rows": len(backtest),
            "folds": fold_boundaries,
        },
        "note": "24시간 전·168시간 전 값만 사용하며 미래값은 입력하지 않음",
        "targets": {},
    }

    last_fold_number = int(backtest["fold"].max())
    calibration = backtest[backtest["fold"] < last_fold_number]
    last_fold = backtest[backtest["fold"] == last_fold_number]

    for target, output_column in TARGETS.items():
        target_metrics: dict[str, object] = {
            "ai": metric_dict(backtest[target], backtest[output_column]),
            "baselines": {},
        }
        for name in ("24_hours_ago", "168_hours_ago", "average_24_168"):
            column = f"{target}_baseline_{name}"
            target_metrics["baselines"][name] = metric_dict(
                backtest[target], backtest[column]
            )
        best_name = min(
            target_metrics["baselines"],
            key=lambda name: target_metrics["baselines"][name]["mae"],
        )
        best_mae = target_metrics["baselines"][best_name]["mae"]
        ai_mae = target_metrics["ai"]["mae"]
        target_metrics["best_baseline"] = best_name
        target_metrics["mae_improvement_percent"] = round(
            100 * (best_mae - ai_mae) / best_mae, 2
        )

        calibration_error = np.abs(
            calibration[target] - calibration[output_column]
        )
        interval_half_width = float(np.quantile(calibration_error, 0.90))
        lower = last_fold[output_column] - interval_half_width
        upper = last_fold[output_column] + interval_half_width
        coverage = float(
            ((last_fold[target] >= lower) & (last_fold[target] <= upper)).mean()
        )
        target_metrics["approx_90_interval_half_width"] = round(interval_half_width, 4)
        target_metrics["last_fold_interval_coverage"] = round(coverage, 4)
        metrics["targets"][target] = target_metrics

    # 2025-12-10은 마지막 시험 구간에 속하므로 모델이 해당 정답을 학습하지 않았다.
    demo_date = pd.Timestamp("2025-12-10").date()
    demo_prediction = backtest[
        backtest["timestamp"].dt.date == demo_date
    ].copy()
    demo_source = frame[frame["timestamp"].dt.date == demo_date].copy()
    if len(demo_prediction) != 24:
        raise ValueError("발표용 2025-12-10 예측 24개를 만들지 못했습니다.")

    weather_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]
    result = demo_prediction[
        ["timestamp", "smp", "renewable_mwh", "predicted_smp", "predicted_renewable_mwh"]
    ].rename(
        columns={"smp": "actual_smp", "renewable_mwh": "actual_renewable_mwh"}
    )
    result = result.merge(demo_source[["timestamp", *weather_columns]], on="timestamp")

    smp_interval = metrics["targets"]["smp"]["approx_90_interval_half_width"]
    renewable_interval = metrics["targets"]["renewable_mwh"][
        "approx_90_interval_half_width"
    ]
    result["predicted_smp_lower"] = result["predicted_smp"] - smp_interval
    result["predicted_smp_upper"] = result["predicted_smp"] + smp_interval
    result["predicted_renewable_lower"] = np.maximum(
        result["predicted_renewable_mwh"] - renewable_interval, 0
    )
    result["predicted_renewable_upper"] = (
        result["predicted_renewable_mwh"] + renewable_interval
    )

    # 하루 안의 상대평가가 아니라 2025년 과거 분포를 기준으로 점수를 계산한다.
    history_for_score = frame[frame["timestamp"] < pd.Timestamp("2025-12-10")]
    result["price_opportunity_score"] = score_against_history(
        result["predicted_smp"], history_for_score["smp"], higher_is_better=False
    ).round(1)
    result["renewable_opportunity_score"] = score_against_history(
        result["predicted_renewable_mwh"],
        history_for_score["renewable_mwh"],
        higher_is_better=True,
    ).round(1)
    result["green_score"] = (
        0.45 * result["price_opportunity_score"]
        + 0.55 * result["renewable_opportunity_score"]
    ).round(1)

    # 예보가 틀려도 무리한 추천을 하지 않도록 불리한 경우를 기준으로 한 보수적 점수.
    # 가격은 예상 상한(더 비싼 경우), 재생에너지는 예상 하한(더 적은 경우)을 사용한다.
    result["conservative_price_score"] = score_against_history(
        result["predicted_smp_upper"],
        history_for_score["smp"],
        higher_is_better=False,
    ).round(1)
    result["conservative_renewable_score"] = score_against_history(
        result["predicted_renewable_lower"],
        history_for_score["renewable_mwh"],
        higher_is_better=True,
    ).round(1)
    result["planning_score"] = (
        0.45 * result["conservative_price_score"]
        + 0.55 * result["conservative_renewable_score"]
    ).round(1)
    result["forecast_risk_points"] = (
        result["green_score"] - result["planning_score"]
    ).clip(lower=0).round(1)

    # 과거 재현 데모에서만 알 수 있는 실제 결과 점수다. 미래 서비스에서는
    # 충전이 끝난 뒤 관측값이 들어왔을 때 성과형 보너스를 정산하는 데 사용한다.
    result["actual_price_score"] = score_against_history(
        result["actual_smp"], history_for_score["smp"], higher_is_better=False
    ).round(1)
    result["actual_renewable_score"] = score_against_history(
        result["actual_renewable_mwh"],
        history_for_score["renewable_mwh"],
        higher_is_better=True,
    ).round(1)
    result["actual_green_score"] = (
        0.45 * result["actual_price_score"]
        + 0.55 * result["actual_renewable_score"]
    ).round(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        final_model = build_model(target)
        fit_model(final_model, features, frame, target)
        joblib.dump(
            {
                "model": final_model,
                "features": FEATURE_COLUMNS,
                "target": target,
                "trained_until": str(frame["timestamp"].max()),
                "validation": metrics["validation"]["method"],
                "approach": (
                    "average-lag baseline + AI residual correction"
                    if target == "smp"
                    else "direct prediction"
                ),
                "smp_correction_weight": (
                    SMP_CORRECTION_WEIGHT if target == "smp" else None
                ),
            },
            MODEL_DIR / f"{target}_model.joblib",
        )

    result.to_csv(DEMO_PREDICTION_PATH, index=False, encoding="utf-8-sig")
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    demo_source.to_csv(DEMO_PATH, index=False, encoding="utf-8-sig")
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, metrics


if __name__ == "__main__":
    predictions, evaluation = train()
    print("\n=== 강화된 AI 모델 학습 완료 ===")
    print("검증: 서로 다른 5개 기간 × 각 30일")
    for target, values in evaluation["targets"].items():
        best = values["best_baseline"]
        print(f"\n[{target}]")
        print(f"AI MAE: {values['ai']['mae']}")
        print(f"가장 강한 단순 기준: {best}")
        print(f"단순 기준 MAE: {values['baselines'][best]['mae']}")
        print(f"공정한 개선율: {values['mae_improvement_percent']}%")
        print(
            "약 90% 예측범위 반폭: "
            f"±{values['approx_90_interval_half_width']}"
        )
    print(f"\n발표용 24시간 예측: {DEMO_PREDICTION_PATH}")
    print(f"전체 백테스트 예측: {BACKTEST_PATH}")
    print(f"평가 결과: {METRICS_PATH}")
