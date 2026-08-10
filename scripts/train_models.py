"""제주 SMP와 재생에너지 발전량 예측 모델을 학습하고 발표용 예측을 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_utils import FEATURE_COLUMNS, make_features, percentile_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DEMO_PATH = ROOT / "data" / "demo" / "demo_forecast.csv"
DEMO_PREDICTION_PATH = OUTPUT_DIR / "demo_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
TARGETS = {
    "smp": "predicted_smp",
    "renewable_mwh": "predicted_renewable_mwh",
}


def baseline_predictions(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> np.ndarray:
    """AI와 비교할 단순 기준: 같은 월·같은 시각의 중앙값."""
    lookup_df = train_df.copy()
    lookup_df["month"] = lookup_df["timestamp"].dt.month
    lookup_df["hour"] = lookup_df["timestamp"].dt.hour
    lookup = lookup_df.groupby(["month", "hour"])[target].median()
    fallback = float(train_df[target].median())
    return np.array(
        [
            lookup.get((timestamp.month, timestamp.hour), fallback)
            for timestamp in test_df["timestamp"]
        ]
    )


def metric_dict(actual: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "rmse": round(float(mean_squared_error(actual, prediction) ** 0.5), 4),
        "r2": round(float(r2_score(actual, prediction)), 4),
    }


def train() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    if df.isna().any().any():
        raise ValueError("학습 데이터에 빈칸이 있습니다. validate_data.py를 먼저 실행하세요.")

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    train_x = make_features(train_df)
    test_x = make_features(test_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_end": str(train_df["timestamp"].max()),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
        "note": "마지막 20% 기간을 학습에서 제외한 시간순 검증 결과",
        "targets": {},
    }
    models = {}

    for target, output_column in TARGETS.items():
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(train_x, train_df[target])
        prediction = model.predict(test_x)
        baseline = baseline_predictions(train_df, test_df, target)
        target_metrics = {
            "ai": metric_dict(test_df[target], prediction),
            "baseline": metric_dict(test_df[target], baseline),
        }
        target_metrics["mae_improvement_percent"] = round(
            100
            * (
                target_metrics["baseline"]["mae"] - target_metrics["ai"]["mae"]
            )
            / target_metrics["baseline"]["mae"],
            2,
        )
        metrics["targets"][target] = target_metrics
        models[target] = model
        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLUMNS,
                "target": target,
                "trained_until": str(train_df["timestamp"].max()),
            },
            MODEL_DIR / f"{target}_model.joblib",
        )

    # 학습에 사용하지 않은 2025-12-10을 미래 하루처럼 재현한다.
    demo_date = pd.Timestamp("2025-12-10").date()
    demo = test_df[test_df["timestamp"].dt.date == demo_date].copy()
    if len(demo) != 24:
        fallback_date = test_df["timestamp"].dt.date.iloc[-24]
        demo = test_df[test_df["timestamp"].dt.date == fallback_date].copy()
    demo_x = make_features(demo)
    result = demo[["timestamp", "smp", "renewable_mwh", *FEATURE_COLUMNS[-4:]]].copy()
    result = result.rename(
        columns={"smp": "actual_smp", "renewable_mwh": "actual_renewable_mwh"}
    )
    for target, output_column in TARGETS.items():
        prediction = models[target].predict(demo_x)
        if target == "renewable_mwh":
            prediction = np.maximum(prediction, 0)
        result[output_column] = prediction

    result["price_opportunity_score"] = percentile_score(
        result["predicted_smp"], higher_is_better=False
    ).round(1)
    result["renewable_opportunity_score"] = percentile_score(
        result["predicted_renewable_mwh"], higher_is_better=True
    ).round(1)
    result["green_score"] = (
        0.45 * result["price_opportunity_score"]
        + 0.55 * result["renewable_opportunity_score"]
    ).round(1)
    result.to_csv(DEMO_PREDICTION_PATH, index=False, encoding="utf-8-sig")
    demo.to_csv(DEMO_PATH, index=False, encoding="utf-8-sig")
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, metrics


if __name__ == "__main__":
    predictions, evaluation = train()
    print("\n=== AI 모델 학습 완료 ===")
    for target, values in evaluation["targets"].items():
        print(f"\n[{target}]")
        print(f"AI MAE: {values['ai']['mae']}")
        print(f"단순 기준 MAE: {values['baseline']['mae']}")
        print(f"MAE 개선율: {values['mae_improvement_percent']}%")
    print(f"\n발표용 24시간 예측: {DEMO_PREDICTION_PATH}")
    print(f"평가 결과: {METRICS_PATH}")
