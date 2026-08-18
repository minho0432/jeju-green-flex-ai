"""다년도 공식 데이터로 후보 모델을 시간순 비교·검증하고 최종 모델을 저장합니다."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from model_builders import MODEL_CANDIDATES, build_model
from model_utils import (
    FEATURE_COLUMNS, LIVE_FEATURE_COLUMNS, WEATHER_COLUMNS, add_lag_features,
    attach_supply_margin_scores, make_features, make_live_features,
    month_hour_baseline, score_against_history, supply_margin,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DEMO_PATH = ROOT / "data" / "demo" / "demo_forecast.csv"
DEMO_PREDICTION_PATH = OUTPUT_DIR / "demo_predictions.csv"
BACKTEST_PATH = OUTPUT_DIR / "backtest_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
PRIMARY_TARGETS = ("renewable_mwh", "demand_mwh")
ALL_TARGETS = PRIMARY_TARGETS
N_SPLITS = 5
TEST_HOURS = 24 * 30
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
    overlap = set(np.argsort(actual)[-count:]) & set(np.argsort(prediction)[-count:])
    return round(len(overlap) / count, 4)


def _prepare_frame(raw: pd.DataFrame, target: str, live: bool):
    frame = raw.dropna(subset=[target, *WEATHER_COLUMNS]).copy()
    if live:
        x = make_live_features(frame)
        required = LIVE_FEATURE_COLUMNS
    else:
        frame = add_lag_features(frame)
        x = make_features(frame)
        required = FEATURE_COLUMNS
    valid = x[required].notna().all(axis=1) & frame[target].notna()
    return frame.loc[valid].reset_index(drop=True), x.loc[valid].reset_index(drop=True)


def _evaluate_mode(raw: pd.DataFrame, target: str, live: bool):
    """앞 4개 구간으로 모델을 선택하고 마지막 30일은 성능 확인에만 씁니다."""
    frame, x = _prepare_frame(raw, target, live)
    splits = list(TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_HOURS).split(frame))
    candidates = MODEL_CANDIDATES
    candidate_parts = {name: [] for name in candidates}

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_frame, test_frame = frame.iloc[train_idx], frame.iloc[test_idx]
        baseline = month_hour_baseline(train_frame, target, test_frame).to_numpy()
        for candidate in candidates:
            model = build_model(target, candidate)
            model.fit(x.iloc[train_idx], train_frame[target])
            prediction = model.predict(x.iloc[test_idx])
            prediction = np.maximum(prediction, 0)
            candidate_parts[candidate].append(pd.DataFrame({
                "timestamp": test_frame["timestamp"].to_numpy(),
                "actual": test_frame[target].to_numpy(),
                "prediction": prediction, "baseline": baseline, "fold": fold,
            }))

    all_results = {name: pd.concat(parts, ignore_index=True) for name, parts in candidate_parts.items()}
    leaderboard = []
    for candidate, result in all_results.items():
        selection = result[result["fold"] < N_SPLITS]
        leaderboard.append({
            "model": candidate,
            "selection_mae": round(float(mean_absolute_error(selection["actual"], selection["prediction"])), 4),
            "selection_rows": int(len(selection)),
        })
    leaderboard.sort(key=lambda row: row["selection_mae"])
    winner = str(leaderboard[0]["model"])
    selected = all_results[winner]
    holdout = selected[selected["fold"] == N_SPLITS].copy()
    baseline_metrics = metric_dict(holdout["actual"], holdout["baseline"])
    ai_metrics = metric_dict(holdout["actual"], holdout["prediction"])
    calibration = selected[selected["fold"] < N_SPLITS]
    half_width = float(np.quantile(np.abs(calibration["actual"] - calibration["prediction"]), 0.90))
    coverage = float(((holdout["actual"] >= holdout["prediction"] - half_width) &
                      (holdout["actual"] <= holdout["prediction"] + half_width)).mean())
    details = {
        "selected_model": winner,
        "selection_method": "앞 4개 시간 구간 MAE 최소 모델 선택, 마지막 30일 별도 검증",
        "candidate_leaderboard": leaderboard,
        "ai": ai_metrics, "baseline_month_hour": baseline_metrics,
        "mae_improvement_percent": round(
            100 * (baseline_metrics["mae"] - ai_metrics["mae"]) / baseline_metrics["mae"], 2),
        "approx_90_interval_half_width": round(half_width, 4),
        "last_fold_interval_coverage": round(coverage, 4),
        "holdout_rows": int(len(holdout)), "training_rows_available": int(len(frame)),
    }
    return selected, details, winner


def _select_green_blend(renewable: pd.DataFrame, demand: pd.DataFrame) -> dict[str, float]:
    """마지막 홀드아웃을 보지 않고 Green Time 상위 30% 일치율이 높은 혼합비를 고릅니다."""
    paired = renewable.merge(demand, on=["timestamp", "fold"], suffixes=("_r", "_d"))
    selection = paired[paired["fold"] < N_SPLITS]
    actual = supply_margin(selection["actual_r"], selection["actual_d"])
    choices = []
    for renewable_alpha in np.linspace(0, 1, 11):
        for demand_alpha in np.linspace(0, 1, 11):
            predicted = supply_margin(
                renewable_alpha * selection["prediction_r"] + (1 - renewable_alpha) * selection["baseline_r"],
                demand_alpha * selection["prediction_d"] + (1 - demand_alpha) * selection["baseline_d"],
            )
            choices.append((
                green_time_overlap(actual, predicted),
                float(mean_absolute_error(actual, predicted)),
                float(renewable_alpha), float(demand_alpha),
            ))
    # 일치율 최대, 같은 경우 공급비율 오차 최소
    best = sorted(choices, key=lambda row: (-row[0], row[1]))[0]
    return {
        "renewable_ai_alpha": best[2], "demand_ai_alpha": best[3],
        "selection_top_30_overlap": best[0], "selection_supply_margin_mae": round(best[1], 6),
    }


def train():
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    required = {"demand_mwh", "renewable_mwh", *WEATHER_COLUMNS}
    missing = required - set(raw.columns)
    if missing or raw[list(required)].isna().any().any():
        raise ValueError(f"핵심 학습 데이터 누락 또는 결측: {sorted(missing)}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "data_rows_total": int(len(raw)), "additional_rows_vs_2025": int(len(raw) - 8760),
        "data_period": {"start": str(raw["timestamp"].min()), "end": str(raw["timestamp"].max())},
        "official_demand_rows": int(raw["demand_mwh"].notna().sum()),
        "demand_source": "KPX 제주 전력수급현황 계통수요(단위 정규화 완료)",
        "validation": {"method": "rolling time-series; folds 1-4 model selection, final 30-day holdout", "splits": N_SPLITS, "test_hours_per_split": TEST_HOURS},
        "score_definition": "Green Score는 예측 재생에너지/예측 수요의 과거 백분위",
        "targets": {}, "forecast_only_targets": {},
    }
    backtests = {"full": {}, "live": {}}
    for target in ALL_TARGETS:
        for mode, live in (("full", False), ("live", True)):
            evaluated, details, winner = _evaluate_mode(raw, target, live)
            backtests[mode][target] = evaluated
            bucket = "forecast_only_targets" if live else "targets"
            if live:
                details["note"] = "시간·기상예보형 입력 검증이며 실제 날씨예보 오차는 별도"
            metrics[bucket][target] = details
            frame, x = _prepare_frame(raw, target, live)
            final_model = build_model(target, winner)
            final_model.fit(x, frame[target])
            joblib.dump(final_model, MODEL_DIR / f"{target.replace('_mwh', '')}_{mode}.joblib")

    blend = _select_green_blend(
        backtests["live"]["renewable_mwh"], backtests["live"]["demand_mwh"]
    )
    for target, alpha_key in (("renewable_mwh", "renewable_ai_alpha"), ("demand_mwh", "demand_ai_alpha")):
        alpha = blend[alpha_key]
        selected = backtests["live"][target]
        selected["prediction"] = alpha * selected["prediction"] + (1 - alpha) * selected["baseline"]
        holdout = selected[selected["fold"] == N_SPLITS]
        calibration = selected[selected["fold"] < N_SPLITS]
        ai_metrics = metric_dict(holdout["actual"], holdout["prediction"])
        baseline_metrics = metric_dict(holdout["actual"], holdout["baseline"])
        half_width = float(np.quantile(np.abs(calibration["actual"] - calibration["prediction"]), 0.90))
        coverage = float(((holdout["actual"] >= holdout["prediction"] - half_width) &
                          (holdout["actual"] <= holdout["prediction"] + half_width)).mean())
        detail = metrics["forecast_only_targets"][target]
        detail["blend_alpha_ai"] = alpha
        detail["ai"] = ai_metrics
        detail["baseline_month_hour"] = baseline_metrics
        detail["mae_improvement_percent"] = round(
            100 * (baseline_metrics["mae"] - ai_metrics["mae"]) / baseline_metrics["mae"], 2
        )
        detail["approx_90_interval_half_width"] = round(half_width, 4)
        detail["last_fold_interval_coverage"] = round(coverage, 4)

    base = backtests["live"]["renewable_mwh"].query("fold == @N_SPLITS").copy()
    backtest = base.rename(columns={"actual": "renewable_mwh", "prediction": "live_renewable_mwh", "baseline": "baseline_renewable_mwh"})
    for target in ("demand_mwh",):
        part = backtests["live"][target].query("fold == @N_SPLITS").copy()
        part = part.rename(columns={"actual": target, "prediction": f"live_{target}", "baseline": f"baseline_{target}"})
        backtest = backtest.merge(part.drop(columns="fold"), on="timestamp", how="left")
    for target in ALL_TARGETS:
        part = backtests["full"][target].query("fold == @N_SPLITS")[["timestamp", "prediction"]]
        backtest = backtest.merge(part.rename(columns={"prediction": f"full_{target}"}), on="timestamp", how="left")

    actual_margin = supply_margin(backtest["renewable_mwh"], backtest["demand_mwh"])
    predicted_margin = supply_margin(backtest["live_renewable_mwh"], backtest["live_demand_mwh"])
    baseline_margin = supply_margin(backtest["baseline_renewable_mwh"], backtest["baseline_demand_mwh"])
    metrics["green_time"] = {
        "evaluation_rows": int(len(backtest)),
        "definition": "percentile(predicted renewable / predicted demand) versus historical distribution",
        "supply_margin_mae_ai": round(float(mean_absolute_error(actual_margin, predicted_margin)), 6),
        "supply_margin_mae_baseline": round(float(mean_absolute_error(actual_margin, baseline_margin)), 6),
        "top_30_percent_overlap_ai": green_time_overlap(actual_margin, predicted_margin),
        "top_30_percent_overlap_baseline": green_time_overlap(actual_margin, baseline_margin),
        "deployment_blend": blend,
    }

    demo = backtest[backtest["timestamp"].dt.normalize() == DEMO_DATE].copy()
    if len(demo) != 24:
        raise ValueError("발표용 2025-12-10 예측 24개를 만들지 못했습니다.")
    source = raw[raw["timestamp"].dt.normalize() == DEMO_DATE][["timestamp", *WEATHER_COLUMNS]]
    result = pd.DataFrame({
        "timestamp": demo["timestamp"],
        "actual_renewable_mwh": demo["renewable_mwh"], "actual_demand_mwh": demo["demand_mwh"],
        "predicted_renewable_mwh": demo["live_renewable_mwh"],
        "predicted_demand_mwh": demo["live_demand_mwh"],
    }).merge(source, on="timestamp")
    for target, center, prefix in (
        ("renewable_mwh", "predicted_renewable_mwh", "predicted_renewable"),
        ("demand_mwh", "predicted_demand_mwh", "predicted_demand"),
    ):
        half_width = metrics["forecast_only_targets"][target]["approx_90_interval_half_width"]
        result[f"{prefix}_lower"] = result[center] - half_width
        result[f"{prefix}_upper"] = result[center] + half_width
        result[f"{prefix}_lower"] = result[f"{prefix}_lower"].clip(lower=0)

    history = raw[raw["timestamp"] < DEMO_DATE]
    result = attach_supply_margin_scores(result, history)
    actual_margin_demo = supply_margin(result["actual_renewable_mwh"], result["actual_demand_mwh"])
    historical_margin = supply_margin(history["renewable_mwh"], history["demand_mwh"])
    result["actual_green_score"] = score_against_history(actual_margin_demo, historical_margin, True).round(1)
    result["actual_renewable_score"] = score_against_history(result["actual_renewable_mwh"], history["renewable_mwh"], True).round(1)

    result.to_csv(DEMO_PREDICTION_PATH, index=False, encoding="utf-8-sig")
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    raw[raw["timestamp"].dt.normalize() == DEMO_DATE].to_csv(DEMO_PATH, index=False, encoding="utf-8-sig")
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, metrics


if __name__ == "__main__":
    _, evaluation = train()
    print("다년도 모델 선택·학습 및 최종 홀드아웃 검증 완료")
    print(f"총 {evaluation['data_rows_total']:,}시간 / 추가 {evaluation['additional_rows_vs_2025']:,}시간")
    for target, values in evaluation["forecast_only_targets"].items():
        print(f"{target}: {values['selected_model']}, holdout MAE={values['ai']['mae']}, baseline={values['baseline_month_hour']['mae']}, 개선율={values['mae_improvement_percent']}%")
    print("Green Time:", evaluation["green_time"])
