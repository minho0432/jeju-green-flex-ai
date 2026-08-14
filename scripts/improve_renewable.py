"""재생에너지 예측 개선: 특성 보강 · 하이브리드 α · 시간순 검증 · Green Time 지표."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_utils import (
    DEFAULT_RENEWABLE_AI_ALPHA,
    add_lag_features,
    hybrid_blend,
    make_features,
    make_live_features,
    month_hour_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "train.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "renewable_mwh"
FOLD_HOURS = 30 * 24
N_FOLDS = 4
ALPHAS = [0.35, 0.5, 0.65, 0.75, 0.85, 1.0]
GREEN_TOP_PCT = 0.30


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0, 4),
    }


def _green_time_overlap(y_true: pd.Series, y_pred: pd.Series, top_pct: float = GREEN_TOP_PCT) -> float:
    n = len(y_true)
    k = max(1, int(n * top_pct))
    true_top = set(y_true.nlargest(k).index)
    pred_top = set(pd.Series(y_pred, index=y_true.index).nlargest(k).index)
    return len(true_top & pred_top) / k


def _make_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )


def run() -> dict:
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    raw = raw.dropna(subset=[TARGET, "shortwave_radiation", "wind_speed_10m"]).reset_index(drop=True)
    framed = add_lag_features(raw)
    framed = framed.dropna(subset=["renewable_lag_24h", "renewable_lag_168h"]).reset_index(drop=True)

    n = len(framed)
    test_block = min(FOLD_HOURS * N_FOLDS, n // 3)
    fold_size = test_block // N_FOLDS
    results_ai, results_base = [], []
    results_hybrid = {a: [] for a in ALPHAS}
    green_ai, green_base = [], []
    green_hybrid = {a: [] for a in ALPHAS}
    fold_rows = []

    for fold in range(N_FOLDS):
        test_end = n - (N_FOLDS - fold - 1) * fold_size
        test_start = test_end - fold_size
        train = framed.iloc[:test_start]
        test = framed.iloc[test_start:test_end]
        if len(train) < 24 * 14 or len(test) < 24:
            continue

        model = _make_lgbm()
        model.fit(make_features(train), train[TARGET])
        ai_pred = model.predict(make_features(test))
        base_pred = month_hour_baseline(train, TARGET, test)
        y_test = test[TARGET]

        results_ai.append(_metrics(y_test, ai_pred))
        results_base.append(_metrics(y_test, base_pred))
        green_ai.append(_green_time_overlap(y_test, pd.Series(ai_pred, index=y_test.index)))
        green_base.append(_green_time_overlap(y_test, base_pred))

        for a in ALPHAS:
            hyb = hybrid_blend(ai_pred, base_pred, a)
            results_hybrid[a].append(_metrics(y_test, hyb))
            green_hybrid[a].append(_green_time_overlap(y_test, pd.Series(hyb, index=y_test.index)))

        fold_rows.append({
            "fold": fold + 1,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_start": str(test["timestamp"].iloc[0]),
            "test_end": str(test["timestamp"].iloc[-1]),
            "ai_mae": results_ai[-1]["mae"],
            "baseline_mae": results_base[-1]["mae"],
        })

    def avg_metric(lst, key="mae"):
        return round(float(np.mean([d[key] for d in lst])), 4) if lst else None

    ai_mae = avg_metric(results_ai)
    base_mae = avg_metric(results_base)
    best_alpha = DEFAULT_RENEWABLE_AI_ALPHA
    best_hybrid_mae = 1e18
    hybrid_summary = {}
    for a in ALPHAS:
        m = avg_metric(results_hybrid[a])
        g = round(float(np.mean(green_hybrid[a])), 4) if green_hybrid[a] else None
        hybrid_summary[str(a)] = {
            "mae": m,
            "rmse": avg_metric(results_hybrid[a], "rmse"),
            "r2": avg_metric(results_hybrid[a], "r2"),
            "green_time_overlap": g,
        }
        if m is not None and m < best_hybrid_mae:
            best_hybrid_mae = m
            best_alpha = a

    live_maes = []
    for fold in range(N_FOLDS):
        test_end = n - (N_FOLDS - fold - 1) * fold_size
        test_start = test_end - fold_size
        train = framed.iloc[:test_start]
        test = framed.iloc[test_start:test_end]
        if len(train) < 24 * 14 or len(test) < 24:
            continue
        model = _make_lgbm()
        model.fit(make_live_features(train), train[TARGET])
        pred = model.predict(make_live_features(test))
        live_maes.append(mean_absolute_error(test[TARGET], pred))

    improvement = None
    if base_mae and best_hybrid_mae < 1e17:
        improvement = round((1 - best_hybrid_mae / base_mae) * 100, 2)

    report = {
        "target": TARGET,
        "folds": fold_rows,
        "baseline_month_hour": {
            "mae": base_mae,
            "green_time_overlap": round(float(np.mean(green_base)), 4) if green_base else None,
        },
        "ai_full_features": {
            "mae": ai_mae,
            "rmse": avg_metric(results_ai, "rmse"),
            "r2": avg_metric(results_ai, "r2"),
            "green_time_overlap": round(float(np.mean(green_ai)), 4) if green_ai else None,
        },
        "hybrid_by_alpha": hybrid_summary,
        "best_hybrid_alpha": best_alpha,
        "best_hybrid_mae": best_hybrid_mae if best_hybrid_mae < 1e17 else None,
        "mae_improvement_vs_baseline_percent": improvement,
        "forecast_only_features_mae": round(float(np.mean(live_maes)), 4) if live_maes else None,
        "notes": [
            "시간순 검증",
            "Green Time overlap = 실제 상위 30%와 예측 상위 30% 교집합 비율",
            "하이브리드 = alpha*AI + (1-alpha)*월시간 기준표",
            "100% 정확도는 기상 불확실성으로 불가능",
        ],
    }

    final_model = _make_lgbm()
    final_model.fit(make_features(framed), framed[TARGET])
    live_model = _make_lgbm()
    live_model.fit(make_live_features(framed), framed[TARGET])
    try:
        import joblib
        model_dir = ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(final_model, model_dir / "renewable_full.joblib")
        joblib.dump(live_model, model_dir / "renewable_live.joblib")
        report["models_saved"] = ["models/renewable_full.joblib", "models/renewable_live.joblib"]
    except Exception as e:
        report["models_saved"] = []
        report["model_save_error"] = str(e)

    (OUT_DIR / "improved_renewable_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "hybrid_alpha.json").write_text(
        json.dumps({"renewable_ai_alpha": best_alpha, "source": "improve_renewable.py"}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
