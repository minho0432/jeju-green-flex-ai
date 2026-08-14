"""수요 예측 + 공급여력 Green Score 검증 파이프라인."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model_utils import (
    ensure_demand_column,
    make_live_features,
    month_hour_baseline,
    supply_margin,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "train.csv"
OUT = ROOT / "outputs"


def make_lgbm():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42
        )


def time_folds(n: int, n_folds: int = 4, test_days: int = 30):
    test_h = test_days * 24
    folds = []
    end = n
    for _ in range(n_folds):
        te_end = end
        te_start = end - test_h
        if te_start < test_h * 2:
            break
        folds.append((np.arange(0, te_start), np.arange(te_start, te_end)))
        end = te_start
    return list(reversed(folds))


def green_time_overlap(actual: np.ndarray, pred: np.ndarray, top_frac: float = 0.3) -> float:
    n = len(actual)
    k = max(1, int(n * top_frac))
    a_idx = set(np.argsort(actual)[-k:])
    p_idx = set(np.argsort(pred)[-k:])
    return len(a_idx & p_idx) / k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp")
    df = ensure_demand_column(df)
    df.to_csv(DATA, index=False)

    features = make_live_features(df)
    y_re = df["renewable_mwh"].to_numpy()
    y_dem = df["demand_mwh"].to_numpy()
    y_margin = supply_margin(df["renewable_mwh"], df["demand_mwh"]).to_numpy()

    folds = time_folds(len(df))
    rows = []
    for i, (tr, te) in enumerate(folds):
        m_d = make_lgbm()
        m_d.fit(features.iloc[tr], y_dem[tr])
        pred_d = np.maximum(m_d.predict(features.iloc[te]), 0)
        base_d = month_hour_baseline(df.iloc[tr], "demand_mwh", df.iloc[te]).to_numpy()

        m_r = make_lgbm()
        m_r.fit(features.iloc[tr], y_re[tr])
        pred_r = np.maximum(m_r.predict(features.iloc[te]), 0)
        base_r = month_hour_baseline(df.iloc[tr], "renewable_mwh", df.iloc[te]).to_numpy()

        margin_true = y_margin[te]
        margin_ai = supply_margin(pred_r, pred_d).to_numpy()
        margin_base = supply_margin(base_r, base_d).to_numpy()

        rows.append(
            {
                "fold": i,
                "demand_mae_ai": mean_absolute_error(y_dem[te], pred_d),
                "demand_mae_base": mean_absolute_error(y_dem[te], base_d),
                "renewable_mae_ai": mean_absolute_error(y_re[te], pred_r),
                "renewable_mae_base": mean_absolute_error(y_re[te], base_r),
                "margin_mae_ai": mean_absolute_error(margin_true, margin_ai),
                "margin_mae_base": mean_absolute_error(margin_true, margin_base),
                "green_time_margin_ai": green_time_overlap(margin_true, margin_ai),
                "green_time_margin_base": green_time_overlap(margin_true, margin_base),
            }
        )
        print(
            f"fold {i}: demand MAE {rows[-1]['demand_mae_ai']:.2f}, "
            f"margin GT {rows[-1]['green_time_margin_ai']:.3f}"
        )

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "fold"}
    summary["folds"] = len(rows)
    summary["demand_source"] = "proxy_generation_sum_solar_wind_lng_bio"
    summary["green_score_definition"] = "percentile(renewable/demand) vs history"

    path = OUT / "demand_supply_margin_metrics.json"
    path.write_text(
        json.dumps({"folds": rows, "mean": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved", path)


if __name__ == "__main__":
    main()
