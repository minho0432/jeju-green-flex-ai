"""실측 도착 기반 예측 보정과 MW→MWh 변환 규칙을 검사한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from realtime_adjustment import (  # noqa: E402
    adjust_forecast_with_live_renewables,
    adjust_forecast_with_observations,
    five_minute_mw_to_hourly_mwh,
)


def sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-01", periods=200, freq="h"),
            "smp": list(range(50, 250)),
            "renewable_mwh": list(range(200)),
        }
    )


def sample_forecast() -> pd.DataFrame:
    timestamps = pd.date_range("2025-12-10", periods=24, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "predicted_smp": [100.0] * 24,
            "predicted_smp_lower": [80.0] * 24,
            "predicted_smp_upper": [120.0] * 24,
            "predicted_renewable_mwh": [100.0] * 24,
            "predicted_renewable_lower": [70.0] * 24,
            "predicted_renewable_upper": [130.0] * 24,
            "actual_smp": [110.0] * 24,
            "actual_renewable_mwh": [70.0] * 24,
            "actual_green_score": [50.0] * 24,
        }
    )
    return frame


class RealtimeAdjustmentTests(unittest.TestCase):
    def test_future_actual_values_are_hidden(self):
        forecast = sample_forecast()
        adjusted, metadata = adjust_forecast_with_observations(
            forecast,
            sample_history(),
            as_of=pd.Timestamp("2025-12-10 10:00"),
        )
        future = adjusted[adjusted["timestamp"] > pd.Timestamp("2025-12-10 10:00")]
        observed = adjusted[adjusted["timestamp"] <= pd.Timestamp("2025-12-10 10:00")]
        self.assertTrue(future["observed_actual_smp"].isna().all())
        self.assertTrue(future["observed_actual_renewable_mwh"].isna().all())
        self.assertTrue(observed["observed_actual_smp"].notna().all())
        self.assertNotIn("actual_green_score", adjusted.columns)
        self.assertEqual(metadata["observed_hours"], 11)
        self.assertLessEqual(
            pd.Timestamp(metadata["score_reference_end"]),
            pd.Timestamp("2025-12-10 10:00"),
        )

    def test_recent_miss_changes_future_in_same_direction(self):
        adjusted, metadata = adjust_forecast_with_observations(
            sample_forecast(),
            sample_history(),
            as_of=pd.Timestamp("2025-12-10 10:00"),
        )
        future = adjusted[adjusted["timestamp"] > pd.Timestamp("2025-12-10 10:00")]
        self.assertLess(future.iloc[0]["predicted_renewable_mwh"], 100)
        self.assertGreater(future.iloc[0]["predicted_smp"], 100)
        self.assertEqual(metadata["recent_renewable_bias_mwh"], -30)
        self.assertEqual(metadata["recent_smp_bias"], 10)

    def test_correction_decays_for_distant_hours(self):
        adjusted, _ = adjust_forecast_with_observations(
            sample_forecast(),
            sample_history(),
            as_of=pd.Timestamp("2025-12-10 10:00"),
        )
        future = adjusted[adjusted["timestamp"] > pd.Timestamp("2025-12-10 10:00")]
        first = abs(future.iloc[0]["renewable_realtime_correction_mwh"])
        last = abs(future.iloc[-1]["renewable_realtime_correction_mwh"])
        self.assertGreater(first, last)
        self.assertTrue(adjusted["green_score"].between(0, 100).all())
        self.assertTrue(adjusted["planning_score"].between(0, 100).all())

    def test_five_minute_mw_converts_to_hourly_mwh(self):
        samples = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=12, freq="5min"),
                "solar_mw": [60.0] * 12,
                "wind_mw": [12.0] * 12,
            }
        )
        hourly = five_minute_mw_to_hourly_mwh(samples)
        self.assertEqual(len(hourly), 1)
        self.assertAlmostEqual(hourly.loc[0, "solar_mwh"], 60.0)
        self.assertAlmostEqual(hourly.loc[0, "wind_mwh"], 12.0)
        self.assertAlmostEqual(hourly.loc[0, "renewable_mwh"], 72.0)
        self.assertAlmostEqual(hourly.loc[0, "coverage_ratio"], 1.0)

    def test_missing_five_minute_sample_is_reported(self):
        samples = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=11, freq="5min"),
                "solar_mw": [60.0] * 11,
                "wind_mw": [12.0] * 11,
            }
        )
        hourly = five_minute_mw_to_hourly_mwh(samples)
        self.assertAlmostEqual(hourly.loc[0, "coverage_ratio"], 11 / 12)

    def test_official_renewables_adjust_only_renewable_forecast(self):
        forecast = sample_forecast()
        observations = pd.DataFrame(
            {
                "timestamp": forecast["timestamp"].iloc[:11],
                "actual_renewable_mwh": (
                    forecast["predicted_renewable_mwh"].iloc[:11] - 10
                ),
                "coverage_ratio": 1.0,
            }
        )
        as_of = forecast["timestamp"].iloc[10]
        adjusted, metadata = adjust_forecast_with_live_renewables(
            forecast.drop(columns=["actual_smp", "actual_renewable_mwh"]),
            sample_history(),
            observations,
            as_of=as_of,
        )
        future = adjusted[adjusted["timestamp"] > as_of]
        self.assertTrue(
            (future["predicted_smp"] == future["raw_predicted_smp"]).all()
        )
        self.assertTrue(
            (
                future["predicted_renewable_mwh"]
                < future["raw_predicted_renewable_mwh"]
            ).all()
        )
        self.assertEqual(metadata["renewable_observed_hours"], 11)


if __name__ == "__main__":
    unittest.main()
