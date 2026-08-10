"""충전 최적화와 Green Point 정책의 핵심 규칙을 검사한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from optimizer import make_plan  # noqa: E402


def sample_forecast() -> pd.DataFrame:
    timestamps = pd.date_range("2025-12-10", periods=24, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "predicted_smp": [200 - hour * 3 for hour in range(24)],
            "predicted_smp_upper": [220 - hour * 3 for hour in range(24)],
            "predicted_renewable_mwh": [hour * 5 for hour in range(24)],
            "green_score": [min(hour * 4, 100) for hour in range(24)],
            "planning_score": [min(hour * 3, 100) for hour in range(24)],
            "actual_green_score": [min(hour * 4, 100) for hour in range(24)],
        }
    )


class OptimizerTests(unittest.TestCase):
    def make_default_plan(self, **overrides):
        arguments = {
            "forecast": sample_forecast(),
            "current_soc": 30,
            "target_soc": 80,
            "battery_kwh": 60,
            "charger_kw": 7,
            "efficiency": 0.9,
            "start_hour": 8,
            "departure_hour": 20,
            "retail_price": 320,
            "base_point_rate": 10,
            "bonus_point_rate": 20,
            "reward_threshold": 70,
            "session_point_cap": 1500,
            "continuous": True,
            "conservative": True,
        }
        arguments.update(overrides)
        return make_plan(**arguments)

    def test_reaches_target_soc(self):
        plan = self.make_default_plan()
        self.assertTrue(plan["feasible"])
        self.assertAlmostEqual(plan["reached_soc"], 80.0)

    def test_continuous_window_has_no_internal_gap(self):
        plan = self.make_default_plan()
        used = plan["ai_schedule"][plan["ai_schedule"]["scheduled_kwh"] > 0]
        differences = used["timestamp"].sort_values().diff().dropna()
        self.assertTrue((differences == pd.Timedelta(hours=1)).all())

    def test_guaranteed_points_survive_forecast_miss(self):
        forecast = sample_forecast()
        forecast["actual_green_score"] = 0
        plan = self.make_default_plan(forecast=forecast)
        self.assertGreater(plan["ai"]["guaranteed_points"], 0)
        self.assertEqual(plan["ai"]["settled_bonus_points"], 0)
        self.assertEqual(
            plan["ai"]["settled_total_points"],
            plan["ai"]["guaranteed_points"],
        )

    def test_future_without_actual_data_is_pending(self):
        forecast = sample_forecast().drop(columns="actual_green_score")
        plan = self.make_default_plan(forecast=forecast)
        self.assertEqual(plan["ai"]["settlement_status"], "pending_actual_data")
        self.assertGreaterEqual(
            plan["ai"]["expected_total_points"],
            plan["ai"]["guaranteed_points"],
        )

    def test_session_point_cap_is_enforced(self):
        plan = self.make_default_plan(
            base_point_rate=100,
            bonus_point_rate=200,
            session_point_cap=500,
        )
        self.assertLessEqual(plan["ai"]["expected_total_points"], 500)
        self.assertLessEqual(plan["ai"]["settled_total_points"], 500)

    def test_impossible_schedule_is_reported(self):
        plan = self.make_default_plan(
            current_soc=10,
            target_soc=100,
            battery_kwh=100,
            charger_kw=3,
            start_hour=18,
            departure_hour=20,
        )
        self.assertFalse(plan["feasible"])
        self.assertLess(plan["reached_soc"], 100)

    def test_invalid_time_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_default_plan(start_hour=20, departure_hour=8)


if __name__ == "__main__":
    unittest.main()
