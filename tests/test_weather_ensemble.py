"""Open-Meteo 앙상블 파싱과 AI 예상범위 확대를 검사한다."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from weather_ensemble import (  # noqa: E402
    WeatherEnsembleError,
    apply_ensemble_uncertainty,
    build_ensemble_url,
    parse_ensemble_response,
)


class RadiationModel:
    def predict(self, features):
        return features["shortwave_radiation"].to_numpy(dtype=float) / 10


class WindModel:
    def predict(self, features):
        return features["wind_speed_10m"].to_numpy(dtype=float) * 5


def ensemble_payload(member_count: int = 6) -> dict[str, object]:
    times = pd.date_range("2026-08-12", periods=24, freq="h")
    hourly: dict[str, object] = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
    for member in range(member_count):
        suffix = "" if member == 0 else f"_member{member:02d}"
        hourly[f"temperature_2m{suffix}"] = [25 + member] * 24
        hourly[f"relative_humidity_2m{suffix}"] = [70 - member] * 24
        hourly[f"wind_speed_10m{suffix}"] = [3 + member] * 24
        hourly[f"shortwave_radiation{suffix}"] = [100 + member * 50] * 24
    return {"hourly": hourly}


class WeatherEnsembleTests(unittest.TestCase):
    def test_url_uses_jeju_korea_time_and_mps(self):
        url = build_ensemble_url()
        self.assertIn("timezone=Asia%2FSeoul", url)
        self.assertIn("wind_speed_unit=ms", url)
        self.assertIn("models=icon_seamless_eps", url)

    def test_parser_makes_24_hours_for_every_member(self):
        frame = parse_ensemble_response(
            json.dumps(ensemble_payload()), date(2026, 8, 12)
        )
        self.assertEqual(frame["ensemble_member"].nunique(), 6)
        self.assertEqual(len(frame), 24 * 6)
        self.assertFalse(frame.isna().any().any())

    def test_too_few_members_are_rejected(self):
        with self.assertRaises(WeatherEnsembleError):
            parse_ensemble_response(ensemble_payload(4), date(2026, 8, 12))

    def test_ensemble_expands_intervals_and_reports_members(self):
        weather = parse_ensemble_response(ensemble_payload(), date(2026, 8, 12))
        timestamps = pd.date_range("2026-08-12", periods=24, freq="h")
        forecast = pd.DataFrame(
            {
                "timestamp": timestamps,
                "predicted_renewable_mwh": [10.0] * 24,
                "predicted_renewable_lower": [7.0] * 24,
                "predicted_renewable_upper": [13.0] * 24,
                "predicted_demand_mwh": [20.0] * 24,
                "predicted_demand_lower": [15.0] * 24,
                "predicted_demand_upper": [25.0] * 24,
                "green_score": [60.0] * 24,
            }
        )
        history = pd.DataFrame(
            {
                "renewable_mwh": np.linspace(0, 100, 200),
                "demand_mwh": np.linspace(100, 200, 200),
            }
        )
        result, metadata = apply_ensemble_uncertainty(
            forecast,
            {"demand_mwh": WindModel(), "renewable_mwh": RadiationModel()},
            history,
            weather,
        )
        self.assertEqual(metadata["member_count"], 6)
        self.assertTrue((result["predicted_renewable_upper"] >= 13).all())
        self.assertTrue((result["predicted_demand_upper"] >= 25).all())
        self.assertIn("planning_score", result.columns)


if __name__ == "__main__":
    unittest.main()
