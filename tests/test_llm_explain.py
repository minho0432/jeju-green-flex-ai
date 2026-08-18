import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

from scripts.llm_explain import explain_recommendation


class LlmExplainTests(unittest.TestCase):
    def setUp(self):
        self.forecast = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-08-18", periods=2, freq="h"),
                "planning_score": [42.0, 78.0],
                "forecast_risk_points": [20.0, 5.0],
                "wind_speed_10m": [2.0, 4.0],
                "shortwave_radiation": [50.0, 180.0],
            }
        )
        self.plan = {
            "ai_schedule": pd.DataFrame(
                {
                    "timestamp": self.forecast["timestamp"],
                    "scheduled_kwh": [0.0, 5.0],
                }
            ),
            "score_column": "planning_score",
            "required_grid_kwh": 5.0,
            "reached_soc": 80.0,
        }

    def test_without_key_uses_local_fallback(self):
        explanation = explain_recommendation(self.forecast, self.plan, api_key="")
        self.assertIn("01:00~02:00", explanation)
        self.assertIn("바람", explanation)

    @patch("scripts.llm_explain.requests.post")
    def test_openai_compatible_response_is_returned(self, post):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "추천 설명"}}]}
        post.return_value = response

        explanation = explain_recommendation(
            self.forecast,
            self.plan,
            api_key="test-key",
            model="test-model",
            endpoint="https://example.test/v1/chat/completions",
        )

        self.assertEqual(explanation, "추천 설명")
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["model"], "test-model")

    @patch(
        "scripts.llm_explain.requests.post",
        side_effect=requests.RequestException("network"),
    )
    def test_api_failure_does_not_break_recommendation(self, post):
        explanation = explain_recommendation(self.forecast, self.plan, api_key="test-key")
        self.assertIn("추천 충전시간", explanation)