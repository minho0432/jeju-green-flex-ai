import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_utils import add_lag_features


class LagFeatureTests(unittest.TestCase):
    def test_missing_calendar_hour_does_not_create_false_24h_lag(self):
        timestamps = list(pd.date_range("2025-01-01", periods=24, freq="h"))
        timestamps.append(pd.Timestamp("2025-01-02 01:00"))
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "renewable_mwh": range(25),
            "demand_mwh": range(100, 125),
        })
        result = add_lag_features(frame)
        self.assertTrue(pd.isna(result.iloc[-1]["renewable_lag_24h"]))
        self.assertTrue(pd.isna(result.iloc[-1]["demand_lag_24h"]))


if __name__ == "__main__":
    unittest.main()
