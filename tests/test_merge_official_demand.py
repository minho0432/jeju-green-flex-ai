from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from merge_official_demand import load_demand_csv  # noqa: E402


class MergeOfficialDemandTests(unittest.TestCase):
    def test_cp949_wide_file_maps_1_to_midnight_and_24_to_23(self):
        source = pd.DataFrame(
            [{"날짜": "2025-01-01", **{f"{hour}시": 600 + hour for hour in range(1, 25)}}]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "계통수요.csv"
            source.to_csv(path, index=False, encoding="cp949")
            result = load_demand_csv(path)

        self.assertEqual(len(result), 24)
        self.assertEqual(result.iloc[0]["timestamp"], pd.Timestamp("2025-01-01 00:00"))
        self.assertEqual(result.iloc[-1]["timestamp"], pd.Timestamp("2025-01-01 23:00"))
        self.assertEqual(result.iloc[0]["demand_mwh"], 601)
        self.assertEqual(result.iloc[-1]["demand_mwh"], 624)


if __name__ == "__main__":
    unittest.main()
