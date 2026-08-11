"""제주 충전소 위치·상태 API를 네트워크 없이 검사한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ev_charger_live import (  # noqa: E402
    EvChargerApiError,
    assess_charger_compatibility,
    build_request_url,
    charger_status_summary,
    operating_hours_cover_slots,
    parse_charger_response,
)


def response_xml() -> str:
    return """<response><header><resultCode>00</resultCode><resultMsg>OK</resultMsg></header>
    <body><items>
      <item><statNm>제주 A</statNm><statId>JA</statId><chgerId>01</chgerId>
      <chgerType>06</chgerType><addr>제주시</addr><lat>33.50</lat><lng>126.53</lng>
      <useTime>24시간</useTime><busiNm>A사</busiNm><stat>2</stat>
      <statUpdDt>20260811123000</statUpdDt><output>50</output><method>단독</method>
      <parkingFree>Y</parkingFree><limitYn>N</limitYn></item>
      <item><statNm>제주 B</statNm><statId>JB</statId><chgerId>01</chgerId>
      <chgerType>04</chgerType><addr>서귀포시</addr><lat>33.25</lat><lng>126.56</lng>
      <useTime>09:00~18:00</useTime><busiNm>B사</busiNm><stat>3</stat>
      <statUpdDt>20260811123100</statUpdDt><output>100</output><method>동시</method>
      <parkingFree>N</parkingFree><limitYn>N</limitYn></item>
    </items></body></response>"""


class EvChargerLiveTests(unittest.TestCase):
    def test_url_limits_results_to_jeju(self):
        url = build_request_url("abc%2Bdef%3D")
        self.assertIn("zcode=50", url)
        self.assertIn("serviceKey=abc%2Bdef%3D", url)
        self.assertNotIn("%252B", url)

    def test_xml_is_parsed_with_location_and_status(self):
        frame = parse_charger_response(response_xml())
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["status_label"], "충전대기")
        self.assertAlmostEqual(frame.iloc[0]["latitude"], 33.5)
        self.assertEqual(frame.iloc[1]["output_kw"], 100)

    def test_status_summary_counts_each_state(self):
        summary = charger_status_summary(parse_charger_response(response_xml()))
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["available"], 1)
        self.assertEqual(summary["charging"], 1)

    def test_gateway_error_keeps_reason_without_key(self):
        payload = """<OpenAPI_ServiceResponse><cmmMsgHeader>
        <returnReasonCode>30</returnReasonCode>
        <returnAuthMsg>SERVICE KEY IS NOT REGISTERED</returnAuthMsg>
        </cmmMsgHeader></OpenAPI_ServiceResponse>"""
        with self.assertRaises(EvChargerApiError) as context:
            parse_charger_response(payload)
        self.assertIn("30", str(context.exception))
        self.assertNotIn("serviceKey=", str(context.exception))

    def test_only_waiting_charger_matches_current_conditions(self):
        frame = parse_charger_response(response_xml())
        result = assess_charger_compatibility(
            frame,
            [pd.Timestamp("2026-08-11 12:00")],
            requested_output_kw=50,
        )
        self.assertEqual(result.iloc[0]["recommendation_status"], "현재 조건 일치")
        self.assertEqual(result.iloc[1]["recommendation_status"], "현재 이용 불가")
        self.assertEqual(int(result["matches_current_conditions"].sum()), 1)

    def test_charger_with_too_little_output_is_not_matched(self):
        frame = parse_charger_response(response_xml()).iloc[[0]].copy()
        result = assess_charger_compatibility(
            frame,
            [pd.Timestamp("2026-08-11 12:00")],
            requested_output_kw=100,
        )
        self.assertEqual(result.iloc[0]["recommendation_status"], "출력 부족")

    def test_daily_hours_must_cover_every_recommended_slot(self):
        slots = [pd.Timestamp("2026-08-11 17:00")]
        self.assertTrue(operating_hours_cover_slots("09:00~18:00", slots))
        self.assertFalse(
            operating_hours_cover_slots(
                "09:00~18:00", [pd.Timestamp("2026-08-11 18:00")]
            )
        )

    def test_overnight_hours_are_supported(self):
        self.assertTrue(
            operating_hours_cover_slots(
                "22:00~06:00",
                [pd.Timestamp("2026-08-11 23:00"), pd.Timestamp("2026-08-12 05:00")],
            )
        )
        self.assertFalse(
            operating_hours_cover_slots(
                "22:00~06:00", [pd.Timestamp("2026-08-11 21:00")]
            )
        )

    def test_complex_weekday_hours_require_manual_confirmation(self):
        result = operating_hours_cover_slots(
            "평일 09:00~18:00 / 주말 휴무",
            [pd.Timestamp("2026-08-11 12:00")],
        )
        self.assertIsNone(result)

    def test_restricted_charger_requires_confirmation(self):
        frame = parse_charger_response(response_xml()).iloc[[0]].copy()
        frame.loc[:, "user_limit"] = "Y"
        result = assess_charger_compatibility(
            frame,
            [pd.Timestamp("2026-08-11 12:00")],
            requested_output_kw=50,
        )
        self.assertEqual(
            result.iloc[0]["recommendation_status"], "이용 제한 확인 필요"
        )


if __name__ == "__main__":
    unittest.main()
