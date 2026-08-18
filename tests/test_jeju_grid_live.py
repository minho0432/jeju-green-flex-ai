"""공식 제주 5분 API 파서와 MW→MWh 변환을 네트워크 없이 검사한다."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from jeju_grid_live import (  # noqa: E402
    JejuGridApiError,
    JejuGridNoDataError,
    build_legacy_request_url,
    build_request_url,
    fetch_jeju_grid_live,
    grid_samples_to_hourly,
    latest_complete_hour,
    observation_age_minutes,
    parse_api_response,
)


def sample_item(minute: int) -> dict[str, object]:
    return {
        "rn": minute // 5 + 1,
        "baseDatetime": int(f"2026081110{minute:02d}"),
        "suppAbility": 1813,
        "currPwrTot": 900,
        "renewPwrTot": 180,
        "renewPwrSolar": 60,
        "renewPwrWind": 120,
        "currNtPwrTot": 883,
    }


class JejuGridLiveTests(unittest.TestCase):
    def test_json_response_is_parsed_and_sorted(self):
        payload = {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE"},
                "body": {"items": {"item": [sample_item(5), sample_item(0)]}},
            }
        }
        frame = parse_api_response(json.dumps(payload))
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["timestamp"].strftime("%H:%M"), "10:00")
        self.assertEqual(frame.iloc[0]["solar_mw"], 60)
        self.assertEqual(frame.iloc[0]["wind_mw"], 120)

    def test_gateway_items_array_is_supported(self):
        payload = {
            "response": {
                "header": {"resultCode": "00"},
                "body": {"items": [sample_item(0), sample_item(5)], "totalCount": 2},
            }
        }
        frame = parse_api_response(payload)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["timestamp"].strftime("%H:%M"), "10:00")

    def test_empty_success_response_is_not_reported_as_auth_failure(self):
        payload = {
            "response": {
                "header": {"resultCode": "00"},
                "body": {"items": [], "totalCount": 0},
            }
        }
        with self.assertRaises(JejuGridNoDataError) as context:
            parse_api_response(payload)
        self.assertIn("연결과 인증은 성공", str(context.exception))
        self.assertIn("totalCount=0", str(context.exception))

    def test_single_item_and_top_level_response_are_supported(self):
        payload = {
            "header": {"resultCode": "00"},
            "body": {"items": {"item": sample_item(0)}},
        }
        frame = parse_api_response(payload)
        self.assertEqual(len(frame), 1)

    def test_api_error_does_not_expose_service_key(self):
        payload = {
            "response": {
                "header": {"resultCode": "30", "resultMsg": "SERVICE KEY IS NOT REGISTERED"},
                "body": {},
            }
        }
        with self.assertRaises(JejuGridApiError) as context:
            parse_api_response(payload)
        self.assertNotIn("serviceKey", str(context.exception))

    def test_common_gateway_xml_error_keeps_official_reason(self):
        payload = """<?xml version="1.0" encoding="UTF-8"?>
        <OpenAPI_ServiceResponse><cmmMsgHeader>
        <errMsg>SERVICE ERROR</errMsg><returnAuthMsg>SERVICE KEY IS NOT REGISTERED</returnAuthMsg>
        <returnReasonCode>30</returnReasonCode>
        </cmmMsgHeader></OpenAPI_ServiceResponse>"""
        with self.assertRaises(JejuGridApiError) as context:
            parse_api_response(payload)
        self.assertIn("30", str(context.exception))
        self.assertIn("SERVICE KEY IS NOT REGISTERED", str(context.exception))

    def test_successful_legacy_xml_response_is_parsed(self):
        payload = """<?xml version="1.0" encoding="UTF-8"?>
        <response><header><resultCode>00</resultCode><resultMsg>NORMAL SERVICE</resultMsg></header>
        <body><items><item>
        <baseDatetime>202608111000</baseDatetime><suppAbility>1813</suppAbility>
        <currPwrTot>900</currPwrTot><renewPwrTot>180</renewPwrTot>
        <renewPwrSolar>60</renewPwrSolar><renewPwrWind>120</renewPwrWind>
        <currNtPwrTot>883</currNtPwrTot>
        </item></items><totalCount>1</totalCount></body></response>"""
        frame = parse_api_response(payload)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["timestamp"].strftime("%Y-%m-%d %H:%M"), "2026-08-11 10:00")
        self.assertEqual(frame.iloc[0]["solar_mw"], 60)
        self.assertEqual(frame.iloc[0]["wind_mw"], 120)

    def test_expired_key_error_has_actionable_guidance(self):
        payload = {
            "response": {
                "header": {
                    "resultCode": "31",
                    "resultMsg": "DEADLINE_HAS_EXPIRED_ERROR",
                },
                "body": {},
            }
        }
        with self.assertRaises(JejuGridApiError) as context:
            parse_api_response(payload)
        self.assertIn("사용기간이 만료", str(context.exception))
        self.assertIn("연장 승인", str(context.exception))

    def test_request_limit_error_has_actionable_guidance(self):
        payload = {
            "response": {
                "header": {
                    "resultCode": "22",
                    "resultMsg": "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR",
                },
                "body": {},
            }
        }
        with self.assertRaises(JejuGridApiError) as context:
            parse_api_response(payload)
        self.assertIn("호출한도", str(context.exception))

    def test_encoded_key_is_not_double_encoded(self):
        url = build_request_url("abc%2Bdef%3D", base_date="20260811")
        self.assertIn("serviceKey=abc%2Bdef%3D", url)
        self.assertNotIn("%252B", url)
        self.assertIn("baseDate=20260811", url)

    def test_request_url_can_omit_base_date(self):
        url = build_request_url("abc%2Bdef%3D", include_base_date=False)
        self.assertNotIn("baseDate=", url)
        self.assertIn("serviceKey=abc%2Bdef%3D", url)

    def test_legacy_request_url_uses_xml_endpoint_without_date_or_json(self):
        url = build_legacy_request_url("abc%2Bdef%3D")
        self.assertIn("openapi.kpx.or.kr", url)
        self.assertIn("serviceKey=abc%2Bdef%3D", url)
        self.assertNotIn("%252B", url)
        self.assertNotIn("baseDate=", url)
        self.assertNotIn("dataType=", url)

    def test_live_fetch_uses_legacy_xml_after_gw_zero_rows(self):
        empty_payload = json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00"},
                    "body": {"items": [], "totalCount": 0},
                }
            }
        ).encode()
        success_payload = json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00"},
                    "body": {"items": [sample_item(0)], "totalCount": 1},
                }
            }
        ).encode()
        empty_response = MagicMock()
        empty_response.__enter__.return_value.read.return_value = empty_payload
        success_response = MagicMock()
        success_response.__enter__.return_value.read.return_value = success_payload

        with patch(
            "jeju_grid_live.urlopen",
            side_effect=[empty_response, success_response],
        ) as mocked_urlopen:
            frame = fetch_jeju_grid_live("abc%2Bdef%3D")

        self.assertEqual(len(frame), 1)
        self.assertEqual(mocked_urlopen.call_count, 2)
        first_url = mocked_urlopen.call_args_list[0].args[0].full_url
        second_url = mocked_urlopen.call_args_list[1].args[0].full_url
        self.assertNotIn("baseDate=", first_url)
        self.assertIn("apis.data.go.kr", first_url)
        self.assertIn("openapi.kpx.or.kr", second_url)
        self.assertEqual(frame.attrs["api_source"], "구형 KPX XML")

    def test_live_fetch_uses_legacy_after_gw_api_error(self):
        denied_payload = json.dumps(
            {
                "response": {
                    "header": {"resultCode": "20", "resultMsg": "SERVICE ACCESS DENIED"},
                    "body": {},
                }
            }
        ).encode()
        success_payload = json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00"},
                    "body": {"items": [sample_item(0)], "totalCount": 1},
                }
            }
        ).encode()
        denied_response = MagicMock()
        denied_response.__enter__.return_value.read.return_value = denied_payload
        success_response = MagicMock()
        success_response.__enter__.return_value.read.return_value = success_payload

        with patch(
            "jeju_grid_live.urlopen",
            side_effect=[denied_response, success_response],
        ) as mocked_urlopen:
            frame = fetch_jeju_grid_live("abc%2Bdef%3D")

        self.assertEqual(len(frame), 1)
        self.assertEqual(mocked_urlopen.call_count, 2)
        self.assertEqual(frame.attrs["api_source"], "구형 KPX XML")

    def test_explicit_date_does_not_trigger_live_fallback(self):
        empty_payload = json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00"},
                    "body": {"items": [], "totalCount": 0},
                }
            }
        ).encode()
        empty_response = MagicMock()
        empty_response.__enter__.return_value.read.return_value = empty_payload

        with patch("jeju_grid_live.urlopen", return_value=empty_response) as mocked:
            with self.assertRaises(JejuGridNoDataError):
                fetch_jeju_grid_live("abc%2Bdef%3D", base_date="20260814")

        self.assertEqual(mocked.call_count, 1)

    def test_twelve_five_minute_samples_make_one_hour_mwh(self):
        payload = {
            "response": {
                "header": {"resultCode": "00"},
                "body": {"items": {"item": [sample_item(m) for m in range(0, 60, 5)]}},
            }
        }
        hourly = grid_samples_to_hourly(parse_api_response(payload))
        self.assertEqual(len(hourly), 1)
        self.assertAlmostEqual(hourly.iloc[0]["solar_mwh"], 60.0)
        self.assertAlmostEqual(hourly.iloc[0]["wind_mwh"], 120.0)
        self.assertAlmostEqual(hourly.iloc[0]["actual_renewable_mwh"], 180.0)
        self.assertAlmostEqual(hourly.iloc[0]["coverage_ratio"], 1.0)
        self.assertEqual(latest_complete_hour(hourly).strftime("%H:%M"), "10:00")

    def test_eleven_samples_are_not_treated_as_a_complete_hour(self):
        payload = {
            "response": {
                "header": {"resultCode": "00"},
                "body": {"items": {"item": [sample_item(m) for m in range(0, 55, 5)]}},
            }
        }
        hourly = grid_samples_to_hourly(parse_api_response(payload))
        self.assertAlmostEqual(hourly.iloc[0]["coverage_ratio"], 11 / 12)
        with self.assertRaises(JejuGridApiError):
            latest_complete_hour(hourly)

    def test_observation_age_uses_korea_local_timestamp(self):
        payload = {
            "response": {
                "header": {"resultCode": "00"},
                "body": {"items": {"item": sample_item(0)}},
            }
        }
        samples = parse_api_response(payload)
        age = observation_age_minutes(samples, now="2026-08-11 10:17:00")
        self.assertAlmostEqual(age, 17.0)


if __name__ == "__main__":
    unittest.main()
