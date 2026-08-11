"""한국환경공단 API에서 제주 전기차 충전소 위치·상태를 읽는다."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import pandas as pd


API_URL = "https://apis.data.go.kr/B552584/EvCharger/getChargerInfo"
JEJU_REGION_CODE = "50"
STATUS_LABELS = {
    "1": "통신이상",
    "2": "충전대기",
    "3": "충전중",
    "4": "운영중지",
    "5": "점검중",
    "9": "상태미확인",
}
DISPLAY_COLUMNS = [
    "station_name",
    "station_id",
    "charger_id",
    "charger_type",
    "address",
    "latitude",
    "longitude",
    "available_time",
    "operator_name",
    "status_code",
    "status_updated_at",
    "output_kw",
    "charging_method",
    "parking_free",
    "user_limit",
]
API_TO_CANONICAL = {
    "statNm": "station_name",
    "statId": "station_id",
    "chgerId": "charger_id",
    "chgerType": "charger_type",
    "addr": "address",
    "lat": "latitude",
    "lng": "longitude",
    "useTime": "available_time",
    "busiNm": "operator_name",
    "stat": "status_code",
    "statUpdDt": "status_updated_at",
    "output": "output_kw",
    "method": "charging_method",
    "parkingFree": "parking_free",
    "limitYn": "user_limit",
}


class EvChargerApiError(RuntimeError):
    """비밀키를 노출하지 않는 충전소 API 오류."""


def build_request_url(service_key: str, num_of_rows: int = 9999) -> str:
    key = unquote(service_key.strip())
    if not key:
        raise ValueError("충전소 API 인증키가 비어 있습니다.")
    if not 10 <= num_of_rows <= 9999:
        raise ValueError("충전소 API 행 수는 10~9999여야 합니다.")
    query = urlencode(
        {
            "serviceKey": key,
            "pageNo": 1,
            "numOfRows": num_of_rows,
            "zcode": JEJU_REGION_CODE,
            "dataType": "XML",
        }
    )
    return f"{API_URL}?{query}"


def _text(node: ElementTree.Element | None, tag: str, default: str = "") -> str:
    if node is None:
        return default
    child = node.find(tag)
    return default if child is None or child.text is None else child.text.strip()


def parse_charger_response(payload: str | bytes) -> pd.DataFrame:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise EvChargerApiError("충전소 API 응답을 읽을 수 없습니다.") from error

    common_error = root.find(".//cmmMsgHeader")
    if common_error is not None:
        code = _text(common_error, "returnReasonCode", "GATEWAY_ERROR")
        message = (
            _text(common_error, "returnAuthMsg")
            or _text(common_error, "errMsg")
            or "공공데이터포털 게이트웨이 오류"
        )
        raise EvChargerApiError(f"충전소 API 오류({code}): {message}")

    header = root.find(".//header")
    result_code = _text(header, "resultCode", "00")
    if result_code not in {"0", "00"}:
        message = _text(header, "resultMsg", "알 수 없는 오류")
        raise EvChargerApiError(f"충전소 API 오류({result_code}): {message}")

    items = []
    for item in root.findall(".//item"):
        row: dict[str, Any] = {}
        for api_name, canonical_name in API_TO_CANONICAL.items():
            row[canonical_name] = _text(item, api_name)
        items.append(row)
    if not items:
        raise EvChargerApiError(
            "제주 충전소 자료가 없거나 이 API의 별도 활용신청이 필요합니다."
        )
    frame = pd.DataFrame(items, columns=DISPLAY_COLUMNS)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["output_kw"] = pd.to_numeric(frame["output_kw"], errors="coerce")
    timestamp_text = frame["status_updated_at"].replace("", pd.NA)
    frame["status_updated_at"] = pd.to_datetime(
        timestamp_text, format="%Y%m%d%H%M%S", errors="coerce"
    )
    frame["status_label"] = frame["status_code"].map(STATUS_LABELS).fillna(
        "기타상태"
    )
    frame = frame.drop_duplicates(
        ["station_id", "charger_id"], keep="last"
    ).reset_index(drop=True)
    return frame


def fetch_jeju_ev_chargers(
    service_key: str, timeout_seconds: int = 15
) -> pd.DataFrame:
    request = Request(
        build_request_url(service_key),
        headers={"User-Agent": "JejuGreenFlexAI/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except Exception as error:
        raise EvChargerApiError(
            "충전소 API에 연결하지 못했습니다. 해당 API 활용신청 승인과 인증키를 확인하세요."
        ) from error
    return parse_charger_response(payload)


def charger_status_summary(frame: pd.DataFrame) -> dict[str, int]:
    statuses = frame["status_code"].astype(str)
    return {
        "total": int(len(frame)),
        "available": int((statuses == "2").sum()),
        "charging": int((statuses == "3").sum()),
        "unavailable": int(statuses.isin(["1", "4", "5"]).sum()),
        "unknown": int((~statuses.isin(["1", "2", "3", "4", "5"])).sum()),
    }
