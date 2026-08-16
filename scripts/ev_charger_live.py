"""한국환경공단 API에서 제주 전기차 충전소 위치·상태를 읽는다."""

from __future__ import annotations

import re
from collections.abc import Iterable
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
CHARGER_FIT_LABELS = {
    "matched": "현재 조건 일치",
    "unavailable": "현재 이용 불가",
    "restricted": "이용 제한 확인 필요",
    "output_unknown": "출력 확인 필요",
    "output_low": "출력 부족",
    "hours_unknown": "운영시간 확인 필요",
    "hours_closed": "추천시간 운영 안 함",
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


def _parse_daily_operating_minutes(value: Any) -> tuple[int, int] | None:
    """단순한 매일 운영시간을 분 단위로 바꾼다.

    요일마다 시간이 다른 문구는 잘못 해석하지 않도록 None으로 남긴다.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    compact = re.sub(r"\s+", "", text)
    if "24시간" in compact or compact.lower() in {"24h", "24hours"}:
        return 0, 24 * 60
    if re.search(r"평일|주말|공휴일|토요일|일요일|월요일|화요일|수요일|목요일|금요일", text):
        return None

    ranges = re.findall(
        r"(?<!\d)(\d{1,2})(?::(\d{2}))?\s*[~∼～-]\s*"
        r"(\d{1,2})(?::(\d{2}))?(?!\d)",
        text,
    )
    if len(ranges) != 1:
        return None
    start_hour, start_minute, end_hour, end_minute = ranges[0]
    start_hour = int(start_hour)
    start_minute = int(start_minute or 0)
    end_hour = int(end_hour)
    end_minute = int(end_minute or 0)
    if not (0 <= start_hour <= 23 and 0 <= start_minute <= 59):
        return None
    if not (0 <= end_hour <= 24 and 0 <= end_minute <= 59):
        return None
    if end_hour == 24 and end_minute != 0:
        return None
    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute
    if start == end:
        return None
    return start, end


def operating_hours_cover_slots(
    available_time: Any, charging_slots: Iterable[pd.Timestamp]
) -> bool | None:
    """운영시간이 추천된 각 1시간 충전칸을 모두 포함하는지 검사한다.

    True/False를 확정할 수 없는 복잡한 문구는 None을 반환한다.
    """
    operating_range = _parse_daily_operating_minutes(available_time)
    if operating_range is None:
        return None
    slots = [pd.Timestamp(value) for value in charging_slots]
    if not slots:
        return None

    start, end = operating_range
    if start == 0 and end == 24 * 60:
        return True
    for timestamp in slots:
        slot_start = timestamp.hour * 60 + timestamp.minute
        slot_end = slot_start + 60
        if end > start:
            covered = start <= slot_start and slot_end <= end
        else:
            adjusted_start = slot_start + (24 * 60 if slot_start < end else 0)
            covered = start <= adjusted_start and slot_end + (
                24 * 60 if slot_start < end else 0
            ) <= end + 24 * 60
        if not covered:
            return False
    return True


def assess_charger_compatibility(
    frame: pd.DataFrame,
    charging_slots: Iterable[pd.Timestamp],
    requested_output_kw: float,
) -> pd.DataFrame:
    """현재 상태·출력·운영시간으로 충전소가 추천 계획과 맞는지 표시한다.

    API의 현재 상태는 예약이나 미래 이용 가능성을 보장하지 않으므로, 모든 조건이
    맞아도 '현재 조건 일치'라고만 표시한다.
    """
    if requested_output_kw <= 0:
        raise ValueError("비교할 충전기 출력은 0보다 커야 합니다.")
    slots = [pd.Timestamp(value) for value in charging_slots]
    result = frame.copy()
    result["operating_hours_match"] = result["available_time"].apply(
        lambda value: operating_hours_cover_slots(value, slots)
    )

    def fit_label(row: pd.Series) -> str:
        if str(row["status_code"]) != "2":
            return CHARGER_FIT_LABELS["unavailable"]
        if str(row.get("user_limit", "")).strip().upper() == "Y":
            return CHARGER_FIT_LABELS["restricted"]
        if pd.isna(row["output_kw"]):
            return CHARGER_FIT_LABELS["output_unknown"]
        if float(row["output_kw"]) < requested_output_kw:
            return CHARGER_FIT_LABELS["output_low"]
        hours_match = row["operating_hours_match"]
        if pd.isna(hours_match):
            return CHARGER_FIT_LABELS["hours_unknown"]
        if not bool(hours_match):
            return CHARGER_FIT_LABELS["hours_closed"]
        return CHARGER_FIT_LABELS["matched"]

    result["recommendation_status"] = result.apply(fit_label, axis=1)
    result["matches_current_conditions"] = (
        result["recommendation_status"] == CHARGER_FIT_LABELS["matched"]
    )
    return result
