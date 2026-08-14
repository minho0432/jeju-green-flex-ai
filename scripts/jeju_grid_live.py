"""공공데이터포털의 제주 5분 계통·재생에너지 실측을 안전하게 읽는다.

공식 API:
https://www.data.go.kr/data/15158505/openapi.do

API 원본의 단위는 MW(그 순간의 발전 세기)다. 현재 AI가 학습한 태양광+풍력
정답은 시간별 MWh(한 시간 동안 생산한 전기의 양)이므로, 5분 표본을
``MW × 5/60``으로 바꿔 한 시간 안에서 합산한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import pandas as pd

from realtime_adjustment import five_minute_mw_to_hourly_mwh


API_URL = (
    "https://apis.data.go.kr/B552115/JejuSukub5mToday/"
    "getJejuSukub5mToday"
)
KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")

API_TO_CANONICAL = {
    "baseDatetime": "timestamp",
    "suppAbility": "supply_capacity_mw",
    "currPwrTot": "demand_generation_mw",
    "renewPwrTot": "renewable_total_mw",
    "renewPwrSolar": "solar_mw",
    "renewPwrWind": "wind_mw",
    "currNtPwrTot": "demand_transmission_mw",
}
NUMERIC_COLUMNS = [
    "supply_capacity_mw",
    "demand_generation_mw",
    "renewable_total_mw",
    "solar_mw",
    "wind_mw",
    "demand_transmission_mw",
]
REQUIRED_COLUMNS = ["timestamp", *NUMERIC_COLUMNS]


class JejuGridApiError(RuntimeError):
    """API 오류를 비밀키가 노출되지 않는 사용자용 메시지로 바꾼 예외."""


def build_request_url(
    service_key: str,
    base_date: str | None = None,
    page_no: int = 1,
    num_of_rows: int = 300,
) -> str:
    """공식 명세에 맞는 URL을 만든다.

    공공데이터포털은 인코딩키와 디코딩키를 모두 보여 준다. 어느 쪽을 복사해도
    이중 인코딩되지 않도록 먼저 한 번 디코딩한 뒤 URL 쿼리로 인코딩한다.
    """
    key = unquote(service_key.strip())
    if not key:
        raise ValueError("공공데이터포털 인증키가 비어 있습니다.")
    if page_no < 1 or num_of_rows < 1:
        raise ValueError("페이지와 행 수는 1 이상이어야 합니다.")
    if base_date is None:
        base_date = datetime.now(KOREA_TIMEZONE).strftime("%Y%m%d")
    if len(base_date) != 8 or not base_date.isdigit():
        raise ValueError("기준일은 YYYYMMDD 형식이어야 합니다.")
    query = urlencode(
        {
            "serviceKey": key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            "dataType": "json",
            "baseDate": base_date,
        }
    )
    return f"{API_URL}?{query}"


def _xml_to_mapping(payload: str) -> dict[str, Any]:
    """JSON 요청이 오류 XML로 돌아오는 경우까지 읽을 수 있게 변환한다."""
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise JejuGridApiError("제주 실시간 API 응답을 읽을 수 없습니다.") from error

    header_node = root.find(".//header")
    common_error_node = root.find(".//cmmMsgHeader")
    body_node = root.find(".//body")
    header: dict[str, Any] = {}
    if header_node is not None:
        header = {child.tag: child.text for child in header_node}
    elif common_error_node is not None:
        common_error = {child.tag: child.text for child in common_error_node}
        header = {
            "resultCode": common_error.get("returnReasonCode", "GATEWAY_ERROR"),
            "resultMsg": (
                common_error.get("returnAuthMsg")
                or common_error.get("errMsg")
                or "공공데이터포털 게이트웨이 오류"
            ),
        }
    items = []
    if body_node is not None:
        for item_node in body_node.findall(".//item"):
            items.append({child.tag: child.text for child in item_node})
    return {"response": {"header": header, "body": {"items": {"item": items}}}}


def _normalise_payload(payload: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        stripped = payload.lstrip()
        if stripped.startswith("<"):
            return _xml_to_mapping(payload)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise JejuGridApiError("제주 실시간 API가 올바른 JSON을 보내지 않았습니다.") from error
        if not isinstance(parsed, dict):
            raise JejuGridApiError("제주 실시간 API 응답의 구조가 예상과 다릅니다.")
        return parsed
    if not isinstance(payload, dict):
        raise TypeError("API 응답은 JSON 문자열, 바이트 또는 사전이어야 합니다.")
    return payload


def parse_api_response(payload: str | bytes | dict[str, Any]) -> pd.DataFrame:
    """공식 JSON/XML 응답을 단위가 표시된 일정한 표로 바꾼다."""
    parsed = _normalise_payload(payload)
    response = parsed.get("response", parsed)
    header = response.get("header", {}) or {}
    result_code = str(header.get("resultCode", "00")).strip()
    if result_code not in {"0", "00"}:
        result_message = str(header.get("resultMsg", "알 수 없는 오류"))
        raise JejuGridApiError(
            f"제주 실시간 API 오류({result_code}): {result_message}"
        )

    body = response.get("body", {}) or {}
    items_container = body.get("items", {}) or {}
    items = items_container.get("item", []) if isinstance(items_container, dict) else []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise JejuGridApiError("제주 실시간 API에 오늘 관측값이 아직 없습니다.")

    frame = pd.DataFrame(items).rename(columns=API_TO_CANONICAL)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise JejuGridApiError(
            f"제주 실시간 API 응답에 필요한 열이 없습니다: {sorted(missing)}"
        )
    frame = frame[REQUIRED_COLUMNS].copy()
    timestamp_text = frame["timestamp"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame["timestamp"] = pd.to_datetime(
        timestamp_text, format="%Y%m%d%H%M", errors="coerce"
    )
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise JejuGridApiError("제주 실시간 API에 시간 또는 숫자로 읽을 수 없는 값이 있습니다.")
    return (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def fetch_jeju_grid_live(
    service_key: str,
    base_date: str | None = None,
    timeout_seconds: int = 12,
) -> pd.DataFrame:
    """오늘의 제주 5분 관측값을 호출한다. 인증키는 오류문에 포함하지 않는다."""
    request_url = build_request_url(service_key, base_date=base_date)
    request = Request(request_url, headers={"User-Agent": "JejuGreenFlexAI/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except Exception as error:
        raise JejuGridApiError(
            "제주 실시간 API에 연결하지 못했습니다. 인증키 승인 상태와 네트워크를 확인하세요."
        ) from error
    return parse_api_response(payload)


def grid_samples_to_hourly(samples: pd.DataFrame) -> pd.DataFrame:
    """5분 실측을 모델과 같은 시간별 태양광+풍력 MWh로 바꾼다."""
    required = {"timestamp", *NUMERIC_COLUMNS}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"시간 변환에 필요한 열이 없습니다: {sorted(missing)}")

    energy = five_minute_mw_to_hourly_mwh(
        samples,
        power_columns=("solar_mw", "wind_mw"),
    ).rename(columns={"renewable_mwh": "actual_renewable_mwh"})
    context = samples.copy()
    context["timestamp"] = pd.to_datetime(context["timestamp"]).dt.floor("h")
    context = (
        context.groupby("timestamp", as_index=False)[
            [
                "supply_capacity_mw",
                "demand_generation_mw",
                "renewable_total_mw",
                "demand_transmission_mw",
            ]
        ]
        .mean()
    )
    # 공식 계통 수요: 시간 평균 MW ≈ 그 시간 MWh (부하 에너지 근사)
    # API 필드 currPwrTot → demand_generation_mw (이미 매핑됨)
    context["actual_demand_mwh"] = context["demand_generation_mw"]
    return energy.merge(context, on="timestamp", how="left")


def latest_complete_hour(
    hourly: pd.DataFrame, minimum_coverage: float = 1.0
) -> pd.Timestamp:
    """5분 표본 12개가 모두 모인 최신 시간을 고른다.

    11개만 더해도 한 시간 에너지가 약 8.3% 작게 계산될 수 있으므로 기본값은
    100%다. 연구 목적으로 기준을 낮출 수는 있지만 앱에서는 기본값만 쓴다.
    """
    if not 0 < minimum_coverage <= 1:
        raise ValueError("관측 완성도 기준은 0보다 크고 1 이하여야 합니다.")
    complete = hourly[hourly["coverage_ratio"] >= minimum_coverage]
    if complete.empty:
        raise JejuGridApiError(
            "5분 관측이 충분히 모인 완전한 시간대가 아직 없습니다."
        )
    return pd.Timestamp(complete["timestamp"].max())


def observation_age_minutes(
    samples: pd.DataFrame, now: pd.Timestamp | datetime | None = None
) -> float:
    """가장 최근 5분 실측이 현재보다 몇 분 전인지 계산한다."""
    if samples.empty or "timestamp" not in samples.columns:
        raise JejuGridApiError("최신성을 확인할 제주 실시간 관측값이 없습니다.")
    timestamps = pd.to_datetime(samples["timestamp"], errors="coerce")
    if timestamps.isna().all():
        raise JejuGridApiError("제주 실시간 관측시각을 읽을 수 없습니다.")
    latest = pd.Timestamp(timestamps.max())
    current = (
        pd.Timestamp.now(tz=KOREA_TIMEZONE)
        if now is None
        else pd.Timestamp(now)
    )
    if current.tzinfo is not None:
        current = current.tz_convert(KOREA_TIMEZONE).tz_localize(None)
    if latest.tzinfo is not None:
        latest = latest.tz_convert(KOREA_TIMEZONE).tz_localize(None)
    return max(0.0, float((current - latest).total_seconds() / 60))
