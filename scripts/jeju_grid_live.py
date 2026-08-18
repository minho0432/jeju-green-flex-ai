"""공공데이터포털의 제주 5분 계통·재생에너지 실측을 안전하게 읽는다.

공식 API:
https://www.data.go.kr/data/15158505/openapi.do

API 원본의 단위는 MW(그 순간의 발전 세기)다. 현재 AI가 학습한 태양광+풍력
정답은 시간별 MWh(한 시간 동안 생산한 전기의 양)이므로, 5분 표본을
``MW × 5/60``으로 바꿔 한 시간 안에서 합산한다.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import pandas as pd

from realtime_adjustment import five_minute_mw_to_hourly_mwh


GW_API_URL = (
    "https://apis.data.go.kr/B552115/JejuSukub5mToday/"
    "getJejuSukub5mToday"
)
LEGACY_API_URL = (
    "https://openapi.kpx.or.kr/openapi/chejusukub5mToday/"
    "getChejuSukub5mToday"
)
# 기존 코드와 테스트에서 참조할 수 있도록 신규 GW 주소를 기본 API로 유지한다.
API_URL = GW_API_URL
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


class JejuGridNoDataError(JejuGridApiError):
    """연결과 인증은 성공했지만 요청일 관측값이 0건인 경우."""


ERROR_GUIDANCE = {
    "01": "공공데이터포털 내부 오류입니다. 잠시 후 다시 시도하세요.",
    "04": "호출 주소 또는 요청 방식이 허용되지 않았습니다.",
    "05": "공공데이터포털 또는 제공기관 응답이 지연되고 있습니다.",
    "10": "요청 날짜나 페이지 파라미터 형식이 올바르지 않습니다.",
    "11": "필수 요청값인 기준일(baseDate)이 빠졌습니다.",
    "12": "요청한 API 서비스 주소가 변경됐거나 존재하지 않습니다.",
    "20": "이 API의 활용신청·승인 상태와 Streamlit Secrets의 인증키를 확인하세요.",
    "22": "오늘의 API 호출한도를 초과했습니다. 한도 초기화 후 다시 시도하세요.",
    "23": "짧은 시간에 호출이 몰렸습니다. 잠시 후 다시 시도하세요.",
    "29": "배포 서버의 접속 IP가 공공데이터포털에서 차단됐습니다.",
    "30": "등록되지 않은 인증키입니다. 해당 API에 발급된 키인지 확인하세요.",
    "31": "인증키 사용기간이 만료됐습니다. 연장 승인과 적용 상태를 확인하세요.",
}


def _api_error_message(result_code: str, result_message: str) -> str:
    code = str(result_code).strip()
    official = str(result_message).strip() or "사유 없음"
    upper = official.upper()
    if "DEADLINE_HAS_EXPIRED" in upper:
        code = "31"
    elif "SERVICE_KEY_IS_NOT_REGISTERED" in upper:
        code = "30"
    elif "LIMITED_NUMBER" in upper and "PER_SECOND" in upper:
        code = "23"
    elif "LIMITED_NUMBER" in upper:
        code = "22"
    elif "PERMISSION_DENIED" in upper or "SERVICE_ACCESS_DENIED" in upper:
        code = "20"
    guidance = ERROR_GUIDANCE.get(
        code,
        "공식 응답의 오류코드와 공공데이터포털 활용신청 상태를 확인하세요.",
    )
    return f"제주 실시간 API 오류({code}): {guidance} [공식 응답: {official}]"


def build_request_url(
    service_key: str,
    base_date: str | None = None,
    page_no: int = 1,
    num_of_rows: int = 300,
    include_base_date: bool = True,
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
    query_parameters: dict[str, str | int] = {
        "serviceKey": key,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "dataType": "json",
    }
    if include_base_date:
        if base_date is None:
            base_date = datetime.now(KOREA_TIMEZONE).strftime("%Y%m%d")
        if len(base_date) != 8 or not base_date.isdigit():
            raise ValueError("기준일은 YYYYMMDD 형식이어야 합니다.")
        query_parameters["baseDate"] = base_date
    query = urlencode(query_parameters)
    return f"{API_URL}?{query}"


def build_legacy_request_url(
    service_key: str,
    page_no: int = 1,
    num_of_rows: int = 300,
) -> str:
    """GW 전환 전 KPX XML API의 오늘 자료 요청 URL을 만든다.

    구형 API도 공공데이터포털 일반인증키를 사용하며 날짜 파라미터 없이
    오늘 5분 자료만 제공한다. 인코딩키·디코딩키 어느 쪽도 안전하게 받는다.
    """
    key = unquote(service_key.strip())
    if not key:
        raise ValueError("공공데이터포털 인증키가 비어 있습니다.")
    if page_no < 1 or num_of_rows < 1:
        raise ValueError("페이지와 행 수는 1 이상이어야 합니다.")
    query = urlencode(
        {
            "serviceKey": key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
        }
    )
    return f"{LEGACY_API_URL}?{query}"


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
        # 공공데이터포털 게이트웨이 오류는 정상 API와 다른 XML 구조를 쓴다.
        # 인증키 원문은 절대 오류문에 넣지 않고, 공식 오류코드와 사유만 옮긴다.
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
            _api_error_message(result_code, result_message)
        )

    body = response.get("body", {}) or {}
    items_container = body.get("items", []) or []
    if isinstance(items_container, dict):
        items = items_container.get("item", []) or []
    elif isinstance(items_container, list):
        # GW 전환 응답은 items가 곧 배열로 오는 형식도 지원한다.
        items = items_container
    else:
        items = []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        total_count = body.get("totalCount", 0)
        raise JejuGridNoDataError(
            "KPX API 연결과 인증은 성공했지만 요청일 관측값이 0건입니다 "
            f"(totalCount={total_count}). 제공기관의 오늘 자료 등록 상태를 확인하세요."
        )

    frame = pd.DataFrame(items).rename(columns=API_TO_CANONICAL)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise JejuGridApiError(
            f"제주 실시간 API 응답에 필요한 열이 없습니다: {sorted(missing)}"
        )
    frame = frame[REQUIRED_COLUMNS].copy()
    # 실제 GW 응답은 초까지 포함한 14자리 YYYYMMDDHHMMSS를 보내기도 한다.
    # 기존 12자리 YYYYMMDDHHMM도 지원하고 5분 집계에 불필요한 초는 제거한다.
    timestamp_digits = (
        frame["timestamp"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )
    timestamp_text = timestamp_digits.str.slice(0, 12)
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
    """오늘의 제주 5분 관측값을 호출하고 실패 원인을 구분한다.

    실제 GW는 문서의 선택 표기와 달리 기준일을 요구하므로 한국시간 오늘
    날짜를 명시해 요청한다. GW가 0건이거나 일시적으로 실패하면, 같은
    태양광·풍력 필드를 제공하는 KPX의
    구형 공식 XML 주소를 한 번만 대체 호출한다. 이 방식은 호출당 최대 2회라
    앱의 60분 오류 캐시와 함께 개발계정 일 100회 한도를 넘지 않는다.

    사용자가 과거 날짜를 명시한 진단 호출은 과거 조회가 불가능한 구형
    `Today` 주소로 대체하지 않고 신규 GW에 그 날짜만 한 번 요청한다.
    """
    if timeout_seconds <= 0:
        raise ValueError("API 제한시간은 0보다 커야 합니다.")
    if base_date is None:
        requests = [
            ("신규 GW", build_request_url(service_key, include_base_date=True)),
            ("구형 KPX XML", build_legacy_request_url(service_key)),
        ]
    else:
        requests = [("신규 GW", build_request_url(service_key, base_date=base_date))]

    def request_frame(request_url: str) -> pd.DataFrame:
        request = Request(
            request_url, headers={"User-Agent": "JejuGreenFlexAI/1.0"}
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
        except HTTPError as error:
            # 게이트웨이가 HTTP 오류와 함께 공식 XML/JSON 사유를 보내면 우선 해석한다.
            try:
                payload = error.read()
                if payload:
                    return parse_api_response(payload)
            except JejuGridApiError:
                raise
            except Exception:
                pass
            raise JejuGridApiError(
                f"제주 실시간 API HTTP 오류({error.code})입니다. "
                "호출 주소와 공공데이터포털 서비스 상태를 확인하세요."
            ) from error
        except (TimeoutError, socket.timeout) as error:
            raise JejuGridApiError(
                f"제주 실시간 API가 {timeout_seconds}초 안에 응답하지 않았습니다. "
                "잠시 후 다시 시도하세요."
            ) from error
        except URLError as error:
            reason = getattr(error, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise JejuGridApiError(
                    f"제주 실시간 API가 {timeout_seconds}초 안에 응답하지 않았습니다. "
                    "잠시 후 다시 시도하세요."
                ) from error
            raise JejuGridApiError(
                "Streamlit 서버에서 공공데이터포털에 연결하지 못했습니다. "
                "배포 서버 네트워크와 공공데이터포털 상태를 확인하세요."
            ) from error
        except OSError as error:
            raise JejuGridApiError(
                "Streamlit 서버의 네트워크 오류로 제주 실시간 API를 호출하지 못했습니다."
            ) from error
        return parse_api_response(payload)

    failures: list[tuple[str, JejuGridApiError]] = []
    for source_name, request_url in requests:
        try:
            frame = request_frame(request_url)
            frame.attrs["api_source"] = source_name
            return frame
        except JejuGridApiError as error:
            failures.append((source_name, error))

    if len(requests) > 1 and all(
        isinstance(error, JejuGridNoDataError) for _, error in failures
    ):
        raise JejuGridNoDataError(
            "신규 GW와 구형 KPX XML API의 오늘 관측값이 모두 0건입니다. "
            "제공기관의 오늘 자료 등록 상태를 확인하세요."
        ) from failures[-1][1]
    if len(requests) > 1:
        failure_summary = " / ".join(
            f"{source}: {error}" for source, error in failures
        )
        raise JejuGridApiError(
            "신규 GW와 구형 KPX XML 대체 경로가 모두 실패했습니다. "
            "구형 API도 별도 활용신청 대상인지 확인하세요. "
            f"[{failure_summary}]"
        ) from failures[-1][1]
    raise failures[-1][1]


def grid_samples_to_hourly(samples: pd.DataFrame) -> pd.DataFrame:
    """5분 실측을 모델과 같은 시간별 재생에너지·전력수요 MWh로 바꾼다."""
    required = {"timestamp", *NUMERIC_COLUMNS}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"시간 변환에 필요한 열이 없습니다: {sorted(missing)}")

    # 태양광 + 풍력 실측을 시간별 MWh로 변환
    energy = five_minute_mw_to_hourly_mwh(
        samples,
        power_columns=("solar_mw", "wind_mw"),
    ).rename(columns={"renewable_mwh": "actual_renewable_mwh"})

    # 송전단 수요 실측을 시간별 MWh로 변환
    demand_energy = five_minute_mw_to_hourly_mwh(
        samples,
        power_columns=("demand_transmission_mw",),
    )[["timestamp", "demand_transmission_mwh"]].rename(
        columns={"demand_transmission_mwh": "actual_demand_mwh"}
    )

    energy = energy.merge(
        demand_energy,
        on="timestamp",
        how="left",
    )

    # 화면/상태 표시용 시간별 평균값 유지
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

    return energy.merge(context, on="timestamp", how="left")
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
