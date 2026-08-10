"""KPX EPSIS에서 2025년 제주 시간별 SMP를 내려받아 CSV로 저장한다."""

from __future__ import annotations

import re
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "smp_jeju_2025.csv"
PAGE_URL = "https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202"
DATA_URL = "https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShd.ajax"


def download_smp() -> pd.DataFrame:
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    headers = {"User-Agent": "Mozilla/5.0"}
    with opener.open(Request(PAGE_URL, headers=headers), timeout=60):
        pass

    body = urlencode(
        {
            "beginDate": "20250101",
            "endDate": "20251231",
            "selYear": "",
            "selMonth": "",
            "selKind": "jeju",
            "locale": "",
        }
    ).encode("utf-8")
    request = Request(DATA_URL, data=body, headers=headers, method="POST")
    with opener.open(request, timeout=120) as response:
        text = response.read().decode("utf-8")

    block_pattern = re.compile(
        r"((?:\s*c\d+\s*=\s*textFormmat\(\"[^\"]+\",count\);\s*){27})"
        r"\s*gridData\.push\(\{\"Date\":\"(\d{4}/\d{2}/\d{2})\"",
        re.DOTALL,
    )
    value_pattern = re.compile(
        r"c(\d+)\s*=\s*textFormmat\(\"([^\"]+)\",count\);"
    )

    rows: list[dict[str, object]] = []
    for assignments, date_text in block_pattern.findall(text):
        values = {int(k): float(v) for k, v in value_pattern.findall(assignments)}
        date = pd.to_datetime(date_text, format="%Y/%m/%d")
        for hour_ending in range(1, 25):
            # KPX의 1시는 00:00~01:00 구간이므로 구간 시작시각으로 저장한다.
            timestamp = date + pd.Timedelta(hours=hour_ending - 1)
            rows.append(
                {
                    "timestamp": timestamp,
                    "smp": values[hour_ending],
                }
            )

    if not rows:
        raise RuntimeError("SMP 데이터를 찾지 못했습니다. EPSIS 화면 구조를 확인하세요.")

    df = (
        pd.DataFrame(rows)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    result = download_smp()
    print(f"저장 완료: {OUTPUT}")
    print(f"행 수: {len(result):,}")
    print(result.head())
    print(result.tail())
