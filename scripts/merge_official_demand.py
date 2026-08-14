"""공식 시간별 제주 전력수요 CSV를 train에 demand_mwh로 병합한다.

출처 예: 공공데이터포털「한국전력거래소_시간별 제주전력수요」
https://www.data.go.kr/data/15065239/fileData.do

지원 형식
1) long: timestamp, demand_mwh (또는 일자, 시간, 수요)
2) wide: 일자 + 1시~24시(또는 01~24) 열

사용:
  python scripts/merge_official_demand.py \\
    --demand-csv data/raw/jeju_hourly_demand.csv \\
    --train data/processed/train.csv \\
    --out data/processed/train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_date_col(columns: list[str]) -> str | None:
    for c in columns:
        cl = str(c).lower()
        if any(k in cl for k in ("일자", "날짜", "date", "ymd", "기준")):
            return c
    return columns[0] if columns else None


def load_demand_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = list(df.columns)

    if "timestamp" in cols and (
        "demand_mwh" in cols or "demand" in cols or "수요" in cols
    ):
        dcol = "demand_mwh" if "demand_mwh" in cols else (
            "demand" if "demand" in cols else "수요"
        )
        out = df[["timestamp", dcol]].copy()
        out = out.rename(columns={dcol: "demand_mwh"})
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        out["demand_mwh"] = pd.to_numeric(out["demand_mwh"], errors="coerce")
        return out.dropna().sort_values("timestamp").drop_duplicates("timestamp")

    date_col = _find_date_col(cols)
    hour_cols = []
    for c in cols:
        if c == date_col:
            continue
        s = str(c).replace("시", "").replace("h", "").replace("H", "").strip()
        if s.isdigit() and 1 <= int(s) <= 24:
            hour_cols.append((c, int(s) % 24))

    if date_col and hour_cols:
        rows = []
        for _, row in df.iterrows():
            base = pd.to_datetime(str(row[date_col]))
            for c, h in hour_cols:
                ts = base + pd.Timedelta(hours=h)
                rows.append(
                    {
                        "timestamp": ts,
                        "demand_mwh": pd.to_numeric(row[c], errors="coerce"),
                    }
                )
        out = pd.DataFrame(rows).dropna()
        return out.sort_values("timestamp").drop_duplicates("timestamp")

    raise ValueError(
        "수요 CSV 형식을 모릅니다. "
        "timestamp+demand_mwh 또는 일자+1~24시 열을 사용해 주세요."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="공식 수요를 train에 병합")
    parser.add_argument("--demand-csv", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/train.csv"))
    args = parser.parse_args()

    demand = load_demand_csv(args.demand_csv)
    train = pd.read_csv(args.train)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    before = train["demand_mwh"].notna().sum() if "demand_mwh" in train.columns else 0
    train = train.drop(columns=["demand_mwh"], errors="ignore")
    merged = train.merge(demand, on="timestamp", how="left")
    matched = int(merged["demand_mwh"].notna().sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(
        f"병합 완료: 수요 매칭 {matched}/{len(merged)}행 "
        f"(이전 demand 유효 {before}) → {args.out}"
    )
    if matched == 0:
        print("경고: 한 행도 매칭되지 않았습니다. 날짜/시간 형식을 확인하세요.")


if __name__ == "__main__":
    main()
