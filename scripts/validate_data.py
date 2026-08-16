"""완성된 학습 데이터가 AI 담당자에게 전달 가능한 상태인지 검사한다."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "processed" / "train.csv"
REQUIRED = [
    "timestamp",
    "solar_mwh",
    "wind_mwh",
    "renewable_mwh",
    "demand_mwh",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]


def validate() -> None:
    df = pd.read_csv(PATH, parse_dates=["timestamp"])
    missing_columns = [col for col in REQUIRED if col not in df.columns]
    if missing_columns:
        raise AssertionError(f"필수 열 누락: {missing_columns}")

    if df["timestamp"].duplicated().any():
        raise AssertionError("중복 timestamp가 있습니다.")

    missing = df[REQUIRED].isna().sum()
    if missing.sum() > 0:
        raise AssertionError(f"결측값이 있습니다:\n{missing[missing > 0]}")

    calculated = (df["solar_mwh"] + df["wind_mwh"]).round(6)
    stored = df["renewable_mwh"].round(6)
    if not calculated.equals(stored):
        raise AssertionError("renewable_mwh 계산이 맞지 않습니다.")

    if (df["demand_mwh"] <= 0).any():
        raise AssertionError("demand_mwh에 0 이하 값이 있습니다.")
    if not df["demand_mwh"].between(300, 2_000).all():
        raise AssertionError("demand_mwh가 제주 계통수요의 점검 범위(300~2,000MW)를 벗어났습니다.")

    expected_interval = df["timestamp"].diff().dropna().value_counts().index[0]
    if expected_interval != pd.Timedelta(hours=1):
        raise AssertionError(f"대표 시간 간격이 1시간이 아닙니다: {expected_interval}")

    print("데이터 검사 통과")
    print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"행 수: {len(df):,}")
    print(f"열 수: {len(df.columns)}")
    print(f"공식 수요 유효 행: {df['demand_mwh'].notna().sum():,}/{len(df):,}")
    if len(df) != 25_559:
        raise AssertionError(f"예상한 25,559시간과 다릅니다: {len(df):,}")
    year_counts = df.groupby(df["timestamp"].dt.year).size().to_dict()
    if year_counts != {2023: 8_015, 2024: 8_784, 2025: 8_760}:
        raise AssertionError(f"연도별 행 수가 예상과 다릅니다: {year_counts}")


if __name__ == "__main__":
    validate()
