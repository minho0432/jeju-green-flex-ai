"""2023~2025 제주 재생에너지·전력수요·기상을 시간 단위로 병합합니다.

공식 발전량/수요 파일의 서로 다른 형식과 2023년 수요 단위(kWh)를 정규화해
모델이 바로 사용할 수 있는 ``data/processed/train.csv``를 만듭니다.
"""

from __future__ import annotations

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "train.csv"
CURRENT_TRAIN = OUTPUT


def read_korean_csv(path: Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"CSV 인코딩을 확인할 수 없습니다: {path}")


def clean_name(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).replace(" ", "").lower()


def find_raw_name(fragment: str) -> Path:
    target = unicodedata.normalize("NFC", fragment)
    for path in RAW.iterdir():
        if target in unicodedata.normalize("NFC", path.name):
            return path
    raise FileNotFoundError(f"data/raw에서 {fragment!r} 파일을 찾지 못했습니다.")


def to_timestamp(date: pd.Series, hour_ending: pd.Series) -> pd.Series:
    return pd.to_datetime(date.astype(str)) + pd.to_timedelta(
        pd.to_numeric(hour_ending, errors="raise").astype(int) - 1, unit="h"
    )


def _generation_legacy(path: Path) -> pd.DataFrame:
    frame = read_korean_csv(path)
    normalized = {clean_name(column): column for column in frame.columns}
    date_col = next(c for n, c in normalized.items() if "거래일" in n)
    hour_col = next(c for n, c in normalized.items() if "거래시간" in n)
    region_col = next(c for n, c in normalized.items() if n in {"지역", "지역명"})
    solar_col = next(c for n, c in normalized.items() if "태양광발전량" in n)
    wind_col = next(c for n, c in normalized.items() if "풍력발전량" in n)
    frame = frame[frame[region_col].astype(str).str.strip() == "제주도"].copy()
    frame["timestamp"] = to_timestamp(frame[date_col], frame[hour_col])
    frame["solar_mwh"] = pd.to_numeric(frame[solar_col], errors="coerce")
    frame["wind_mwh"] = pd.to_numeric(frame[wind_col], errors="coerce")
    return frame[["timestamp", "solar_mwh", "wind_mwh"]]


def load_generation_2023() -> pd.DataFrame:
    files = [
        RAW / "kpx_solar_wind_2023_01_02.csv",
        RAW / "kpx_solar_wind_2023_03_05.csv",
        RAW / "kpx_solar_wind_2023_06_08.csv",
        RAW / "kpx_solar_wind_2023.csv",
    ]
    result = pd.concat([_generation_legacy(path) for path in files], ignore_index=True)
    result = result[result["timestamp"].between("2023-01-01", "2023-11-30 23:00")]
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    _validate_complete(result, "2023-01-01", "2023-11-30 23:00", "2023 발전량")
    return result.reset_index(drop=True)


def load_generation_2024() -> pd.DataFrame:
    frame = read_korean_csv(RAW / "kpx_solar_wind_2024.csv")
    frame["timestamp"] = to_timestamp(frame["거래일자"], frame["거래시간"])
    frame["전력거래량(MWh)"] = pd.to_numeric(frame["전력거래량(MWh)"], errors="coerce")
    solar = frame[(frame["지역"] == "제주도") & (frame["연료원"] == "태양광")][
        ["timestamp", "전력거래량(MWh)"]
    ].rename(columns={"전력거래량(MWh)": "solar_mwh"})
    wind = frame[(frame["지역"] == "제주") & (frame["연료원"] == "풍력")][
        ["timestamp", "전력거래량(MWh)"]
    ].rename(columns={"전력거래량(MWh)": "wind_mwh"})
    result = solar.merge(wind, on="timestamp", how="inner").sort_values("timestamp")
    _validate_complete(result, "2024-01-01", "2024-12-31 23:00", "2024 발전량")
    return result.reset_index(drop=True)


def wide_demand(path: Path, divide_by_1000: bool) -> pd.DataFrame:
    frame = read_korean_csv(path)
    date_col = frame.columns[0]
    hour_columns: list[tuple[object, int]] = []
    for column in frame.columns[1:]:
        name = clean_name(column).replace("시", "")
        if name.isdigit() and 1 <= int(name) <= 24:
            hour_columns.append((column, int(name)))
    long = frame.melt(
        id_vars=[date_col],
        value_vars=[column for column, _ in hour_columns],
        var_name="hour_column",
        value_name="demand_mwh",
    )
    hour_map = {clean_name(column): hour for column, hour in hour_columns}
    long["hour_ending"] = long["hour_column"].map(lambda value: hour_map[clean_name(value)])
    long["timestamp"] = to_timestamp(long[date_col], long["hour_ending"])
    long["demand_mwh"] = pd.to_numeric(
        long["demand_mwh"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    if divide_by_1000:
        long["demand_mwh"] = long["demand_mwh"] / 1000.0
    return long[["timestamp", "demand_mwh"]].dropna()


def load_demand() -> pd.DataFrame:
    # 과거 파일은 2023년 9월까지도 kWh 숫자로 저장되어 있어 명시적으로 변환합니다.
    early = wide_demand(RAW / "kpx_demand_through_2023_02.csv", divide_by_1000=True)
    middle = wide_demand(RAW / "kpx_demand_2023_03_09.csv", divide_by_1000=True)
    current = wide_demand(find_raw_name("계통수요.csv"), divide_by_1000=False)
    early = early[early["timestamp"].between("2023-01-01", "2023-02-28 23:00")]
    middle = middle[middle["timestamp"].between("2023-03-01", "2023-08-31 23:00")]
    current = current[current["timestamp"] >= pd.Timestamp("2023-09-01")]
    result = pd.concat([early, middle, current], ignore_index=True)
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if not result["demand_mwh"].between(300, 2_000).all():
        bad = result.loc[~result["demand_mwh"].between(300, 2_000)].head()
        raise ValueError(f"수요 단위 변환 오류가 의심됩니다:\n{bad}")
    return result.reset_index(drop=True)


def load_weather_2023_2024() -> pd.DataFrame:
    frame = pd.read_csv(RAW / "weather_jeju_2023_2024.csv", skiprows=3)
    frame = frame.rename(
        columns={
            "time": "timestamp",
            "temperature_2m (°C)": "temperature_2m",
            "relative_humidity_2m (%)": "relative_humidity_2m",
            "wind_speed_10m (km/h)": "wind_speed_10m",
            "shortwave_radiation (W/m²)": "shortwave_radiation",
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    columns = [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]
    return frame[columns].sort_values("timestamp")


def _validate_complete(frame: pd.DataFrame, start: str, end: str, label: str) -> None:
    expected = pd.date_range(start, end, freq="h")
    actual = pd.DatetimeIndex(frame["timestamp"])
    if len(frame) != len(expected) or not actual.equals(expected):
        missing = expected.difference(actual)[:5].astype(str).tolist()
        raise ValueError(
            f"{label} 시간축이 완전하지 않습니다: {len(frame)}/{len(expected)}, 누락 {missing}"
        )


def build() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_TRAIN, parse_dates=["timestamp"])
    current_2025 = current[current["timestamp"].dt.year == 2025].copy()
    if len(current_2025) != 8760:
        raise ValueError(f"기존 2025 학습자료가 8,760행이 아닙니다: {len(current_2025)}")

    generation = pd.concat([load_generation_2023(), load_generation_2024()], ignore_index=True)
    demand = load_demand()
    weather = load_weather_2023_2024()
    historical = generation.merge(demand, on="timestamp", how="left")
    historical = historical.merge(weather, on="timestamp", how="left")
    historical["renewable_mwh"] = historical["solar_mwh"] + historical["wind_mwh"]
    historical["smp"] = np.nan
    historical["lng_mwh"] = np.nan
    historical["bio_mwh"] = np.nan

    columns = list(current_2025.columns)
    for column in columns:
        if column not in historical:
            historical[column] = np.nan
    # 공식 원본 자체에 비어 있는 발전량은 추정으로 채우지 않고 해당 시간만 제외합니다.
    historical = historical.dropna(
        subset=["solar_mwh", "wind_mwh", "renewable_mwh", "demand_mwh"]
    )
    combined = pd.concat([historical[columns], current_2025[columns]], ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    required = [
        "solar_mwh",
        "wind_mwh",
        "renewable_mwh",
        "demand_mwh",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    ]
    if combined[required].isna().any().any():
        raise ValueError(f"핵심 열 결측값:\n{combined[required].isna().sum()}")
    if combined["timestamp"].duplicated().any():
        raise ValueError("중복 timestamp가 있습니다.")

    expected_rows = 8015 + 8784 + 8760
    if len(combined) != expected_rows:
        raise ValueError(f"예상 행 수 {expected_rows:,}와 다릅니다: {len(combined):,}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    return combined


if __name__ == "__main__":
    result = build()
    print("다년도 학습 데이터 병합 완료")
    print(f"기간: {result['timestamp'].min()} ~ {result['timestamp'].max()}")
    print(f"행 수: {len(result):,} (기존 8,760 + 추가 {len(result) - 8_760:,})")
    print("제외: 2023-01-26 00:00 풍력 원본 결측 1시간(임의 보간 없음)")
    print(result.groupby(result["timestamp"].dt.year).size().to_string())
    print(f"공식 수요 유효: {result['demand_mwh'].notna().sum():,}/{len(result):,}")
    print(f"SMP 유효: {result['smp'].notna().sum():,}/{len(result):,} (Green Score 미사용)")
