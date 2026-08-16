"""발전량과 날씨 데이터를 한 시간 단위 학습 데이터로 합친다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GEN_DIR = RAW / "generation"
OUTPUT = ROOT / "data" / "processed" / "train.csv"
DEMO_OUTPUT = ROOT / "data" / "demo" / "demo_forecast.csv"


def read_korean_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"인코딩 확인 필요: {path}")


def generation_to_long(path: Path, value_name: str) -> pd.DataFrame:
    df = read_korean_csv(path)
    date_col = df.columns[0]
    hour_cols = [col for col in df.columns if str(col).endswith("시")]
    long_df = df.melt(
        id_vars=[date_col],
        value_vars=hour_cols,
        var_name="hour_ending",
        value_name=value_name,
    )
    long_df["hour"] = (
        long_df["hour_ending"].str.replace("시", "", regex=False).astype(int) - 1
    )
    long_df["timestamp"] = pd.to_datetime(long_df[date_col]) + pd.to_timedelta(
        long_df["hour"], unit="h"
    )
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    return long_df[["timestamp", value_name]].sort_values("timestamp")


def read_weather() -> pd.DataFrame:
    path = RAW / "weather_jeju_2025.csv"
    df = pd.read_csv(path, skiprows=3)
    df = df.rename(
        columns={
            "time": "timestamp",
            "temperature_2m (°C)": "temperature_2m",
            "relative_humidity_2m (%)": "relative_humidity_2m",
            "wind_speed_10m (km/h)": "wind_speed_10m",
            "shortwave_radiation (W/m²)": "shortwave_radiation",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def prepare() -> pd.DataFrame:
    solar = generation_to_long(
        GEN_DIR / "25년 제주지역 태양광 시간대별 발전량.csv", "solar_mwh"
    )
    wind = generation_to_long(
        GEN_DIR / "25년 제주지역 풍력 시간대별 발전량.csv", "wind_mwh"
    )
    lng = generation_to_long(
        GEN_DIR / "25년 제주지역 LNG 시간대별 발전량.csv", "lng_mwh"
    )
    bio = generation_to_long(
        GEN_DIR / "25년 제주지역 바이오중유 시간대별 발전량.csv", "bio_mwh"
    )
    weather = read_weather()

    df = solar.merge(wind, on="timestamp", how="inner")
    df = df.merge(lng, on="timestamp", how="inner")
    df = df.merge(bio, on="timestamp", how="inner")
    df = df.merge(weather, on="timestamp", how="inner")
    df["renewable_mwh"] = df["solar_mwh"] + df["wind_mwh"]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    # 발표용 24시간 예시는 2025-08-10을 사용한다.
    demo = df[df["timestamp"].dt.date == pd.Timestamp("2025-08-10").date()].copy()
    DEMO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(DEMO_OUTPUT, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    result = prepare()
    print(f"저장 완료: {OUTPUT}")
    print(f"행 수: {len(result):,}, 열 수: {len(result.columns)}")
    print(result.head())
