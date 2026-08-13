"""JEJU Green Time — 날씨 연동 + 연속 충전 구간 최적화."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from live_forecast import (  # noqa: E402
    build_live_prediction,
    fetch_open_meteo_forecast,
    train_forecast_only_models,
)
from optimizer import make_plan  # noqa: E402

FORECAST_PATH = ROOT / "outputs" / "demo_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
HISTORY_PATH = ROOT / "data" / "processed" / "train.csv"

st.set_page_config(page_title="JEJU Green Time", page_icon="🌿", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; max-width: 1100px; }
div[data-testid="stMetric"] {
  background: #f3faf6; border: 1px solid #d8efe3; border-radius: 16px; padding: 12px 16px;
}
.gt-hero {
  background: linear-gradient(135deg, #0f766e 0%, #14b8a6 55%, #5eead4 100%);
  color: white; border-radius: 20px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}
.gt-hero h2 { color: white !important; margin: 0 0 0.35rem 0; font-size: 1.45rem; }
.gt-hero p { margin: 0; opacity: 0.95; }
.gt-pill { display: inline-block; padding: 0.22rem 0.65rem; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; margin-right: 0.3rem; }
.gt-pill-ok { background: #d1fae5; color: #065f46; }
.gt-pill-off { background: #f3f4f6; color: #4b5563; }
section[data-testid="stSidebar"] { background: #f8fafc; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="gt-hero">
  <h2>🌿 JEJU Green Time</h2>
  <p>실제 날씨 예보 + 연속 충전 구간 최적화 · 날·일정마다 추천이 달라집니다</p>
</div>
""",
    unsafe_allow_html=True,
)


def get_data_go_kr_service_key() -> str:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("DATA_GO_KR_SERVICE_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource
def load_forecast_only_models():
    return train_forecast_only_models()


@st.cache_data(ttl=1800, show_spinner="오늘 제주 날씨를 가져오는 중…")
def load_weather(day_offset: int):
    return fetch_open_meteo_forecast(target_day_offset=day_offset)


@st.cache_data
def load_demo_forecast():
    return pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])


def ensure_score_columns(forecast: pd.DataFrame) -> pd.DataFrame:
    out = forecast.copy()
    if "green_score" not in out.columns and "renewable_opportunity_score" in out.columns:
        out["green_score"] = out["renewable_opportunity_score"]
    if "planning_score" not in out.columns:
        if "conservative_renewable_score" in out.columns:
            out["planning_score"] = out["conservative_renewable_score"]
        elif "green_score" in out.columns:
            out["planning_score"] = out["green_score"]
    if "predicted_smp" not in out.columns:
        out["predicted_smp"] = 100.0
    if "predicted_smp_upper" not in out.columns:
        out["predicted_smp_upper"] = out["predicted_smp"] * 1.1
    return out


with st.sidebar:
    st.header("① 어떤 날을 볼까요")
    mode = st.radio(
        "데이터 기준",
        [
            "오늘 실제 날씨 예보",
            "내일 실제 날씨 예보",
            "검증용 과거 하루 (고정 데모)",
        ],
        index=0,
        help="오늘/내일은 Open-Meteo 실시간 예보로 매일 추천이 바뀝니다.",
    )
    st.header("② 내 차 · 일정")
    current_soc = st.slider("지금 배터리 (%)", 5, 90, 30, 5)
    target_soc = st.slider("출발 때 목표 (%)", 10, 100, 80, 5)
    battery_kwh = st.number_input("배터리 용량 (kWh)", 20.0, 120.0, 60.0, 1.0)
    charger_kw = st.selectbox("충전기 출력 (kW)", [3.0, 7.0, 11.0, 50.0], index=1)
    efficiency_pct = st.slider("충전 효율 (%)", 70, 100, 90)
    start_hour = st.slider("충전 가능 시작", 0, 22, 8)
    departure_hour = st.slider("출발 시각", 1, 24, 20)
    continuous = st.checkbox("한 번에 이어서 충전 (권장)", value=True)

efficiency = efficiency_pct / 100.0

forecast = None
weather_note = ""

if mode.startswith("검증용"):
    if not FORECAST_PATH.exists():
        st.error("데모 예측 파일이 없습니다.")
        st.stop()
    forecast = ensure_score_columns(load_demo_forecast())
    weather_note = "고정된 과거 검증일 (날씨가 바뀌지 않음)"
else:
    day_offset = 0 if mode.startswith("오늘") else 1
    try:
        with st.spinner("제주 날씨 예보 수신 중…"):
            weather = load_weather(day_offset)
            models, history = load_forecast_only_models()
            forecast = build_live_prediction(models, history, weather)
            forecast = ensure_score_columns(forecast)
        peak = weather.loc[weather["shortwave_radiation"].idxmax()]
        weather_note = (
            f"Open-Meteo 실예보 · 일사량 최대 "
            f"{peak['timestamp']:%m/%d %H시} ({peak['shortwave_radiation']:.0f} W/m²)"
        )
    except Exception as error:
        st.error(f"날씨/예측을 가져오지 못했습니다: {error}")
        st.info("「검증용 과거 하루」로 전환하거나 잠시 후 다시 시도하세요.")
        st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**날씨**")
    st.markdown(
        '<span class="gt-pill gt-pill-ok">Open-Meteo</span>'
        if not mode.startswith("검증")
        else '<span class="gt-pill gt-pill-off">데모 파일</span>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown("**추천 엔진**")
    st.markdown('<span class="gt-pill gt-pill-ok">연속 구간 최적화</span>', unsafe_allow_html=True)
with c3:
    st.markdown("**KPX 실측 키**")
    st.markdown(
        '<span class="gt-pill gt-pill-ok">등록됨</span>'
        if get_data_go_kr_service_key()
        else '<span class="gt-pill gt-pill-off">없음 (예보만 사용)</span>',
        unsafe_allow_html=True,
    )

st.caption(weather_note)

try:
    plan = make_plan(
        forecast=forecast,
        current_soc=float(current_soc),
        target_soc=float(target_soc),
        battery_kwh=float(battery_kwh),
        charger_kw=float(charger_kw),
        efficiency=float(efficiency),
        start_hour=int(start_hour),
        departure_hour=int(departure_hour),
        retail_price=250.0,
        continuous=continuous,
        conservative=True,
    )
except ValueError as error:
    st.warning(str(error))
    st.stop()

schedule = plan["ai_schedule"]
used = schedule[schedule["scheduled_kwh"] > 1e-6].sort_values("timestamp")

st.subheader("✨ 추천 충전 구간")
if used.empty:
    st.error("가능한 충전 구간이 없습니다. 시작·출발 시각이나 목표 배터리를 조정해 보세요.")
else:
    start_ts = used["timestamp"].iloc[0]
    end_ts = used["timestamp"].iloc[-1]
    hours = sorted(used["timestamp"].dt.hour.tolist())
    hour_txt = ", ".join(f"{h}시" for h in hours)
    score_col = plan["score_column"]
    avg_score = float(
        (used["scheduled_kwh"] * used[score_col]).sum() / used["scheduled_kwh"].sum()
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("추천 시작", f"{start_ts:%H시}")
    m2.metric("추천 종료", f"{end_ts:%H시}")
    m3.metric("채울 전기", f"{plan['required_grid_kwh']:.1f} kWh")
    m4.metric("예상 도착 SOC", f"{plan['reached_soc']:.0f}%")

    if plan["feasible"]:
        st.success(
            f"**{start_ts:%m/%d %H시} ~ {end_ts:%H시}** 에 이어서 충전하세요. "
            f"선택 시각: {hour_txt} · 구간 평균 Green 점수 **{avg_score:.0f}**"
        )
    else:
        st.warning(
            f"시간·출력만으로는 목표 SOC에 못 미칠 수 있습니다. "
            f"가능 시 SOC 약 {plan['reached_soc']:.0f}% · 부족 {plan['remaining_kwh']:.1f} kWh"
        )

score_col = plan.get("score_column") or "green_score"
plot_df = forecast.sort_values("timestamp").copy()
if score_col not in plot_df.columns:
    score_col = "green_score"

fig = go.Figure()
fig.add_trace(
    go.Bar(x=plot_df["timestamp"], y=plot_df[score_col], name="Green 점수", marker_color="#99f6e4")
)
if not used.empty:
    fig.add_trace(
        go.Bar(x=used["timestamp"], y=used[score_col], name="추천 충전 구간", marker_color="#0f766e")
    )
fig.update_layout(
    barmode="overlay",
    height=360,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h"),
    yaxis_title="점수",
    xaxis_title="시각",
)
st.plotly_chart(fig, use_container_width=True)

if "predicted_renewable_mwh" in forecast.columns:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["predicted_renewable_mwh"],
            mode="lines+markers",
            name="예측 재생(MWh)",
            line=dict(color="#059669", width=3),
        )
    )
    if "shortwave_radiation" in plot_df.columns:
        fig2.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["shortwave_radiation"],
                mode="lines",
                name="일사량 (W/m²)",
                yaxis="y2",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            )
        )
        fig2.update_layout(yaxis2=dict(overlaying="y", side="right", title="일사량"))
    fig2.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
        yaxis_title="재생 MWh",
    )
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("추천 구간 상세"):
    if not used.empty:
        show = used.copy()
        show["시각"] = show["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        cols = ["시각", "scheduled_kwh", score_col]
        if "predicted_renewable_mwh" in show.columns:
            cols.append("predicted_renewable_mwh")
        st.dataframe(
            show[cols].rename(
                columns={
                    "scheduled_kwh": "충전 kWh",
                    score_col: "점수",
                    "predicted_renewable_mwh": "예측 재생 MWh",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

with st.expander("왜 날이 바뀌면 추천이 달라지나요?"):
    st.markdown(
        """
1. **Open-Meteo**에서 제주 오늘/내일 **실제 기상예보**를 받습니다.
2. 모델이 예보로 **시간별 재생에너지**를 예측합니다.
3. 과거 분포와 비교해 **Green 점수**를 매깁니다.
4. **시작~출발** 안에서 목표 SOC만큼 **연속 구간**을 고릅니다.

구름 많은 날·맑은 날, 출발을 오전으로 잡는지에 따라 추천이 달라집니다.
「검증용 과거 하루」만 고르면 예전처럼 고정 데모가 나옵니다.
"""
    )

st.caption(
    "JEJU Green Time · 추천은 참고용이며 실제 요금·포인트·충전기 제어를 보장하지 않습니다."
)
