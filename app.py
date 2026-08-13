"""JEJU Green Time — 제주 전기차 친환경 충전 안내."""

from __future__ import annotations

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

st.set_page_config(
    page_title="JEJU Green Time",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', sans-serif;
}
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 480px !important;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.hero { text-align: center; padding: 0.5rem 0 1.25rem 0; }
.hero-badge {
  display: inline-block; background: #ecfdf5; color: #047857;
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
  padding: 0.35rem 0.75rem; border-radius: 999px; margin-bottom: 0.75rem;
}
.hero h1 {
  font-size: 1.65rem !important; font-weight: 700 !important;
  color: #0f172a !important; margin: 0 0 0.35rem 0 !important;
  letter-spacing: -0.02em;
}
.hero p { color: #64748b; font-size: 0.95rem; margin: 0; line-height: 1.45; }

.rec-card {
  background: linear-gradient(160deg, #0f766e 0%, #0d9488 45%, #14b8a6 100%);
  border-radius: 24px; padding: 1.5rem 1.35rem 1.35rem; color: #fff;
  box-shadow: 0 12px 40px rgba(15, 118, 110, 0.28);
  margin: 0.5rem 0 1.25rem 0;
}
.rec-card .label { font-size: 0.8rem; opacity: 0.9; font-weight: 500; margin-bottom: 0.35rem; }
.rec-card .time {
  font-size: 2rem; font-weight: 700; letter-spacing: -0.03em;
  line-height: 1.15; margin-bottom: 0.5rem;
}
.rec-card .sub { font-size: 0.9rem; opacity: 0.92; line-height: 1.4; }
.rec-card .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 1rem; }
.rec-card .chip {
  background: rgba(255,255,255,0.18); border-radius: 999px;
  padding: 0.35rem 0.7rem; font-size: 0.78rem; font-weight: 600;
}

.section-title {
  font-size: 0.85rem; font-weight: 600; color: #475569; margin: 1.25rem 0 0.6rem 0;
}
div[data-testid="stMetric"] {
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 0.85rem 1rem;
}
div[data-testid="stMetric"] label { color: #64748b !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-size: 1.25rem !important; color: #0f172a !important;
}
div[data-testid="stExpander"] {
  border: 1px solid #e2e8f0; border-radius: 16px; overflow: hidden; background: #fff;
}
.warning-box {
  background: #fff7ed; border: 1px solid #fed7aa; border-radius: 16px;
  padding: 0.9rem 1rem; color: #9a3412; font-size: 0.9rem; margin: 0.75rem 0;
}
.footer-note {
  text-align: center; color: #94a3b8; font-size: 0.75rem;
  margin-top: 2rem; line-height: 1.5;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-badge">JEJU · GREEN TIME</div>
  <h1>언제 충전할까요?</h1>
  <p>재생에너지가 많은 시간에 맞춰<br/>충전 구간을 알려 드려요</p>
</div>
""",
    unsafe_allow_html=True,
)


def get_service_key() -> str:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("DATA_GO_KR_SERVICE_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource
def load_models():
    return train_forecast_only_models()


@st.cache_data(ttl=1800, show_spinner=False)
def load_weather(day_offset: int):
    return fetch_open_meteo_forecast(target_day_offset=day_offset)


@st.cache_data
def load_demo():
    return pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])


def ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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


day_choice = st.radio(
    "언제 기준인가요?",
    ["오늘", "내일", "데모(과거)"],
    horizontal=True,
    label_visibility="collapsed",
)
st.caption("오늘 · 내일은 제주 실제 날씨 예보를 사용합니다")

with st.expander("내 차 정보 입력", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        current_soc = st.slider("지금 배터리", 5, 90, 30, 5, format="%d%%")
        battery_kwh = st.number_input("배터리 용량", 20.0, 120.0, 60.0, 1.0, help="kWh")
        start_hour = st.slider("충전 시작 가능", 0, 22, 8, format="%d시")
    with c2:
        target_soc = st.slider("목표 배터리", 10, 100, 80, 5, format="%d%%")
        charger_kw = st.selectbox(
            "충전기", [3.0, 7.0, 11.0, 50.0], index=1,
            format_func=lambda x: f"{x:g} kW",
        )
        departure_hour = st.slider("출발 시각", 1, 24, 20, format="%d시")
    efficiency_pct = st.slider("충전 효율", 70, 100, 90, format="%d%%")

weather_note = ""
try:
    if day_choice == "데모(과거)":
        if not FORECAST_PATH.exists():
            st.error("데모 데이터가 없습니다.")
            st.stop()
        forecast = ensure_scores(load_demo())
        weather_note = "과거 검증 데이터"
    else:
        offset = 0 if day_choice == "오늘" else 1
        with st.spinner("날씨와 추천을 준비하는 중…"):
            weather = load_weather(offset)
            models, history = load_models()
            forecast = ensure_scores(build_live_prediction(models, history, weather))
        peak = weather.loc[weather["shortwave_radiation"].idxmax()]
        weather_note = f"제주 예보 · 일사 최대 {peak['timestamp']:%H시}"
except Exception as err:
    st.markdown(
        f'<div class="warning-box">날씨를 불러오지 못했어요. 잠시 후 다시 시도하거나 데모를 선택해 주세요.<br/><small>{err}</small></div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    plan = make_plan(
        forecast=forecast,
        current_soc=float(current_soc),
        target_soc=float(target_soc),
        battery_kwh=float(battery_kwh),
        charger_kw=float(charger_kw),
        efficiency=float(efficiency_pct) / 100.0,
        start_hour=int(start_hour),
        departure_hour=int(departure_hour),
        retail_price=250.0,
        continuous=True,
        conservative=True,
    )
except ValueError as err:
    st.markdown(f'<div class="warning-box">{err}</div>', unsafe_allow_html=True)
    st.stop()

used = plan["ai_schedule"][plan["ai_schedule"]["scheduled_kwh"] > 1e-6].sort_values("timestamp")
score_col = plan.get("score_column") or "green_score"

if used.empty:
    st.markdown(
        '<div class="warning-box">이 일정 안에서는 충전 구간을 만들기 어려워요. 시작·출발 시간이나 목표 배터리를 조정해 보세요.</div>',
        unsafe_allow_html=True,
    )
else:
    start_ts = used["timestamp"].iloc[0]
    end_ts = used["timestamp"].iloc[-1]
    avg_score = float(
        (used["scheduled_kwh"] * used[score_col]).sum() / used["scheduled_kwh"].sum()
    )
    hours = sorted(used["timestamp"].dt.hour.unique().tolist())
    if len(hours) == 1:
        time_label = f"{hours[0]}시"
    else:
        time_label = f"{hours[0]}시 – {hours[-1] + 1}시"

    st.markdown(
        f"""
<div class="rec-card">
  <div class="label">추천 충전 시간</div>
  <div class="time">{time_label}</div>
  <div class="sub">{start_ts:%m월 %d일} · {weather_note}</div>
  <div class="chips">
    <span class="chip">도착 배터리 약 {plan['reached_soc']:.0f}%</span>
    <span class="chip">{plan['required_grid_kwh']:.0f} kWh</span>
    <span class="chip">친환경 {avg_score:.0f}점</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if not plan["feasible"]:
        st.markdown(
            f'<div class="warning-box">이 일정만으로는 목표에 조금 못 미칠 수 있어요. '
            f"가능 시 약 {plan['reached_soc']:.0f}%까지 충전됩니다.</div>",
            unsafe_allow_html=True,
        )

st.markdown('<p class="section-title">오늘의 친환경 흐름</p>', unsafe_allow_html=True)

plot_df = forecast.sort_values("timestamp")
if score_col not in plot_df.columns:
    score_col = "green_score" if "green_score" in plot_df.columns else plot_df.columns[-1]

colors = []
used_set = set(used["timestamp"]) if not used.empty else set()
for _, row in plot_df.iterrows():
    colors.append("#0f766e" if row["timestamp"] in used_set else "#ccfbf1")

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=plot_df["timestamp"],
        y=plot_df[score_col],
        marker_color=colors,
        hovertemplate="%{x|%H시}<br>점수 %{y:.0f}<extra></extra>",
    )
)
fig.update_layout(
    height=220,
    margin=dict(l=8, r=8, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, tickformat="%H시", dtick=1000 * 60 * 60 * 3),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 100], title=None),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption("진한 초록 = 추천 충전 구간")

m1, m2, m3 = st.columns(3)
m1.metric("시작", f"{start_hour}시")
m2.metric("출발", f"{departure_hour}시")
m3.metric("충전기", f"{charger_kw:g} kW")

with st.expander("충전 구간 자세히"):
    if not used.empty:
        show = used.copy()
        show["시각"] = show["timestamp"].dt.strftime("%H:%M")
        st.dataframe(
            show[["시각", "scheduled_kwh", score_col]].rename(
                columns={"scheduled_kwh": "충전량(kWh)", score_col: "점수"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("추천 구간이 없습니다.")

with st.expander("이 서비스가 하는 일"):
    st.markdown(
        """
제주 날씨 예보로 재생에너지가 많을 시간을 예측하고,
배터리·출발 일정에 맞춰 **한 번에 이어서 충전하기 좋은 구간**을 고릅니다.

- 오늘·내일: 실시간 기상 예보
- 데모: 검증용 과거 하루
- 추천은 참고용이며 실제 요금·포인트·충전기 제어와 무관합니다.
"""
    )

st.markdown(
    '<p class="footer-note">JEJU Green Time<br/>친환경 충전 시간을 위한 안내</p>',
    unsafe_allow_html=True,
)
