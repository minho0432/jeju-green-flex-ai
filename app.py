"""JEJU Green Time — 대시보드 시각화 포함."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
KST = ZoneInfo("Asia/Seoul")
AUTO_REFRESH_MS = 10 * 60 * 1000

st.set_page_config(
    page_title="JEJU Green Time",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

_refresh_count = 0
try:
    from streamlit_autorefresh import st_autorefresh

    _refresh_count = st_autorefresh(interval=AUTO_REFRESH_MS, key="jeju_gt_auto")
except Exception:
    pass

st.markdown(
    """
<style>
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', sans-serif;
}
.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 3rem !important;
  max-width: 720px !important;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.hero { text-align: center; padding: 0.35rem 0 0.85rem 0; }
.hero-badge {
  display: inline-block; background: #ecfdf5; color: #047857;
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
  padding: 0.35rem 0.75rem; border-radius: 999px; margin-bottom: 0.6rem;
}
.hero h1 {
  font-size: 1.55rem !important; font-weight: 700 !important;
  color: #0f172a !important; margin: 0 0 0.3rem 0 !important;
}
.hero p { color: #64748b; font-size: 0.92rem; margin: 0; }
.rec-card {
  background: linear-gradient(160deg, #0f766e 0%, #0d9488 45%, #14b8a6 100%);
  border-radius: 22px; padding: 1.35rem 1.25rem; color: #fff;
  box-shadow: 0 12px 36px rgba(15, 118, 110, 0.28);
  margin: 0.4rem 0 1rem 0;
}
.rec-card .label { font-size: 0.78rem; opacity: 0.9; margin-bottom: 0.3rem; }
.rec-card .time { font-size: 1.85rem; font-weight: 700; margin-bottom: 0.4rem; }
.rec-card .sub { font-size: 0.88rem; opacity: 0.92; }
.rec-card .chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.85rem; }
.rec-card .chip {
  background: rgba(255,255,255,0.18); border-radius: 999px;
  padding: 0.3rem 0.65rem; font-size: 0.76rem; font-weight: 600;
}
.section-title {
  font-size: 0.9rem; font-weight: 700; color: #0f172a; margin: 1.1rem 0 0.5rem 0;
}
div[data-testid="stMetric"] {
  background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 0.75rem 0.9rem;
}
.status-bar {
  display: flex; flex-wrap: wrap; gap: 0.35rem; justify-content: center; margin: 0 0 0.5rem 0;
}
.status-chip {
  font-size: 0.7rem; font-weight: 600; padding: 0.28rem 0.6rem; border-radius: 999px;
}
.status-on { background: #d1fae5; color: #065f46; }
.status-off { background: #f1f5f9; color: #64748b; }
.live-line { text-align: center; color: #64748b; font-size: 0.75rem; margin-bottom: 0.6rem; }
.warning-box {
  background: #fff7ed; border: 1px solid #fed7aa; border-radius: 14px;
  padding: 0.85rem 1rem; color: #9a3412; font-size: 0.88rem; margin: 0.6rem 0;
}
.footer-note {
  text-align: center; color: #94a3b8; font-size: 0.72rem; margin-top: 1.75rem;
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
  <p>재생에너지가 많은 시간에 맞춰 충전 구간을 알려 드려요</p>
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


def now_kst() -> datetime:
    return datetime.now(KST)


@st.cache_resource
def load_models():
    return train_forecast_only_models()


@st.cache_data(ttl=600, show_spinner=False)
def load_weather(day_offset: int, _bucket: int):
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


def chart_green_score(plot_df, used, score_col):
    used_set = set(used["timestamp"]) if not used.empty else set()
    colors = ["#0f766e" if t in used_set else "#99f6e4" for t in plot_df["timestamp"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["timestamp"],
            y=plot_df[score_col],
            marker_color=colors,
            hovertemplate="%{x|%H시}<br>점수 %{y:.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#94a3b8", annotation_text="70점", annotation_position="right")
    fig.update_layout(
        height=260,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickformat="%H시"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 100], title="점수"),
        showlegend=False,
    )
    return fig


def chart_renewable_weather(plot_df, used):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "predicted_renewable_mwh" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["predicted_renewable_mwh"],
                mode="lines+markers",
                name="예측 재생 (MWh)",
                line=dict(color="#059669", width=3),
            ),
            secondary_y=False,
        )
        if "predicted_renewable_lower" in plot_df.columns and "predicted_renewable_upper" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(plot_df["timestamp"]) + list(plot_df["timestamp"][::-1]),
                    y=list(plot_df["predicted_renewable_upper"])
                    + list(plot_df["predicted_renewable_lower"][::-1]),
                    fill="toself",
                    fillcolor="rgba(16, 185, 129, 0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="예측 구간",
                    hoverinfo="skip",
                ),
                secondary_y=False,
            )
    if "shortwave_radiation" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["shortwave_radiation"],
                mode="lines",
                name="일사량 (W/m²)",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            ),
            secondary_y=True,
        )
    if not used.empty:
        for _, row in used.iterrows():
            fig.add_vrect(
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=1),
                fillcolor="rgba(15, 118, 110, 0.12)",
                line_width=0,
            )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showgrid=False, tickformat="%H시"),
    )
    fig.update_yaxes(title_text="재생 MWh", secondary_y=False, gridcolor="#f1f5f9")
    fig.update_yaxes(title_text="일사량", secondary_y=True, showgrid=False)
    return fig


def chart_charge_schedule(used):
    fig = go.Figure(
        go.Bar(
            x=used["timestamp"],
            y=used["scheduled_kwh"],
            marker_color="#0d9488",
            text=used["scheduled_kwh"].round(1),
            textposition="outside",
            hovertemplate="%{x|%H시}<br>%{y:.1f} kWh<extra></extra>",
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickformat="%H시", showgrid=False),
        yaxis=dict(title="kWh", gridcolor="#f1f5f9"),
        showlegend=False,
    )
    return fig


def chart_soc_progress(current, target, reached):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=reached,
            delta={"reference": current, "increasing": {"color": "#059669"}},
            number={"suffix": "%"},
            title={"text": "예상 도착 배터리"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [0, current], "color": "#e2e8f0"},
                    {"range": [current, target], "color": "#ccfbf1"},
                ],
                "threshold": {
                    "line": {"color": "#f59e0b", "width": 3},
                    "thickness": 0.8,
                    "value": target,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


_now = now_kst()
_bucket = int(_now.timestamp() // 600) + int(_refresh_count)
_kpx_ok = bool(get_service_key())
st.markdown(
    f"""
<div class="status-bar">
  <span class="status-chip status-on">날씨 예보 연동됨</span>
  <span class="status-chip {"status-on" if _kpx_ok else "status-off"}">
    {"제주 실측 연동됨" if _kpx_ok else "제주 실측 미연결"}
  </span>
  <span class="status-chip status-on">10분 자동 갱신</span>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="live-line">{_now.strftime("%Y-%m-%d %H:%M")} KST · 갱신 #{_refresh_count}</p>',
    unsafe_allow_html=True,
)

day_choice = st.radio("기준 일", ["오늘", "내일", "데모(과거)"], horizontal=True, label_visibility="collapsed")

with st.expander("내 차 · 일정", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        current_soc = st.slider("지금 배터리", 5, 90, 30, 5, format="%d%%")
        battery_kwh = st.number_input("배터리 용량 (kWh)", 20.0, 120.0, 60.0, 1.0)
        start_hour = st.slider("충전 시작 가능", 0, 22, 8, format="%d시")
    with c2:
        target_soc = st.slider("목표 배터리", 10, 100, 80, 5, format="%d%%")
        charger_kw = st.selectbox("충전기", [3.0, 7.0, 11.0, 50.0], index=1, format_func=lambda x: f"{x:g} kW")
        departure_hour = st.slider("출발 시각", 1, 24, 20, format="%d시")
    efficiency_pct = st.slider("충전 효율", 70, 100, 90, format="%d%%")

effective_start = int(start_hour)
current_hour = _now.hour
if day_choice == "오늘":
    effective_start = max(effective_start, current_hour)
    if effective_start >= departure_hour:
        st.markdown(
            '<div class="warning-box">오늘 남은 충전 가능 시간이 거의 없어요. 출발 시각을 늘리거나 내일을 선택해 보세요.</div>',
            unsafe_allow_html=True,
        )

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
        with st.spinner("날씨·예측 준비 중…"):
            weather = load_weather(offset, _bucket)
            models, history = load_models()
            forecast = ensure_scores(build_live_prediction(models, history, weather))
        peak = weather.loc[weather["shortwave_radiation"].idxmax()]
        weather_note = f"제주 예보 · 일사 최대 {peak['timestamp']:%H시}"
except Exception as err:
    st.markdown(f'<div class="warning-box">날씨를 불러오지 못했어요.<br/><small>{err}</small></div>', unsafe_allow_html=True)
    st.stop()

try:
    plan = make_plan(
        forecast=forecast,
        current_soc=float(current_soc),
        target_soc=float(target_soc),
        battery_kwh=float(battery_kwh),
        charger_kw=float(charger_kw),
        efficiency=float(efficiency_pct) / 100.0,
        start_hour=int(effective_start),
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
plot_df = forecast.sort_values("timestamp")
if score_col not in plot_df.columns:
    score_col = "green_score" if "green_score" in plot_df.columns else plot_df.columns[-1]

if used.empty:
    st.markdown('<div class="warning-box">이 일정으로는 추천 구간을 만들기 어려워요.</div>', unsafe_allow_html=True)
else:
    start_ts = used["timestamp"].iloc[0]
    avg_score = float((used["scheduled_kwh"] * used[score_col]).sum() / used["scheduled_kwh"].sum())
    hours = sorted(used["timestamp"].dt.hour.unique().tolist())
    time_label = f"{hours[0]}시" if len(hours) == 1 else f"{hours[0]}시 – {hours[-1] + 1}시"
    st.markdown(
        f"""
<div class="rec-card">
  <div class="label">추천 충전 시간</div>
  <div class="time">{time_label}</div>
  <div class="sub">{start_ts:%m월 %d일} · {weather_note}</div>
  <div class="chips">
    <span class="chip">도착 약 {plan['reached_soc']:.0f}%</span>
    <span class="chip">{plan['required_grid_kwh']:.0f} kWh</span>
    <span class="chip">친환경 {avg_score:.0f}점</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("시작(적용)", f"{effective_start}시")
k2.metric("출발", f"{departure_hour}시")
k3.metric("충전기", f"{charger_kw:g} kW")
k4.metric("최고 점수", f"{float(plot_df[score_col].max()):.0f}")

st.markdown('<p class="section-title">📊 Green 점수 · 추천 구간</p>', unsafe_allow_html=True)
st.plotly_chart(chart_green_score(plot_df, used, score_col), use_container_width=True, config={"displayModeBar": False})
st.caption("진한 초록 = 추천 충전 시간 · 점선 = 70점 참고선")

st.markdown('<p class="section-title">☀️ 예측 재생에너지 · 일사량</p>', unsafe_allow_html=True)
st.plotly_chart(chart_renewable_weather(plot_df, used), use_container_width=True, config={"displayModeBar": False})
st.caption("초록 밴드 = 예측 구간 · 배경 음영 = 추천 충전 시간")

d1, d2 = st.columns(2)
with d1:
    st.markdown('<p class="section-title">🔋 예상 배터리</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_soc_progress(current_soc, target_soc, plan["reached_soc"]), use_container_width=True, config={"displayModeBar": False})
with d2:
    st.markdown('<p class="section-title">⚡ 시간별 충전량</p>', unsafe_allow_html=True)
    if not used.empty:
        st.plotly_chart(chart_charge_schedule(used), use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("추천 구간이 없습니다.")

if st.button("지금 새로고침", use_container_width=True):
    load_weather.clear()
    st.rerun()

with st.expander("표로 보기"):
    if not used.empty:
        show = used.copy()
        show["시각"] = show["timestamp"].dt.strftime("%H:%M")
        cols = ["시각", "scheduled_kwh", score_col]
        if "predicted_renewable_mwh" in show.columns:
            cols.append("predicted_renewable_mwh")
        st.dataframe(
            show[cols].rename(columns={"scheduled_kwh": "충전 kWh", score_col: "점수", "predicted_renewable_mwh": "예측 재생 MWh"}),
            hide_index=True,
            use_container_width=True,
        )
    view = plot_df.copy()
    view["시각"] = view["timestamp"].dt.strftime("%H:%M")
    show_cols = ["시각", score_col]
    for c in ("predicted_renewable_mwh", "shortwave_radiation", "temperature_2m"):
        if c in view.columns:
            show_cols.append(c)
    st.dataframe(view[show_cols], hide_index=True, use_container_width=True)

st.markdown('<p class="footer-note">JEJU Green Time · 추천은 참고용입니다</p>', unsafe_allow_html=True)
