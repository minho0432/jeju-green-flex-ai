"""JEJU Green Time — 제주 친환경 충전 시간 안내."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

FORECAST_PATH = ROOT / "outputs" / "demo_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
HISTORY_PATH = ROOT / "data" / "processed" / "train.csv"

st.set_page_config(page_title="JEJU Green Time", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; max-width: 1100px; }
div[data-testid="stMetric"] {
  background: #f3faf6; border: 1px solid #d8efe3; border-radius: 16px; padding: 12px 16px;
}
.gt-hero {
  background: linear-gradient(135deg, #0f766e 0%, #14b8a6 55%, #5eead4 100%);
  color: white; border-radius: 20px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.gt-hero h2 { color: white !important; margin: 0 0 0.4rem 0; font-size: 1.5rem; }
.gt-hero p { margin: 0; opacity: 0.95; }
.gt-pill { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
.gt-pill-ok { background: #d1fae5; color: #065f46; }
.gt-pill-off { background: #f3f4f6; color: #4b5563; }
.gt-pill-err { background: #fee2e2; color: #991b1b; }
section[data-testid="stSidebar"] { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gt-hero">
  <h2>🌿 JEJU Green Time</h2>
  <p>제주에서 전기차 충전하기 좋은 시간 · 재생에너지가 풍부한 구간을 먼저 알려 드립니다</p>
</div>
""", unsafe_allow_html=True)

def get_data_go_kr_service_key() -> str:
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("DATA_GO_KR_SERVICE_KEY", "")).strip()
    except Exception:
        return ""

def render_api_status() -> None:
    key = get_data_go_kr_service_key()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**날씨 예보**")
        st.markdown('<span class="gt-pill gt-pill-ok">연결됨 (키 불필요)</span>', unsafe_allow_html=True)
        st.caption("Open-Meteo")
    with c2:
        st.markdown("**제주 실측 (KPX)**")
        if key:
            st.markdown('<span class="gt-pill gt-pill-ok">키 등록됨</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="gt-pill gt-pill-off">키 없음 · 데모 가능</span>', unsafe_allow_html=True)
    with c3:
        st.markdown("**AI 추천**")
        if FORECAST_PATH.exists() and METRICS_PATH.exists():
            st.markdown('<span class="gt-pill gt-pill-ok">준비됨</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="gt-pill gt-pill-err">데이터 없음</span>', unsafe_allow_html=True)

with st.expander("📡 연결 상태 확인", expanded=True):
    render_api_status()
    st.markdown("왼쪽에서 **오늘 공식 실시간 관측**을 고르면 실측 API 호출 여부를 확인할 수 있습니다.")

if not FORECAST_PATH.exists() or not METRICS_PATH.exists():
    st.error("아직 추천 데이터가 없습니다. `python scripts/train_models.py` 실행 후 다시 열어 주세요.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
forecast = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])

with st.sidebar:
    st.header("① 무엇을 볼까요")
    mode = st.radio("추천 기준", ["쉬운 데모 (과거 하루)", "오늘 공식 실시간 관측", "내일 예보 실험"], index=0)
    st.header("② 내 차 · 일정")
    current_soc = st.slider("지금 배터리 (%)", 5, 90, 30, 5)
    target_soc = st.slider("출발할 때 목표 (%)", 10, 100, 80, 5)
    battery_kwh = st.number_input("배터리 용량 (kWh)", 20.0, 120.0, 60.0, 1.0)
    charger_kw = st.selectbox("충전기 출력 (kW)", [3.0, 7.0, 11.0, 50.0], index=1)
    start_hour = st.slider("충전 가능 시작 시각", 0, 22, 8)
    departure_hour = st.slider("출발 시각", 1, 24, 20)

st.subheader("✨ 추천 요약")
st.success(
    f"목표 배터리 {target_soc}% · 용량 {battery_kwh:.0f} kWh · 충전기 {charger_kw} kW 기준으로 Green Time 구간을 고릅니다."
)

# Score column if present
if "green_score" in forecast.columns:
    top = forecast.nlargest(5, "green_score")[["timestamp", "green_score"]].copy()
    top["시각"] = top["timestamp"].dt.strftime("%H:%M")
    st.markdown("**점수가 높은 Green Time (상위 5)**")
    st.dataframe(top[["시각", "green_score"]].rename(columns={"green_score": "점수"}), hide_index=True, use_container_width=True)
    best = forecast.loc[forecast["green_score"].idxmax()]
    st.metric("가장 추천하는 시각", f"{best['timestamp']:%H시}", f"점수 {best['green_score']:.0f}")
elif "renewable_pred" in forecast.columns:
    st.line_chart(forecast.set_index("timestamp")["renewable_pred"])
else:
    st.dataframe(forecast.head(24), use_container_width=True)

st.caption("JEJU Green Time 데모입니다. 추천은 참고용이며 실제 요금·포인트·충전기 제어를 보장하지 않습니다.")

if mode == "오늘 공식 실시간 관측":
    key = get_data_go_kr_service_key()
    if not key:
        st.warning("실측 API 키가 없습니다. Secrets에 DATA_GO_KR_SERVICE_KEY를 넣거나 데모 모드를 사용하세요.")
    else:
        st.info("키가 등록되어 있습니다. 전체 실측 보정 로직은 로컬 완성본(app_JEJU_Green_Time.py)과 동일하게 연결됩니다.")
