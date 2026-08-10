"""Jeju Green Flex AI 해커톤 데모 화면."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from optimizer import make_plan  # noqa: E402


FORECAST_PATH = ROOT / "outputs" / "demo_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"

st.set_page_config(page_title="Jeju Green Flex AI", page_icon="🌱", layout="wide")
st.title("🌱 Jeju Green Flex AI")
st.caption("제주 재생에너지와 SMP 예측을 이용한 개인 전기차 충전시간 추천 데모")

if not FORECAST_PATH.exists():
    st.error("AI 예측 파일이 없습니다. 터미널에서 `python scripts/train_models.py`를 먼저 실행하세요.")
    st.stop()

forecast = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
demo_date = forecast["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]

with st.sidebar:
    st.header("내 차량과 일정")
    current_soc = st.slider("현재 배터리(%)", 5, 90, 30, 5)
    target_soc = st.slider("출발할 때 목표 배터리(%)", 10, 100, 80, 5)
    battery_kwh = st.number_input("배터리 전체 용량(kWh)", 20.0, 120.0, 60.0, 1.0)
    charger_kw = st.selectbox("충전기 출력(kW)", [3.0, 7.0, 11.0, 50.0], index=1)
    efficiency_percent = st.slider("충전 효율(%)", 70, 100, 90)
    start_hour = st.slider("충전을 시작할 수 있는 시각", 0, 22, 8)
    departure_hour = st.slider("차량을 사용해야 하는 시각", 1, 24, 20)

    st.header("요금·리워드 가정")
    retail_price = st.number_input("사용자 충전 단가(원/kWh)", 0.0, 1000.0, 320.0, 10.0)
    reward_rate = st.number_input("Green Time 리워드(원/kWh)", 0.0, 200.0, 30.0, 5.0)
    reward_threshold = st.slider("리워드 지급 기준 점수", 0, 100, 70, 5)

try:
    plan = make_plan(
        forecast=forecast,
        current_soc=current_soc,
        target_soc=target_soc,
        battery_kwh=battery_kwh,
        charger_kw=charger_kw,
        efficiency=efficiency_percent / 100,
        start_hour=start_hour,
        departure_hour=departure_hour,
        retail_price=retail_price,
        reward_rate=reward_rate,
        reward_threshold=reward_threshold,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

st.info(
    f"이 화면은 {demo_date}의 과거 조건을 미래 하루처럼 예측한 해커톤 시뮬레이션입니다. "
    "SMP는 도매가격 지표이며 소비자 충전요금과 동일하지 않습니다."
)

if not plan["feasible"]:
    st.warning(
        f"선택한 시간과 충전기 출력으로는 {target_soc}%까지 도달할 수 없습니다. "
        f"예상 도달 배터리는 약 {plan['reached_soc']:.1f}%입니다."
    )

used = plan["ai_schedule"][plan["ai_schedule"]["scheduled_kwh"] > 0]
recommended_times = ", ".join(used["timestamp"].dt.strftime("%H시").tolist())

col1, col2, col3, col4 = st.columns(4)
col1.metric("필요한 충전량", f"{plan['required_grid_kwh']:.1f} kWh")
col2.metric("추천 충전시간", recommended_times or "없음")
col3.metric("가상 Green Reward", f"{plan['ai']['reward_won']:,.0f}원")
col4.metric("예상 도달 배터리", f"{plan['reached_soc']:.1f}%")

st.subheader("24시간 예측과 AI 추천")
chart_data = forecast.merge(
    plan["ai_schedule"][["timestamp", "scheduled_kwh"]],
    on="timestamp",
    how="left",
).fillna({"scheduled_kwh": 0})
fig = go.Figure()
fig.add_bar(
    x=chart_data["timestamp"],
    y=chart_data["green_score"],
    name="Green Score",
    marker_color=["#20a464" if value > 0 else "#d9e2dc" for value in chart_data["scheduled_kwh"]],
)
fig.add_scatter(
    x=chart_data["timestamp"],
    y=chart_data["predicted_smp"],
    name="예측 SMP",
    yaxis="y2",
    line={"color": "#ed8b38", "width": 2},
)
fig.update_layout(
    height=420,
    xaxis_title="시간",
    yaxis={"title": "Green Score", "range": [0, 105]},
    yaxis2={"title": "예측 SMP(원/kWh)", "overlaying": "y", "side": "right"},
    legend={"orientation": "h", "y": 1.12},
    margin={"t": 30, "b": 30},
)
st.plotly_chart(fig, use_container_width=True)
st.caption("초록색 막대는 실제로 충전하도록 선택된 시간입니다. 주황색 선은 AI가 예측한 제주 SMP입니다.")

st.subheader("AI 추천과 즉시 충전 비교")
compare1, compare2, compare3 = st.columns(3)
compare1.metric(
    "Green Score",
    f"{plan['ai']['weighted_green_score']:.1f}점",
    f"{plan['ai']['weighted_green_score'] - plan['baseline']['weighted_green_score']:+.1f}점",
)
compare2.metric(
    "도매가격 비용지수",
    f"{plan['ai']['market_cost_proxy_won']:,.0f}원",
    f"{plan['ai']['market_cost_proxy_won'] - plan['baseline']['market_cost_proxy_won']:+,.0f}원",
    delta_color="inverse",
)
compare3.metric(
    "리워드 반영 체감비용",
    f"{plan['ai']['effective_cost_won']:,.0f}원",
    f"-{plan['ai']['reward_won']:,.0f}원",
)

st.subheader("시간별 계산 내역")
table = plan["ai_schedule"].copy()
table["시간"] = table["timestamp"].dt.strftime("%H:%M")
table["예측 SMP"] = table["predicted_smp"].round(1)
table["예측 재생에너지(MWh)"] = table["predicted_renewable_mwh"].round(1)
table["Green Score"] = table["green_score"].round(1)
table["충전량(kWh)"] = table["scheduled_kwh"].round(2)
table["예상 리워드(원)"] = (
    (table["green_score"] >= reward_threshold)
    * table["scheduled_kwh"]
    * reward_rate
).round(0)
st.dataframe(
    table[["시간", "예측 SMP", "예측 재생에너지(MWh)", "Green Score", "충전량(kWh)", "예상 리워드(원)"]],
    hide_index=True,
    use_container_width=True,
)

with st.expander("페이백은 어떤 돈인가요?"):
    st.markdown(
        f"""
        현재 데모의 페이백은 **실제 전력시장에서 자동 발생하는 돈이 아닙니다.**

        운영자가 Green Time 참여를 유도하기 위해 정한 캠페인 예산이라고 가정합니다.

        `가상 페이백 = Green Score {reward_threshold}점 이상에서 충전한 양 × {reward_rate:,.0f}원/kWh`

        실제 지급 서비스로 발전하려면 충전사업자·렌터카사·지자체 중 한 곳이 리워드 예산을 제공하고,
        충전 세션 ID와 실제 충전량을 확인할 수 있어야 합니다.
        """
    )

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    with st.expander("AI 모델 성능 확인"):
        for target, label in [("smp", "SMP"), ("renewable_mwh", "재생에너지")]:
            values = metrics["targets"][target]
            st.write(
                f"**{label}** — AI MAE {values['ai']['mae']}, "
                f"단순 기준 MAE {values['baseline']['mae']}, "
                f"개선율 {values['mae_improvement_percent']}%"
            )

st.caption(
    "주의: 본 MVP의 가격·리워드·환경효과는 시뮬레이션입니다. 실제 할인, REC 인증, "
    "충전기 제어를 구현했다고 주장하지 않습니다."
)
