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
st.caption("예측이 틀릴 가능성까지 고려하는 개인 전기차 Green Time 추천 데모")

if not FORECAST_PATH.exists():
    st.error("AI 예측 파일이 없습니다. 터미널에서 `python scripts/train_models.py`를 먼저 실행하세요.")
    st.stop()

forecast = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
demo_date = forecast["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]

with st.sidebar:
    st.header("1. 내 차량과 일정")
    current_soc = st.slider("현재 배터리(%)", 5, 90, 30, 5)
    target_soc = st.slider("출발할 때 목표 배터리(%)", 10, 100, 80, 5)
    battery_kwh = st.number_input("배터리 전체 용량(kWh)", 20.0, 120.0, 60.0, 1.0)
    charger_kw = st.selectbox("충전기 출력(kW)", [3.0, 7.0, 11.0, 50.0], index=1)
    efficiency_percent = st.slider("충전 효율(%)", 70, 100, 90)
    start_hour = st.slider("충전을 시작할 수 있는 시각", 0, 22, 8)
    departure_hour = st.slider("차량을 사용해야 하는 시각", 1, 24, 20)
    continuous = st.checkbox("충전시간을 연속으로 추천", value=True)
    conservative = st.checkbox("예측 오차까지 고려해 보수적으로 추천", value=True)

    st.header("2. 요금·리워드 가정")
    retail_price = st.number_input("사용자 충전 단가(원/kWh)", 0.0, 1000.0, 320.0, 10.0)
    base_reward_rate = st.number_input("참여 보장 리워드(원/kWh)", 0.0, 200.0, 10.0, 5.0)
    bonus_reward_rate = st.number_input("성과형 추가 리워드(원/kWh)", 0.0, 200.0, 20.0, 5.0)
    reward_threshold = st.slider("성과형 지급 기준 실제 점수", 0, 100, 70, 5)

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
        base_reward_rate=base_reward_rate,
        bonus_reward_rate=bonus_reward_rate,
        reward_threshold=reward_threshold,
        continuous=continuous,
        conservative=conservative,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

st.info(
    f"{demo_date}의 과거 하루를 미래처럼 가리고 예측한 재현 실험입니다. "
    "실제 서비스 요금·페이백·충전기 제어가 아니라 검증 가능한 해커톤 시뮬레이션입니다."
)

if not plan["feasible"]:
    st.warning(
        f"선택한 시간과 충전기 출력으로는 {target_soc}%까지 도달할 수 없습니다. "
        f"가능한 범위에서는 약 {plan['reached_soc']:.1f}%까지 충전합니다."
    )

used = plan["ai_schedule"][plan["ai_schedule"]["scheduled_kwh"] > 0]
if used.empty:
    recommended_times = "없음"
else:
    first_time = used["timestamp"].min()
    last_time = used["timestamp"].max() + pd.Timedelta(hours=1)
    recommended_times = f"{first_time:%H:%M}~{last_time:%H:%M}"

col1, col2, col3, col4 = st.columns(4)
col1.metric("필요한 전력", f"{plan['required_grid_kwh']:.1f} kWh")
col2.metric("추천 충전시간", recommended_times)
col3.metric("재현 실험 정산 리워드", f"{plan['ai']['settled_reward_won']:,.0f}원")
col4.metric("예상 도달 배터리", f"{plan['reached_soc']:.1f}%")

reward1, reward2, reward3 = st.columns(3)
reward1.metric("예보가 틀려도 보장", f"{plan['ai']['guaranteed_reward_won']:,.0f}원")
reward2.metric("예측 당시 기대 보너스", f"{plan['ai']['expected_bonus_won']:,.0f}원")
reward3.metric("실제값 확인 후 보너스", f"{plan['ai']['settled_bonus_won']:,.0f}원")

st.subheader("24시간 예측과 보수적인 추천")
chart_data = forecast.merge(
    plan["ai_schedule"][["timestamp", "scheduled_kwh"]],
    on="timestamp",
    how="left",
).fillna({"scheduled_kwh": 0})
score_to_show = "planning_score" if plan["conservative"] else "green_score"
fig = go.Figure()
fig.add_bar(
    x=chart_data["timestamp"],
    y=chart_data[score_to_show],
    name="추천에 사용한 점수",
    marker_color=["#20a464" if value > 0 else "#d9e2dc" for value in chart_data["scheduled_kwh"]],
)
if plan["conservative"]:
    fig.add_scatter(
        x=chart_data["timestamp"],
        y=chart_data["green_score"],
        name="중심 예측 Green Score",
        line={"color": "#277da1", "width": 2, "dash": "dot"},
    )
fig.add_scatter(
    x=chart_data["timestamp"],
    y=chart_data["predicted_smp"],
    name="예측 SMP",
    yaxis="y2",
    line={"color": "#ed8b38", "width": 2},
)
fig.update_layout(
    height=430,
    xaxis_title="시간",
    yaxis={"title": "점수", "range": [0, 105]},
    yaxis2={"title": "예측 SMP(원/kWh)", "overlaying": "y", "side": "right"},
    legend={"orientation": "h", "y": 1.14},
    margin={"t": 35, "b": 30},
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "초록 막대는 선택된 충전시간입니다. 보수적 모드에서는 가격은 예상 상한, "
    "재생에너지는 예상 하한을 사용해 추천합니다."
)

if not used.empty:
    best_slot = used.sort_values([score_to_show, "predicted_smp"], ascending=[False, True]).iloc[0]
    candidates = forecast[
        (forecast["timestamp"].dt.hour >= start_hour)
        & (forecast["timestamp"].dt.hour < departure_hour)
    ]
    score_difference = best_slot[score_to_show] - candidates[score_to_show].mean()
    st.success(
        f"추천 이유: {best_slot['timestamp']:%H시}의 보수적 기회점수가 충전 가능시간 평균보다 "
        f"{score_difference:+.1f}점 높습니다. SMP는 {best_slot['predicted_smp']:.1f}원/kWh로 "
        f"예측됐고 재생에너지는 {best_slot['predicted_renewable_mwh']:.1f}MWh로 예측됐습니다."
    )

st.subheader("AI 예측이 실제값과 얼마나 비슷했는가")
smp_tab, renewable_tab = st.tabs(["SMP", "재생에너지"])
with smp_tab:
    smp_fig = go.Figure()
    smp_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["predicted_smp_lower"],
        line={"width": 0}, name="예상 하한", showlegend=False,
    )
    smp_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["predicted_smp_upper"],
        line={"width": 0}, fill="tonexty", fillcolor="rgba(237,139,56,0.18)",
        name="약 90% 예상 범위",
    )
    smp_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["predicted_smp"],
        name="AI 예측", line={"color": "#ed8b38", "width": 3},
    )
    smp_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["actual_smp"],
        name="실제값", line={"color": "#243447", "width": 2, "dash": "dot"},
    )
    smp_fig.update_layout(height=350, yaxis_title="원/kWh", margin={"t": 20, "b": 30})
    st.plotly_chart(smp_fig, use_container_width=True)

with renewable_tab:
    renewable_fig = go.Figure()
    renewable_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["predicted_renewable_lower"],
        line={"width": 0}, name="예상 하한", showlegend=False,
    )
    renewable_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["predicted_renewable_upper"],
        line={"width": 0}, fill="tonexty", fillcolor="rgba(32,164,100,0.18)",
        name="약 90% 예상 범위",
    )
    renewable_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["predicted_renewable_mwh"],
        name="AI 예측", line={"color": "#20a464", "width": 3},
    )
    renewable_fig.add_scatter(
        x=forecast["timestamp"], y=forecast["actual_renewable_mwh"],
        name="실제값", line={"color": "#243447", "width": 2, "dash": "dot"},
    )
    renewable_fig.update_layout(height=350, yaxis_title="MWh", margin={"t": 20, "b": 30})
    st.plotly_chart(renewable_fig, use_container_width=True)

st.caption(
    "색칠된 영역은 과거 검증 오차로 만든 약 90% 예상 범위입니다. 확률을 엄밀하게 보장하는 구간은 아닙니다."
)

st.subheader("AI 추천과 즉시 충전 비교")
compare1, compare2, compare3 = st.columns(3)
compare1.metric(
    "보수적 기회점수",
    f"{plan['ai']['weighted_planning_score']:.1f}점",
    f"{plan['ai']['weighted_planning_score'] - plan['baseline']['weighted_planning_score']:+.1f}점",
)
compare2.metric(
    "도매가격 비용지수",
    f"{plan['ai']['market_cost_proxy_won']:,.0f}",
    f"{plan['ai']['market_cost_proxy_won'] - plan['baseline']['market_cost_proxy_won']:+,.0f}",
    delta_color="inverse",
)
compare3.metric(
    "재현 실험 체감비용",
    f"{plan['ai']['settled_cost_won']:,.0f}원",
    f"-{plan['ai']['settled_reward_won']:,.0f}원",
)
st.caption("도매가격 비용지수는 충전사업자의 실제 원가나 소비자 요금이라고 주장하지 않습니다.")

st.subheader("시간별 계산 내역")
table = plan["ai_schedule"].copy()
table["시간"] = table["timestamp"].dt.strftime("%H:%M")
table["예측 SMP"] = table["predicted_smp"].round(1)
table["예측 재생에너지"] = table["predicted_renewable_mwh"].round(1)
table["중심 점수"] = table["green_score"].round(1)
table["보수적 점수"] = table.get("planning_score", table["green_score"]).round(1)
table["실제 점수"] = table.get("actual_green_score", pd.Series(index=table.index, dtype=float)).round(1)
table["충전량(kWh)"] = table["scheduled_kwh"].round(2)
table["보장 리워드"] = (table["scheduled_kwh"] * base_reward_rate).round(0)
table["성과 보너스"] = (
    (table.get("actual_green_score", 0) >= reward_threshold)
    * table["scheduled_kwh"]
    * bonus_reward_rate
).round(0)
st.dataframe(
    table[[
        "시간", "예측 SMP", "예측 재생에너지", "중심 점수", "보수적 점수",
        "실제 점수", "충전량(kWh)", "보장 리워드", "성과 보너스",
    ]],
    hide_index=True,
    use_container_width=True,
)

with st.expander("페이백이 왜 논리적으로 가능한가요?"):
    st.markdown(
        f"""
        현재 금액은 **실제 전력시장에서 자동으로 생기는 돈이 아니라 제휴사가 미리 정한 캠페인 예산**입니다.

        1. **참여 보장분:** 추천시간을 실제로 따르면 `{base_reward_rate:,.0f}원/kWh`를 지급합니다.
           예보가 틀려도 돌려받지 않는다는 정책입니다.
        2. **성과형 보너스:** 충전 뒤 실제 데이터의 Green Score가 {reward_threshold}점 이상이면
           `{bonus_reward_rate:,.0f}원/kWh`를 추가 지급합니다.
        3. 실제 서비스에는 `충전 세션 ID`, `실제 충전량`, `결제 기록`, `리워드 예산 제공자`가 필요합니다.

        이 데모에서는 과거 실제값이 있으므로 두 번째 보너스까지 **재현 실험으로 정산**할 수 있습니다.
        """
    )

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    with st.expander("AI 성능을 어떻게 검증했나요?"):
        st.write("시간순서를 섞지 않고 서로 다른 5개 기간에서 각각 30일씩 시험했습니다.")
        for target, label in [("smp", "SMP"), ("renewable_mwh", "재생에너지")]:
            values = metrics["targets"][target]
            best_name = values["best_baseline"]
            baseline_mae = values["baselines"][best_name]["mae"]
            st.write(
                f"**{label}** — AI MAE {values['ai']['mae']}, "
                f"강한 단순 기준({best_name}) MAE {baseline_mae}, "
                f"개선율 {values['mae_improvement_percent']}%, "
                f"약 90% 예상범위 ±{values['approx_90_interval_half_width']}"
            )

with st.expander("현재 구현한 것과 아직 구현하지 않은 것"):
    st.markdown(
        """
        **구현:** 2025년 시간별 데이터 학습, 5구간 시간순 검증, SMP·재생에너지 예측,
        예측범위, 보수적 연속 충전시간, 리워드 재현 정산, 즉시 충전 비교.

        **미구현:** 실시간 기상예보 연결, 2023~2025 다년도 통합, 충전사업자 결제 연동,
        실제 페이백 지급, 실제 충전기 제어, REC/K-RE100 인증.
        """
    )

st.caption(
    "주의: 본 MVP는 의사결정 시뮬레이션입니다. 소비자 실제 요금, 실제 할인, 탄소감축량, "
    "REC 인증 또는 충전기 제어를 구현했다고 주장하지 않습니다."
)
