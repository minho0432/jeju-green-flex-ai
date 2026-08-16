"""Jeju Green Flex AI 해커톤 데모 화면."""

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
from jeju_grid_live import (  # noqa: E402
    JejuGridApiError,
    fetch_jeju_grid_live,
    grid_samples_to_hourly,
    latest_complete_hour,
    observation_age_minutes,
)
from optimizer import (  # noqa: E402
    bonus_rate_for_score,
    derive_point_policy,
    make_plan,
)
from realtime_adjustment import (  # noqa: E402
    adjust_forecast_with_live_renewables,
    adjust_forecast_with_observations,
)


FORECAST_PATH = ROOT / "outputs" / "demo_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
HISTORY_PATH = ROOT / "data" / "processed" / "train.csv"

st.set_page_config(page_title="Jeju Green Flex AI", page_icon="🌱", layout="wide")
st.title("🌱 Jeju Green Flex AI")
st.caption("가격절약 기회와 친환경 충전을 함께 고려하는 제주 개인 EV 충전시간 추천")

if not FORECAST_PATH.exists() or not METRICS_PATH.exists():
    st.error("AI 결과가 없습니다. 터미널에서 `python scripts/train_models.py`를 먼저 실행하세요.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_forecast_only_models():
    return train_forecast_only_models()


@st.cache_data(ttl=1800)
def load_tomorrow_weather():
    return fetch_open_meteo_forecast(target_day_offset=1)


@st.cache_data(ttl=1800)
def load_today_weather():
    return fetch_open_meteo_forecast(target_day_offset=0)


@st.cache_data(ttl=1200, show_spinner=False)
def load_official_grid(_service_key: str):
    # 20분 캐시: 개발계정의 일 100회 한도를 넘기지 않도록 보호한다.
    return fetch_jeju_grid_live(_service_key)


@st.cache_data
def load_history():
    return pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])


def get_data_go_kr_service_key() -> str:
    """로컬 환경변수 또는 Streamlit 비밀설정에서 키를 읽는다."""
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("DATA_GO_KR_SERVICE_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


with st.sidebar:
    st.header("0. 시연 모드")
    mode = st.radio(
        "어떤 결과를 볼까요?",
        [
            "검증된 과거 재현",
            "오늘 공식 실시간 관측",
            "실시간 보정 재현",
            "내일 예보 실험",
        ],
        help=(
            "과거 재현은 성능 검증, 오늘 공식 실시간 관측은 KPX 5분 자료, "
            "실시간 보정 재현은 API 없이 흐름 검증, 내일 예보는 날씨예보 실험입니다."
        ),
    )

is_official_live = False
live_samples = None
live_hourly = None
if mode == "검증된 과거 재현":
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    has_actual = True
    has_observed = False
    realtime_metadata = None
    display_date = forecast["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]
    mode_label = "검증 모드"
elif mode == "오늘 공식 실시간 관측":
    service_key = get_data_go_kr_service_key()
    if not service_key:
        st.error("공공데이터포털 인증키가 아직 설정되지 않았습니다.")
        st.markdown(
            "[한국전력거래소 제주계통운영정보 API 활용신청]"
            "(https://www.data.go.kr/data/15158505/openapi.do) 후 "
            "Streamlit Cloud의 **Settings → Secrets**에 아래 한 줄을 넣으세요."
        )
        st.code('DATA_GO_KR_SERVICE_KEY = "발급받은_일반인증키"', language="toml")
        st.info("키가 없어도 다른 세 가지 모드는 정상 작동합니다.")
        st.stop()
    try:
        live_models, live_history = load_forecast_only_models()
        today_weather = load_today_weather()
        live_samples = load_official_grid(service_key)
        live_hourly = grid_samples_to_hourly(live_samples)
        observation_as_of = latest_complete_hour(live_hourly)
        live_age_minutes = observation_age_minutes(live_samples)
        original_forecast = build_live_prediction(
            live_models, live_history, today_weather
        )
        forecast, realtime_metadata = adjust_forecast_with_live_renewables(
            original_forecast,
            live_history,
            live_hourly,
            as_of=observation_as_of,
        )
    except (JejuGridApiError, RuntimeError, ValueError, OSError) as error:
        st.error(str(error))
        st.info(
            "API가 잠시 실패해도 `검증된 과거 재현` 또는 `실시간 보정 재현`은 사용할 수 있습니다."
        )
        st.stop()
    observation_hour = int(observation_as_of.hour)
    has_actual = False
    has_observed = True
    is_official_live = True
    display_date = forecast["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]
    mode_label = "KPX 공식 5분 실측"
elif mode == "실시간 보정 재현":
    original_forecast = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    with st.sidebar:
        observation_hour = st.slider(
            "실측값이 도착한 현재 시각",
            0,
            18,
            10,
            help="선택 시각까지의 실제값만 공개하고 이후 시간은 다시 예측합니다.",
        )
    replay_date = original_forecast["timestamp"].dt.normalize().iloc[0]
    observation_as_of = replay_date + pd.Timedelta(hours=observation_hour)
    forecast, realtime_metadata = adjust_forecast_with_observations(
        original_forecast,
        load_history(),
        as_of=observation_as_of,
    )
    has_actual = False
    has_observed = True
    display_date = forecast["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]
    mode_label = "실시간 보정 재현"
else:
    try:
        live_models, live_history = load_forecast_only_models()
        live_weather = load_tomorrow_weather()
        forecast = build_live_prediction(live_models, live_history, live_weather)
    except (RuntimeError, ValueError, OSError) as error:
        st.error(str(error))
        st.info("왼쪽에서 `검증된 과거 재현`을 선택하면 인터넷 없이도 시연할 수 있습니다.")
        st.stop()
    has_actual = False
    has_observed = False
    realtime_metadata = None
    display_date = forecast["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]
    mode_label = "실험 모드"

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

    st.header("2. Green 충전 크레딧")
    retail_price = st.number_input("사용자 충전 단가 가정(원/kWh)", 0.0, 1000.0, 320.0, 10.0)
    st.caption("1P를 다음 충전에서 1원처럼 쓰는 정책 시뮬레이션입니다.")
    with st.expander("운영자 캠페인 예산 가정", expanded=True):
        monthly_budget_won = st.number_input(
            "월 캠페인 예산(원)", 100000.0, 100000000.0, 3000000.0, 100000.0
        )
        target_shifted_kwh = st.number_input(
            "월 목표 Green Time 충전량(kWh)", 1000.0, 1000000.0, 100000.0, 1000.0
        )
        session_point_cap = st.number_input(
            "한 번 충전 크레딧 상한(P)", 100.0, 10000.0, 1500.0, 100.0
        )

point_policy = derive_point_policy(monthly_budget_won, target_shifted_kwh)
base_point_rate = point_policy["base_point_rate"]
partial_bonus_rate = point_policy["partial_bonus_rate"]
bonus_point_rate = point_policy["maximum_bonus_rate"]
partial_reward_threshold = 50
full_reward_threshold = 70

# 실시간 보정 재현에서는 현재 시각 이전을 다시 추천하지 않는다.
planning_start_hour = start_hour
if has_observed:
    planning_start_hour = max(start_hour, observation_hour + 1)
    if planning_start_hour != start_hour:
        st.info(
            f"실측값이 {observation_hour:02d}:00까지 도착했으므로 "
            f"아직 지나지 않은 {planning_start_hour:02d}:00부터 충전계획을 다시 계산합니다."
        )

if planning_start_hour >= departure_hour:
    st.error(
        "출발시각 전의 추천 가능한 시간이 없습니다. 출발시각을 늦추거나, "
        "공식 실시간 모드에서는 내일 예보 실험으로 바꿔 주세요."
    )
    st.stop()

try:
    plan = make_plan(
        forecast=forecast,
        current_soc=current_soc,
        target_soc=target_soc,
        battery_kwh=battery_kwh,
        charger_kw=charger_kw,
        efficiency=efficiency_percent / 100,
        start_hour=planning_start_hour,
        departure_hour=departure_hour,
        retail_price=retail_price,
        base_point_rate=base_point_rate,
        bonus_point_rate=bonus_point_rate,
        partial_reward_threshold=partial_reward_threshold,
        full_reward_threshold=full_reward_threshold,
        session_point_cap=session_point_cap,
        continuous=continuous,
        conservative=conservative,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

if has_actual:
    st.info(
        f"**{mode_label} · {display_date}** — 이 날짜를 미래 하루처럼 가리고 예측한 뒤 실제값과 비교합니다."
    )
elif has_observed:
    if is_official_live:
        latest = live_samples.iloc[-1]
        status_message = (
            f"**{mode_label} · 최근 자료 {latest['timestamp']:%Y-%m-%d %H:%M} "
            f"({live_age_minutes:.0f}분 전)** — 공식 태양광·풍력 실측으로 오늘 남은 "
            "예측과 충전계획을 보정했습니다. API 호출은 일일 한도를 보호하기 위해 "
            "20분 동안 저장됩니다."
        )
        if live_age_minutes <= 35:
            st.success(status_message)
        else:
            st.warning(
                status_message
                + " 최신 자료가 35분 넘게 지연되어 결과를 참고용으로만 보세요."
            )
        live1, live2, live3, live4 = st.columns(4)
        live1.metric("현재 신재생 발전", f"{latest['renewable_total_mw']:.1f} MW")
        live2.metric("현재 태양광", f"{latest['solar_mw']:.1f} MW")
        live3.metric("현재 풍력", f"{latest['wind_mw']:.1f} MW")
        live4.metric("현재 송전단수요", f"{latest['demand_transmission_mw']:.1f} MW")
        st.metric(
            "최근 재생에너지 예측 편차",
            f"{realtime_metadata['recent_renewable_bias_mwh']:+.1f} MWh",
            help="완료된 최근 3시간의 실제 태양광+풍력 MWh에서 AI 예측을 뺀 값입니다.",
        )
        st.caption(
            "화면의 MW는 순간 발전 세기이고, AI 보정에는 한 시간의 5분 표본 "
            "12개가 모두 모인 경우에만 합산한 MWh를 사용합니다."
        )
        complete_hours = int((live_hourly["coverage_ratio"] >= 1.0).sum())
        with st.expander("공식 자료 연결 상태와 계산 기준"):
            st.write(f"5분 관측 개수: **{len(live_samples):,}개**")
            st.write(f"완료된 시간 수: **{complete_hours}시간**")
            st.write(f"보정에 사용한 최신 완료시간: **{observation_as_of:%H:%M}**")
            st.write(f"가장 최근 관측의 나이: **{live_age_minutes:.0f}분**")
            st.write("완료 기준: **5분 관측 12개 중 12개(100%)**")
            st.write("공식 출처: **한국전력거래소 제주계통운영정보 API**")
    else:
        st.info(
            f"**{mode_label} · {display_date} {observation_hour:02d}:00 기준** — "
            "선택 시각까지 도착한 과거 실측값만 이용해 이후 예측과 충전계획을 다시 계산합니다. "
            "공식 API가 없어도 전체 흐름을 확인하는 재현 모드입니다."
        )
        st.metric(
            "최근 재생에너지 예측 편차",
            f"{realtime_metadata['recent_renewable_bias_mwh']:+.1f} MWh",
            help="실제값-예측값입니다. 음수면 실제 발전량이 예측보다 적었다는 뜻입니다.",
        )
else:
    st.warning(
        f"**{mode_label} · {display_date}** — 실제 내일 날씨예보를 사용하지만 과거 기상예보 오차까지 검증한 모델은 아닙니다. "
        "발표의 성능 근거는 검증 모드 결과를 사용하세요."
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

display_points = (
    plan["ai"]["settled_total_points"]
    if has_actual
    else plan["ai"]["expected_total_points"]
)
point_label = "재현 정산 충전 크레딧" if has_actual else "예상 충전 크레딧"

col1, col2, col3, col4 = st.columns(4)
col1.metric("필요한 전력", f"{plan['required_grid_kwh']:.1f} kWh")
col2.metric("추천 충전시간", recommended_times)
col3.metric(point_label, f"{display_points:,.0f}P")
col4.metric("예상 도달 배터리", f"{plan['reached_soc']:.1f}%")

point1, point2, point3 = st.columns(3)
point1.metric("예보가 틀려도 참여 보장", f"{plan['ai']['guaranteed_points']:,.0f}P")
point2.metric("예측 당시 기대 보너스", f"{plan['ai']['expected_bonus_points']:,.0f}P")
if has_actual:
    point3.metric("실제값 확인 후 보너스", f"{plan['ai']['settled_bonus_points']:,.0f}P")
else:
    point3.metric("실제값 확인 후 보너스", "충전 후 정산")

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
fig.update_layout(
    height=430,
    xaxis_title="시간",
    yaxis={"title": "기회점수", "range": [0, 105]},
    legend={"orientation": "h", "y": 1.14},
    margin={"t": 35, "b": 30},
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "추천점수는 예측 재생에너지를 예측 제주 전력수요로 나눈 공급여력의 과거 백분위입니다."
)

if not used.empty:
    best_slot = used.sort_values([score_to_show, "timestamp"], ascending=[False, True]).iloc[0]
    candidates = forecast[
        (forecast["timestamp"].dt.hour >= planning_start_hour)
        & (forecast["timestamp"].dt.hour < departure_hour)
    ]
    score_difference = best_slot[score_to_show] - candidates[score_to_show].mean()
    st.success(
        f"추천 이유: {best_slot['timestamp']:%H시}의 보수적 기회점수가 사용 가능시간 평균보다 "
        f"{score_difference:+.1f}점 높습니다. 재생에너지는 {best_slot['predicted_renewable_mwh']:.1f}MWh, "
        f"제주 전력수요는 {best_slot['predicted_demand_mwh']:.1f}MWh로 예상됩니다."
    )

if has_actual:
    st.subheader("AI 예측과 실제값 비교")
    renewable_tab, demand_tab = st.tabs(["재생에너지", "제주 전력수요"])
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

    with demand_tab:
        demand_fig = go.Figure()
        demand_fig.add_scatter(
            x=forecast["timestamp"], y=forecast["predicted_demand_lower"],
            line={"width": 0}, name="예상 하한", showlegend=False,
        )
        demand_fig.add_scatter(
            x=forecast["timestamp"], y=forecast["predicted_demand_upper"],
            line={"width": 0}, fill="tonexty", fillcolor="rgba(39,125,161,0.18)",
            name="약 90% 예상 범위",
        )
        demand_fig.add_scatter(
            x=forecast["timestamp"], y=forecast["predicted_demand_mwh"],
            name="AI 예측", line={"color": "#277da1", "width": 3},
        )
        demand_fig.add_scatter(
            x=forecast["timestamp"], y=forecast["actual_demand_mwh"],
            name="실제값", line={"color": "#243447", "width": 2, "dash": "dot"},
        )
        demand_fig.update_layout(height=350, yaxis_title="MWh", margin={"t": 20, "b": 30})
        st.plotly_chart(demand_fig, use_container_width=True)
elif has_observed:
    st.subheader("실측 도착 전후 예측 보정")
    with st.container():
        renewable_fig = go.Figure()
        renewable_fig.add_scatter(
            x=forecast["timestamp"], y=forecast["raw_predicted_renewable_mwh"],
            name="보정 전 예측", line={"color": "#aab4bd", "width": 2, "dash": "dot"},
        )
        renewable_fig.add_scatter(
            x=forecast["timestamp"], y=forecast["predicted_renewable_mwh"],
            name="실측 반영 후 예측", line={"color": "#20a464", "width": 3},
        )
        renewable_fig.add_scatter(
            x=forecast["timestamp"], y=forecast["observed_actual_renewable_mwh"],
            name="현재까지 도착한 실측", mode="lines+markers",
            line={"color": "#243447", "width": 2},
        )
        renewable_fig.update_layout(height=350, yaxis_title="MWh", margin={"t": 20, "b": 30})
        st.plotly_chart(renewable_fig, use_container_width=True)
    st.caption(
        "완료된 최근 3시간의 실제-예측 오차를 사용합니다. 다음 한 시간은 크게 고치고, "
        "먼 시간일수록 보정 영향이 줄어듭니다. 미래 실제값은 계산에 사용하지 않습니다."
    )
else:
    st.subheader("내일 날씨예보 입력")
    weather_table = forecast[[
        "timestamp", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "shortwave_radiation",
    ]].copy()
    weather_table.columns = ["시간", "기온(°C)", "습도(%)", "풍속(km/h)", "일사량(W/m²)"]
    st.dataframe(weather_table, hide_index=True, use_container_width=True)
    st.caption("Open-Meteo의 제주시 기준 내일 시간별 예보를 30분 동안 저장해 사용합니다.")

st.subheader("추천과 즉시 충전 비교")
compare1, compare2 = st.columns(2)
compare1.metric(
    "보수적 기회점수",
    f"{plan['ai']['weighted_planning_score']:.1f}점",
    f"{plan['ai']['weighted_planning_score'] - plan['baseline']['weighted_planning_score']:+.1f}점",
)
simulated_cost = (
    plan["ai"]["simulated_settled_cost_won"]
    if has_actual
    else plan["ai"]["simulated_expected_cost_won"]
)
compare2.metric(
    "1P=1원 가정 체감비용",
    f"{simulated_cost:,.0f}원",
    f"-{display_points:,.0f}P",
)
st.caption(
    "체감비용은 포인트를 1P=1원으로 가정한 시뮬레이션입니다. 실제 제휴 전에는 충전요금 절감을 보장하지 않습니다."
)

st.subheader("추천 시간별 계산")
table = plan["ai_schedule"].copy()
table["시간"] = table["timestamp"].dt.strftime("%H:%M")
table["예측 재생에너지"] = table["predicted_renewable_mwh"].round(1)
table["예측 제주 전력수요"] = table["predicted_demand_mwh"].round(1)
table["중심 점수"] = table["green_score"].round(1)
table["보수적 점수"] = table.get("planning_score", table["green_score"]).round(1)
table["충전량(kWh)"] = table["scheduled_kwh"].round(2)
table["보장 P"] = (table["scheduled_kwh"] * base_point_rate).round(0)
table["예상 보너스 단가"] = table["green_score"].apply(
    lambda score: bonus_rate_for_score(
        score,
        bonus_point_rate,
        partial_reward_threshold,
        full_reward_threshold,
    )
)
table["예상 보너스 P"] = (
    table["scheduled_kwh"] * table["예상 보너스 단가"]
).round(0)
columns = [
    "시간", "예측 재생에너지", "예측 제주 전력수요", "중심 점수", "보수적 점수",
    "충전량(kWh)", "보장 P", "예상 보너스 단가", "예상 보너스 P",
]
if has_actual:
    table["실제 점수"] = table["actual_green_score"].round(1)
    table["정산 보너스 단가"] = table["actual_green_score"].apply(
        lambda score: bonus_rate_for_score(
            score,
            bonus_point_rate,
            partial_reward_threshold,
            full_reward_threshold,
        )
    )
    table["정산 보너스 P"] = (
        table["scheduled_kwh"] * table["정산 보너스 단가"]
    ).round(0)
    columns.extend(["실제 점수", "정산 보너스 단가", "정산 보너스 P"])
st.dataframe(table[columns], hide_index=True, use_container_width=True)
st.caption(f"총 지급 포인트는 한 번 충전당 최대 {session_point_cap:,.0f}P로 제한합니다.")

with st.expander("Green 충전 크레딧은 어떻게 정했나요?"):
    st.markdown(
        f"""
        현재 크레딧은 **충전사업자·지자체·후원기업 중 한 곳이 월 {monthly_budget_won:,.0f}원의 캠페인 예산을 제공한다고 가정한 시뮬레이션**입니다.

        - 최대 단가 근거: {monthly_budget_won:,.0f}원 ÷ {target_shifted_kwh:,.0f}kWh = {point_policy['maximum_total_rate']:,.1f}P/kWh
        - 참여 보장: 추천시간과 겹친 충전량 × {base_point_rate:,.1f}P/kWh
        - 50점 미만: 성과 보너스 0P/kWh
        - 50~69점: 성과 보너스 {partial_bonus_rate:,.1f}P/kWh
        - 70점 이상: 성과 보너스 {bonus_point_rate:,.1f}P/kWh
        - 세션 상한: {session_point_cap:,.0f}P

        1P는 다음 충전에서 1원처럼 사용하는 충전 크레딧으로 정의합니다. 실제 서비스에는 충전 세션 ID,
        실제 충전량, 결제기록, 중복지급 방지 원장과 예산 제공자 계약이 필요합니다.
        """
    )

with st.expander("AI 성능을 어떻게 검증했나요?"):
    st.write("앞선 4개 시간 구간으로 모델을 선택하고, 마지막 30일은 선택에 쓰지 않고 최종 확인했습니다.")
    for target, label in [("renewable_mwh", "재생에너지"), ("demand_mwh", "제주 전력수요")]:
        values = metrics["forecast_only_targets"][target]
        baseline_mae = values["baseline_month_hour"]["mae"]
        st.write(
            f"**{label} 내일 예보형 모델** — 최종 30일 MAE {values['ai']['mae']}, "
            f"월·시간 기준 MAE {baseline_mae}, "
            f"개선율 {values['mae_improvement_percent']}%"
        )
    st.write("실제 과거 기상예보 원본이 아닌 관측날씨로 재현했으므로, 실제 기상예보 오차는 별도 한계입니다.")

with st.expander("현재 구현한 것과 아직 구현하지 않은 것"):
    st.markdown(
        """
        **구현:** 2023~2025년 25,559시간 병합, 후보 모델 시간순 비교, 오늘·내일 날씨예보 조회,
        재생에너지·전력수요 예측,
        KPX 제주 5분 태양광·풍력·수요·공급 실측 연결, 도착한 재생에너지 실측으로 남은 예측과
        충전시간을 다시 계산하는 보정, API 장애용 과거 재현, 보수적 연속 충전시간,
        Green Point 정책, 목표 SOC 불가능 경고, 자동검사.

        **미구현:** 충전사업자 결제 연동, 실제 포인트 지급, 실제 충전기 제어,
        HVDC·발전기 정비·출력제어 예고, 공식 탄소감축·REC 인증.
        """
    )

st.caption(
    "주의: 본 MVP는 충전 의사결정과 Green Point 정책 시뮬레이션입니다. 실제 충전요금 절감, "
    "포인트 지급, 탄소감축량, REC 인증 또는 충전기 제어를 보장하지 않습니다."
)
