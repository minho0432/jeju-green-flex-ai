# JEJU Green Time — 코드 설명 (한국어)

이 문서는 프로젝트 코드가 **무엇을 하는지**를 한국어로 설명합니다.

---

## 1. 전체 흐름

```
날씨 예보(Open-Meteo)
    → AI가 시간별 재생에너지 예측
    → Green Score(0~100) 계산
    → 내 배터리·출발 시간에 맞춰 연속 충전 구간 추천
    → Streamlit 화면에 카드·그래프로 표시
```

**Green Score는 재생에너지 발전량을 제주 전력수요로 나눈 공급여력 기준입니다.**

---

## 2. 주요 파일

| 파일 | 역할 |
|------|------|
| `app.py` | 사용자 화면 + 대시보드 그래프 + 자동 갱신 |
| `scripts/model_utils.py` | 시간·날씨 특성, Green Score 가중치, 하이브리드 혼합 |
| `scripts/live_forecast.py` | Open-Meteo 예보 조회, 예보용 모델 학습·예측 |
| `scripts/optimizer.py` | 목표 SOC를 맞추는 연속 충전 구간·Green Point 계산 |
| `scripts/improve_renewable.py` | 재생 예측 정확도 검증·하이브리드 α 탐색 |
| `scripts/jeju_grid_live.py` | KPX 제주 5분 실측 API (키 필요) |
| `scripts/realtime_adjustment.py` | 실측으로 남은 시간 예측 보정 |
| `scripts/weather_ensemble.py` | 날씨 시나리오(앙상블)로 불확실 구간 확대 |
| `data/processed/train.csv` | 학습용 시간별 데이터 |
| `outputs/` | 검증 지표·데모 예측 결과 |

---

## 3. `app.py` (화면)

- **오늘 / 내일 / 데모**: 데이터 기준 선택
- **오늘 모드**: 이미 지난 시각은 추천에서 제외
- **10분마다 자동 갱신**: 탭이 열려 있으면 예보·추천 다시 계산
- **연동 상태**: 날씨(키 없음)·제주 실측(키 있으면 연동됨)
- **그래프**
  - Green 점수 막대 (추천 구간 강조)
  - 예측 재생 + 일사량
  - 배터리 게이지
  - 시간별 충전 kWh

키 이름: `DATA_GO_KR_SERVICE_KEY` (환경변수 또는 Streamlit Secrets)

---

## 4. `model_utils.py` (특성·점수)

### 하는 일
- 시각을 숫자 특성으로 변환 (시, 요일, 월, sin/cos 주기)
- 날씨: 기온, 습도, 풍속, 일사 + 일사×풍속 등
- 과거 같은 시각 값(lag 24h, 48h, 168h) — **미래 값은 쓰지 않음**
- `RENEWABLE_WEIGHT = 1.0`, `MARKET_WEIGHT = 0.0` → 점수는 재생만

### 주요 함수
| 함수 | 설명 |
|------|------|
| `make_features` | 학습용 전체 특성 (lag 포함) |
| `make_live_features` | 예보 모드: 시간+날씨만 (누수 없는 입력) |
| `month_hour_baseline` | 월·시간 평균 기준표 예측 |
| `hybrid_blend` | α×AI + (1−α)×기준표 |
| `score_against_history` | 과거 분포 대비 0~100점 |

---

## 5. `live_forecast.py` (실예보 예측)

1. Open-Meteo에서 제주 **오늘(0) 또는 내일(1)** 24시간 예보 수신
2. 과거 `train.csv`로 예보용 모델 학습 (시간·날씨만 입력)
3. 예보를 넣어 재생량 예측
4. 과거 분포와 비교해 `green_score` 등 부여
5. (선택) `hybrid_alpha.json`의 α로 기준표와 혼합 — 검증상 최선은 α=1.0

---

## 6. `optimizer.py` (충전 구간 추천)

### 입력
- 24시간 점수표, 현재 SOC, 목표 SOC, 배터리 kWh, 충전기 kW, 효율
- 충전 가능 시작 시각 ~ 출발 시각

### 로직
1. 필요한 전기량(kWh) 계산
2. 가능 시간대만 후보로 둠
3. **연속된 시간** 중 점수×충전량이 가장 좋은 구간 선택 (한 번에 이어서 충전)
4. 목표 SOC 도달 가능 여부·예상 Green Point 요약

`conservative=True`이면 보수적 점수(`planning_score`)를 우선합니다.

---

## 7. `improve_renewable.py` (정확도 검증)

- 시간순으로 여러 구간을 나눠 검증 (미래 데이터가 학습에 들어가지 않음)
- AI vs 월·시간 기준표 MAE 비교
- Green Time: 실제 상위 30% 시각과 예측 상위 30%가 겹치는 비율
- 하이브리드 α 그리드 탐색 후 `outputs/hybrid_alpha.json` 저장

**최근 결과 요약**
- AI MAE ≈ 10.34 / 기준표 ≈ 18.04 (약 42.7% 개선)
- Green Time 겹침 ≈ 92.5%
- 최적 α = 1.0

---

## 8. 실측 보정 (키 있을 때)

| 파일 | 설명 |
|------|------|
| `jeju_grid_live.py` | KPX 5분 태양광·풍력·수요 조회, MW→시간 MWh 변환 |
| `realtime_adjustment.py` | 최근 실측−예측 편차로 **아직 안 지난 시간**만 보정 |

키가 없으면 날씨 예보 추천만으로도 동작합니다.

---

## 9. 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
# 정확도 재검증
python scripts/improve_renewable.py
```

---

## 10. 용어 짧게

| 용어 | 의미 |
|------|------|
| 재생에너지 | 태양광 + 풍력 발전량(합계) |
| Green Score | 재생이 많을수록 높은 0~100점 |
| Green Time | 점수가 높아 충전하기 좋은 시간대 |
| Green Point | 점수 구간에 따른 예상 리워드(시뮬레이션) |
| SOC | 배터리 잔량(%) |
| lag | 과거 같은 시각 값 (예: 24시간 전) |
| 하이브리드 α | AI와 기준표를 섞는 비율 |
