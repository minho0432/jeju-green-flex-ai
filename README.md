# 🌱 Jeju Green Flex AI

제주 개인 전기차 사용자가 출발 전 목표 배터리를 채우면서, 재생에너지 활용에 유리한 시간에 충전하고 Green Point 혜택을 받을 수 있도록 돕는 해커톤 MVP입니다.

> 핵심 가치는 두 가지입니다. **친환경 시간 선택**은 재생에너지 예측이 담당하고, **사용자 가격 혜택**은 Green Point가 담당합니다. SMP는 소비자 충전요금이 아니라 추천을 보조하는 전력시장 지표입니다.

## 사용자가 보는 결과

사용자는 현재 배터리, 목표 배터리, 충전기 출력과 출발시간을 입력합니다. 서비스는 다음을 보여줍니다.

- 출발 전 목표 배터리를 채울 수 있는지
- 추천하는 연속 충전시간
- 예상 Green Point
- 추천시간의 재생에너지 기회와 SMP 보조지표
- 예측이 틀릴 가능성을 고려한 보수적 결과

## 세 가지 시연 모드

| 모드 | 무엇을 사용하나 | 발표에서의 역할 |
|---|---|---|
| 검증된 과거 재현 | 2025년의 한 날짜를 미래처럼 가린 AI 예측과 그날 실제값 | 성능과 사후 포인트 정산을 증명하는 기본 데모 |
| 실시간 보정 재현 | 선택한 현재 시각까지만 과거 실측값을 순서대로 공개 | 실제값 도착→오차 확인→남은 예측 보정→충전시간 재계산을 검증 |
| 내일 예보 실험 | Open-Meteo의 제주시 내일 시간별 기상예보 | 실제 서비스처럼 보이는 24시간 추천 실험 |

내일 예보 실험은 실제 API를 쓰지만, 과거에 발표됐던 기상예보 자체의 오차까지 검증한 모델은 아닙니다. 따라서 공식 성능 수치는 검증된 과거 재현 결과만 사용합니다.

`실시간 보정 재현`은 공식 제주 실시간 발전량 API가 연결된 것처럼 꾸미지 않습니다. 과거 실측을 시간 순서대로 한 시간씩 공개하며 최근 3시간의 `실제값-예측값` 편차로 아직 지나지 않은 예측만 수정합니다. 다음 한 시간에 편차를 가장 크게 반영하고 먼 시간일수록 영향력을 줄인 뒤 Green Score와 충전계획을 다시 계산합니다. 미래 실제값은 화면·계획·포인트 정산뿐 아니라 Green Score의 과거 비교 기준에서도 차단합니다.

## Green Score와 SMP

```text
Green Score = 재생에너지 기회점수 × 80% + SMP 시장기회점수 × 20%
```

- 재생에너지 기회점수: 과거보다 태양광·풍력 발전량이 많을수록 높습니다.
- SMP 시장기회점수: 과거보다 제주 전력시장의 시간별 도매가격 지표가 낮을수록 높습니다.
- SMP가 낮아도 소비자가 내는 충전요금이 자동으로 싸지는 것은 아닙니다.

가격절약은 SMP를 충전요금으로 바꾸어 계산하지 않고, 제휴사가 제공한다고 가정한 Green Point로 따로 표현합니다.

## 예산으로 역산하는 Green 충전 크레딧

```text
최대 지급 단가 = 월 캠페인 예산 ÷ 월 목표 Green Time 충전량 ÷ 1원/P
기본 예시 = 3,000,000원 ÷ 100,000kWh = 30P/kWh

참여 보장 = 추천시간과 겹친 충전량 × 10P/kWh
실제 점수 50점 미만 = 보너스 0P/kWh
실제 점수 50~69점 = 보너스 10P/kWh
실제 점수 70점 이상 = 보너스 20P/kWh
한 번 충전의 총 포인트 상한 = 1,500P
```

기존처럼 사용자가 보상 단가를 임의로 정하지 않습니다. 운영자가 입력한 월 예산과 목표 이동 충전량으로 최대 단가를 먼저 계산하고, 그중 1/3은 예보가 틀려도 유지하는 참여 보장분, 2/3은 실제 성과에 따른 최대 보너스로 배분합니다.

기본 조건 SOC 30%→80%, 배터리 60kWh, 효율 90%에서는 다음과 같습니다.

```text
필요한 계통 충전량 = 60 × (80%-30%) ÷ 0.9 = 33.33kWh
참여 보장 = 약 333P
과거 재현 성과 보너스 = 약 667P
총 재현 정산 = 1,000P
```

`1P=1원`은 다음 충전에서 1원처럼 사용하는 충전 크레딧의 정책 가정입니다. 실제 지급을 위해서는 충전사업자, 제주도 또는 후원기업의 캠페인 예산과 충전 세션 연동이 필요합니다.

## AI 검증 결과

시간순서를 지킨 5회 백테스트에서 서로 다른 기간을 각각 30일씩 시험했습니다. 전체 시험시간은 3,600시간입니다.

| 대상 | 검증 AI MAE | 가장 강한 단순 기준 MAE | 개선율 |
|---|---:|---:|---:|
| SMP | 10.7418원/kWh | 11.2154 | 4.22% |
| 재생에너지 | 10.9011MWh | 15.7213 | 30.66% |

MAE는 예측과 실제의 평균적인 차이이며 작을수록 좋습니다. 내일 예보용 모델은 미래에 알 수 있는 시간·날씨만 사용하며, 과거 관측날씨 기준 MAE는 SMP 14.1874원/kWh, 재생에너지 11.9452MWh입니다. 이 수치에는 실제 기상예보 오차가 포함되지 않습니다.

## 가장 쉬운 실행 방법

### macOS

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_data.py
python scripts/train_models.py
python scripts/validate_ai.py
python -m streamlit run app.py
```

### Windows

1. `SETUP_WINDOWS.bat` 실행
2. `RUN_WINDOWS.bat` 실행

## 발표 전 전체 검사

```bash
python scripts/validate_data.py
python scripts/train_models.py
python scripts/validate_ai.py
python -m unittest discover -s tests -v
```

현재 자동검사는 데이터 8,760행, AI 결과 24시간, 목표 SOC 달성, 80:20 점수, 예산 기반 단가, 3단계 보너스, 포인트 상한, 예보 실패 대응, 실시간 보정의 미래값 차단과 MW→MWh 단위 변환을 확인합니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `scripts/prepare_data.py` | 원본 데이터 통합 |
| `scripts/validate_data.py` | 데이터 품질 검사 |
| `scripts/model_utils.py` | AI 입력 특징과 80:20 점수 계산 |
| `scripts/train_models.py` | 모델 학습·5회 백테스트·예측범위 생성 |
| `scripts/live_forecast.py` | Open-Meteo 내일 예보 조회와 실험용 예측 |
| `scripts/realtime_adjustment.py` | 도착 실측으로 미래 예측 보정·5분 MW를 시간별 MWh로 변환 |
| `scripts/optimizer.py` | 목표 SOC를 지키는 충전시간·포인트 계산 |
| `scripts/validate_ai.py` | 발표용 결과 자동검사 |
| `tests/test_optimizer.py` | 최적화·포인트 규칙 단위테스트 9개 |
| `tests/test_realtime_adjustment.py` | 실시간 보정·미래값 차단·단위 변환 테스트 5개 |
| `app.py` | Streamlit 데모 화면 |

## 현재 구현한 것과 남은 것

### 구현 완료

- 2025년 제주 시간별 데이터 8,760행 검사
- 두 AI 모델과 5회 시간순 백테스트
- 약 90% 예상범위와 보수적 점수
- 목표 SOC·출발시간·충전출력을 지키는 연속 시간 추천
- 검증된 과거 재현과 내일 기상예보 실험 모드
- 실측 도착에 따른 미래 예측·Green Score·충전계획 재계산 구조와 과거 재현
- 5분 MW 실측을 시간별 MWh로 바꾸면서 누락률을 확인하는 변환 함수
- 월 예산 기반 Green 충전 크레딧·3단계 성과 보너스·세션 상한
- GitHub Actions 자동검사

### 아직 구현하지 않음

- 실제 충전 세션·결제·포인트 지급
- 충전기 예약 또는 원격제어
- 제주 실시간 전력수요·HVDC·발전기 상태·출력제어 예고
- 공식 제주 태양광·풍력 실시간 API 연동
- 다년도 학습과 과거 기상예보 원본을 이용한 완전한 실시간 검증
- 공식 탄소감축·REC 인증

## 문서

- [TEAM_EXPLANATION.md](TEAM_EXPLANATION.md): 비전공자용 전체 해설과 팀 공유 대본
- [MODEL_CARD.md](MODEL_CARD.md): 모델 구조·검증·한계
- [CHANGELOG.md](CHANGELOG.md): 무엇이 바뀌었는지
- [data_dictionary.md](data_dictionary.md): 데이터 열 설명

## 데이터·API 출처

- 제주 SMP: https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpSmpChart.do?menuId=040202
- 제주 연료원별 발전량: https://www.data.go.kr/data/15138838/fileData.do
- 과거 날씨: https://open-meteo.com/en/docs/historical-weather-api
- 내일 날씨예보: https://open-meteo.com/en/docs/gfs-api

## 한 문장 주의사항

이 MVP는 **충전시간 추천과 Green Point 정책 시뮬레이션**이며, 실제 충전요금 할인·포인트 지급·탄소감축량·REC 인증·충전기 제어를 보장하지 않습니다.
