# 🌱 Jeju Green Flex AI

제주 SMP와 재생에너지 발전량을 예측하고, 전기차 사용자의 목표 SOC와 출발시간을 지키면서 보수적인 연속 충전시간을 추천하는 해커톤 MVP입니다.

> 현재 버전은 2025년 과거 데이터를 이용한 재현 시뮬레이션입니다. 실제 요금 할인, 실제 페이백, 실제 충전기 제어 또는 REC 인증을 구현한 서비스가 아닙니다.

## 가장 쉬운 실행 방법

### macOS

VS Code 터미널에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_data.py
python scripts/train_models.py
python scripts/validate_ai.py
python -m streamlit run app.py
```

브라우저가 열리면 왼쪽에서 현재 SOC, 목표 SOC, 출발시간과 리워드 가정을 바꿔봅니다.

### Windows

1. `SETUP_WINDOWS.bat` 실행
2. `RUN_WINDOWS.bat` 실행

## 발표 전에 한 번에 검사

```bash
python scripts/validate_data.py
python scripts/train_models.py
python scripts/validate_ai.py
python -m unittest discover -s tests -v
```

모든 단계가 통과해야 발표용 버전입니다. GitHub에 변경사항을 올리면 같은 검사가 자동으로 다시 실행됩니다.

## 핵심 구현

- 2025년 제주 시간별 데이터 8,760행
- 24시간·168시간 시차 특징
- 서로 다른 5개 기간 × 30일 시간순 백테스트
- 세 가지 강한 단순 기준과 비교
- SMP와 재생에너지 예측의 약 90% 예상 범위
- 불리한 예측값으로 계산하는 보수적 `planning_score`
- 목표 SOC·출발시간·충전출력을 지키는 연속 시간 최적화
- 참여 보장 리워드 + 실제 성과 보너스 분리
- 데이터·AI 결과·최적화 자동검사

## 검증 결과

| 대상 | AI MAE | 가장 강한 기준 MAE | 개선율 |
|---|---:|---:|---:|
| SMP | 10.7418원/kWh | 11.2154 | 4.22% |
| 재생에너지 | 10.9011MWh | 15.7213 | 30.66% |

MAE는 예측과 실제의 평균적인 절대 차이이며 작을수록 좋습니다. 실행 환경에 따라 마지막 자릿수는 조금 달라질 수 있으므로 발표에는 직접 실행한 `outputs/model_metrics.json`의 값을 사용합니다.

## 문서 안내

- [TEAM_EXPLANATION.md](TEAM_EXPLANATION.md): 비전공자도 발표할 수 있는 전체 해설과 예상 질문
- [MODEL_CARD.md](MODEL_CARD.md): 모델 구조, 검증 결과, 한계, 가능한 주장
- [CHANGELOG.md](CHANGELOG.md): 초기 버전에서 무엇이 바뀌었는지
- [data_dictionary.md](data_dictionary.md): 데이터 열의 의미

## 주요 파일

| 파일 | 역할 |
|---|---|
| `scripts/prepare_data.py` | 원본 데이터 통합 |
| `scripts/validate_data.py` | 데이터 품질 검사 |
| `scripts/train_models.py` | 모델 학습·5회 백테스트·예측범위 생성 |
| `scripts/optimizer.py` | 충전시간과 리워드 계산 |
| `scripts/validate_ai.py` | 발표용 결과 자동검사 |
| `tests/test_optimizer.py` | 최적화·리워드 규칙 단위테스트 |
| `app.py` | Streamlit 데모 화면 |

## 기본 리워드 예시

기본 조건은 SOC 30%→80%, 배터리 60kWh, 효율 90%, 7kW 충전기입니다.

```text
필요한 계통 충전량 = 60 × (80%-30%) ÷ 0.9 = 33.33kWh
참여 보장 = 33.33 × 10원 ≈ 333원
성과 보너스 = 실제 기준을 통과한 충전량 × 20원
```

기본 과거 재현에서는 총 리워드가 약 860원으로 계산됩니다. 이 단가는 법정요금이나 실제 제휴단가가 아니라 MVP 정책 가정입니다.

## 현재 한계

- 데이터 기간은 2025년 1년입니다.
- 과거 관측 날씨로 과거 예측을 재현하며 실시간 예보 API는 연결하지 않았습니다.
- 제주 전력수요·발전기 상태·HVDC·출력제어 예고는 모델 입력에 없습니다.
- SMP는 소비자 실제 충전요금이 아닙니다.
- 실제 충전 세션, 결제, 리워드 지급과 충전기 제어는 구현하지 않았습니다.

## 데이터 출처

- 제주 SMP: https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202
- 제주 연료원별 발전량: https://www.data.go.kr/data/15138838/fileData.do
- 날씨: https://open-meteo.com/en/docs/historical-weather-api
