# 🌱 Jeju Green Flex AI

제주 개인 전기차 사용자가 출발 전 목표 배터리를 채우면서, 재생에너지 활용에 유리한 시간에 충전하고 Green Point 혜택을 받을 수 있도록 돕는 해커톤 MVP입니다.

> 핵심 가치는 두 가지입니다. **친환경 시간 선택**은 재생에너지 발전량과 제주 전력수요 예측이 담당하고, **사용자 가격 혜택**은 Green Point가 담당합니다.

## 사용자가 보는 결과

사용자는 현재 배터리, 목표 배터리, 충전기 출력과 출발시간을 입력합니다. 서비스는 다음을 보여줍니다.

- 출발 전 목표 배터리를 채울 수 있는지
- 추천하는 연속 충전시간
- 예상 Green Point
- 추천시간의 재생에너지 공급여력
- 예측이 틀릴 가능성을 고려한 보수적 결과

## 네 가지 시연 모드

| 모드 | 무엇을 사용하나 | 발표에서의 역할 |
|---|---|---|
| 검증된 과거 재현 | 2025년의 한 날짜를 미래처럼 가린 AI 예측과 그날 실제값 | 성능과 사후 포인트 정산을 증명하는 기본 데모 |
| 오늘 공식 실시간 관측 | KPX의 제주 5분 신재생·태양광·풍력·수요·공급 API | 최근 실측으로 오늘 남은 재생에너지 예측과 충전계획을 다시 계산 |
| 실시간 보정 재현 | 선택한 현재 시각까지만 과거 실측값을 순서대로 공개 | 실제값 도착→오차 확인→남은 예측 보정→충전시간 재계산을 검증 |
| 내일 예보 실험 | Open-Meteo의 제주시 내일 시간별 기상예보 | 실제 서비스처럼 보이는 24시간 추천 실험 |

내일 예보 실험은 실제 API를 쓰지만, 과거에 발표됐던 기상예보 자체의 오차까지 검증한 모델은 아닙니다. 따라서 공식 성능 수치는 검증된 과거 재현 결과만 사용합니다.

`오늘 공식 실시간 관측`은 공공데이터포털 인증키를 설정했을 때 작동합니다. 5분 단위 MW를 시간별 태양광+풍력 MWh로 변환하고, 한 시간의 5분 자료 12개가 모두 모인 완료 시간만 사용합니다. 11개만 합치면 에너지가 약 8.3% 작게 계산될 수 있기 때문입니다. 최근 3시간의 `실제값-예측값` 편차로 아직 지나지 않은 재생에너지 예측만 수정합니다. 다음 한 시간에 편차를 가장 크게 반영하고 먼 시간일수록 영향력을 줄인 뒤 Green Score와 충전계획을 다시 계산합니다.

`실시간 보정 재현`은 인증키가 없거나 API가 멈춘 날에도 같은 계산 흐름을 보여 주는 안전한 대체 데모입니다. 과거 실측을 시간 순서대로 한 시간씩 공개하며 미래 실제값은 화면·계획·포인트 정산뿐 아니라 Green Score의 과거 비교 기준에서도 차단합니다.

## 공식 실시간 API 연결 방법

1. [한국전력거래소 제주계통운영정보 API](https://www.data.go.kr/data/15158505/openapi.do)에서 `활용신청`을 누릅니다.
2. 발급된 일반 인증키를 복사합니다. 인코딩키와 디코딩키 중 어느 쪽도 코드가 처리합니다.
3. 로컬에서는 `.streamlit/secrets.toml.example`을 `.streamlit/secrets.toml`로 복사하고 키를 넣습니다.
4. Streamlit Cloud에서는 앱의 `Settings → Secrets`에 다음을 넣고 저장합니다.

```toml
DATA_GO_KR_SERVICE_KEY = "발급받은_일반인증키"
```

실제 키를 GitHub, 카카오톡, 이 대화창에 올리면 안 됩니다. 앱은 개발계정의 하루 100회 제한을 보호하기 위해 API 결과를 20분 동안 저장합니다. 따라서 일반 새로고침은 즉시 추가 호출하지 않으며 최대 약 20분 이내 자료를 보여 줄 수 있습니다.

## 추천 결과 LLM 설명

LLM은 충전시간·Green Score를 계산하지 않습니다. 기존 LightGBM/최적화 결과를 바탕으로
예보 오차 가능성, 낮은 점수의 이유, 실시간 보정 필요성을 한국어로 설명하는 선택 기능입니다.
API 키가 없거나 호출이 실패해도 수치 기반 설명으로 자동 전환되므로 추천 화면은 중단되지 않습니다.

Streamlit Cloud의 `Settings → Secrets` 또는 로컬 환경변수에 다음을 설정합니다.

```toml
OPENAI_API_KEY = "발급받은_API_키"
OPENAI_MODEL = "gpt-4o-mini"
# 다른 OpenAI 호환 서버를 사용할 때만 지정합니다.
# OPENAI_BASE_URL = "https://example.com/v1/chat/completions"
```

실제 API 키는 GitHub에 커밋하지 마세요.

## Green Score

```text
시간별 공급여력 = 예측 태양광·풍력 발전량 ÷ 예측 제주 전력수요
Green Score = 공급여력을 과거와 비교한 백분위 점수(0~100)
```

- 발전량이 많아도 수요가 더 많으면 점수가 무조건 높아지지 않습니다.
- 70점은 해당 시간이 과거 비교 구간의 약 70%보다 공급여력이 높다는 뜻입니다.

가격 혜택은 제휴사가 제공한다고 가정한 Green Point로 따로 표현합니다.

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

2023~2025년 25,559시간을 사용했습니다. 앞선 4개 시간 구간으로 후보 모델과 혼합비를 선택하고, 마지막 30일 720시간은 선택에 쓰지 않고 최종 확인했습니다.

| 대상 | 검증 AI MAE | 가장 강한 단순 기준 MAE | 개선율 |
|---|---:|---:|---:|
| 재생에너지 | 37.353MWh | 39.297MWh | 4.95% |
| 제주 전력수요 | 22.4568MWh | 41.9477MWh | 46.46% |

MAE는 예측과 실제의 평균적인 차이이며 작을수록 좋습니다. 공급여력 MAE는 0.0531로 월·시간 기준 0.0569보다 낮았습니다. Green Time 상위 30% 일치율은 양쪽 모두 92.13%였습니다. 이 수치에는 실제 기상예보 자체의 오차가 포함되지 않습니다.

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

현재 자동검사는 데이터 25,559행, AI 결과 24시간, 공급여력 100% 점수, 목표 SOC 달성, 예산 기반 단가, 3단계 보너스, 포인트 상한, 예보 실패 대응, 실시간 보정의 미래값 차단, MW→MWh 단위 변환과 원본 누락 구간의 잘못된 시차 생성을 확인합니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `scripts/prepare_data.py` | 원본 데이터 통합 |
| `scripts/validate_data.py` | 데이터 품질 검사 |
| `scripts/model_utils.py` | AI 입력 특징과 공급여력 점수 계산 |
| `scripts/train_models.py` | 후보 모델 비교·선택·최종 홀드아웃 검증 |
| `scripts/build_multiyear_data.py` | 2023~2025년 공식 자료 병합·단위 통일 |
| `scripts/live_forecast.py` | Open-Meteo 내일 예보 조회와 실험용 예측 |
| `scripts/jeju_grid_live.py` | KPX 제주 5분 실측 조회·검증·MW→시간별 MWh 변환 |
| `scripts/realtime_adjustment.py` | 도착 실측으로 미래 예측 보정·5분 MW를 시간별 MWh로 변환 |
| `scripts/optimizer.py` | 목표 SOC를 지키는 충전시간·포인트 계산 |
| `scripts/validate_ai.py` | 발표용 결과 자동검사 |
| `tests/test_optimizer.py` | 최적화·포인트 규칙 단위테스트 10개 |
| `tests/test_realtime_adjustment.py` | 실시간 보정·미래값 차단·단위 변환 테스트 6개 |
| `tests/test_jeju_grid_live.py` | 공식 API JSON/XML 파싱·오류·키 인코딩·완성도·최신성·단위 변환 테스트 8개 |
| `app.py` | Streamlit 데모 화면 |

## 현재 구현한 것과 남은 것

### 구현 완료

- 2023~2025년 제주 시간별 데이터 25,559행 검사
- 발전량·전력수요 후보 모델 비교와 마지막 30일 홀드아웃 검증
- 약 90% 예상범위와 보수적 점수
- 목표 SOC·출발시간·충전출력을 지키는 연속 시간 추천
- 검증된 과거 재현과 내일 기상예보 실험 모드
- 실측 도착에 따른 미래 예측·Green Score·충전계획 재계산 구조와 과거 재현
- 공공데이터포털 KPX 제주 5분 신재생·태양광·풍력·수요·공급 실측 연결 코드
- 실시간 API 결과 검증, 20분 캐시, 관측 최신성·완료시간 표시, 키 누락·장애 시 안전한 기존 데모 유지
- 5분 MW 실측을 시간별 MWh로 바꾸면서 누락률을 확인하는 변환 함수
- 월 예산 기반 Green 충전 크레딧·3단계 성과 보너스·세션 상한
- GitHub Actions 자동검사

### 아직 구현하지 않음

- 실제 충전 세션·결제·포인트 지급
- 충전기 예약 또는 원격제어
- HVDC·개별 발전기 고장/정비·출력제어 예고
- 소비자 실제 충전요금 연동
- 과거에 실제 발표된 기상예보 원본을 이용한 완전한 예보 성능 검증
- 공식 탄소감축·REC 인증

## 문서

- [TEAM_EXPLANATION.md](TEAM_EXPLANATION.md): 비전공자용 전체 해설과 팀 공유 대본
- [MODEL_CARD.md](MODEL_CARD.md): 모델 구조·검증·한계
- [docs/MULTIYEAR_DATA.md](docs/MULTIYEAR_DATA.md): 다년도 데이터 출처·단위·결측 처리
- [CHANGELOG.md](CHANGELOG.md): 무엇이 바뀌었는지
- [data_dictionary.md](data_dictionary.md): 데이터 열 설명

## 데이터·API 출처

- 제주 연료원별 발전량: https://www.data.go.kr/data/15138838/fileData.do
- 지역별 시간별 태양광·풍력: https://www.data.go.kr/tcs/dss/selectFileDataDetailView.do?publicDataPk=15065269
- 제주 시간별 계통수요: https://www.data.go.kr/data/15065239/fileData.do?recommendDataYn=Y
- 과거 날씨: https://open-meteo.com/en/docs/historical-weather-api
- 내일 날씨예보: https://open-meteo.com/en/docs/gfs-api
- 제주 5분 계통·신재생 실측: https://www.data.go.kr/data/15158505/openapi.do

다음 API를 무엇부터 붙여야 하는지는 [API_ROADMAP.md](API_ROADMAP.md)에 별도로 정리했습니다. 가장 먼저 필요한 것은 충전소 위치·운영상태이고, 실제 포인트 지급에는 공개 API가 아니라 충전사업자의 세션·결제 제휴 API가 필요합니다.

## 한 문장 주의사항

이 MVP는 **충전시간 추천과 Green Point 정책 시뮬레이션**이며, 실제 충전요금 할인·포인트 지급·탄소감축량·REC 인증·충전기 제어를 보장하지 않습니다.
