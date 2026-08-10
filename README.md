# 이민호 담당 데이터 패키지

제주 시간별 SMP, 연료원별 발전량, 날씨를 하나의 AI 학습용 파일로 합치는 패키지입니다.

## 이미 포함된 원본 데이터

- 2025년 제주 태양광 발전량
- 2025년 제주 풍력 발전량
- 2025년 제주 LNG 발전량
- 2025년 제주 바이오중유 발전량
- 2025년 제주 날씨

## 실행 방법

### 맥에서 지금부터 실행할 명령

Pandas와 데이터 검사를 통과했다면 VS Code 터미널에서 아래 두 줄만 실행합니다.

```bash
python scripts/train_models.py
python -m streamlit run app.py
```

브라우저가 자동으로 열리면 왼쪽에서 차량 조건을 바꾸며 데모합니다.

더블클릭으로 실행하려면 `RUN_DEMO_MAC.command`를 사용합니다. macOS가 실행 권한을 요구하면
터미널에서 한 번만 `chmod +x RUN_DEMO_MAC.command SETUP_MAC.command`를 실행합니다.

### 윈도우에서 가장 쉬운 방법

1. 압축을 푼 뒤 `SETUP_WINDOWS.bat`을 먼저 더블클릭합니다.
2. 마지막에 `설치와 검사가 모두 성공했습니다`가 나오는지 확인합니다.
3. 데이터를 다시 만들고 싶을 때 `RUN_WINDOWS.bat`을 더블클릭합니다.
4. VS Code에서 `Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv\\Scripts\\python.exe`를 선택합니다.
5. `.ipynb` 파일을 열었다면 오른쪽 위 `커널 선택`에서도 같은 `.venv`를 선택합니다.

### 직접 실행하는 방법

터미널에서 이 폴더로 이동한 뒤 아래 명령을 순서대로 실행합니다.

```bash
pip install -r requirements.txt
python scripts/download_smp.py
python scripts/prepare_data.py
python scripts/validate_data.py
```

## 최종 결과

AI 담당자에게 아래 파일을 전달합니다.

```text
data/processed/train.csv
data/demo/demo_forecast.csv
data_dictionary.md
```

AI 학습 후에는 아래 파일도 생성됩니다.

```text
models/smp_model.joblib
models/renewable_mwh_model.joblib
outputs/model_metrics.json
outputs/demo_predictions.csv
```

## 데모에서 사실대로 말해야 하는 것

- SMP는 전력 도매가격 지표이며 소비자의 실제 충전요금과 같지 않습니다.
- Green Reward는 운영자 캠페인 예산을 가정한 시뮬레이션입니다.
- 실제 지급에는 충전사업자·렌터카사·지자체 중 한 곳과의 제휴가 필요합니다.
- 실제 충전기 제어와 REC 인증을 구현했다고 주장하지 않습니다.

## 이민호님이 팀에 설명할 말

> 제주 시간별 SMP, 태양광·풍력 발전량과 날씨를 같은 시간 기준으로 합쳐 AI 학습용 데이터셋을 만들었습니다. 전력거래소의 시간 표기 차이를 맞췄고, 중복·결측값과 재생에너지 합계도 검사했습니다.

## 공식 출처

- 제주 SMP: https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202
- 제주 발전량: https://www.data.go.kr/data/15138838/fileData.do
- 날씨: https://open-meteo.com/en/docs/historical-weather-api
