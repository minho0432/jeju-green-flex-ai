# 🌱 JEJU Green Time

제주 개인 전기차 사용자가 출발 전 목표 SOC까지 충전할 때, **재생에너지가 많은 Green Time**을 우선 추천하는 MVP입니다.

> 저장소 URL 이름은 기존 `jeju-green-flex-ai` 를 유지합니다. 제품·화면·문서 명칭은 모두 **JEJU Green Time** 입니다.

## 한 줄 소개
재생에너지(태양광+풍력) 예측으로 친환경 충전 구간을 고르고, 목표 배터리(SOC)를 맞춥니다.

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 시연 모드
- 검증된 과거 재현 (키 없이 가능)
- 오늘 공식 실시간 관측 (공공데이터포털 키 필요)
- 실시간 보정 재현
- 내일 예보 실험 (Open-Meteo, 키 없음)

## API 키 (선택)
실측 보정만 필요합니다. Streamlit Secrets:
```toml
DATA_GO_KR_SERVICE_KEY = "디코딩_일반인증키"
```
발급: https://www.data.go.kr/data/15158505/openapi.do

## 모델
- 핵심: **LightGBM** 재생에너지 예측
- Green Score: 재생 공급 기회(백분위) 0~100
- SMP는 점수·추천에 사용하지 않습니다.

## 링크
- 저장소: https://github.com/minho0432/jeju-green-flex-ai
- KPX 제주 5분 API: https://www.data.go.kr/data/15158505/openapi.do
- Open-Meteo: https://open-meteo.com
