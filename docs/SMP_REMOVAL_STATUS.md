# SMP 제거 반영 상태 (2026-08-13)

## 이미 main에 반영됨
- `scripts/model_utils.py`: Green Score = 재생 100% (`MARKET_WEIGHT=0`)
- `scripts/model_builders.py`: 재생 LightGBM
- 추천 점수·리워드 로직에서 SMP 미사용

## 앱 UI (SMP 차트/탭/문구 제거 완료본)

완성된 Streamlit 앱은 로컬 아티팩트 `app_NO_SMP.py`와 동일합니다.
저장소 루트 `app.py`를 아래로 교체하세요.

```bash
# 프로젝트 루트에서
# app_NO_SMP.py 내용을 app.py로 복사한 뒤
streamlit run app.py
```

### 화면에서 제거한 것
- 메인 차트 SMP 보조축
- SMP 예측 비교 탭 / SMP 보정 탭
- 표의 예측 SMP 열
- "SMP와 가격절약" 설명 expander
- 성능 검증의 SMP MAE 행
- 실시간 SMP 편차 메트릭

### 남는 메시지
- Green Score = 재생 공급 기회(과거 대비 백분위)만 사용
- 사용자 혜택 = Green Point만 표시
