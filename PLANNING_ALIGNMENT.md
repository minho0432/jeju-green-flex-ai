# 기획서 대조 (Green Time JEJU)

기준 문서: [프로젝트 기획서](https://docs.google.com/document/d/1Tik5HTZQu5YIWoi4o3886RAq5pQx2UUPnpBcGlkB3F4/edit)

## 핵심 결정 (기획서 = 구현 기준)

| 항목 | 기획서 | 구현 방향 |
|------|--------|----------|
| 예측 대상 | 재생(태양광+풍력) + **전력수요** | RE + Demand |
| **SMP** | 공급여력 정의에 **없음** | **사용하지 않음** |
| Green Score | 공급여력의 과거 계절 대비 백분위 0~100 | RE/Demand 백분위 |
| Green Time | 상위 공급여력 / 70점+ 후보, 85+ 우선 | 동일 |
| 보수적 점수 | RE 하한, 수요 상한 | asymmetric interval |
| 충전 우선순위 | **목표 SOC > 친환경** | 제약 최적화 |
| 리워드 | E(≥70) × 100P/kWh × (S/100) | 정책 시뮬레이션 |
| 날씨 | Open-Meteo | 연동 |
| 실측 보정 | KPX 5분 → 남은 계획만 재계산 | 로직 |

## 점수 해석 (기획서)

- 0~49: 공급여력 낮음
- 50~69: 일반
- 70~84: Green Time 추천 후보
- 85~100: 우선 추천

## 서비스 흐름

데이터 수집 → 발전량·전력수요 예측 → Green Score → 차량 SOC·출발 입력 → 추천 충전시간 → 실측 시 남은 구간만 수정

## 의도적 미구현

- 충전사업자 세션 API / 실제 포인트 지급
- 공식 탄소·REC 인증
- 충전기 원격 제어

## 모델 방침

기획서: 트리 기반, HistGBR·XGBoost·LightGBM MAE 비교 후 결정.  
SMP residual / ExtraTrees SMP 경로는 제품 정의에서 제외한다.
