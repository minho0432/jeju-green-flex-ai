# JEJU Green Time — 재생 예측 개선 결과

## 한 줄
시간순 검증 기준 **순수 AI(특성 보강 LightGBM)** 가 월·시간 기준표보다 MAE **약 42.7% 낮음**.  
하이브리드는 이 데이터에서는 α=1.0(AI 100%)이 최선.

## 지표 (4-fold, ~30일씩)

| 방법 | MAE | Green Time 상위30% 겹침 |
|------|-----|-------------------------|
| 월·시간 기준표 | 18.04 | 88.3% |
| AI (보강 특성) | **10.34** | **92.5%** |
| 예보만 특성 (라이브) | 10.84 | — |

## 적용 내용
1. 특성: lag 48h, 월 주기, 주간/야간, 일사×풍속, 제곱항
2. 하이브리드 α 그리드 탐색 → **α=1.0 채택**
3. Green Time overlap 지표 추가
4. `scripts/improve_renewable.py` 재실행 가능
5. `outputs/hybrid_alpha.json`, `improved_renewable_metrics.json` 저장

## 한계
- 관측 날씨 기준 검증 (실예보 오차 미포함)
- KPX 당일 보정은 API 키 있을 때 앱에서 추가 가능
- **100% 정확도는 불가능** (기상 불확실성)

## 재실행
```bash
python scripts/improve_renewable.py
```
