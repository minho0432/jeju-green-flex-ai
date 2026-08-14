# 수요·공급여력 Green Score

## 정의
- **수요 대리변수** `demand_mwh` = solar + wind + lng + bio
  (공식 시간별 전력수요 시계열이 없을 때 시장 참여 발전 합계 사용)
- **공급여력** = renewable_mwh / max(demand_mwh, 50)
- **Green Score** = 공급여력의 과거 분포 대비 백분위 (0~100)
- 보수 점수 = (재생 하한) / (수요 상한) 백분위

## 검증 요약 (시간순 4-fold)
- 수요 AI MAE ≈ 54.5 vs 기준표 ≈ 70.9
- 공급여력 Green Time 겹침 AI ≈ 90.4% vs 기준표 ≈ 86.6%

## 재실행
```bash
python scripts/improve_demand_green_score.py
```
