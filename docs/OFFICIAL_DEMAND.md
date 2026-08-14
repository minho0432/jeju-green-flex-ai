# 공식 전력수요 연동

## 실시간 (추가 API 없음)

기존 **제주 5분 계통 API** (`JejuSukub5mToday`) 필드:

| API 필드 | 코드 열 | 의미 |
|----------|---------|------|
| `currPwrTot` | `demand_generation_mw` → `actual_demand_mwh` | 현재(계통) 수요 |
| `renewPwrSolar` / `renewPwrWind` | 재생 실측 | 기존과 동일 |

앱·`adjust_forecast_with_live_renewables`가 **재생 + 수요** 오차로
남은 시간 예측을 보정한 뒤 **공급여력 Green Score**를 다시 계산한다.

## 학습 (파일)

1. [시간별 제주전력수요](https://www.data.go.kr/data/15065239/fileData.do) CSV 다운로드
2. 프로젝트에 저장 예: `data/raw/jeju_hourly_demand.csv`
3. 실행:

```bash
python scripts/merge_official_demand.py \
  --demand-csv data/raw/jeju_hourly_demand.csv \
  --train data/processed/train.csv \
  --out data/processed/train.csv
python scripts/improve_demand_green_score.py
```

공식 수요가 없으면 기존처럼 `solar+wind+lng+bio` **proxy**를 사용한다
(`model_utils.ensure_demand_column`).
