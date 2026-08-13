# train_models.py 복구 안내

일시적으로 `scripts/train_models.py`가 스텁 상태일 수 있습니다.

## 즉시 복구

1. 로컬 아티팩트 또는 이 저장소 이슈에 올린 전체 `train_models.py`를 사용
2. 또는 아래 명령:

```bash
git checkout c76d497 -- scripts/train_models.py
# 그 다음 model_builders import 한 줄 추가 (재생 = LightGBM)
```

## 이미 반영된 기획서 정렬 (main)

- `model_utils.py`: `RENEWABLE_WEIGHT=1.0`, `MARKET_WEIGHT=0.0` → **Green Score = 재생만**
- `model_builders.py`: 재생 **LightGBM HPO** 설정
- `requirements.txt`: `lightgbm>=4.0`
- `outputs/model_metrics.json`: 재학습 지표
- `PLANNING_ALIGNMENT.md`: 기획서 대조

SMP는 **추천 점수·리워드에 사용하지 않습니다.**
