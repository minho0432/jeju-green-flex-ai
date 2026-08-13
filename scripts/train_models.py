"""Train entry: fetch last complete train_models from git history, inject model_builders."""
from __future__ import annotations

import re
import runpy
import tempfile
from pathlib import Path

import urllib.request

_COMMIT = "c76d497b8e0c63e0404a460a51c0ebe466b739d5"
_URL = (
    f"https://raw.githubusercontent.com/minho0432/jeju-green-flex-ai/{_COMMIT}/"
    "scripts/train_models.py"
)

_OLD_BUILD = '''def build_model(target: str):
    if target == "smp":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        )
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
'''

_NEW_BUILD = "from model_builders import build_model  # LightGBM RE; SMP not in Green Score\n"


def _load_source() -> str:
    with urllib.request.urlopen(_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    if _OLD_BUILD not in text:
        raise RuntimeError("expected build_model block not found in historical train_models")
    text = text.replace(_OLD_BUILD, _NEW_BUILD, 1)
    text = text.replace(
        '"note": "재생에너지는 핵심 신호, SMP는 소비자 요금이 아닌 보조 시장지표"',
        '"note": "기획서 기준 Green Score는 재생에너지 100%. SMP는 점수 미사용"',
    )
    return text


_source = _load_source()
_path = Path(__file__).with_name("_train_models_full.py")
_path.write_text(_source, encoding="utf-8")

if __name__ == "__main__":
    runpy.run_path(str(_path), run_name="__main__")
else:
    _ns = runpy.run_path(str(_path), run_name="train_models")
    globals().update({k: v for k, v in _ns.items() if not k.startswith("__")})
