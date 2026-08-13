"""JEJU Green Time train entry.

Builds LightGBM models via model_builders and writes metrics under outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    try:
        from model_builders import build_models  # type: ignore
    except Exception:
        build_models = None
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "model_metrics.json"
    if build_models is not None:
        metrics = build_models()
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("trained", metrics_path)
    else:
        print("model_builders not available; skip full train")

if __name__ == "__main__":
    main()
