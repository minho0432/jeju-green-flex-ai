"""Jeju Green Flex AI Streamlit app (SMP UI removed per planning doc)."""
from __future__ import annotations
import base64
from pathlib import Path

def _load(prefix: str) -> str:
    parts = sorted(Path(__file__).parent.glob(f"{prefix}.b64.*"))
    if not parts:
        raise SystemExit(f"missing {prefix}.b64.* next to app.py")
    return base64.b64decode("".join(p.read_text() for p in parts)).decode("utf-8")

_code = _load("_app_part1") + _load("_app_part2")
exec(compile(_code, str(Path(__file__).resolve()), "exec"), globals())
