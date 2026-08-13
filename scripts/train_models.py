"""Decode and run full train_models implementation."""
from __future__ import annotations
from pathlib import Path
import base64
import runpy

_here = Path(__file__).resolve().parent
parts = sorted(_here.glob("_tm.b64.*"))
if not parts:
    raise SystemExit("missing _tm.b64.* fragments — git pull origin main")
_target = _here / "_train_models_full.py"
_target.write_bytes(base64.b64decode("".join(p.read_text() for p in parts)))
if __name__ == "__main__":
    runpy.run_path(str(_target), run_name="__main__")
else:
    _ns = runpy.run_path(str(_target), run_name="train_models")
    globals().update({k: v for k, v in _ns.items() if not k.startswith("__")})
