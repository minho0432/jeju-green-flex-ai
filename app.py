"""Jeju Green Flex AI (SMP UI removed). Reconstructs app from gzip+b64 fragments."""
from __future__ import annotations
import base64, gzip
from pathlib import Path

parts = sorted(Path(__file__).parent.glob("_app.gz.b64.*"))
if not parts:
    raise SystemExit("missing _app.gz.b64.* next to app.py — git pull origin main")
raw = gzip.decompress(base64.b64decode("".join(p.read_text() for p in parts)))
exec(compile(raw.decode("utf-8"), str(Path(__file__).resolve()), "exec"), globals())
