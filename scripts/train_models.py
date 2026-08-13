"""Train entry — loads split parts to avoid oversized single blob on deploy."""
from __future__ import annotations
from pathlib import Path

_code = "".join(
    Path(__file__).with_name(name).read_text(encoding="utf-8")
    for name in ("_train_part1.py", "_train_part2.py")
)
exec(compile(_code, str(Path(__file__).with_name("_train_combined.py")), "exec"), globals())
