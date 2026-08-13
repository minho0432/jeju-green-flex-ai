"""Jeju Green Flex AI Streamlit entry — loads _app_part1/_app_part2 (SMP UI removed)."""
from pathlib import Path

_code = "".join(
    Path(__file__).with_name(n).read_text(encoding="utf-8")
    for n in ("_app_part1.py", "_app_part2.py")
)
exec(compile(_code, str(Path(__file__).resolve()), "exec"), globals())
