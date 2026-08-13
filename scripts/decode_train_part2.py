"""Decode _train_part2 from base64 fragments."""
from pathlib import Path
import base64
parts = sorted(Path(__file__).parent.glob("_train_part2.b64.*"))
data = "".join(p.read_text() for p in parts)
Path(__file__).with_name("_train_part2.py").write_bytes(base64.b64decode(data))
print("wrote _train_part2.py", len(data))
