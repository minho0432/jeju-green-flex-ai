"""Restore scripts/train_models.py from base64 chunks."""
from pathlib import Path
import base64
root = Path(__file__).parent
parts = sorted(root.glob("_tm.b64.*"))
data = "".join(p.read_text() for p in parts)
(root / "train_models.py").write_bytes(base64.b64decode(data))
print("restored train_models.py", (root / "train_models.py").stat().st_size)
