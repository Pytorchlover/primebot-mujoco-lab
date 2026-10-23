"""Keep the downloaded upstream intact; make a small inspectable local fork."""

from pathlib import Path
import difflib

root = Path(__file__).resolve().parents[1]
src = root / "upstream/src/task_carry_clothes.py"
old = src.read_text()
new = old.replace(
    "scan_yaw_speed: float = 0.35",
    "washer_confidence: float = 0.5          # Configurable score gate; keep upstream default\n    scan_yaw_speed: float = 0.35",
)
needle = "and d.confidence >= 0.5 for d in dets):"
assert old.count(needle) == 1
new = new.replace(
    needle, "and d.confidence >= self.cfg.washer_confidence for d in dets):"
)
assert new != old
(root / "patched").mkdir(exist_ok=True)
(root / "patched/task_carry_clothes.py").write_text(new)
(root / "results/upstream.patch").write_text(
    "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="upstream/src/task_carry_clothes.py",
            tofile="patched/task_carry_clothes.py",
        )
    )
)
