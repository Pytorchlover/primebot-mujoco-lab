"""Exercise the real FIND_WASHER method; no ROS2 imports or real motion."""

import json, sys, time, importlib.util
from pathlib import Path
from detector import OwlDetector
from primebot_iface import MockPrimeBot
from vision import HeuristicDetector

root = Path(__file__).resolve().parents[1]


def load_task(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class PhotoBot(MockPrimeBot):
    def __init__(self, files, log):
        super().__init__(log=log)
        self.files = files
        self.index = 0
        self.commands = []

    def capture_image(self, *args):
        p = self.files[self.index % len(self.files)]
        self.index += 1
        return p.read_bytes()

    def drive(self, f, l, a):
        self.commands.append(
            {"t": time.monotonic(), "forward": f, "lateral": l, "angular": a}
        )
        super().drive(f, l, a)

    def stop(self):
        self.commands.append(
            {"t": time.monotonic(), "forward": 0, "lateral": 0, "angular": 0}
        )
        super().stop()


from skill_demo_helpers import ConfirmDetector

model = OwlDetector(root / "model")
pos = root / "assets/washer_normal.jpg"
neg = root / "assets/no_washer.png"
for _ in range(5):
    model.detect(pos.read_bytes())
original = load_task("original_task", root / "upstream/src/task_carry_clothes.py")
patched = load_task("patched_task", root / "patched/task_carry_clothes.py")
results = []
for name, mod, files, required, threshold in [
    ("placeholder", original, [pos], 1, 0.5),
    ("real_positive", patched, [neg, pos, pos, pos], 3, 0.5),
    ("target_absent", patched, [neg], 3, 0.5),
    ("intermittent_target", patched, [pos, neg], 3, 0.5),
    ("occluded_strict", patched, [root / "assets/washer_occluded.jpg"], 3, 0.5),
    ("occluded_relaxed", patched, [root / "assets/washer_occluded.jpg"], 3, 0.12),
]:
    logs = []
    bot = PhotoBot(files, logs.append)
    model.threshold = threshold
    gate = ConfirmDetector(model, required)
    detector = HeuristicDetector(log=logs.append) if name == "placeholder" else gate
    cfg = mod.TaskConfig(scan_max_s=1.2, set_action_interval_s=0.0)
    if hasattr(cfg, "washer_confidence"):
        cfg.washer_confidence = threshold
    task = mod.CarryClothesTask(bot, detector=detector, cfg=cfg, log=logs.append)
    start = time.monotonic()
    try:
        task._s7_find_washer()
        outcome = "FOUND"
        reason = "FIND_WASHER returned normally; GO_WASHER would be next"
    except mod.TaskAborted as e:
        outcome = "ABORTED"
        reason = e.reason
    finally:
        bot.stop()
    result = {
        "case": name,
        "outcome": outcome,
        "reason": reason,
        "elapsed_s": time.monotonic() - start,
        "trace": gate.trace,
        "commands": bot.commands,
        "logs": logs,
        "frames": bot.index,
        "threshold": threshold,
    }
    results.append(result)
    print(name, outcome, bot.index, flush=True)
assert results[1]["outcome"] == "FOUND"
assert all(r["outcome"] == "ABORTED" for r in results[2:5])
assert results[0]["outcome"] == "ABORTED"
assert results[5]["outcome"] == "FOUND"
assert all(all(c["forward"] == 0 for c in r["commands"]) for r in results)
assert all(r["commands"][-1]["angular"] == 0 for r in results)
(root / "results/skill.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2)
)
