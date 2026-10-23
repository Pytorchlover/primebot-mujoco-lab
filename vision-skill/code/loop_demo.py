"""50 Hz Mock command loop + independent inference, including camera freeze.
Only a yaw alignment example: never approaches a target from uncalibrated photos.
"""

import json, threading, time
from pathlib import Path
from detector import OwlDetector
from skill_demo_helpers import ConfirmDetector
from primebot_iface import MockPrimeBot

root = Path(__file__).resolve().parents[1]
det = OwlDetector(root / "model")
image = (root / "assets/washer_normal.jpg").read_bytes()
for _ in range(5):
    det.detect(image)
gate = ConfirmDetector(det)
lock = threading.Lock()
latest = None
stop = threading.Event()
ready = threading.Event()
start = time.monotonic()
ttl = 0.30
freeze = 1.2
duration = 2.0
errors = []
observations = []


def worker():
    global latest
    try:
        while not stop.is_set():
            captured = time.monotonic()
            if captured - start >= freeze:
                return  # Deliberately stop new camera frames / inference publications.
            detections = gate.detect(image)
            published = time.monotonic()
            with lock:
                latest = (captured, published, detections)
            observations.append(
                {
                    "capture_s": captured - start,
                    "publish_s": published - start,
                    "accepted": bool(detections),
                }
            )
            ready.set()
    except Exception as exc:
        errors.append(repr(exc))
        stop.set()


thread = threading.Thread(target=worker, daemon=True)
thread.start()
bot = MockPrimeBot(log=lambda _: None)
records = []
tick = start
try:
    while time.monotonic() - start < duration:
        now = time.monotonic()
        with lock:
            snapshot = latest
        age = (now - snapshot[0]) if snapshot else None
        fresh = snapshot is not None and age <= ttl
        accepted = fresh and bool(snapshot[2])
        yaw = max(-0.25, min(0.25, snapshot[2][0].cx * 0.4)) if accepted else 0.0
        bot.drive(0.0, 0.0, yaw)
        records.append(
            {
                "t": now - start,
                "age_s": age,
                "fresh": fresh,
                "accepted": bool(accepted),
                "yaw": yaw,
            }
        )
        tick += 0.02
        delay = tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            tick = (
                time.monotonic()
            )  # Skip missed deadlines; do not issue a burst of old commands.
finally:
    bot.stop()
    stop.set()
    thread.join(timeout=10)
assert not thread.is_alive()
assert not errors, errors
assert any(r["accepted"] for r in records)
assert all(
    not r["accepted"] and r["yaw"] == 0
    for r in records
    if r["t"] >= freeze + ttl + 0.05
)
(root / "results/loop.json").write_text(
    json.dumps(
        {
            "ttl_s": ttl,
            "camera_freeze_s": freeze,
            "records": records,
            "observations": observations,
            "worker_errors": errors,
            "stopped_in_finally": True,
        },
        indent=2,
    )
)
print("ticks", len(records), "observations", len(observations), "stale stop passed")
