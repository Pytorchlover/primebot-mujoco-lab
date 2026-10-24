"""Verify real stepping, sequential state continuity and cancellation in MuJoCo."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root = ROOT / "results/mujoco"
suite = json.loads((root / "suite.json").read_text())
checks = []


def check(name, passed):
    checks.append({"check": name, "passed": bool(passed)})


expected = {
    "wave": ["gesture.wave"],
    "welcome": ["greeting.say", "gesture.wave", "dance.perform"],
    "heart": ["gesture.heart", "dance.perform"],
    "ambiguous": [],
    "unsupported": [],
    "failure": ["gesture.wave"],
    "cancel": ["dance.perform"],
}
for c in suite["cases"]:
    starts = [e for e in c["backend_events"] if e["state"] == "start"]
    check(
        c["id"] + " expected order", [e["action"] for e in starts] == expected[c["id"]]
    )
    check(
        c["id"] + " valid outcome",
        c["outcome"] == ("cancelled" if c["id"] == "cancel" else "complete"),
    )
    previous = None
    for e in starts:
        p = root / c["id"] / "simulation" / e["action"]
        m = json.loads((p / "metrics.json").read_text())
        if previous is not None:
            check(
                c["id"] + " state continuity", abs(m["start_sim_s"] - previous) < 1e-8
            )
        previous = m["end_sim_s"]
        check(
            c["id"] + " actual sim clock",
            abs(m["end_sim_s"] - m["start_sim_s"] - m["physics_steps"] * 0.001) < 1e-7,
        )
        if m["status"] == "success":
            check(c["id"] + " stepped", m["physics_steps"] > 0 and m["frames"] > 0)
            a = np.load(p / "trajectory.npz")["samples"]
            check(c["id"] + " finite state", np.isfinite(a).all())
            if e["action"] != "greeting.say":
                check(
                    c["id"] + " visible joint motion", m["max_joint_excursion_deg"] > 5
                )
        if c["id"] == "cancel":
            check(
                "cancel stops before 3 seconds",
                0 < m["end_sim_s"] - m["start_sim_s"] < 3
                and m["status"] == "cancelled",
            )
    if c["id"] == "failure":
        check(
            "failure skips dance",
            c["batches"][0]["results"]["t2"]["status"] == "skipped",
        )
    check(
        c["id"] + " released",
        sum(e["state"] == "released" for e in c["backend_events"]) == len(starts),
    )
(root / "checks.json").write_text(json.dumps(checks, indent=2))
for c in checks:
    print("PASS" if c["passed"] else "FAIL", c["check"])
assert all(c["passed"] for c in checks)
