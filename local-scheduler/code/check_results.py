"""Check behavior against requests defined before running the model."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
suite = json.loads((ROOT / "results/suite.json").read_text())
checks = []


def check(name, ok):
    checks.append({"check": name, "passed": bool(ok)})


for c in suite["cases"]:
    events = c["backend_events"]
    starts = [e["action"] for e in events if e["state"] == "start"]
    i = c["id"]
    check(i + ": no runtime error", c["outcome"] in ["complete", "cancelled"])
    if i == "wave":
        check("wave: correct action", starts == ["gesture.wave"])
    if i == "welcome":
        check(
            "welcome: exact order",
            starts == ["greeting.say", "gesture.wave", "dance.perform"],
        )
    if i == "heart":
        check("heart: exact order", starts == ["gesture.heart", "dance.perform"])
    if i in ["ambiguous", "unsupported"]:
        check(i + ": zero actions", not starts)
    if i == "failure":
        check("failure: no dance and no retry", starts == ["gesture.wave"])
        check(
            "failure: reported by backend", any(e["state"] == "failed" for e in events)
        )
    if i == "cancel":
        check("cancel: no following heart", starts == ["dance.perform"])
        check(
            "cancel: interrupted and released",
            c["outcome"] == "cancelled"
            and any(e["state"] == "released" for e in events),
        )
    # For this deliberately sequential scenario, each start must follow prior release.
    active = set()
    for e in events:
        if e["state"] == "start":
            check(i + ": sequential execution", not active)
            active.add(e["action"])
        elif e["state"] == "released":
            active.discard(e["action"])
    check(i + ": mock released", not active)
check(
    "guard tests reject all three invalid plans",
    all(x["rejected"] for x in suite["guard_tests"]),
)
(ROOT / "results/checks.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2)
)
for c in checks:
    print(("PASS" if c["passed"] else "FAIL"), c["check"])
if not all(c["passed"] for c in checks):
    raise SystemExit(1)
