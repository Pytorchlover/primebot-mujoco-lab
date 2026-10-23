import argparse, copy, json, platform, time
from bootstrap import ROOT
from local_model import LocalChat
from primebuddy.core.engine import Context, execute, validate_plan
from primebuddy.skills.registry import Registry
import torch, transformers

CASES = [
    {"id": "wave", "text": "跟门口的客人挥挥手。"},
    {"id": "welcome", "text": "先说一句欢迎光临，然后挥手，最后跳1秒的舞。"},
    {"id": "heart", "text": "比个心，做完之后再跳两秒舞。"},
    {"id": "ambiguous", "text": "来一个吧。"},
    {"id": "unsupported", "text": "去厨房给客人倒杯水。"},
    {"id": "failure", "text": "先挥手，再跳一秒舞。", "fail_action": "gesture.wave"},
    {"id": "cancel", "text": "先跳三秒舞，然后比心。", "cancel_after_start": 0.3},
]


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_case(client, case):
    folder = ROOT / "results" / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "backend.jsonl"
    log.unlink(missing_ok=True)
    context = Context(
        runtime=ROOT / ".runtime" / case["id"],
        simulate=True,
        skill_configs={
            "demo": {"events_path": str(log), "fail_action": case.get("fail_action")}
        },
    )
    registry = Registry(ROOT / "skills")
    start = time.monotonic()
    call_start = len(client.calls)
    trace = []
    batches = []
    spoken = []
    final = ""
    error = None
    cancel_sent = False
    outcome = "complete"

    def emit(event):
        trace.append({"t": time.monotonic() - start, **event})

    def checkpoint():
        nonlocal cancel_sent
        if "cancel_after_start" in case and not cancel_sent:
            events = read_events(log)
            starts = [
                e
                for e in events
                if e["state"] == "start" and e["action"] == "dance.perform"
            ]
            if (
                starts
                and time.monotonic() - starts[0]["t"] >= case["cancel_after_start"]
            ):
                cancel_sent = True
                context.cancelled.set()
                emit({"state": "cancel_requested"})

    try:
        decision = client.decide(case["text"], registry)
        if decision["decision"] != "execute":
            proposal = {"reply": decision["reply"], "tasks": []}
        else:
            feedback = previous = None
            for attempt in range(2):
                proposal = client.propose(case["text"], registry, feedback, previous)
                try:
                    validate_plan(proposal, registry)
                    break
                except ValueError as exc:
                    if attempt == 1:
                        raise
                    feedback = str(exc)
                    previous = client.calls[-1]["raw"]
        if proposal["tasks"]:
            batch = {"tasks": copy.deepcopy(proposal["tasks"])}
            batches.append(batch)
            result = execute(
                proposal,
                registry,
                context,
                announce=spoken.append,
                emit=emit,
                checkpoint=checkpoint,
            )
            batch["results"] = result
            labels = {
                t["id"]: t["skill"] + "." + t["action"] for t in proposal["tasks"]
            }
            words = {
                "success": "模拟完成",
                "failed": "模拟失败",
                "skipped": "未执行（前置任务失败）",
            }
            final = (
                "；".join(
                    labels[i] + "：" + words.get(r["status"], r["status"])
                    for i, r in result.items()
                )
                + "。"
            )
        else:
            final = proposal["reply"]

    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        if cancel_sent:
            outcome = "cancelled"
            final = "已处理取消请求，当前 Mock 任务已退出，后续任务未启动。"
        else:
            outcome = "error"
    finally:
        registry.close()
    result = {
        **case,
        "outcome": outcome,
        "final": final,
        "error": error,
        "cancel_sent": cancel_sent,
        "elapsed_s": time.monotonic() - start,
        "batches": batches,
        "engine_events": trace,
        "announcements": spoken,
        "backend_events": read_events(log),
        "model_calls": client.calls[call_start:],
    }
    (folder / "trace.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(case["id"], outcome, round(result["elapsed_s"], 2), final, error, flush=True)
    return result


def guard_tests():
    registry = Registry(ROOT / "skills")

    def task(i, skill="gesture", after=None):
        return {
            "id": i,
            "skill": skill,
            "action": "perform",
            "args": {"name": "wave"},
            "after": after or [],
        }

    tests = {
        "unknown_skill": [task("a", "teleport")],
        "same_body_without_order": [task("a"), task("b")],
        "dependency_cycle": [task("a", after=["b"]), task("b", after=["a"])],
    }
    results = []
    for name, tasks in tests.items():
        try:
            validate_plan({"reply": "", "tasks": tasks}, registry)
        except ValueError as exc:
            results.append({"case": name, "rejected": True, "reason": str(exc)})
        else:
            results.append({"case": name, "rejected": False})
    registry.close()
    assert all(r["rejected"] for r in results)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text")
    p.add_argument("--suite", action="store_true")
    p.add_argument("--case", choices=[c["id"] for c in CASES])
    a = p.parse_args()
    if not (a.text or a.suite or a.case):
        p.error("Use --text, --suite, or --case")
    client = LocalChat()
    torch.cuda.reset_peak_memory_stats()
    cases = (
        CASES
        if a.suite
        else (
            [next(c for c in CASES if c["id"] == a.case)]
            if a.case
            else [{"id": "interactive", "text": a.text}]
        )
    )
    results = [run_case(client, c) for c in cases]
    report = {
        "cases": results,
        "guard_tests": guard_tests(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
            "model": "Qwen/Qwen3-4B",
            "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "primebuddy_revision": "8731c2fcbe4aed7183882cab14f403df96bab62e",
            "do_sample": False,
            "planner": "intent_then_schema_constrained",
            "enable_thinking": {"intent": True, "plan": False},
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    out = ROOT / "results" / ("suite.json" if a.suite else "latest.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(out, flush=True)


if __name__ == "__main__":
    main()
