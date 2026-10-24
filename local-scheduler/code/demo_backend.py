"""Mock actuator events only. No ROS, audio output, shell, or network."""

import json, os, time


def run_action(context, action, args, duration):
    cfg = context.config_for("demo")

    def emit(state, **extra):
        event = {
            "t": time.monotonic(),
            "action": action,
            "args": args,
            "state": state,
            **extra,
        }
        fd = os.open(cfg["events_path"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(event, ensure_ascii=False) + "\n").encode())
        finally:
            os.close(fd)

    if not context.simulate:
        raise RuntimeError("This package supports Mock only")
    if cfg.get("backend") == "mujoco":
        from mujoco_backend import run_simulation
        return run_simulation(context, action, args, emit)
    emit("start")
    try:
        if cfg.get("fail_action") == action:
            emit("failed", reason="MOCK_ACTUATOR_BUSY")
            return {
                "status": "failed",
                "message": f"Mock {action} 失败：执行器忙，未完成动作。",
            }
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            context.checkpoint()
            time.sleep(0.02)
        emit("success")
        return {
            "status": "success",
            "message": f"Mock {action} 已完成；参数 {args}。",
            "simulated": True,
        }
    finally:
        emit("released")
