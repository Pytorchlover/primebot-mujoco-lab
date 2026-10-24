"""Physical stepping of a fixed-base Q1; no live hardware or learned balance policy."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image


def make_model(source):
    source = Path(source).resolve()
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", str(source.parent.parent / "meshes"))
    pelvis = root.find(".//body[@name='pelvis']")
    pelvis.remove(pelvis.find("freejoint"))
    pelvis.set("pos", "0 0 0.55")
    for geom in pelvis.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    world = root.find("worldbody")
    for parent in root.iter():
        for old_floor in parent.findall("geom[@name='floor']"):
            parent.remove(old_floor)
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="3 3 .1",
        rgba=".84 .88 .88 1",
        group="1",
    )
    ET.SubElement(
        world,
        "geom",
        name="support",
        type="box",
        pos="-.14 0 .28",
        size=".035 .055 .28",
        rgba=".20 .32 .36 1",
        group="1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        name="base",
        type="cylinder",
        pos="0 0 .015",
        size=".34 .015",
        rgba=".28 .42 .43 1",
        group="1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(world, "light", pos="1 -1 3", dir="-.3 .2 -1", diffuse=".8 .8 .8")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    m.vis.global_.offwidth = 960
    m.vis.global_.offheight = 720
    return m


def neutral(names):
    q = np.zeros(len(names))
    for side in ["left", "right"]:
        q[names.index(side + "_elbow_joint")] = 1.1
    return q


def target(action, t, duration, names):
    q = neutral(names)

    def put(name, val):
        q[names.index(name + "_joint")] = val

    # Smooth entry/exit; target is never assigned directly to the simulated state.
    envelope = min(1.0, t / 0.45, max(0.0, (duration - t) / 0.45))
    envelope = envelope * envelope * (3 - 2 * envelope)
    if action == "gesture.wave":
        put("left_shoulder_pitch", -1.25)
        put("left_shoulder_roll", 0.8 + 0.32 * np.sin(2 * np.pi * 1.5 * t))
        put("left_elbow", -0.35)
    elif action == "gesture.heart":
        for side, sign in [("left", 1), ("right", -1)]:
            put(side + "_shoulder_pitch", -0.65)
            put(side + "_shoulder_roll", sign * 0.25)
            put(side + "_shoulder_yaw", -sign * 1.0)
            put(side + "_elbow", -0.55)
    elif action == "dance.perform":
        phase = np.sin(2 * np.pi * t)
        put("waist_roll", 0.15 * phase)
        put("waist_yaw", 0.28 * np.sin(2 * np.pi * t + 0.4))
        for side, sign in [("left", 1), ("right", -1)]:
            put(side + "_shoulder_pitch", -0.65 - 0.3 * sign * phase)
            put(side + "_shoulder_roll", sign * (0.65 + 0.3 * sign * phase))
            put(side + "_elbow", 0.1 + 0.35 * sign * phase)
    base = neutral(names)
    return base + envelope * (q - base)


def run_simulation(context, action, args, emit):
    cfg = context.config_for("demo")
    out = Path(cfg["sim_dir"])
    out.mkdir(parents=True, exist_ok=True)
    m = make_model(cfg["model_xml"])
    d = mujoco.MjData(m)
    joints = m.actuator_trnid[:, 0]
    qa = m.jnt_qposadr[joints]
    va = m.jnt_dofadr[joints]
    assert np.allclose(m.actuator_gear[:, 0], 1)
    names = [m.joint(int(j)).name for j in joints]
    state_path = out / "state.npz"
    if state_path.exists():
        old = np.load(state_path)
        d.qpos[:] = old["qpos"]
        d.qvel[:] = old["qvel"]
        d.time = float(old["time"])
    else:
        d.qpos[qa] = neutral(names)
    mujoco.mj_forward(m, d)
    duration = (
        2.4
        if action.startswith("gesture.")
        else (float(args["seconds"]) if action == "dance.perform" else 0.8)
    )
    path = out / action
    path.mkdir(exist_ok=True)
    renderer = mujoco.Renderer(m, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.58]
    camera.distance = 1.65
    camera.azimuth = 155
    camera.elevation = -12
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    option.geomgroup[1] = 1
    writer = imageio.get_writer(path / "motion.mp4", fps=25, macro_block_size=1)
    rows = []
    frames = []
    start_sim = float(d.time)
    start_wall = time.monotonic()
    status = "cancelled"
    steps = 0
    emit("start", backend="mujoco", sim_time=start_sim)
    try:
        if cfg.get("fail_action") == action:
            status = "failed"
            emit(
                "failed",
                reason="INJECTED_ACTUATOR_BUSY",
                backend="mujoco",
                sim_time=float(d.time),
            )
            return {"status": "failed", "message": "MuJoCo 动作启动前注入执行器忙"}
        for k in range(round(duration / 0.04)):
            context.checkpoint()
            for _ in range(40):
                t = float(d.time) - start_sim
                q = target(action, t, duration, names)
                q = np.clip(q, m.jnt_range[joints, 0], m.jnt_range[joints, 1])
                # Gravity compensation plus torque-limited PD, not kinematic playback.
                tau = 35 * (q - d.qpos[qa]) - 1.2 * d.qvel[va] + d.qfrc_bias[va]
                d.ctrl[:] = np.clip(
                    tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
                )
                mujoco.mj_step(m, d)
                steps += 1
            if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
                raise RuntimeError("Invalid MuJoCo state")
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=camera, scene_option=option)
            rgb = renderer.render().copy()
            writer.append_data(rgb)
            Image.fromarray(rgb).save(path / f"frame_{k:03d}.png")
            frames.append(k)
            rows.append([float(d.time), *q, *d.qpos[qa]])
            # Keep wall-time cancellation observable; slow renderers may run below real time.
            remaining = (k + 1) * 0.04 - (time.monotonic() - start_wall)
            if remaining > 0:
                time.sleep(remaining)
        status = "success"
        emit("success", backend="mujoco", sim_time=float(d.time), physics_steps=steps)
        return {
            "status": "success",
            "message": "MuJoCo 关节动作完成",
            "physics_steps": steps,
        }
    except BaseException:
        emit(
            "interrupted", backend="mujoco", sim_time=float(d.time), physics_steps=steps
        )
        raise
    finally:
        writer.close()
        renderer.close()
        np.savez(state_path, qpos=d.qpos, qvel=d.qvel, time=d.time)
        np.savez(
            path / "trajectory.npz",
            samples=np.asarray(rows),
            joint_names=np.asarray(names),
        )
        metrics = {
            "action": action,
            "status": status,
            "physics_steps": steps,
            "start_sim_s": start_sim,
            "end_sim_s": float(d.time),
            "frames": len(frames),
            "timestep": float(m.opt.timestep),
            "base": "fixed pelvis, gravity enabled, robot collisions disabled",
            "control": "35*(q_ref-q)-1.2*qvel+qfrc_bias, clipped to actuator limits",
        }
        if rows:
            a = np.asarray(rows)
            n = len(names)
            metrics["max_joint_excursion_deg"] = float(
                np.rad2deg(np.ptp(a[:, 1 + n :], axis=0)).max()
            )
            metrics["tracking_rmse_deg"] = float(
                np.rad2deg(np.sqrt(np.mean((a[:, 1 : 1 + n] - a[:, 1 + n :]) ** 2)))
            )
        (path / "metrics.json").write_text(json.dumps(metrics, indent=2))
        emit("released", backend="mujoco", sim_time=float(d.time))
