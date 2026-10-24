"""Render recorded physics states from a front camera, then build article figures."""

from mujoco_backend import make_model
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "results/mujoco"
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)
m = make_model(ROOT / "sources/q1-urdf/xml/q1_v3_mc.xml")
d = mujoco.MjData(m)
r = mujoco.Renderer(m, height=720, width=960)
c = mujoco.MjvCamera()
c.lookat[:] = [0, 0, 0.58]
c.distance = 1.65
c.azimuth = 155
c.elevation = -12
opt = mujoco.MjvOption()
opt.geomgroup[0] = 0
opt.geomgroup[1] = 1
for file in sorted(base.glob("*/simulation/*/trajectory.npz")):
    a = np.load(file)["samples"]
    if not len(a):
        continue
    with imageio.get_writer(
        file.parent / "motion.mp4", fps=25, macro_block_size=1
    ) as writer:
        for k, row in enumerate(a):
            d.qpos[:] = row[1 + m.nu :]
            d.time = row[0]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=c, scene_option=opt)
            rgb = r.render().copy()
            Image.fromarray(rgb).save(file.parent / f"frame_{k:03d}.png")
            writer.append_data(rgb)
r.close()
plt.rcParams.update({"font.size": 13, "figure.facecolor": "#f5f7f6"})


def panel(ax, case, action, k, title):
    p = base / case / "simulation" / action
    a = np.load(p / "trajectory.npz")["samples"]
    k = min(k, len(a) - 1)
    ax.imshow(Image.open(p / f"frame_{k:03d}.png"))
    ax.axis("off")
    ax.set_title(f"{title}\nMuJoCo time = {a[k, 0]:.2f} s", pad=12)


fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
panel(axs[0], "welcome", "greeting.say", 10, "1. Welcome text / neutral pose")
panel(axs[1], "welcome", "gesture.wave", 20, "2. Wave: left arm raised")
panel(axs[2], "welcome", "dance.perform", 12, "3. Dance: waist + both arms")
fig.suptitle("Local Qwen plan → PrimeBuddy engine → Q1 MuJoCo motion", fontsize=19)
fig.text(
    0.5,
    0.035,
    "Fixed pelvis • gravity + torque-limited PD • robot collisions disabled • rendered from recorded physics states",
    ha="center",
    fontsize=11,
)
fig.subplots_adjust(top=0.82, bottom=0.09, wspace=0.02)
fig.savefig(figures / "mujoco_sequence.png", dpi=160)
plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
for ax, k, title in zip(axs, [1, 20, 57], ["Raise arm", "Wave", "Return to neutral"]):
    panel(ax, "wave", "gesture.wave", k, title)
fig.suptitle("gesture.wave: the same skill call now drives Q1 joints", fontsize=19)
fig.text(
    0.5,
    0.035,
    "2.4 seconds of physics, 2,400 mj_step calls; snapshots are measured poses, not target poses.",
    ha="center",
    fontsize=12,
)
fig.subplots_adjust(top=0.82, bottom=0.09, wspace=0.02)
fig.savefig(figures / "mujoco_wave.png", dpi=160)
plt.close(fig)
# Measured trajectory verifies physical movement rather than just matching pictures.
p = base / "welcome/simulation/gesture.wave"
a = np.load(p / "trajectory.npz")
names = list(a["joint_names"])
x = a["samples"]
n = len(names)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, joint in zip(axs, ["left_shoulder_pitch_joint", "left_shoulder_roll_joint"]):
    j = names.index(joint)
    ax.plot(x[:, 0], np.rad2deg(x[:, 1 + j]), "--", label="PD target", color="#84999b")
    ax.plot(
        x[:, 0], np.rad2deg(x[:, 1 + n + j]), label="MuJoCo observed", color="#226c54"
    )
    ax.set(title=joint, xlabel="Simulation time (s)", ylabel="Angle (degrees)")
    ax.legend()
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(figures / "mujoco_tracking.png", dpi=170)
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
panel(axs[0], "cancel", "dance.perform", 0, "Dance started")
a = np.load(base / "cancel/simulation/dance.perform/trajectory.npz")["samples"]
panel(axs[1], "cancel", "dance.perform", len(a) - 1, "Cancellation: last saved pose")
fig.suptitle("Cancel dance → no gesture.heart starts", fontsize=19)
fig.text(
    0.5,
    0.03,
    "Cancellation stops this simulation run; it is not a validated physical braking controller.",
    ha="center",
    fontsize=12,
)
fig.subplots_adjust(top=0.82, bottom=0.09, wspace=0.02)
fig.savefig(figures / "mujoco_cancel.png", dpi=160)
plt.close(fig)
# One continuous welcome clip with measured simulation time, no fabricated movement.
with imageio.get_writer(
    base / "welcome_sequence.mp4", fps=25, macro_block_size=1
) as writer:
    for action in ["greeting.say", "gesture.wave", "dance.perform"]:
        for frame in sorted((base / "welcome/simulation" / action).glob("frame_*.png")):
            writer.append_data(imageio.imread(frame))
print("Rendered media and figures")
