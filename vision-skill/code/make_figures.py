import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import fontManager, FontProperties
from PIL import Image

root = Path(__file__).resolve().parents[1]
out = root / "figures"
out.mkdir(exist_ok=True)
font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font.exists():
    fontManager.addfont(str(font))
    plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
plt.rcParams.update(
    {
        "axes.unicode_minus": False,
        "font.size": 12,
        "figure.facecolor": "#f5f7fa",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
p = json.loads((root / "results/probe.json").read_text())
rows = {r["file"]: r for r in p["rows"]}


def photo(ax, name, title, draw_low=False):
    im = Image.open(root / "assets" / name)
    r = rows[name]
    ax.imshow(im)
    if r["score"] >= 0.5 or draw_low:
        x1, y1, x2, y2 = r["box"]
        c = "#08a687" if r["score"] >= 0.5 else "#ef7236"
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, lw=3)
        )
    ax.set_title(title + "\nscore = " + f"{r['score']:.3f}", fontweight="bold", pad=12)
    ax.axis("off")


fig, axs = plt.subplots(1, 2, figsize=(12, 6.8))
photo(axs[0], "washer_normal.jpg", "实拍图 A：找到整台洗衣机")
photo(axs[1], "laundry_normal.jpg", "实拍图 B：复杂背景下定位")
fig.suptitle("文字查询 → 检测框 → 启元 Skill", fontsize=22, fontweight="bold", y=0.98)
fig.text(
    0.5,
    0.02,
    "OWL-ViT / RTX 5070 Ti   |   实际推理结果，图片不是机器人相机采集",
    ha="center",
    color="#536072",
)
fig.subplots_adjust(top=0.82, bottom=0.08, wspace=0.05)
fig.savefig(out / "vision_detection.png", dpi=160)
plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(14, 6))
photo(axs[0], "washer_dark.jpg", "亮度降至 25%：仍通过")
photo(axs[1], "washer_occluded.jpg", "灰条遮挡：只框住上半截", True)
photo(axs[2], "no_washer.png", "无目标输入：不通过")
fig.suptitle("分数低了，不能只想着降低阈值", fontsize=22, fontweight="bold", y=0.98)
fig.text(
    0.5,
    0.025,
    "中图是人为遮挡测试；橙框为低于 0.5 的候选框，正常门槛不会放行。",
    ha="center",
    color="#536072",
)
fig.subplots_adjust(top=0.81, bottom=0.10, wspace=0.1)
fig.savefig(out / "vision_failure.png", dpi=160)
plt.close(fig)
s = json.loads((root / "results/skill.json").read_text())
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, i, title in zip(
    axs,
    [1, 3, 5],
    [
        "正常输入：无目标 → 连续三帧确认 → 搜索成功",
        "间歇出现：每次被无目标帧打断 → 超时",
        "门槛降为 0.12：局部错误框连续通过 → 误放行风险",
    ],
):
    r = s[i]
    t = r["trace"]
    xs = np.arange(1, len(t) + 1)
    scores = [x["score"] for x in t]
    ax.plot(xs, scores, "o-", lw=1.5, color="#4277b9", label="真实模型分数")
    ax.axhline(r["threshold"], ls="--", color="#ef7236", label=f"门槛 {r['threshold']}")
    for k, x in enumerate(t):
        if x["passed"]:
            ax.scatter(
                k + 1,
                x["score"],
                s=170,
                marker="*",
                color="#08a687",
                zorder=4,
                label="连续三帧通过" if k == 2 else None,
            )
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("score")
    ax.grid(alpha=0.15)
axs[0].legend(loc="upper right", ncol=2, fontsize=10)
axs[-1].set_xlabel("实际推理调用序号（静态照片按脚本重复输入）")
fig.tight_layout()
fig.savefig(out / "vision_skill_trace.png", dpi=160)
plt.close(fig)
l = json.loads((root / "results/loop.json").read_text())
rr = l["records"]
sync = np.diff([c["t"] for c in s[2]["commands"] if c["angular"] != 0]) * 1000
async_ = np.diff([c["t"] for c in rr]) * 1000
fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 7.6))
ax.plot(np.arange(len(sync)), sync, "o-", ms=3, color="#ef7236", label="同步搜索循环")
ax.plot(
    np.arange(len(async_)),
    async_,
    "-",
    color="#4277b9",
    label="独立 50 Hz Mock 指令循环",
)
ax.axhline(20, ls="--", lw=1, color="#606a77")
ax.set_ylabel("相邻指令间隔 / ms")
ax.set_xlabel("指令序号")
ax.legend(loc="upper right")
ax.grid(alpha=0.15)
ax.set_title(
    "35 ms 的视觉推理，不能塞进 20 ms 的指令周期", loc="left", fontweight="bold"
)
bx.plot(
    [r["t"] for r in rr],
    [r["age_s"] * 1000 if r["age_s"] is not None else np.nan for r in rr],
    color="#4277b9",
    label="当前观测年龄",
)
bx.axvline(l["camera_freeze_s"], ls="--", color="#ef7236", label="停止产生新画面")
bx.axhline(300, ls="--", color="#939da8", label="300 ms 过期门槛")
for r in rr:
    if not r["fresh"] and r["t"] > 1:
        bx.axvspan(r["t"], r["t"] + 0.02, color="#ef7236", alpha=0.12, lw=0)
bx.set_ylabel("距图像获取时刻 / ms")
bx.set_xlabel("运行时间 / s")
bx.legend(loc="upper left", fontsize=10)
bx.grid(alpha=0.15)
bx.set_title("冻结画面后：结果过期，Mock 指令归零", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(out / "vision_timing.png", dpi=160)
plt.close(fig)
print("4 figures saved")
