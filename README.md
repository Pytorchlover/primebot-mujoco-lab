# 启元 Q1 MuJoCo：控制与 VLA 动作接口实验

两篇教程的原创代码、实际渲染图与可复核结果。实测：Ubuntu 22.04.5、Python 3.10.12、RTX 5070 Ti、MuJoCo 3.17.0。未加载 VLA 权重、未训练策略、未连接真机。

## 复现

```bash
git clone https://github.com/Pytorchlover/primebot-mujoco-lab.git
cd primebot-mujoco-lab
# Ubuntu 缺 venv 时：sudo apt-get install python3-venv
bash code/setup.sh
MUJOCO_GL=egl .venv/bin/python code/validate.py
```

`setup.sh` 下载两个官方模型仓库并固定提交。模型网格不在此镜像分发，其权利归原作者，使用须遵循上游许可。官方来源：

- https://atomgit.com/primebot/q1-urdf — `562b764a18613d459140ed9b255fa30434e346ef`
- https://atomgit.com/primebot/t1-urdf — `e6482e6b0883e93e378b8b6fb40fca6704050a6f`

本仓库原创脚本采用 MIT 许可；不改变上游模型及第三方内容许可。

## 实验条件

1. 原始 Q1：浮动基座、默认重力和接触、零控制输入，运行 2 秒。
2. 接口测试台：从 `q1_v3_mc.xml` 在内存中生成；移除骨盆 freejoint，骨盆升至 0.55 m；关闭机器人 geom 的碰撞，保留重力、质量、惯性、关节限位与 motor；可视夹具本身不参与碰撞。用于排除平衡与接触变量，不能证明站立、抓取、行走或真机性能。
3. 22 维绝对关节目标，仅左肩 pitch 随正弦变化，其余为 0；Kp=15、Kd=0.5；1 kHz 物理步进、20 Hz 零阶保持目标；所有力矩按原模型 ctrlrange 裁剪。
4. 每组 6 秒/120 控制样本；H=10，另测 H=1/5/20 和幅度 0.2/0.4/0.6 rad。19 组确定性实验，不是统计成功率评测。
5. Delta 以每个 chunk 起始实测状态为统一锚点；正确还原、移动锚点、错误累计三组比较。不是对真实 OpenPI 策略的评测。

## 文件

- `code/probe.py`：检查所有 XML 的 nq/nv/nu，物理步进 100 次。
- `code/experiment.py`：生成全部 19 组实验、曲线、视频、原始截图与 episode。
- `code/validate.py`：检查时间间隔、数据形状、相邻状态连续性、动作回放、H=1 阴性对照。
- `figures/`：中文排版图；MuJoCo 原始渲染来自 experiment.py，图文说明为作者制作。
- `results/metrics.json`：逐组误差、目标裁剪计数、力矩饱和计数。
- `results/*.csv`：左肩目标、下发目标、动作执行前实测角度，均为弧度。
- `results/episode.npz`：120 帧 RGB + 22 维状态/绝对目标/next_state；格式说明见 schema。
- `results/*.mp4`：20 fps、6 秒实测仿真录像。

回放验证使用相同模型和初始状态，误差为 0 仅代表本环境数值复现，不代表跨版本位级一致。图像为固定外部相机，不是 Q1 真机相机标定。

如无 NVIDIA EGL 环境，可自行配置 Mesa OSMesa 后设置 `MUJOCO_GL=osmesa`；本文实测路径为 NVIDIA EGL。脚本渲染不需桌面会话，普通 MuJoCo 的 mj_step 在 CPU 上执行。

## 多视角数据采集（新增独立教程）

环境安装后运行：

```bash
MUJOCO_GL=egl .venv/bin/python code/collect_multiview.py
```

`multiview/` 是独立的数据采集实验，不使用上一实验的左肩正弦轨迹：

- 三段动作：左臂伸展、右臂伸展、双臂抬起并转腰。
- 每段 4 秒 / 80 个控制时刻，正面、侧面、俯视三路 480×360 RGB；总计 240 个时刻、720 张 PNG。
- 同一时刻渲染完全部相机后才做物理积分；使用仿真时间，20 Hz 目标、1 kHz PD。
- 相机外参、视场角、分辨率、关节顺序与单位见 `multiview/schema.json`。
- 每段包含 `index.jsonl`、`trajectory.npz`、三路 PNG、拼接视频和 `validation.json`。
- index 保存每张图片的源帧号、仿真时间与 SHA256；trajectory 包含 time/state/velocity/action/next_state/snapshot/initial_integration_state。
- `action` 是绝对关节目标；instruction 只是脚本动作标签，不是模型输入。
- 使用相同环境从初态回放 action，所有采样点关节误差为 0。快照重渲染另行报告 RGB 差异，不宣称像素完全相同。实测最大单通道差为 3/255。
- 每段构造 79 条保留源时间的 side 错一帧索引，均检出；删除一行的连续性检查也通过。故障只注入索引副本，交付数据保持完整。

双击 `multiview/viewer.html` 可查看本地数据，或在仓库根目录运行 `python3 -m http.server 8769` 后打开 `http://localhost:8769/multiview/viewer.html`。支持切换 episode、播放、滑动帧号、方向键逐帧查看；渲染图片依赖完整 `multiview/` 目录。

这是采集与格式检查样例，模型使用固定骨盆、关闭机器人碰撞的测试台，保留重力；并非通用任务训练集。

## 启元 Skill 本地视觉接入

新增 [vision-skill](vision-skill/)：在 RTX 5070 Ti 上实际运行 OWL-ViT，接入 carry-clothes 的洗衣机搜索阶段，包含遮挡反例、六组 Mock 分支测试、独立的异步推理与观测过期实验。代码、图片授权和复现步骤见该目录。
