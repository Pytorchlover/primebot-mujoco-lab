# 启元 carry-clothes 本地视觉接入

本项目用真实 OWL-ViT 推理替换视觉占位接口，直接测试启元示例的 FIND_WASHER 搜索方法。动作端只有 Mock，不连接 ROS2 或真机；不执行接衣物、搬运或导航。所有输入来源及图片授权见 [ASSETS.md](ASSETS.md)。

## 复现

需要 NVIDIA CUDA 环境（实测 RTX 5070 Ti / 驱动 580.178.04），以及 uv 或 Python venv。

```bash
bash code/setup.sh
.venv/bin/python code/probe.py
.venv/bin/python code/skill_demo.py
.venv/bin/python code/loop_demo.py
.venv/bin/python code/make_figures.py
# 任意图片；第一次调用含冷启动，勿与预热后的统计直接比较
.venv/bin/python code/infer.py assets/washer_normal.jpg --output results/example.png
```

`setup.sh` 从 AtomGit 获取源码并固定到 `301b02ab86efa62cba1d4fdf70268715af9625d9`，下载 `google/owlvit-base-patch32` 的 `cbc355fb364588351c5d51c7f74465e8e7ec6f72`。网络受限时可自行设置 `HF_ENDPOINT`；本文远程环境模型文件经 hf-mirror.com 下载，权重 SHA256 记录在 results/model-sha256.txt。模型加载后无需网络请求。

系统缺少 venv 模块时，先安装系统对应的 python3-venv，或使用 uv；中文绘图使用系统 Noto Sans CJK 字体。

## 文件

- `code/detector.py`：JPEG 字节 → GPU 推理 → 原始分数、归一化中心、像素框高。
- `code/patch_upstream.py`：原源码保持不变，仅生成阈值可配置的 patched 副本和可审阅 diff。
- `code/skill_demo.py`：直接调用原任务类搜索阶段，六组实际推理/占位对照；断言异常分支和最终停止。
- `code/skill_demo_helpers.py`：连续三帧及相邻框 IoU 门控。
- `code/loop_demo.py`：独立的异步推理与 50 Hz Mock 指令实验，含画面更新中断和 300 ms 过期检查。不是整个上游任务的异步重构。
- `results/`：原始预测、时序、分支日志、环境锁文件。
- `figures/`：由实测结果生成的四张文章配图。

## 实验结果

FP32 / batch=1，预热 5 次、计时 40 次：detect 中位 34.71 ms、P95 35.27 ms，PyTorch 峰值 allocated 643.98 MiB。timing 包含 JPEG 解码、预处理、GPU 与后处理；不含权重加载和文件 I/O。原始记录可用于重新统计。

六组搜索结果依次为：占位超时、真实图片确认成功、无目标超时、交替输入超时、遮挡严格门槛超时、遮挡放宽门槛后错误框被放行。最后一组是反例，不是推荐设置。所有门槛和三帧参数仅用于本实验。

两张照片的亮度/遮挡变换不是独立样本，也不表示暗光、遮挡性能的全面评估；没有在这些图片上训练或微调。三帧测试重复使用静态照片，验证状态逻辑；在真实视频中还需验证跟踪、取帧延迟与目标身份。

本目录新增代码沿用仓库 MIT 许可；下载的启元源码保留其原始许可。图片许可单独遵循 ASSETS.md，含 Fanti Salms 照片的 vision_detection.png 使用 CC BY-SA 3.0。
