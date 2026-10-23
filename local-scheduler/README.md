# 本地技能调度：Qwen3-4B + PrimeBuddy

复用[社区项目 PrimeBuddy](https://github.com/huang-kai/primebuddy) 的 Registry 和 engine，另写本地规划器。没有使用上游 planner，也没有改动上游源码。模型在 GPU 上真实推理，三个自定义技能在受监督子进程中运行 Mock 函数，不连接机器人。

## 运行

在 Linux NVIDIA 环境中执行（实测 Ubuntu 22.04、RTX 5070 Ti 16 GB）：

```bash
bash code/setup.sh
.venv/bin/python code/run.py --text '先说一句欢迎光临，然后挥手，最后跳1秒的舞。'
.venv/bin/python code/run.py --suite
.venv/bin/python code/check_results.py
.venv/bin/python code/extra_cases.py
.venv/bin/python code/build_viewer.py
```

首次下载约 8 GB 模型；可设置 `HF_ENDPOINT=https://hf-mirror.com`。下载后使用 local_files_only=True。建议安装 uv；没有 uv 时使用 Python venv/pip。环境和上游源码均放在项目内。

模型 Qwen/Qwen3-4B 固定 revision `1cfa9a7208912126459214e8b04321603b3df60c`；PrimeBuddy 固定 `8731c2fcbe4aed7183882cab14f403df96bab62e`。依赖见 code/requirements.txt，PyTorch 2.7.1 CUDA 12.8 由 setup.sh 安装。

## 实现

1. `local_model.py`：开启思考判断 execute/clarify/unsupported，严格解析决策；execute 后读取三份技能定义，以关闭思考的 Qwen 和 LM Format Enforcer 生成 JSON Schema 约束计划。不是关键词路由。
2. `run.py`：调用 validate_plan 检查参数、资源与依赖；最多一次执行前纠错，然后 execute。动作回执由真实执行状态生成，不使用模型计划中的 reply 冒充成功。
3. `skills/`：greeting.say、gesture.perform、dance.perform。robot.body 为本工程资源标签。函数拒绝 simulate=False，取消检查间隔约 20ms。
4. `bootstrap.py`：将 PrimeBuddy 的 home/runtime 隔离到本工程 .runtime，不运行机器人安装向导。
5. `运行回放.html`：由保存记录生成，可离线查看，不是实时控制台。

取消测试在舞蹈实际启动后 0.3 秒注入取消事件，不是语音打断测试。所有动作均为 Mock，不代表物理运动验证。

## 实验记录

- results/suite.json：七个主案例的模型输入输出、任务、实际执行事件；check_results.py 的 32 个检查均通过（包含三项人工构造的非法计划检查）。
- results/extra-cases.json：四种补充说法及行为检查。
- results/baseline、thinking_only、plan_only、intent_non_thinking：开发中的失败记录。它们不是当前版本的预期行为，也不意味着当前代码可以原样复现旧版本。
- plan_only 中“来一个吧”误启动动作，是添加独立意图判断的直接原因。
- results/requirements-lock.txt、model-sha256.txt：实测环境和权重校验信息。

七个主案例均以已加载模型开始计时。挥手 9.16 秒、欢迎组合 13.14 秒；PyTorch 峰值 allocated 7822.4 MiB。少量工程案例不构成通用准确率评估。

本目录代码采用 MIT；上游 PrimeBuddy、Qwen 和依赖遵循各自许可。未包含模型权重或上游源码。
