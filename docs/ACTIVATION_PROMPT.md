# MicroDuck RL macOS 项目 — 激活提示词

> 新会话中把下面这段发给 Kimi 即可恢复全部上下文。

---

继续 MicroDuck RL 的 macOS 移植项目。工作区：`/Users/jup33q/Documents/kimi/tasks/2026-09-11/05-47-18-6fff4360`，仓库在 `microduck_rl/`（pollen-robotics/microduck_rl，mjlab + MuJoCo Warp + rsl_rl PPO）。

**先读两个文件**：`microduck_rl/docs/macos_setup.md`（macOS 构建注意点 + 代码流程，我写的）和 `microduck_rl/docs/DEV_PLAN.md`（开发计划，M0 已完成）。

**当前状态**：macOS 全链路已跑通（uv sync → 199 个 CPU 测试通过 → CPU 冒烟训练 → ONNX 导出）。Metal 适配已实测不可行（warp-lang macOS 只有 CPU 设备），不要再试。

**命令前必须加**：`export PYTHONPATH="$PWD/src" WANDB_MODE=offline`（.pth 偶发失效 + wandb 强制登录），训练必须加 `--gpu-ids None`（无 CUDA 机器默认参数会 IndexError）。

**下一步**（DEV_PLAN M1）：Mac 本地 32–64 envs 训 500–1000 迭代验证 reward 曲线，然后 `scripts/infer_policy.py --walking` 彩排 ONNX。

相关资产：B2-W 轮足 demo（`run_b2w_viewer.py`，桌面有启动器）、调研报告 `轮足机器人强化学习流程调研.md`。
