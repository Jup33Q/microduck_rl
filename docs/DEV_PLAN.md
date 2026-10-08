# MicroDuck RL on macOS — 开发计划

> 更新：2026-09-12 · 里程碑 M0、M1、M1.5、M2 已完成

## 里程碑

- [x] **M0 — macOS 移植验证（2026-09-11）**
  - uv sync 全依赖安装成功（CPython 3.12，aarch64）
  - CPU 回归测试 199 passed / 1 skipped
  - CPU 冒烟训练跑通（4 envs × 2 iters；32 envs × 3 iters ≈ 614 steps/s）
  - checkpoint → ONNX 导出跑通（`microduck_smoke.onnx`，normalizer 已烘焙）
  - 离屏渲染验证模型网格/材质完整
  - 排坑记录见 [docs/macos_setup.md](docs/macos_setup.md)（.pth 失效 / wandb 登录 / --gpu-ids / Metal 不可行）

- [x] **M1 — 本地策略质量验证（2026-09-11）**
  - 64 envs × 800 iters（27 min，~780 steps/s）：mean reward 0.08→0.95 单调上行，penalty 项全负，零 NaN，曲线健康 ✓
  - checkpoint → `microduck_m1_800.onnx` 导出 ✓（normalizer 烘焙）
  - `infer_policy.py --walking --new-cmd-obs` 彩排跑通：零命令稳站 trunk_z=118mm；给 0.30 m/s 前进指令原地不动——**欠训的正常表现**（1.2M 样本 vs 步态所需 ~400M），非机械/部署信息缺失（BAM 参数镜像、61D obs 契约均有测试锁定）
  - 踩坑补录：mjpython 需 `DYLD_FALLBACK_LIBRARY_PATH`；61D 策略彩排必须 `--new-cmd-obs`——已写入 [docs/macos_setup.md](docs/macos_setup.md) 注意点⑤⑥
  - 工具沉淀：`<workspace>/CurveBoard/`（SwiftUI 训练曲线面板，TensorBoard 内容/macOS 排版）；`<workspace>/rehearsal_mcp/`（彩排调试 MCP：启停/按键注入/遥测/截图，已注册 `~/.kimi-code/mcp.json`）

- [x] **M1.5 — 本地课程全阶梯验证（2026-09-11）**
  - 64 envs × 2000 iters（3.07M steps，67 min）：**48k-step 课程阶梯全部走完**，终档 action_rate_weight −1.0 / standing_envs 0.20 / head_pose_range 0.91 / com_range 0.015 / head_pose_bias_weight 3.0
  - 曲线健康：mean reward 0.3→13.5（加权值，每次课程加权上阶梯时可见回落，如 ~800 iters 处 action_rate 加码后 2.5→1.1 再爬回）；penalty 项全程 ≤0；nan_state 恒 0；fell_over 2.4→1.0；episode length 35→260（顶格 300）；action std 0.97→0.28；entropy 19.4→2.0
  - **track_linear_velocity 抬头（开放问题：是）**：0.02→0.33–0.46，且在课程参数不变的窗口内也持续爬升（如 ~1360 iters 处 0.10→0.31、1793→1999 处 0.10→0.46），非纯权重阶梯假象；末段 error_vel_xy ~0.05–0.08
  - **吞吐（开放问题 3，64 envs 长稳）**：全程 730–806 steps/s，67 min 无衰减；iter 耗时 1.9–2.4 s，collection 主导（learning 仅 ~0.07 s/iter）
  - checkpoint → `microduck_m1_5_2000.onnx` 导出 ✓（`scripts/export.py`，normalizer 烘焙）
  - 彩排（rehearsal MCP，`--new-cmd-obs`）：零命令稳站 trunk_z=113.1mm（M1 为 118mm，站姿略低）；给 0.30 m/s 指令后 **achieved fwd 持续 ±0.00 约 7 min，仍原地站**——与 M1 相同的欠训表现，sim 里的跟踪进步（3M 样本 vs 步态所需 ~400M）尚不足以在 CPU 彩排中表达为步态；随机推/长时间给令后最终倾倒（trunk_z→41mm，精确时刻因 stdout 缓冲无法定位）
  - 工具观察：mjpython 链 stdout 块缓冲严重（`PYTHONUNBUFFERED=1` 也无效，遥测按 ~16KB 一批延迟数分钟到达）；本次 viewer 窗口未暴露 AX/未上屏，截图通道退化，彩排证据以遥测为准

- [x] **M2 — 本地正式训练（2026-09-11 晚启动，2026-09-12 收官；本地，放弃 --hf-jobs/云 GPU）**
  - 规模实测：256 envs × 12000 iters = **73.7M 样本，总时长 21h44m，平均吞吐 942 steps/s**（夜间 ~1040，白天热沉降/系统干扰时段 ~700–960；无长稳衰减）
  - 曲线（全程健康）：mean reward −0.02 → 84.4（~9000 iters 进平台）；episode length 21 → ~880；fell_over 1.17 → 0.00；**track_linear_velocity 0.011 → 1.31**（M1.5 在 2000 iters 为 0.33–0.46，本次 iter 1598 即达 0.97 反超）；error_vel_xy 在 ~0.5 震荡平台；**nan_state 全程恒 0；penalty 项全程 ≤0 无一例外**
  - checkpoint 策略调整为 `--agent.save_interval 10` + 桌面守护 `early_checkpoints.sh`（抢救 model_10/20/50/80 到桌面 `checkpoints/`，run 目录滚动只留最新 4 个）；`prune_checkpoints.sh` 已加早期 4 个保留规则；新增 `.command` 双击入口（启动训练/查看日志/彩排/清理/扫描，训练运行中双击转为实时看日志）
  - **彩排对比（核心结论：原地站已消除，出现稳定前移，但速度跟踪不足）**
    - `model_11999`（终版，`microduck_m2` 级 ONNX 已导出）：零命令稳站 trunk_z 112–116mm（M1.5 为 113mm）；给 0.30 m/s 指令 → **achieved fwd 持续 +0.07–0.08 m/s 约 7 分钟不倒**（M1.5 为 ±0.00 原地站），lat ≈0，yaw ±0.15 rad/s，trunk_z 稳在 115–120mm
    - `model_80`（早期对照）：零命令稳站 trunk_z 118.9mm；0.30 指令 → fwd ±0.00，精确复现 M1.5 的"原地站"——确认前移能力来自本次训练量
    - 判读：步态/前移信号真实出现且躯干稳定，但速度跟踪仅 ~23%（0.07/0.30），属慢速拖步/欠训残差（73.7M vs 步态所需 ~400M 样本估计）；sim 侧 error_vel_xy ~0.5 平台与之自洽
  - 验收状态：对比 M1.5 的改善明确且方向正确，但 0.30 m/s 指令下的速度跟踪未达标；**是否 publish HF Hub 待与用户讨论**（选项：续训至更大量样本 / 以 experimental 策略发布 / 先查 error_vel_xy 平台原因）
  - 参考（沿用 M2 准备期实测）：并发上限 64→800 / 128→874 / 256→911 / 512→925 / 1024→895 steps/s（256–512 envs 饱和，M5 Max 18 核）；checkpoint 4.8MB/个，部署 ONNX 0.79MB（normalizer 烘焙，61D in / 14D out），策略网络结构不动（MLP 512/256/128，属配方）

- [ ] **M3 — 真机部署（待硬件到货，首发预计 2026 圣诞前）**
  - `robotctl policy add` 加载已发布策略
  - 按 AGENTS.md 的 sim2real 清单做部署验证（BAM 执行器、backlash 变体 A/B）

## 开放问题

1. `.pth` 失效与工作区文件同步机制的关系（是否 iCloud/同步器导致 hidden 标志 + 重复文件）——暂以 `PYTHONPATH` 兜底
2. PyTorch MPS 接入 mjlab/rsl_rl 的改造收益评估（当前判断：不做）
3. ~~Mac 本地训练规模上限实测（64/128 envs 吞吐曲线）~~ → 已实测（M1.5 + M2 准备）：64→800 / 128→874 / 256→911 / 512→925 / 1024→895 steps/s；长稳无衰减（67 min @ 64 envs）；峰值平台 256–512 envs

## 工作约定

- 任何 cfg 改动后先跑冒烟：`--env.scene.num-envs 4 --agent.max_iterations 2 --gpu-ids None`
- 命令前缀固定：`PYTHONPATH="$PWD/src" WANDB_MODE=offline`
- 不动 AGENTS.md 列出的不变量（61D obs、关节布局、BAM、ONNX 导出路径、DR 不累积）
