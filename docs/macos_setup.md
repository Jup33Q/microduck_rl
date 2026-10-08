# macOS 开发环境搭建与代码流程（MicroDuck RL）

> 验证日期：2026-09-11 · 机器：MacBook Pro (Apple Silicon, aarch64)，macOS 26 · 状态：**已跑通全链路**
> （依赖安装 → CPU 测试 199 passed → CPU 冒烟训练 → checkpoint → ONNX 导出）

---

## 1. 结论先行

官方 quickstart 声称需要 CUDA GPU，但实测 **macOS 上完整训练链路可用**，因为：

- `mjlab` 的设备选择在 `mjlab/scripts/train.py:48-50`：**`CUDA_VISIBLE_DEVICES` 为空时自动回退 `cpu`**
- `mujoco-warp`（物理后端）在 Mac 上以 **CPU 模式**运行（warp-lang 1.12.0 的 macOS wheel 只有 `cpu` 设备，无 CUDA 也无 Metal）
- `rsl_rl`（PPO）基于 PyTorch，Mac 上 CPU 可跑（MPS 可用但当前链路未启用）

**Metal 适配结论（实测）**：不可行，无需再尝试。
- warp-lang 1.12.0 macOS wheel 只暴露 `"cpu": "arm"` 设备，无 Metal API
- PyTorch MPS 可用（`torch.backends.mps.is_available() == True`），但 mjlab/rsl_rl 的 device 参数只走 cpu/cuda 分支，接入 MPS 需要改 mjlab 源码（`train.py` 设备选择 + 全程 tensor device 审计），收益不确定，不建议当前做

**Mac 本地定位**：小规模训练/调试/导出（32 envs ≈ 614 steps/s）；正式训练（4096 envs）走 `--hf-jobs`（Hugging Face Jobs）或云 GPU。

---

## 2. 构建步骤与注意点（按踩坑顺序）

### 2.1 环境安装

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
export UV_HTTP_TIMEOUT=600   # 首次下载较大，防超时
uv sync                      # uv 会自动拉 CPython 3.12（项目要求 >=3.12,<3.13）
```

### 2.2 注意点 ①：site-packages 的 `.pth` 偶发失效（本仓库特有坑）

`mjlab-microduck` 以 `.pth` 指向 `src/` 的方式安装，但本机上出现过 `.pth` 存在却未生效的情况
（`uv run python -c "import mjlab_microduck"` 与 `.venv/bin/train` 偶发 `ModuleNotFoundError`，
且 site-packages 中出现 `mjlab_microduck 2.pth` 重复文件，所有 `.pth` 带 macOS `hidden` 标志）。
怀疑与工作区文件同步机制有关。

**对策（任何命令前加这两个环境变量即可免疫）**：

```bash
export PYTHONPATH="$PWD/src"   # 绕过 .pth 失效
export WANDB_MODE=offline      # 见注意点②
```

若复发，执行 `uv sync --reinstall-package mjlab-microduck` 重装。

### 2.3 注意点 ②：wandb 默认要求登录

rsl_rl 的 `OnPolicyRunner` 会调 `wandb.init()`，无 API key 直接报错。
本地实验一律 `WANDB_MODE=offline`（日志存本地，事后可 `wandb sync`）。

### 2.4 注意点 ③：无 GPU 机器必须显式传 `--gpu-ids None`

`--gpu-ids` 默认 `'[0]'`，`mjlab/utils/gpu.py:select_gpus` 会按 CUDA 设备列表索引，
Mac 上 `torch.cuda.device_count()==0` → `IndexError: list index out of range`。
传 `--gpu-ids None` 后走 CPU 分支。

### 2.5 注意点 ④：CPU 训练吞吐预期

| 配置 | 迭代耗时 | 吞吐 |
|---|---|---|
| 4 envs × 2 iters | 0.47 s/iter | ~200 steps/s（冒烟用） |
| 32 envs × 3 iters | 1.25 s/iter | ~614 steps/s |

对照官方 4096 envs on GPU ≈ 数万 steps/s，**Mac 本地不适合正式步态训练**，
适合：改配置后的冒烟验证、reward/obs 调试、ONNX 导出验证、sim2real  rehearsal。

### 2.6 注意点 ⑤：mjpython 路径含空格 + libpython 查找失败

Kimi 托管 Python 的 `mjpython` shebang 因路径含空格无法直接执行，
需 `python <...venv/bin/mjpython> script.py` 方式调用（查看器场景）。
且 uv 托管 CPython 下 mjpython 的 dlopen 找不到 `libpython3.12.dylib`
（`@executable_path/../lib` 相对 venv 解析），需再加：

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib"
```

### 2.7 注意点 ⑥：61D 策略彩排必须加 `--new-cmd-obs`

`infer_policy.py` 默认构建旧版 51D obs（48+3 twist），当前 velocity 配方训出的
策略是 61D（13D 指令块），不加 `--new-cmd-obs` 会报
`Got: 51 Expected: 61`。旧策略才保持默认。

---

## 3. 验证过的命令清单（macOS 可直接复制）

```bash
cd <repo>/microduck_rl
export PYTHONPATH="$PWD/src" WANDB_MODE=offline UV_HTTP_TIMEOUT=600

# 1. CPU 回归测试（不需要 GPU，官方设计如此）
uv run --with pytest python -m pytest tests/        # 199 passed, 1 skipped

# 2. 任务注册表
uv run list-envs

# 3. 冒烟训练（改任何 cfg 后必跑）
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4 --agent.max_iterations 2 --gpu-ids None

# 4. 小规模本地训练
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 32 --agent.max_iterations 100 --gpu-ids None

# 5. ONNX 导出（normalizer 已烘焙，官方强制路径）
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file "$PWD/logs/rsl_rl/velocity/<run>/model_N.pt" \
    --device cpu --onnx-file out.onnx

# 6. CPU MuJoCo 部署彩排（键盘控制；61D 策略必须 --new-cmd-obs）
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib"
uv run python .venv/bin/mjpython scripts/infer_policy.py --walking out.onnx --new-cmd-obs
# 或走 MCP 调试通道：<workspace>/rehearsal_mcp/server.py（kimi-code mcp.json 已注册 microduck-rehearsal）

# 7. 正式训练（无 GPU → 提交 HF Jobs）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

产物位置：`logs/rsl_rl/<experiment>/<run>/{model_*.pt, params/, git/, events.out.tfevents.*}`

---

## 4. 代码流程（一次训练调用发生了什么）

```
uv run train <TASK_ID> [options]
  └─ .venv/bin/train  →  mjlab_microduck.train_cli:main          # 本项目 shim（拦截 --hf-jobs）
       ├─ --hf-jobs ? → hf_jobs.py：打包提交 Hugging Face Jobs，本地不训练
       └─ 否则 → mjlab.scripts.train:main
            ├─ tyro 解析：--env.* / --agent.* 覆盖 cfg（num_envs、max_iterations…）
            ├─ select_gpus(args.gpu_ids)                        # Mac: 必须 None → device="cpu"
            ├─ gym.make(TASK_ID)
            │    └─ mjlab.tasks entry point → mjlab_microduck.tasks.__init__
            │         └─ make_microduck_velocity_env_cfg()      # microduck_velocity_env_cfg.py
            │              ├─ robot cfg: microduck_constants.py  # MJCF + HOME 位姿 + BAM 执行器
            │              ├─ MDP: tasks/mdp.py                  # 全部 reward/obs/event/curriculum
            │              └─ -Backlash- 变体: backlash.py       # 包装任意 cfg 生成齿隙孪生任务
            ├─ ManagerBasedRlEnv: mujoco-warp 物理（Mac=CPU 设备）@ 50 Hz 控制 / decimation=5
            ├─ OnPolicyRunner (rsl_rl, PPO)                     # 61D obs → MLP → 14D action
            │    └─ wandb.init（Mac 需 WANDB_MODE=offline）
            └─ 每 save_interval 写 logs/rsl_rl/<exp>/<run>/model_*.pt

uv run scripts/export.py → mjlab_microduck/export.py
  └─ 加载 checkpoint → 把 obs normalizer 烘焙进计算图 → ONNX（61D in / 14D out）
       └─ 部署：microduck 仓库的 Rust runtime（robotd）以 50 Hz 执行 ONNX
```

### 不变量（改动红线，摘自 AGENTS.md，全部已被 CPU 测试锁定）

- **obs 恒为 61D**：48 本体感知 + 13D 指令块 `[twist(3), head_pose(4), body_pose(6)]`，不用的槽位补零，不许删
- **关节布局**：14 个 XL330 舵机，0–4 左腿 / 5–8 颈头 / 9–13 右腿；mdp 函数里不许硬编码索引，
  用 `_servo_joint_ids` / `_servo_joint_pos` 助手（roller/backlash 模型被动关节会穿插）
- **被动关节一律 `passive_` 前缀**；选择器用 `^(?!passive_).*`
- **执行器是 BAM**（电压控制模型，摩擦由执行器计算）；独立 env cfg 必须注册 `expand_bam_friction_fields` 事件
- **必须经 `scripts/export.py` 导出 ONNX**（烘焙 normalizer），禁止手工转 checkpoint
- **策略不加动作低通滤波**（训练/部署不对称会毁迁移）
- **域随机化不可跨 reset 累积**（restore-then-apply）

---

## 5. 相关仓库与本工作区资产

| 资产 | 位置 |
|---|---|
| microduck_rl（本仓库，训练栈） | `<workspace>/microduck_rl/` |
| 冒烟 ONNX 策略（仅链路验证，无实用步态） | `<workspace>/microduck_rl/microduck_smoke.onnx` |
| MicroDuck 离屏渲染验证图 | `<workspace>/microduck_render.png` |
| 宇树 B2-W 官方模型 + 查看器 demo（另一项目） | `<workspace>/unitree_mujoco/`、`run_b2w_viewer.py` |
| 桌面启动器（B2-W demo） | `~/Desktop/启动B2W轮足机器人.command` |
| 轮足 RL 调研报告 | `<workspace>/轮足机器人强化学习流程调研.md` |
