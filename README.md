# MiniMax Design Plugin

面向 **Antigravity**、**Codex** 与 **MiniMax Code** 的本地视频与设计插件。将 MiniMax Design 本机 MCP、8 个配套工作流 Skill（`minimax-design-colab`、`minimax-h3-design`、`design-character-lookup`、`sd25-pe`、`camera-movement`、`star-entrance`、`scene-reference`、`fuse-bead-pattern`）以及 Google Colab 原生 H3 / ComfyUI 视频执行整合在同一个插件包中。图标取自已安装的 MiniMax Design 客户端。

## 安装

需要 **Antigravity / Codex / MiniMax Code**（任一）、Python 3.11+、uv，以及带 Colab 账户传输模块的 Relax Agent。原有账户信息保存在 Relax Agent 的用户目录，插件包不含 ADC、项目素材或模型权重。

- **Antigravity**：将 `plugins/minimax-design` 安装或同步到 `~/.gemini/config/plugins/minimax-design`（使用 `plugin.json` 与 `mcp_config.json`）。Windows 下可将 `mcp_config.json` 中本地服务的 `command` 设为无控制台 `uvw.exe`；后台监听与工作进程支持自动通过 WMI 脱离受限作业对象。插件可复用 Antigravity 安装的 Relax Agent。[Antigravity 本地插件](https://antigravity.google/docs/ide/plugins/)
- **Codex**：在仓库根目录执行 `python install.py`（macOS / Linux 用 `python3 install.py`）。安装器在 Windows 选择无控制台 `uvw.exe`，其他平台选择 `uv`。
- **MiniMax Code**：将 `plugins/minimax-design` 安装或同步到 `~/.minimax/plugins/minimax-design`（使用 `.minimax-plugin/plugin.json`、`servers.mcp.json` 与内置 `skills/`）。

## 配套 Skill 与适用场景

查找、保存、替换画布素材等直接项目操作走 Design 工具。仅在用户明确要求撰写或优化 Seedance 2.5 提示词时加载 `sd25-pe`；已有定稿要求直接生成时沿用用户原文。

- **`minimax-h3-design`**：配合本地 **MiniMax Design 画布** 整理角色/道具/场景素材、复用画布资产、撰写 H3 分镜提示词，并通过 `read_design_workflow` / `deliver_video_to_design` 与 Design 客户端双向联动。
- **`sd25-pe`**：撰写与优化 **Seedance 2.5（SD 2.5）** 文生视频、多参考生成、关键帧、分镜与音视频编辑提示词。平时直接使用随包版本，仅在用户明确要求升级时更新。
- **`camera-movement`**：创作意图明确且用户要求写或优化完整分镜提示词时，先给可用提示词，再做同版可播放的本地白模。两份一起审阅，只同步点名的修改与必要衔接。讨论想法或积累参考时只做讨论或整理。用户只要文本或局部修改时，按指定范围交付。
- **`scene-reference`**：承接灵感讨论后写好的 Markdown 剧本，根据参考图片设计或复刻场景；先逐场交付可复制的静帧提示词，再按选择生成或交用户在其他 App 使用。
- **`star-entrance`** 用于明星或重要角色出场的分镜设计。按场面需要选用局部亮相、目击者反应、跟拍或环绕，把剪点落在人物动作和视线变化上。
- **`design-character-lookup`**：在写分镜或分配 `<Picture N>` 前，从 MiniMax Design 画布资产中解析角色昵称、别名与固定外观设定。角色换装时从身份原图出发，按需使用本地打码工具处理三视图，核对后按用户要求保存造型。
- **`fuse-bead-pattern`**：先逐张判断是否要转 Q 版。写实照片用宿主的图片生成工具做 Q 版底稿，已有插画直接处理。再用本地代码生成带坐标、色号和用量的拼豆图纸及透明预览，支持批次共用色号。此 Skill 不启动 Design 或 Colab。
- **`minimax-design-colab`**：在 Google Colab GPU 上执行原生 H3（T2V / I2V / Ref2VA）与自定义 ComfyUI 工作流，管理模型卡、LoRA、增量素材传输、断连恢复与算力释放。

Windows、macOS 和 Linux 使用同一包。插件优先在宿主系统运行经过账号验证的官方 Colab CLI，兼容当前 Windows 上的 WSL 账户配置。首次打开插件和每次新任务启动都会检查 PyPI 官方版本，按版本隔离安装，用当前账户的官方 `whoami` 验证。升级失败时尝试已验证的 0.8.0 或已有传输。正运行的任务固定其 CLI 与账户。若 Relax Agent 安装在非标准目录，通过 `H3_COLAB_RELAX_MCP` 指定其 `mcp` 目录。

`ensure_design` 会探测 18810，按需启动客户端，然后调用 `list_projects`。Windows 与 macOS 使用标准安装位置。Linux 在有可运行的 Design 客户端时从 PATH 启动，也可通过 `configure_runtime(design_executable=…)` 指定绝对路径。已有运行中的本地 Design MCP 可以直接连接。无需运行 Design 即可用 Colab 生成。

## 一个生成任务

1. 通过 `colab_accounts` 查看、添加、登录、验证、选择、改名、移除账户并查看余额。登录使用官方 Google ADC 交互。只输入凭证文件路径。每个任务可单独指定 `account`，任务运行中绑定相同身份和会话版本。通过 `cli_transport` 可选择自动（`auto`）、本机受管 CLI（`native`），或账号已保存的连接（`configured`，例如 WSL）。
2. 用 `configure_runtime` 设置 GPU 顺序、缓存路径和离线环境。
3. `list_cards` 查看 `h3-t2v`、`h3-i2v` 或 `h3-r2v`。I2V 传 `first_frame` 与可选 `last_frame`。R2V 使用 `image_1` 到 `image_9` 连续编号，提示词用 `<Picture 1>` 等标记。默认 480×864、5 秒，对齐到 H3 的 24 fps 与 17k+5 帧格。卡内时长和显存只是规划值，实际 GPU 和内存由 Colab 启动后核验。
4. `plan_video` 返回精度从高到低的候选项、模型体积和待上传素材。用获授权的 `job_id` 和 `plan_sha256` 调用 `start_video`。可用候选按精度排序。只有明确分配被拒，或显存、内存检查失败且 VM 已确认释放时，才尝试下一候选。只想检查显卡和磁盘空间时，选择 `preflight_only: true`。这种计划在启动后短暂分配 GPU，生成检查回执并释放会话，期间无需上传项目素材、下载权重或挂载 Drive。开启 Drive 缓存时，单独预检不核验缓存容量。
5. Colab 工作进程在每个视频可下载时发出 `output_ready`。插件钩子下载 MP4 并校验 SHA-256 和大小，独立监控进程在整个任务结束后释放该 Colab 会话。用 `video_status` 查看文件路径与 `runtime_released`。失败、超时或取消也走有上限的清理。若连接中断致使释放未获确认，状态显示 `cleanup_pending` 并保留租约，可用 `retry_cleanup` 做一次额外恢复。
6. 成片需要进入 Design 时，指定项目与现有 Composition，将本地文件通过 `deliver_video_to_design` 写入对应资源。批次指定 `segment_id` 导入单段。Design MCP 对单个工作区文件的交付上限为 50 MiB。任务视频原件保存在本机。

运行、阶段、错误和清理回执分别写到同一个任务目录的 `status.json`、`run.log`、`events.jsonl` 和 `lease.json`。桌面通知在系统通知可用时发送。后台进程独立于 Codex 对话。若电脑休眠、断网、关机或 Colab 服务长期不可用，本地监控不能保证立即释放 VM。远端进程有截止时间，VM 释放仍以 Colab 回执为准。

可恢复的上传瞬断自动重试一次，分段下载按文件大小动态放宽超时并最多重试 3 次；`previous_tail` 接点参考视频采用 `crf=8` 近无损编码，长任务自动在远端脱离后台运行并由本地短 `exec` 增量轮询（支持实时 `/ws` 采样步数与瞬断续接）。终止失败或释放仍未确认时，监控器在 `events.jsonl` 写入 `diagnostic_required`，供后续修复读取。以已有的文件、下载或租约恢复路径处理故障，保留已验证的分段及原任务回执。

### 同一任务分段生成

如果要并行跑两个作品，分别用各自的计划调用 `start_video`。同一 Colab 账号最多同时运行两个独立作业，实际是否分到两台取决于 Colab。插件按作业名称分配、监控、收片和释放，另一台实例可继续运行。

原生 H3 默认每个片段生成三个不同 seed 的候选，使用同一台 GPU。用 `takes: 1` 或 `takes: 2` 可覆盖本次。设置 `continuation: "previous_tail"` 时，每个候选只接自己的上一段。状态里的 `segments` 按 `id` 与下载结果匹配，并记录原 `segment_id` 和候选 `take`。

在一次 `plan_video` 调用里用 `segments` 提交 2–8 段。各段独立设置提示词、秒数和可选种子，共享模型卡、LoRA、参考素材、画幅和采样设置。

```json
{
  "card": "h3-t2v",
  "gpus": ["A100"],
  "segments": [
    {"id": "opening", "prompt": "第一段提示词", "seconds": 11},
    {"id": "ending", "prompt": "第二段提示词", "seconds": 9}
  ]
}
```

一次 `start_video` 启动整个批次，顺序使用同一 Colab 会话和同一 ComfyUI 进程。每段生成后自动下载，`video_status.segment_outputs` 会列出已验证的文件。后段失败时可保留前段成片。整批结束时统一释放算力。批次输出是独立 MP4，可按需要剪辑拼接。

## 自定义模型卡与缓存

用 `register_card` 导入本地 JSON。可从插件的 `default_cards.json` 复制一张原生卡再改，或提供其他已验证内置节点的 ComfyUI API 图。自定义 API 卡设置 `native_h3: false`、`graph` 和 `output_node`，输出节点为原生 `SaveVideo`。如需自定义版本，填写 40 位 `comfy_revision`。

原生 H3 的模型卡和 LoRA 可以分别替换。用 `list_cards` 查看底模与可选精度，用 `register_lora` 登记独立适配器，给定 `card_id`、`lora_id`、明确兼容的 `variants`，再提供 `local_path` 或公开仓库的 `hf_repo`、`hf_revision` 和精确的 `hf_file`。插件绑定底模权重 SHA-256。换底模后使用 `list_loras` 检查哪些登记仍可用。重新确认兼容后，用 `replace: true` 按原 ID 更新登记。登记时检查文件身份，实际兼容性还需 GPU 加载验证。

每次用 `plan_video` 分别选择 `card` 和 `loras`，例如 `card: "h3-t2v"`、`loras: {"my-style": 0.65}`。切换适配器只需选择其他 ID，也可传空对象使用纯底模。旧计划固定原来的组件和 SHA-256。卡片内已有的 LoRA 声明继续可用，自定义 ComfyUI API 卡由原图绑定决定配套 LoRA。

| 字段 | 用法 |
| --- | --- |
| `variants` | 每个条目给定 `id`、`precision`、`min_vram_gib`、`min_ram_gib`、`assets`，以及可选 `gpus`、`quality_rank`、`steps` |
| `assets` | 组件按名称保存模型路径、文件 SHA-256、`size_bytes`。可选 HTTPS `url`、本地 `local_path`，或者提前放入该运行时缓存 |
| `loras` | 按名字定义 `asset`、支持的 `variants`，加速 LoRA 可指定 `steps`。任务中按名称选择并给定强度 |
| `graph` | API 图内可用 `{"$bind":"prompt"}`、`{"$bind":"asset.unet"}`、`{"$bind":"ref.first_frame"}` 或 `{"$bind":"strength.my_lora"}` 绑定 |
| `bindings` | 调用 `plan_video` 时传入图中声明的自定义字段 |

Design 节点可通过 `read_design_workflow` 读取草稿与模型依赖。UI 格式工作流先在 ComfyUI 导出 API 格式，工作流结构核实为 API 图之后再注册卡片。插件运行固定版本的 ComfyUI 内置节点。

可通过 `cache_dir` 复用当前 VM **已经挂载**的可持久化目录。预置权重放置于该目录的 `preseeded/<资产路径>`，固定 ComfyUI checkout 放置于 `comfy/<Git SHA>`，`wheelhouse` 可指向环境对应的离线 wheel 仓。固定版本环境复用已验证的 Python/CUDA 依赖与模型；没挂载缓存的新 VM 会冷启动，完整 BF16 组件约 124 GB，首次的读盘和传输耗时需实测。所有下载受启动时限控制。关闭 Colab VM 后其本地缓存随实例回收。

普通 Colab 也可以通过受管 CLI 配置 `drive_cache_relative`，例如 `H3-Colab/models`。插件先核验 GPU 和可用空间，再挂载 Drive。该目录用于权重和 LoRA，ComfyUI 环境安装在 VM 本地。用 `wheelhouse` 指定与 Python、Torch 和 ComfyUI 版本匹配的预建 wheel 仓。带 SHA-256 的权重在下载时同步校验。

调试时，可用 `plan_video(hold_for_generation=true)` 和 `start_video` 先准备并保留实例。状态变为 `awaiting_generation` 后，`resume_video` 接收原计划或同底模的更新计划，支持更换 LoRA、修改提示词以及按 `sha256` 增量更新发生变化的参考素材。计划仍以作业 ID 和 SHA-256 标识。

作业报错且实例状态待确认时，状态进入 `review_pending`。用 `review_runtime` 的 `inspect`（可选 `probe_exec=True` 实测内核 `exec` 可达性，区分真可用与挂死僵尸会话）核对原会话。确认需要保留实例时记录 `keep`，确认可释放或内核不可达时用 `release` 写明原因，插件再核对提供商的释放回执。

用 `configure_runtime(queue_hook="events")` 设置排队 hook，或在单次计划覆盖。默认把排队、分配、重试和结果写入 `queue-hook.json` 及 `events.jsonl`。选择 `notify` 时，排队阶段结束另发本地通知。

开发资料 [MiniMax 官方 ComfyUI 工作流](https://design.minimax.io/tools/minimax-h3-comfyui) · [原生节点模板](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native) · [Google 官方 Colab CLI](https://github.com/googlecolab/google-colab-cli)
