---
name: minimax-design-colab
description: 使用 Google Colab GPU 运行 MiniMax H3 或已配置的本地 ComfyUI 视频工作流，管理自定义模型卡与 LoRA，自动下载成片并释放当前任务专属算力。
---

先用 `doctor` 或 `list_cards` 查看账号、缓存设置和模型卡。用 `colab_accounts` 管理账号，用 `configure_runtime` 核对或修改本地设置。用 `plan_video` 查看模型精度顺序、候选显卡、时长及需要传输的素材。模型卡可以通过 `register_card` 导入。

原生 H3 的 LoRA 用 `register_lora` 独立登记，明确选择底模卡和兼容变体。用 `list_loras` 核对可用绑定。用 `plan_video` 分别选择模型卡 `card` 和 `loras` 名称与强度，使用纯底模时传空的 `loras`。卡中的底模权重变化后，旧绑定会从对应变体停用。登记回执仅表示检查了文件来源和底模绑定，只有实际加载验证后才能报告 GPU 兼容。

用 `preflight_only: true` 核对实际 GPU、内存与磁盘空间。生成任务在同一台 VM 上完成这些检查后再加载模型。模型精度、剪枝和编码器量化从卡里读取。

同一个作品因单次时长限制拆成多段，且模型卡、LoRA、参考素材和画幅相同时，用一个 `plan_video` 的 `segments` 提交。按播放顺序填写各段的 `id`、`prompt`、`seconds` 和可选 `seed`。所有段共享一台 Colab VM 和一个 ComfyUI 进程，模型缓存可跨段复用。每段生成后立即下载并校验，在 `segment_outputs` 查看路径。全批交付完成后释放 VM，已校验的分段继续可用。失败时进入审核流程。

原生 H3 默认每个片段生成三个候选，同一作业复用一台 VM。需要减少候选时，在 `plan_video` 设置 `takes: 1` 或 `takes: 2`。候选使用不同 seed。连续分段按候选分别生成完整顺序，上一段的尾帧仅用于同一候选的下一段。状态里的 `segments` 以 `id` 对应下载结果，并用 `segment_id` 和 `take` 标明原分段和候选编号。

连续动作使用原生 H3 Ref2VA 批次，显式传 `continuation="previous_tail"`。每段使用上一段剪辑终点前 56 帧作为无声视频参考，上一段实际剪辑终帧放进下一段的第 0 帧引导。按状态中的 `edit_frames` 裁去 H3 帧网格补出的帧。下一段提示词从接点的姿势、服装状态和动作写起。先检查接点两侧，确认画面连续。H3 官方单段为 4–15 秒，超出训练区间的本地渲染按实验处理。

`audio_mode="generated"` 保留源工作流音频。`audio_mode="mute"` 输出无音轨 MP4，原生 H3 跳过音频解码，自定义工作流在出片前去掉音轨并复核。若需要纯指定音效，从无音轨底片另配，按最终时间线逐段听审，检查多余的人声、音乐和音效。

同一 Colab 账号可并行启动两个独立作业，各自持有任务 ID、实例名称、进度、产物和释放回执，实际分配取决于 Colab。一个作业内的分段和候选仍共享该作业的同一台 VM。用各自的 `job_id` 查看进度或取消。

提交后给用户任务 ID 和状态查看方法。独立监控进程检查关键日志、超时和算力释放，完成钩子把成片保存到本机。用户询问时用 `video_status` 读一次。只有状态确认算力已释放时才这样报告。需要终止时使用 `cancel_video`，它只停止该作业的运行时。Colab 分配超时后，本地名称查不到仅表示映射缺失。核对提供商对该会话的终止回执，或在分配窗口结束后用服务端会话列表确认该名称已不存在。即使另一台已命名实例还在运行，也可以确认本作业的实例不存在。有未命名分配时继续保留待核查状态。

排队 hook 用 `configure_runtime(queue_hook="events")` 设置全局默认值，或在 `plan_video` 覆盖本次。提交前写入 `queue-hook.json`，记录开始分配、候选变化、重试、拿到实例或分配失败。最新状态从 `video_status` 的 `queue_hook` 和 `queue_hook_path` 读取，完整历史在 `events.jsonl`。`notify` 同样记录并在排队阶段结束时发本地通知，`off` 关闭排队 hook。实例分配成功后另看硬件和环境事件。

排查连接时，分别核对 `Session READY`、远端 `gpu_verified` 和 `environment_ready`。已配置 WSL 的账号在 `auto` 和 `configured` 下使用已保存的 WSL 路由。`doctor(live=true)`、`check_colab_cli_update` 和 `colab_accounts(action="usage")` 支持显式 `cli_transport`。

`events.jsonl` 中的 `preflight_started`、`gpu_verified`、`preflight_ready`、`environment_setup_started`、`environment_ready`、`assets_ready` 与 `setup_failed` 可供后台流程订阅。环境与权重各自完成时立即发事件。用 `setup_failed.stage` 区分环境安装与权重准备，再从任务运行日志定位具体报错。

需要调试 LoRA、提示词或局部更换参考素材时，让 `plan_video` 带 `hold_for_generation: true`，用 `start_video` 开始准备。进入 `awaiting_generation` 后，可用 `resume_video` 提交原计划，或提交同账号、同底模的新计划（支持增补 LoRA、修改分镜或只替换部分参考图/参考视频）。两次调用共用同一台 Colab VM，继续使用已准备的动态权重和 Python 环境，按 `sha256` 增量对比仅上传发生变化的参考素材并清理已移除的旧参考文件，未改动的参考素材直接复用远端副本。生成后的取回、校验和释放仍由同一个作业负责。

官方 CLI 先查询分配，再请求分配。HTTP 请求超时后，CLI 先按原请求标识核对服务端状态。插件为这条路径保留最多 390 秒，受总任务时限约束。外层超时后先尝试接回原会话，只有确认前一笔已清理且剩余预算足够才再提交一次。正式生成任务在分配阶段遇到 `HTTP 502/503/504`（`backend_unavailable`）时，若服务端列表无本作业的旧会话、会话均可识别且剩余预算充足，按指数退避（15s / 30s / 60s）最多有限重分 3 次，且不因先前的 `transport_lost` 耗尽 503 退避机会。预检内核瞬断时在同一会话有限重试。

若生成命令的原生 CLI 在切换 `/content` 的前导 cell 中失去 WebSocket 且尚无正式脚本的事件，先核对原会话是否空闲，确认后在同一实例重连一次。模型准备或生成期间遇到 Jupyter WebSocket 瞬断（`RuntimeError: Connection was lost.`）时，不直接进入 `review_pending`：若远端仍处于 `BUSY`，先等待其执行完成回到 `IDLE`，再探测 `/content/<job_id>/` 下已产出的 `video.mp4`（或分段视频）并计算校验和直接拉回本地完成任务；若尚未开始渲染，则在原实例续传一次，沿用已下载的模型文件和缓存。只有会话确实丢失、超时或不可恢复时才进入 `review_pending` 保留审核。

日常运行中，遇到需要接续动作或审核的真实阶段，在负责该阶段的插件里补 Hook。以持久回执验证触发和下游消费。队列结果触发硬件准备，环境就绪后可继续生成，片段校验后收片，故障进入审核。重连以首条有效远端回执为准，记录 `generation_transport_recovered`。监听掉线时记录 `generation_transport_failed` 及最后确认的远端阶段，先查原会话再决定是否续接。

分配请求收到终态 HTTP 502、503 或 504 时，`video_status.allocation_failure` 会记录 `backend_unavailable` 和 HTTP 状态码。等待服务端确认释放后，再判断是否需要按原任务授权有限退避重试。保持账号与模型方案的绑定，并检查已验证的分段是否可以复用。

ComfyUI 启动时，基础 pip 支持 `--python` 则直接创建不带 pip 的虚拟环境，再向该环境安装依赖。Colab 实测 `ensurepip` 曾退出 1，而此路径成功。已配 `wheelhouse` 时从匹配当前 Python、Torch 和固定 ComfyUI 提交的 wheel 仓安装。排查仍区分虚拟环境创建、pip 安装和依赖导入。

`plan_video(cli_transport="configured")` 将已保存的账号传输方式固定到该任务，比如该账号的 WSL CLI。`native` 要求本机受管 CLI 验证成功，`auto` 对已配置 WSL 的账号保留 WSL 路由。

MiniMax Design 使用本插件附带的本地 MCP。调用 `ensure_design` 启动本机 Design 并确认 `list_projects` 返回项目。用户明确要求撰写或优化 Seedance 2.5 提示词时使用官方 `sd25-pe`。已有提示词直接按用户要求提交。Design 生成返回任务 ID 后，调用 `continue_design_task` 保存项目和任务 ID。任意使用相同本地状态目录的 MCP 客户端可用 `list_task_handoffs` 找回登记的 Design 任务和 Colab 作业。若原客户端还没登记就掉线，用 `find_design_tasks` 从原 Design 项目找回任务。用 `continue_design_task(wait=true)` 等待原任务。任务失败时读取原任务提供的恢复操作，按用户授权处理。Colab 作业用原 ID 调用 `video_status`。生成 H3 时使用内置 `minimax-h3-design` 的素材标记和模型实际支持的输入与秒数。用 `read_design_workflow` 读取现有 ComfyUI 节点和模型依赖，经用户授权用 `deliver_video_to_design` 导入本地成片。导入批次中的单段时传入 `segment_id`。

下载传输失败时插件自动重试一次。作业遇错且租约仍待处理时进入 `review_pending`，事件日志记录 `diagnostic_required`。先用 `review_runtime(action="inspect", probe_exec=True)`（或 `probe_session`）核对实际实例与内核 `exec` 可达性，再查出错阶段、插件调用链和可复用资产。若 `status` 显示 `IDLE` 但 `exec` 探测超时（exit 124 / `exec_reachable: false`），按该作业的命名会话核对是否能继续使用，释放时检查该会话的回执。`wsl.exe` 产生的 UTF-16LE 空字节警告会由 `clean_cli_text` 剥离，避免干扰日志与错误判读。可修复且 `exec_reachable: true` 时用 `keep` 记录继续使用的依据。确认无需保留时用 `release` 记录原因并清理。修复已复现的插件缺陷后验证并更新相关 skill。

`colab new` 在分配 POST 发出后收到 `RemoteDisconnected` 时，记录 `allocation_failure.kind="transport_lost"`。这类结果可能留下尚未返回的会话。先查询任务原会话，只有确认原租约已释放且剩余时限足够，才允许一次有界重试。用同路由的 `sessions` 区分会话查询可达与分配请求断开。

分配问题需要对照时，可以在 `plan_video` 冻结 `runtime_shape="standard"` 来省略 `--high-mem`，或指定 `high_mem`。默认 `auto` 沿用模型卡自动选择。三种模式都校验模型卡要求的实际显存和系统内存，完成核验才上传素材与准备模型。

权重按缓存回执复用。冷启动中下载带 SHA-256 的权重时同步计算哈希，续传从既有前缀接着核验。性能对照分别看缓存命中、首段模型初始化和稳定采样耗时。Drive 缓存由受管 CLI 挂载，`cache_dir` 用于已挂载的持久目录。

用户授权后台唤醒当前 Codex 对话时，按 Relax Agent 的 `log-monitor` 绑定该对话、当前 `job_id` 和本地 `status.json`，只订阅 `failed` 或 `cleanup_pending`。生产前用同一路径验证通知可见。无法验证时保留本地事件及通知。
