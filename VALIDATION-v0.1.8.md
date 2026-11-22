# MiniMax Design 0.1.8 · 分配超时、路由、预检诊断与清理修复

## 修复范围

- **路由选择**：
  - `select_cli` 在 `cli_transport="auto"` 且账号已显式配置 `transport="wsl"` 时优先保留已保存的 WSL 路由，不再被 Windows 本机受管 CLI 抢占；`cli_transport="native"` 仍强制要求本机受管 CLI。
  - `doctor(live=true)` 在身份核验通过后附带实际 `cli_version`（同时覆盖 WSL 与本机受管 CLI）。
  - `check_colab_cli_update` 与 `colab_accounts(action="usage")` 新增 `cli_transport` 参数，可显式指定 `configured` / `native` / `auto`。
- **分配与预检有限恢复**：
  - `colab new` 分配超时由固定 70 秒放宽至最长 120 秒（受任务总剩余预算约束），并同步调整 `allocation_deadline = started + alloc_timeout + 15`，减少 A100 冷启动 60–110 秒区间被误杀的概率。
  - 首次预检 `exec` 若发生瞬时内核断线（`Connection was lost` 或退出码 124 且无远端 `failed` 标记），自动记录 `preflight_retry` 并有限重试一次；若远端明确回报 `runtime_location` 或 `capacity_mismatch` 则立即停止，不重复尝试。
- **报错与结构化诊断**：
  - `runtime_location_diagnostics` 持久写入 `status.json`，不再被紧随其后的 `failed` 标记覆盖。
  - 推理阶段与预检阶段统一使用 `require_remote`，批次后段失败时在 `status.json["error"]` 中直接保留远端 `error_kind` 与原因，同时保留已校验的前段成片 `segment_outputs`。
- **会话释放**：
  - 当 `colab-cli` 因远端异常剪除本地会话映射（`state.prune_session`）导致 `stop` / `status` 报 `not found` 时，`cleanup` 对已返回（`returned`）会话同样复核服务端 `sessions`，仅在服务端明确回报 `No active sessions found on server.` 时确认释放并记录 `no_active_server_sessions`。

## 实测证据（A100 纯预检 `h3-7f156e51ceff4f6a9cfe69c9`）

- 账号：`akirawith51@gmail.com`，WSL `Ubuntu-24.04` CLI `0.8.0`，`cli_transport="configured"`。
- `colab new` 约 60 秒进入 `Session READY.`，远端 `gpu_verified` 确认 `NVIDIA A100-SXM4-80GB`（80.0 GiB VRAM，79.25 GiB 可用，164.52 GiB RAM）。
- 随后 `runtime_location_diagnostics` 实测返回：
  - `metadata.google.internal=http_404`
  - `169.254.169.254=http_404`
  - `metadata.goog=http_404`
  - `[fd20:ce::254]=transport_error`
- 结论：标准消费级 Colab 沙箱的 Google 元数据代理对 `/computeMetadata/v1/instance/zone` 明确返回 `HTTP 404`（平台隐藏实例区域元数据，而非 DNS/超时/连接故障），同时普通 Colab Runtime API 亦不暴露实例区域字段。插件正确阻断后续上传与模型下载，并立即通过 `stop_ack` 释放该 A100 会话（服务端确认 0 活跃会话）。

## 回归验证

- Windows：`python -m unittest discover -p "test_*.py" -v` 收集 46 项，42 项通过，4 项 Linux 专用用例跳过。
- WSL Ubuntu：`python3 -m unittest discover -p "test_*.py" -v` 收集 46 项，46 项全部通过。
- 以 0.1.8 直接通过 MCP 协议核验 `initialize`、`tools/list`、`doctor(live=true, cli_transport="configured")`、`check_colab_cli_update(cli_transport="configured")` 与 `colab_accounts(action="usage", cli_transport="configured")`，均走 WSL 0.8.0 路由并通过。
