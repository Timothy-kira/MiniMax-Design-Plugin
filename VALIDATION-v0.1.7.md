# MiniMax Design 0.1.7 · 连接路由与地区诊断

## 修复范围

- 将 CLI 的显式选择冻结到计划。`configured` 使用账号保存的连接，例如 WSL。`native` 要求本机受管 CLI 通过账号校验。`auto` 沿用旧计划的选择方式。
- Google 实例地区预检增加官方 DNS 别名和 IPv6 入口。无法读取时记录各入口的失败类型和 HTTP 状态。记录只包含固定入口与故障类别。

## 实际 Colab 回执

此前经同一账号的 WSL CLI 0.8.0 成功分配 A100 80 GB，远端硬件预检成功，地区元数据核验失败。随后用更新的探针发起单次 A100 纯预检，申请超时，未进入远端元数据查询。服务端确认无活跃会话，本地租约已关闭。两次尝试均未进入 H3 推理。

托管 Colab Runtime API 目前没有发布实例区域字段。插件继续使用正式地区证据，再执行模型和素材准备。 [Colab Runtime API](https://developers.google.com/colab/api/reference/rest/v1beta/runtimes) · [Compute Engine 元数据](https://docs.cloud.google.com/compute/docs/metadata/querying-metadata)

Windows 收集 42 项，38 项通过，4 项 Linux 专用用例跳过。WSL Ubuntu 收集 42 项并全部通过。以 Antigravity 配置启动 MCP，确认 0.1.7 暴露 17 个工具，并用现有账号实际校验 WSL 路由为 `ready`。
