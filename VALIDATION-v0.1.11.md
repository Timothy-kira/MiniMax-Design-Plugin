# MiniMax Design 0.1.11 · A100 分配连接中断

实际任务 `h3-9babc948742d414a81c3c4fd` 在 Colab CLI 0.8 的分配 POST 链路退出。CLI 输出 `ConnectionError` 和 `RemoteDisconnected`，没有收到分配完成标记。该任务未执行 Python 预检，两段均无产物。

以相同账户与 WSL 路由实测只读 `sessions` 查询成功，服务端返回空会话列表；`usage` 接口也成功返回且余额为正。用户可查找原任务的租约回执确认最终清理结果。这些只读证据不能定位 POST 被哪一端关闭。

插件将分配 POST 断开记录为 `transport_lost`，复用确认原会话、等待租约处理和有限重试的路径。是否可再次分配 GPU 仍以 Colab 回执为准。

新增 `runtime_shape`，可将标准或高内存请求冻结到计划。Windows 收集 59 项，54 项通过，5 项 Linux 专用用例跳过。WSL 59 项全部通过。回归确认标准请求省略 `--high-mem`，而现场核验仍要求相同的 45 GiB 可用系统内存。实际 Colab 对照通过独立任务记录。
