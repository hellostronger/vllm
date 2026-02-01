# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 执行器模块

执行器负责管理 Worker 的生命周期和分布式执行。

执行器类型：

    UniProcExecutor (单进程执行器):
        - 适用于单 GPU 推理
        - 简单直接，无进程间通信开销
        - 默认使用

    MultiprocExecutor (多进程执行器):
        - 适用于多 GPU 场景（张量并行/流水线并行）
        - 启动多个 Worker 进程
        - 通过 RPC 或共享内存通信

    RayExecutor (Ray 分布式执行器):
        - 适用于大规模分布式推理
        - 使用 Ray 框架管理集群
        - 支持跨节点并行

执行器职责：
    1. 初始化 Worker 进程/线程
    2. 加载模型到各设备
    3. 管理 Worker 生命周期
    4. 处理 Worker 间通信

选择执行器：
    - 单 GPU: UniProcExecutor
    - 单节点多 GPU: MultiprocExecutor (TP/PP)
    - 多节点: RayExecutor 或 MultiprocExecutor (需要手动设置)
"""

from .abstract import Executor
from .uniproc_executor import UniProcExecutor

__all__ = ["Executor", "UniProcExecutor"]
