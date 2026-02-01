# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 分布式计算模块

本模块提供了 vLLM 分布式推理所需的并行计算支持。

支持的并行策略：

    张量并行 (Tensor Parallelism, TP):
        - 将模型的不同层分配到不同 GPU
        - 适用于单节点多 GPU

    流水线并行 (Pipeline Parallelism, PP):
        - 将模型的不同层组分配到不同 GPU
        - 适用于多节点场景

    数据并行 (Data Parallelism, DP):
        - 每个 GPU 复制完整模型
        - 独立处理不同数据
        - 吞吐量最高

    上下文并行 (Context Parallelism, CP):
        - 将长上下文分布到多个 GPU
        - 适用于超长上下文场景

    专家并行 (Expert Parallelism, EP):
        - MoE 模型的专家分布到不同 GPU
        - 仅用于 Mixtral 等 MoE 模型

通信操作：
    - all_reduce: 全局求和
    - all_gather: 收集所有 GPU 数据
    - broadcast: 广播数据
    - reduce: 规约操作

使用示例：
    from vllm.distributed import (
        tensor_parallelize,
        get_tensor_model_parallel_world_size,
    )
"""

from .communication_op import *
from .parallel_state import *
from .utils import *
