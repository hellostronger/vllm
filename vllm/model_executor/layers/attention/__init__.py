# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 注意力层模块

本模块实现了各种注意力机制的实现类，供模型层调用。

支持的注意力类型：
    Attention: 标准自注意力（用于 Decoder-only 模型）
    ChunkedLocalAttention: 块状局部注意力（减少长上下文内存占用）
    CrossAttention: 交叉注意力（用于 Encoder-Decoder 模型）
    EncoderOnlyAttention: Encoder 专用注意力
    MLAAttention: MLA（Multi-head Latent Attention）注意力
    MMEncoderAttention: 多模态 Encoder 注意力
    StaticSinkAttention: 静态 Sink 注意力（用于 Long Context）

注意力机制选择：
    - v1 引擎使用 vllm/v1/attention/ 中的后端实现
    - 支持多种后端：FlashAttention、FlashInfer、Triton、xFormers
    - 根据硬件自动选择最优后端

注意事项：
    这些是 PyTorch 层实现，实际推理时 v1 引擎会调用
    vllm/v1/attention/backends/ 中的后端进行高性能计算。
"""

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.chunked_local_attention import (
    ChunkedLocalAttention,
)
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.attention.static_sink_attention import (
    StaticSinkAttention,
)

__all__ = [
    "Attention",
    "ChunkedLocalAttention",
    "CrossAttention",
    "EncoderOnlyAttention",
    "MLAAttention",
    "MMEncoderAttention",
    "StaticSinkAttention",
]
