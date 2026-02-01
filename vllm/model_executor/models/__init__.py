# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 模型注册与接口模块

本模块定义了 vLLM 模型系统的核心接口和注册机制。

模型接口说明：
    VllmModelForTextGeneration: 文本生成模型接口（如 Llama、Qwen）
    VllmModelForPooling: 池化模型接口（如 Embedding、Score 模型）

模型能力接口：
    SupportsLoRA: 支持 LoRA 适配器微调
    SupportsMultiModal: 支持多模态输入（图像+文本）
    SupportsPP: 支持流水线并行
    SupportsMRoPE: 支持 MRoPE 位置编码
    HasInnerState: 模型有内部状态（如 MoE 的专家状态）
    SupportsTranscription: 支持语音转录

模型注册流程：
    1. 在 registry.py 中注册模型
    2. 实现对应的模型接口
    3. 通过 ModelRegistry.lookup_model_class() 获取模型类

使用示例：
    # 检查模型是否支持某种能力
    if supports_multimodal(model_class):
        # 处理多模态输入
        pass
"""

from .interfaces import (
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    has_inner_state,
    supports_lora,
    supports_mrope,
    supports_multimodal,
    supports_pp,
    supports_transcription,
)
from .interfaces_base import (
    VllmModelForPooling,
    VllmModelForTextGeneration,
    is_pooling_model,
    is_text_generation_model,
)
from .registry import ModelRegistry

__all__ = [
    "ModelRegistry",
    "VllmModelForPooling",
    "is_pooling_model",
    "VllmModelForTextGeneration",
    "is_text_generation_model",
    "HasInnerState",
    "has_inner_state",
    "SupportsLoRA",
    "supports_lora",
    "SupportsMultiModal",
    "supports_multimodal",
    "SupportsMRoPE",
    "supports_mrope",
    "SupportsPP",
    "supports_pp",
    "SupportsTranscription",
    "supports_transcription",
]
