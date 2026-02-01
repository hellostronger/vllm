# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 多模态模块

本模块提供了对多模态输入（图像、视频、音频）的支持。

支持的多模态类型：

    图像 (Image):
        - 输入：图片文件路径、URL 或 PIL 图像
        - 模型：LLaVA, Qwen-VL, MiniGPT-4 等

    视频 (Video):
        - 输入：视频文件路径
        - 模型：支持视频理解的多模态模型

    音频 (Audio):
        - 输入：音频文件路径
        - 模型：支持语音输入的多模态模型

    3D (Point Cloud):
        - 输入：点云数据
        - 模型：点云理解模型

多模态数据格式：

    {
        "prompt": "描述这张图片",
        "multi_modal_data": {
            "image": [<image data>, ...]  # 支持多张图片
        }
    }

使用示例：

    from vllm import LLM, TextPrompt, SamplingParams

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # 单张图片
    prompts = [
        TextPrompt(
            prompt="Describe this image.",
            multi_modal_data={"image": "https://example.com/image.jpg"}
        )
    ]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

缓存机制：
    - 多模态数据通过哈希进行缓存
    - 相同图片不会重复处理
    - 缓存键：图片内容哈希

配置选项：
    VLLM_ASSETS_CACHE - 缓存目录
    VLLM_IMAGE_FETCH_TIMEOUT - 图片获取超时
"""

from .hasher import MultiModalHasher
from .inputs import (
    BatchedTensorInputs,
    ModalityData,
    MultiModalDataBuiltins,
    MultiModalDataDict,
    MultiModalKwargsItems,
    MultiModalPlaceholderDict,
    MultiModalUUIDDict,
    NestedTensors,
)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global [`MultiModalRegistry`][vllm.multimodal.registry.MultiModalRegistry]
is used by model runners to dispatch data processing according to the target
model.

Info:
    [mm_processing](../../../design/mm_processing.md)
"""

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHasher",
    "MultiModalKwargsItems",
    "MultiModalPlaceholderDict",
    "MultiModalUUIDDict",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
