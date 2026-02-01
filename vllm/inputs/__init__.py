# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 输入处理模块

本模块定义了 vLLM 接受的各种输入类型和数据结构。

输入类型分类：

    文本输入 (TextPrompt):
        {"prompt": "你好，世界！"}

    Token 输入 (TokensPrompt):
        {"prompt_token_ids": [1, 2, 3, 4, 5]}

    多模态输入 (DataPrompt):
        {"prompt": "描述这张图片", "multi_modal_data": {"image": <image>}}

    Embedding 输入 (EmbedsPrompt):
        用于直接输入 Embedding（用于 Embedding 模型）

    Encoder-Decoder 输入:
        用于 T5 等编码器-解码器模型

输入格式要求：
    - prompt: str - 原始文本
    - prompt_token_ids: List[int] - 已分词的 token ID
    - multi_modal_data: Dict - 多模态数据（如图像）
    - mm_hashes: Dict - 多模态数据的哈希值
    - mm_placeholders: Dict - 多模态占位符

使用示例：
    from vllm import TextPrompt

    prompt = TextPrompt(prompt="给我讲个笑话")
    responses = engine.step(prompt)
"""

from .data import (
    DataPrompt,
    DecoderOnlyInputs,
    EmbedsInputs,
    EmbedsPrompt,
    EncoderDecoderInputs,
    ExplicitEncoderDecoderPrompt,
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    SingletonPrompt,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
    build_explicit_enc_dec_prompt,
    embeds_inputs,
    to_enc_dec_tuple_list,
    token_inputs,
    zip_enc_dec_prompts,
)

__all__ = [
    "DataPrompt",
    "TextPrompt",
    "TokensPrompt",
    "PromptType",
    "SingletonPrompt",
    "ExplicitEncoderDecoderPrompt",
    "TokenInputs",
    "EmbedsInputs",
    "EmbedsPrompt",
    "token_inputs",
    "embeds_inputs",
    "DecoderOnlyInputs",
    "EncoderDecoderInputs",
    "ProcessorInputs",
    "SingletonInputs",
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
]
