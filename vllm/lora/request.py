# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM LoRA 请求模块

本模块定义了 LoRA 适配器的请求数据结构。

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法：
    - 只训练少量低秩参数
    - 保持原始模型不变
    - 可以在运行时动态加载

vLLM 的多 LoRA 支持：
    1. 预训练一个基座模型
    2. 训练多个 LoRA 适配器（针对不同任务）
    3. 运行时动态切换适配器
    4. 内存中缓存多个适配器

使用示例：

    from vllm import LLM, SamplingParams, LoRARequest

    # 创建 LoRA 请求
    lora_request = LoRARequest(
        lora_name="chatglm-lora",
        lora_int_id=1,
        lora_path="/path/to/chatglm-lora-adapter"
    )

    # 在推理时指定使用 LoRA
    outputs = llm.generate(
        "你好",
        sampling_params=SamplingParams(max_tokens=100),
        lora_request=lora_request
    )

LoRA 缓存：
    - vLLM 会自动缓存已加载的 LoRA 适配器
    - 切换到已加载的适配器非常快
    - 使用 load_inplace=True 强制重新加载

API 端点：
    POST /v1/chat/completions 时可以通过 header 指定 LoRA：
    - X-LoRA-Name: LoRA 名称
    - X-LoRA-Int-Id: LoRA ID
"""

import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    LoRA 适配器请求

    用于在推理时指定使用哪个 LoRA 适配器。

    属性说明：
        lora_name: LoRA 适配器的名称（用于识别和比较）
        lora_int_id: LoRA 的整数 ID（必须 > 0，分布式环境中需全局唯一）
        lora_path: LoRA 适配器的路径（HuggingFace 格式或本地路径）
        base_model_name: 基座模型名称
        tensorizer_config_dict: Tensorizer 序列化配置
        load_inplace: 是否强制重新加载（即使缓存中已存在）
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        # Ensure lora_path is not empty
        assert self.lora_path, "lora_path cannot be empty"

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare LoRARequest
        instances based on lora_name. This allows for identification
        and comparison lora adapter across engines.
        """
        return isinstance(value, self.__class__) and self.lora_name == value.lora_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash LoRARequest instances
        based on lora_name. This ensures that LoRARequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.lora_name)
