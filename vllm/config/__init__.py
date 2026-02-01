# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 配置模块

本模块定义了 vLLM 引擎运行所需的所有配置类。

主要配置类：

    VllmConfig: 顶级配置容器，包含所有子配置
    ModelConfig: 模型相关配置（模型名、dtype、分词器等）
    CacheConfig: KV Cache 配置（块大小、块数量等）
    ParallelConfig: 分布式并行配置（TP、PP、DP 等）
    SchedulerConfig: 调度器配置（批处理大小、最大请求数等）
    AttentionConfig: 注意力配置（后端、头数、分块大小等）
    CompilationConfig: 模型编译配置（CUDA Graph、编译模式等）
    LoRAConfig: LoRA 适配器配置
    LoadConfig: 模型加载配置（来源、格式等）
    MultiModalConfig: 多模态配置

配置层次结构：
    VllmConfig
    ├── model_config: ModelConfig
    ├── cache_config: CacheConfig
    ├── parallel_config: ParallelConfig
    ├── scheduler_config: SchedulerConfig
    ├── attention_config: AttentionConfig
    ├── compilation_config: CompilationConfig
    ├── lora_config: LoRAConfig (可选)
    ├── load_config: LoadConfig
    ├── multimodal_config: MultiModalConfig (可选)
    └── ...

配置来源：
    1. 命令行参数 (AsyncEngineArgs.from_cli_args())
    2. 环境变量 (VLLM_*)
    3. 配置文件 (YAML)
    4. 编程方式 (VllmConfig(...))
"""

from vllm.config.attention import AttentionConfig
from vllm.config.cache import CacheConfig
from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    PassConfig,
)
from vllm.config.device import DeviceConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.config.kv_events import KVEventsConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.load import LoadConfig
from vllm.config.lora import LoRAConfig
from vllm.config.model import (
    ModelConfig,
    iter_architecture_defaults,
    str_dtype_to_torch_dtype,
    try_match_architecture_defaults,
)
from vllm.config.multimodal import MultiModalConfig
from vllm.config.observability import ObservabilityConfig
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.config.pooler import PoolerConfig
from vllm.config.profiler import ProfilerConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.config.utils import (
    ConfigType,
    SupportsMetricsInfo,
    config,
    get_attr_docs,
    is_init_field,
    update_config,
)
from vllm.config.vllm import (
    VllmConfig,
    get_cached_compilation_config,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)

# __all__ should only contain classes and functions.
# Types and globals should be imported from their respective modules.
__all__ = [
    # From vllm.config.attention
    "AttentionConfig",
    # From vllm.config.cache
    "CacheConfig",
    # From vllm.config.compilation
    "CompilationConfig",
    "CompilationMode",
    "CUDAGraphMode",
    "PassConfig",
    # From vllm.config.device
    "DeviceConfig",
    # From vllm.config.ec_transfer
    "ECTransferConfig",
    # From vllm.config.kv_events
    "KVEventsConfig",
    # From vllm.config.kv_transfer
    "KVTransferConfig",
    # From vllm.config.load
    "LoadConfig",
    # From vllm.config.lora
    "LoRAConfig",
    # From vllm.config.model
    "ModelConfig",
    "iter_architecture_defaults",
    "str_dtype_to_torch_dtype",
    "try_match_architecture_defaults",
    # From vllm.config.multimodal
    "MultiModalConfig",
    # From vllm.config.observability
    "ObservabilityConfig",
    # From vllm.config.parallel
    "EPLBConfig",
    "ParallelConfig",
    # From vllm.config.pooler
    "PoolerConfig",
    # From vllm.config.scheduler
    "SchedulerConfig",
    # From vllm.config.speculative
    "SpeculativeConfig",
    # From vllm.config.speech_to_text
    "SpeechToTextConfig",
    # From vllm.config.structured_outputs
    "StructuredOutputsConfig",
    # From vllm.config.profiler
    "ProfilerConfig",
    # From vllm.config.utils
    "ConfigType",
    "SupportsMetricsInfo",
    "config",
    "get_attr_docs",
    "is_init_field",
    "update_config",
    # From vllm.config.vllm
    "VllmConfig",
    "get_cached_compilation_config",
    "get_current_vllm_config",
    "get_current_vllm_config_or_none",
    "set_current_vllm_config",
    "get_layers_from_vllm_config",
]
