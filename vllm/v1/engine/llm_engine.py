# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 推理引擎模块

这是 vLLM 的核心引擎类，负责管理整个推理流程。

主要功能：
1. 请求管理：接收用户的推理请求，调度到引擎核心处理
2. 批处理优化：将多个请求组合成批次以提高吞吐量
3. 输出处理：将模型输出转换为用户友好的格式
4. 统计日志：记录推理过程中的各项指标

架构说明：
    LLMEngine 是 v1 引擎的包装层，实际推理工作在 EngineCore 中完成。
    LLMEngine 负责：
    - 输入预处理（分词、格式转换）
    - 输出后处理（解码、格式化）
    - 统计信息收集
    EngineCore 负责：
    - 真正的模型推理
    - KV 缓存管理
    - 请求调度
"""

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any, cast

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """
    vLLM 推理引擎类

    这是用户直接交互的主要入口类（通过 LLM() 或 AsyncLLM 使用），
    内部通过 EngineCoreClient 与 EngineCore 通信。

    核心职责：
    1. 请求路由：将用户请求分发到 EngineCore 处理
    2. 输入处理：预处理分词、多模态数据
    3. 输出处理：将引擎输出转换为 RequestOutput
    4. 统计与监控：收集并上报性能指标

    注意：为了向后兼容保留此命名，实际核心逻辑在 EngineCore 中。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        # ===== 引擎配置 =====
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # ===== 日志开关 =====
        self.log_stats = log_stats

        # ===== 数据并行组初始化 =====
        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        # 判断是否使用外部启动器的数据并行
        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )

        # 重要：在初始化 engine_core 之前先初始化 dp group
        # 在解耦引擎情况下，这部分在 EngineCoreProc 中处理
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            # 初始化数据并行通信组
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None

        # 用于控制是否执行虚拟批次的标志
        self.should_execute_dummy_batch = False

        # ===== 输入处理器 =====
        # 负责处理用户输入：分词、多模态数据预处理等
        self.input_processor = InputProcessor(self.vllm_config)

        # I/O 处理器：处理输入输出的特殊格式（如表格数据）
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # ===== 输出处理器 =====
        # 将 EngineCore 的原始输出转换为用户友好的 RequestOutput
        self.output_processor = OutputProcessor(
            self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
        )

        # 如果配置了 OpenTelemetry 追踪，初始化追踪器
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        # ===== 引擎核心客户端 =====
        # EngineCore 是实际执行推理的组件，通过 IPC 与 LLMEngine 通信
        # 通信方式可以是：同进程、Unix 套接字、HTTP
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        # ===== 统计日志管理器 =====
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        # 兼容 v0：暴露 model_executor 属性
        if not multiprocess_mode:
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        # 如果使用外部启动器的 DP，复用现有的 DP 组
        if self.external_launcher_dp:
            self.dp_group = get_dp_group().cpu_group

        # 清空多模态缓存占位数据
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> None:
        """
        添加一个推理请求到引擎

        参数处理流程：
        1. 验证 request_id 类型（必须是字符串）
        2. 处理输入数据，转换为内部请求格式
        3. 如果是采样参数 n>1，进行请求分叉（生成多个采样结果）
        4. 将请求添加到输出处理器和引擎核心

        参数:
            request_id: 请求的唯一标识符
            prompt: 用户输入（可以是字符串、TokensPrompt、DataPrompt 等）
            params: 采样参数或池化参数
            arrival_time: 请求到达时间（默认当前时间）
            lora_request: LoRA 适配器请求（可选）
            tokenization_kwargs: 分词器额外参数
            trace_headers: 分布式追踪头信息
            priority: 请求优先级（数值越大优先级越高）
            prompt_text: 原始提示文本（自动推断，可不传）
        """
        # 验证 request_id 类型
        if not isinstance(request_id, str):
            raise TypeError(f"request_id 必须是字符串类型，实际得到 {type(request_id)}")

        # 将原始输入处理为内部请求格式
        if isinstance(prompt, EngineCoreRequest):
            # 如果已经是内部格式，直接使用
            request = prompt
            # 警告：如果 request_id 不匹配，使用 request 内部的 id
            if request_id != request.request_id:
                logger.warning_once(
                    "add_request() 传入的 request_id 与 EngineCoreRequest.request_id 不匹配，"
                    "将使用后者。"
                )
        else:
            # 原始输入需要预处理：分词、多模态处理等
            assert prompt_text is None
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
            )
            # 记录原始提示文本用于输出
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        # 为请求分配 ID
        self.input_processor.assign_request_id(request)

        # 使用处理后的参数（可能在 process_inputs 中被更新）
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        # === 单采样情况：直接添加请求 ===
        if n == 1:
            # 在输出处理器中创建请求状态
            self.output_processor.add_request(request, prompt_text, None, 0)
            # 添加到引擎核心开始处理
            self.engine_core.add_request(request)
            return

        # === 多采样情况：分叉请求 ===
        # 当 n>1 时，需要将一个请求拆分为多个子请求
        parent_req = ParentRequest(request)
        for idx in range(n):
            # 获取子请求信息
            request_id, child_params = parent_req.get_child_info(idx)
            # 复制请求（最后一个复用原对象以节省内存）
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # 添加到输出处理器和引擎核心
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            self.engine_core.add_request(child_request)

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        """
        执行一步推理迭代

        核心执行流程：
        1. 从 EngineCore 获取本轮输出
        2. 处理输出（解码、格式化）
        3. 处理因 stop 条件完成的请求
        4. 记录统计信息

        返回:
            本轮产生的可输出结果列表
        """
        # 如果需要执行虚拟批次（用于保持 GPU活跃度），先执行
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # ===== 步骤 1：从引擎核心获取输出 =====
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # ===== 步骤 2：处理引擎输出 =====
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            # 更新调度器统计信息
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # ===== 步骤 3：处理因 stop 字符串/token 而完成的请求 =====
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # ===== 步骤 4：记录统计信息 =====
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if self.logger_manager is not None and outputs.scheduler_stats is not None:
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.input_processor.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.input_processor.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1):
        self.engine_core.sleep(level)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.input_processor.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.input_processor.get_tokenizer()

    @property
    def renderer(self) -> BaseRenderer:
        return self.input_processor.renderer

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
