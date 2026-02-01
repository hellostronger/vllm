# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 分布式追踪模块

本模块集成了 OpenTelemetry (OTel)，用于分布式追踪和可观测性。

追踪用途：
    1. 性能分析 - 识别瓶颈
    2. 调试 - 追踪请求流程
    3. 监控 - 收集延迟、吞吐量指标
    4. 日志关联 - 将追踪与日志关联

追踪头：
    - traceparent: W3C 标准追踪头
    - tracestate: 追踪状态

支持的协议：
    - OTLP/gRPC (默认)
    - OTLP/HTTP

Span 属性（用于 LLM 相关追踪）：

    gen_ai.usage.*:
        - completion_tokens: 生成的 token 数量
        - prompt_tokens: 提示的 token 数量
        - num_sequences: 生成的数量

    gen_ai.latency.*:
        - time_to_first_token: 首 token 延迟
        - time_in_queue: 排队时间
        - time_in_scheduler: 调度时间
        - time_in_model_forward: 前向传播时间
        - time_in_model_decode: 解码时间

配置示例：
    # 设置 OTLP 端点
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="http://localhost:4317"
    export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="grpc"

    # 启动服务
    vllm serve Qwen/Qwen3-0.6B --otlp-traces-endpoint http://localhost:4317

    # 追踪客户端需要在请求头中包含追踪信息
"""

import os
from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

TRACE_HEADERS = ["traceparent", "tracestate"]

logger = init_logger(__name__)

_is_otel_imported = False
otel_import_error_traceback: str | None = None
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    # See https://github.com/vllm-project/vllm/pull/7266#discussion_r1707395458
    import traceback

    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(
    instrumenting_module_name: str, otlp_traces_endpoint: str
) -> Tracer | None:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    trace_provider = TracerProvider()

    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
