# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 服务启动模块

本模块实现了 `vllm serve` 子命令，负责启动 OpenAI 兼容的 HTTP API 服务器。
用户可以通过 HTTP 请求发送推理请求，服务器返回生成的文本结果。

核心功能：
    1. 解析 serve 命令的参数
    2. 创建 LLM 引擎实例
    3. 启动 HTTP 服务器（支持单进程和多进程模式）
    4. 支持数据并行和负载均衡

使用示例：
    vllm serve Qwen/Qwen3-0.6B              # 启动单进程服务
    vllm serve Qwen/Qwen3-0.6B --tp 2       # 启动 2 卡推理
    vllm serve Qwen/Qwen3-0.6B --headless   # 无头模式（无 HTTP 服务）

启动流程：
    serve 命令
        ↓
    解析参数
        ↓
    创建引擎配置 (VllmConfig)
        ↓
    启动模式判断
        ├── 单进程：直接运行 run_server()
        ├── 多进程：启动多个 API 服务器
        └── 无头模式：仅运行引擎
"""

import argparse
import signal
import sys

import vllm
import vllm.envs as envs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import (
    run_server,
    run_server_worker,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_tcp_uri
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager, launch_core_engines
from vllm.v1.executor import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
from vllm.v1.utils import APIServerProcessManager, wait_for_completion_or_failure

logger = init_logger(__name__)

# 条件导入 uvloop（仅在非 Windows 系统上）
USE_UVLOOP = False
if sys.platform != "win32":
    try:
        import uvloop
        USE_UVLOOP = True
    except ImportError:
        pass

DESCRIPTION = """启动本地 OpenAI 兼容的 HTTP API 服务器来提供 LLM 推理服务。

如果不指定模型，默认使用 Qwen/Qwen3-0.6B。

提示：使用 `--help=<配置组>` 可以按组查看选项（例如：
  --help=ModelConfig, --help=Frontend）
  使用 `--help=all` 可以一次性显示所有可用参数。
"""


class ServeSubcommand(CLISubcommand):
    """
    Serve 子命令处理器

    负责处理 `vllm serve` 命令，完成以下工作：
    1. 验证命令行参数
    2. 根据配置选择运行模式（单进程/多进程/无头）
    3. 启动相应的服务
    """

    name = "serve"  # 命令名称

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """
        执行 serve 命令的主逻辑

        参数处理流程：
        1. 如果用户通过位置参数指定了模型名称，优先使用它
        2. 检测负载均衡模式（外部/混合/内部）
        3. 根据 api_server_count 选择运行模式

        参数:
            args: 解析后的命令行参数命名空间

        返回:
            无
        """
        # ===== 参数预处理 =====
        # 如果用户在命令行中通过位置参数指定了模型（model_tag），
        # 它优先于通过选项指定的模型（model）
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        # ===== 无头模式处理 =====
        # 无头模式（headless）表示不启动 HTTP API 服务器，
        # 只需要运行引擎本身，用于与外部调度系统集成
        if args.headless:
            # 无头模式下不能同时指定 api_server_count
            if args.api_server_count is not None and args.api_server_count > 0:
                raise ValueError(
                    f"--api-server-count={args.api_server_count} 不能与 "
                    "--headless 同时使用（无头模式下不启动 API 服务器）。"
                )
            # 无头模式下默认 api_server_count 为 0
            args.api_server_count = 0

        # ===== 检测负载均衡模式 =====
        # vLLM 支持三种负载均衡模式：
        # 1. 外部 LB（external）：外部负载均衡器（如 Nginx）负责分发请求
        # 2. 混合 LB（hybrid）：内部 LB 处理本地 GPU，外部 LB 跨节点分发
        # 3. 内部 LB（internal）：vLLM 自己管理所有请求分发
        is_external_lb = (
            args.data_parallel_external_lb or args.data_parallel_rank is not None
        )
        is_hybrid_lb = (
            args.data_parallel_hybrid_lb or args.data_parallel_start_rank is not None
        )

        # 不能同时使用两种负载均衡模式
        if is_external_lb and is_hybrid_lb:
            raise ValueError(
                "不能同时使用外部负载均衡和混合负载均衡模式。"
                "外部 LB 通过 --data-parallel-external-lb 或 --data-parallel-rank 启用。"
                "混合 LB 通过 --data-parallel-hybrid-lb 或 --data-parallel-start-rank 启用。"
                "请选择其中一种模式。"
            )

        # ===== 设置默认的 api_server_count =====
        # api_server_count 表示要启动多少个 API 服务器进程
        # 如果用户没有明确指定，根据负载均衡模式选择合适的默认值
        if args.api_server_count is None:
            if is_external_lb:
                # 外部 LB：每个节点只启动 1 个 API 服务器，由外部 LB 分发
                args.api_server_count = 1
            elif is_hybrid_lb:
                # 混合 LB：启动与本地 GPU 数量相等的 API 服务器
                args.api_server_count = args.data_parallel_size_local or 1
                if args.api_server_count > 1:
                    logger.info(
                        "在混合 LB 模式下，默认 api_server_count 为 "
                        "data_parallel_size_local（%d）。",
                        args.api_server_count,
                    )
            else:
                # 内部 LB：启动与数据并行规模相等的 API 服务器
                args.api_server_count = args.data_parallel_size
                if args.api_server_count > 1:
                    logger.info(
                        "默认 api_server_count 为 data_parallel_size（%d）。",
                        args.api_server_count,
                    )

        # ===== 选择运行模式 =====
        # 根据 api_server_count 决定启动方式：
        # 1. api_server_count < 1：无头模式（无 HTTP 服务）
        # 2. api_server_count = 1：单进程模式
        # 3. api_server_count > 1：多进程模式
        if args.api_server_count < 1:
            run_headless(args)
        elif args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # 单进程模式：直接在当前进程运行 HTTP 服务器
            # 使用 uvloop 作为事件循环（如果可用）
            if USE_UVLOOP:
                import uvloop
                uvloop.run(run_server(args))
            else:
                # Windows 系统或 uvloop 不可用时使用默认的 asyncio 事件循环
                import asyncio
                asyncio.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            help="Launch a local OpenAI-compatible API server to serve LLM "
            "completions via HTTP.",
            description=DESCRIPTION,
            usage="vllm serve [model_tag] [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]


def run_headless(args: argparse.Namespace):
    if args.api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")

    # Create the EngineConfig.
    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(
        usage_context=usage_context, headless=True
    )

    if engine_args.data_parallel_hybrid_lb:
        raise ValueError("data_parallel_hybrid_lb is not applicable in headless mode")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    shutdown_requested = False

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if parallel_config.node_rank_within_dp > 0:
        from vllm.version import __version__ as VLLM_VERSION

        # Run headless workers (for multi-node PP/TP).
        host = parallel_config.master_addr
        head_node_address = f"{host}:{parallel_config.master_port}"
        logger.info(
            "Launching vLLM (v%s) headless multiproc executor, "
            "with head node address %s for torch.distributed process group.",
            VLLM_VERSION,
            head_node_address,
        )

        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return

    host = parallel_config.data_parallel_master_ip
    port = parallel_config.data_parallel_rpc_port
    handshake_address = get_tcp_uri(host, port)

    logger.info(
        "Launching %d data parallel engine(s) in headless mode, "
        "with head node address %s.",
        local_engine_count,
        handshake_address,
    )

    # Create the engines.
    engine_manager = CoreEngineProcManager(
        target_fn=EngineCoreProc.run_engine_core,
        local_engine_count=local_engine_count,
        start_index=vllm_config.parallel_config.data_parallel_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=Executor.get_class(vllm_config),
        log_stats=not engine_args.disable_log_stats,
    )

    try:
        engine_manager.join_first()
    finally:
        logger.info("Shutting down.")
        engine_manager.close()


def run_multi_api_server(args: argparse.Namespace):
    assert not args.headless
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 0

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

    listen_address, sock = setup_server(args)

    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    engine_args._api_process_count = num_api_servers
    engine_args._api_process_rank = -1

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if num_api_servers > 1 and envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        raise ValueError(
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with api_server_count > 1"
        )

    executor_class = Executor.get_class(vllm_config)
    log_stats = not engine_args.disable_log_stats

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    assert parallel_config.local_engines_only or dp_rank == 0

    api_server_manager: APIServerProcessManager | None = None

    with launch_core_engines(
        vllm_config, executor_class, log_stats, num_api_servers
    ) as (local_engine_manager, coordinator, addresses):
        # Construct common args for the APIServerProcessManager up-front.
        api_server_manager_kwargs = dict(
            target_server_fn=run_api_server_worker_proc,
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=num_api_servers,
            input_addresses=addresses.inputs,
            output_addresses=addresses.outputs,
            stats_update_address=coordinator.get_stats_publish_address()
            if coordinator
            else None,
        )

        # For dp ranks > 0 in external/hybrid DP LB modes, we must delay the
        # start of the API servers until the local engine is started
        # (after the launcher context manager exits),
        # since we get the front-end stats update address from the coordinator
        # via the handshake with the local engine.
        if dp_rank == 0 or not parallel_config.local_engines_only:
            # Start API servers using the manager.
            api_server_manager = APIServerProcessManager(**api_server_manager_kwargs)

    # Start API servers now if they weren't already started.
    if api_server_manager is None:
        api_server_manager_kwargs["stats_update_address"] = (
            addresses.frontend_stats_publish_address
        )
        api_server_manager = APIServerProcessManager(**api_server_manager_kwargs)

    # Wait for API servers
    wait_for_completion_or_failure(
        api_server_manager=api_server_manager,
        engine_manager=local_engine_manager,
        coordinator=coordinator,
    )


def run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Entrypoint for individual API server worker processes."""
    client_config = client_config or {}
    server_index = client_config.get("client_index", 0)

    # Set process title and add process-specific prefix to stdout and stderr.
    set_process_title("APIServer", str(server_index))
    decorate_logs()

    # 使用适当的事件循环运行服务器（Windows 兼容）
    if USE_UVLOOP:
        import uvloop
        uvloop.run(
            run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs)
        )
    else:
        import asyncio
        asyncio.run(
            run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs)
        )