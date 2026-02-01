# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 性能基准测试模块

本模块实现了 `vllm bench` 子命令，用于测试 vLLM 的推理性能。

支持的基准测试类型：

    throughput (吞吐量测试):
        - 测量每秒处理的请求数量
        - 适合评估整体性能

    latency (延迟测试):
        - 测量单个请求的响应时间
        - 适合评估首 token 延迟

    convoy (车队测试):
        - 测量连续请求流的性能
        - 适合评估持续负载

    serving (服务测试):
        - 模拟真实服务场景
        - 测量 QPS (每秒查询数)、TTFT (首 token 时间)

    benchmark 流程：
        1. 准备测试数据（提示词）
        2. 启动/连接 vLLM 引擎
        3. 发送请求并收集指标
        4. 输出性能报告

常用命令：

    # 吞吐量测试
    vllm bench throughput --model Qwen/Qwen3-0.6B

    # 延迟测试
    vllm bench latency --model Qwen/Qwen3-0.6B

    # 服务测试（需要先启动服务）
    vllm bench serving --model Qwen/Qwen3-0.6B --url http://localhost:8000

输出指标：
    - QPS: 每秒查询数
    - TTFT: 首 token 时间 (Time To First Token)
    - TPOT: 每输出 token 时间 (Time Per Output Token)
    - Throughput: 吞吐量 (tokens/s)
"""

import argparse
import typing

from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the vLLM CLI."""

    name = "bench"
    help = "vLLM bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"vllm {self.name} <bench_type> [options]",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"vllm {self.name} {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
            cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(
                subcmd=f"{self.name} {cmd_cls.name}"
            )
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
