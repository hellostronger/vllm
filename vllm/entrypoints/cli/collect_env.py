# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 环境信息收集模块

本模块实现了 `vllm collect-env` 子命令，用于收集并显示系统环境信息。

收集的信息包括：

    Python 环境:
        - Python 版本
        - pip/conda 版本
        - 虚拟环境信息

    硬件信息:
        - GPU 型号、数量
        - CUDA 版本
        - GPU 驱动版本
        - 显存大小

    vLLM 安装:
        - vLLM 版本
        - 安装方式 (pip / 源码)
        - 编译选项

    依赖库:
        - PyTorch 版本
        - Transformers 版本
        - 各类可选依赖状态

使用场景：
    1. 问题排查 - 提交 Issue 时附上环境信息
    2. 环境验证 - 检查 vLLM 是否正确安装
    3. 兼容性检查 - 确认硬件/软件兼容性

常用命令：

    # 收集环境信息并打印
    vllm collect-env

    # 输出示例：
    # Python version: 3.12.0
    # Number of GPUs: 1
    # GPU: NVIDIA GeForce RTX 4090
    # VRAM: 24.00 GB
    # CUDA version: 12.9
    # vLLM version: 0.7.3
"""

import argparse
import typing

from vllm.collect_env import main as collect_env_main
from vllm.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class CollectEnvSubcommand(CLISubcommand):
    """The `collect-env` subcommand for the vLLM CLI."""

    name = "collect-env"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Collect information about the environment."""
        collect_env_main()

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        return subparsers.add_parser(
            "collect-env",
            help="Start collecting environment information.",
            description="Start collecting environment information.",
            usage="vllm collect-env",
        )


def cmd_init() -> list[CLISubcommand]:
    return [CollectEnvSubcommand()]
