# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 命令行入口模块

这个模块是 vLLM 程序的入口点，当用户运行 `vllm` 命令时首先执行。
它负责解析用户输入的命令（如 serve、bench、run-batch 等），然后
将工作委托给对应的子命令处理器。

使用说明：
    vllm serve <model>          # 启动 OpenAI 兼容的 HTTP 服务
    vllm bench <model>          # 运行性能基准测试
    vllm run-batch <model>      # 批量离线推理
    vllm collect-env            # 收集环境信息

注意事项：
    所有子命令模块都在 main() 函数内部导入，而不是在模块顶部导入。
    这是为了避免在模块作用域导入时可能发生的 CUDA 初始化等问题。
"""

import importlib.metadata
import sys

from vllm.logger import init_logger

logger = init_logger(__name__)


def main():
    """
    vLLM CLI 的主入口函数

    执行流程：
    1. 导入所有子命令模块（serve、bench、openai、run-batch、collect-env）
    2. 设置 CLI 环境（环境变量等）
    3. 创建参数解析器，识别用户输入的命令
    4. 将命令分发给对应的处理器执行

    参数：
        无（从 sys.argv 获取命令行参数）

    返回：
        无
    """

    # ===== 步骤 1：导入所有子命令模块 =====
    # 注意：这些导入发生在 main() 函数开头，而不是文件顶部。
    # 如果用户只是运行 `--version` 或 `--help`，这些模块仍会被导入。
    # 这种设计是为了避免在模块作用域导入时可能发生的初始化问题。
    import vllm.entrypoints.cli.benchmark.main
    import vllm.entrypoints.cli.collect_env
    import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve

    # 导入工具函数
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # 定义所有可用的命令模块
    CMD_MODULES = [
        vllm.entrypoints.cli.openai,        # openai 命令：与 OpenAI API 交互
        vllm.entrypoints.cli.serve,         # serve 命令：启动 HTTP 服务
        vllm.entrypoints.cli.benchmark.main, # bench 命令：性能测试
        vllm.entrypoints.cli.collect_env,   # collect-env 命令：收集环境信息
        vllm.entrypoints.cli.run_batch,     # run-batch 命令：批量离线推理
    ]

    # ===== 步骤 2：设置 CLI 环境 =====
    # 初始化环境变量、设置日志等
    cli_env_setup()

    # ===== 步骤 3：特殊处理 bench 命令 =====
    # 当用户运行 `vllm bench` 时，默认使用 CPU 平台
    # 这是因为基准测试可能在没有 GPU 的机器上运行
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "检测到 bench 命令，需要确保当前平台不是 UnspecifiedPlatform，"
            "以避免设备类型推断错误"
        )
        from vllm import platforms

        # 如果平台未指定（没有检测到 GPU），切换到 CPU 平台
        if platforms.current_platform.is_unspecified():
            from vllm.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "未检测到可用平台，已自动切换到 CPU 平台。"
            )

    # ===== 步骤 4：创建命令行参数解析器 =====
    parser = FlexibleArgumentParser(
        description="vLLM CLI - 高吞吐量 LLM 推理引擎",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[子命令]"),
    )

    # 添加全局参数：版本号
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )

    # 创建子命令解析器
    # subparsers 用于将命令分发到不同的处理器
    subparsers = parser.add_subparsers(required=False, dest="subparser")

    # 命令字典：用于快速查找命令对应的处理器
    cmds = {}

    # ===== 步骤 5：注册所有子命令 =====
    # 遍历每个命令模块，让它们注册自己的子命令
    for cmd_module in CMD_MODULES:
        # cmd_init() 返回该模块支持的所有命令
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            # 每个命令都需要：
            # 1. subparser_init(): 创建子命令的参数解析器
            # 2. set_defaults(): 设置dispatch_function为命令的处理函数
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd

    # 记录加载的命令信息，便于调试和验证
    logger.info(f"已加载的CLI命令: {list(cmds.keys())}")

    # ===== 步骤 6：解析并执行命令 =====
    args = parser.parse_args()

    # 如果用户输入了有效的子命令名称
    if args.subparser in cmds:
        # 调用命令的验证函数，检查参数是否合法
        cmds[args.subparser].validate(args)

    # 如果命令有对应的处理函数，调用它
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        # 用户没有输入有效命令，打印帮助信息
        parser.print_help()


if __name__ == "__main__":
    # 当直接运行此脚本时调用 main()
    # 这样可以通过 python vllm/entrypoints/cli/main.py 测试 CLI
    main()