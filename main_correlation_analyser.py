#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABACUS相关性分析独立脚本
功能：对已有的系统指标数据进行相关性分析
调用现有的 CorrelationAnalyser 模块
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.correlation_analyser import CorrelationAnalyser
from src.logmanager import LoggerManager
from src.utils import Constants


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ABACUS相关性分析器 - 分析温度、构象与多样性指标的相关性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
--------
# 分析默认的 system_metrics_summary.csv
python main_correlation_analyser.py

# 指定输入文件和输出目录
python main_correlation_analyser.py -i /path/to/system_metrics_summary.csv -o /path/to/output

# 禁用日志文件输出
python main_correlation_analyser.py --no-log-file
        """
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="auto",
        help="输入CSV文件路径 (默认: auto - 自动查找 system_metrics_summary.csv)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="combined_analysis_results",
        help="输出目录路径 (默认: combined_analysis_results)"
    )

    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="禁用日志文件输出，只输出到控制台"
    )

    args = parser.parse_args()

    # 创建日志记录器
    logger = LoggerManager.create_logger(
        name="CorrelationAnalyser",
        level=Constants.DEFAULT_LOG_LEVEL,
        add_console=True,
        add_file=not args.no_log_file,
        log_format=Constants.DEFAULT_LOG_FORMAT,
        date_format=Constants.DEFAULT_DATE_FORMAT,
    )

    logger.info("=== ABACUS 相关性分析器启动 ===")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出目录: {args.output}")

    try:
        # 创建相关性分析器
        analyser = CorrelationAnalyser(logger=logger)

        # 确定输入文件路径
        if args.input == "auto":
            # 自动查找 system_metrics_summary.csv
            possible_paths = [
                "analysis_results/system_metrics_summary.csv",
                "analysis_results/run_*/system_metrics_summary.csv",
                "system_metrics_summary.csv"
            ]

            input_file = None
            for path_pattern in possible_paths:
                if "*" in path_pattern:
                    # 处理通配符路径
                    import glob
                    matches = glob.glob(path_pattern)
                    if matches:
                        input_file = matches[0]  # 使用第一个匹配的文件
                        break
                elif os.path.exists(path_pattern):
                    input_file = path_pattern
                    break

            if input_file is None:
                logger.error("未找到 system_metrics_summary.csv 文件")
                logger.info("请使用 -i 参数指定输入文件路径")
                return 1

            logger.info(f"自动找到输入文件: {input_file}")
        else:
            input_file = args.input
            if not os.path.exists(input_file):
                logger.error(f"输入文件不存在: {input_file}")
                return 1

        # 执行相关性分析
        logger.info("开始相关性分析...")
        success = analyser.analyse_correlations(input_file, args.output)

        if success:
            logger.info("✅ 相关性分析完成！")
            logger.info(f"结果已保存到: {args.output}")

            # 显示输出文件
            output_files = [
                "parameter_analysis_results.csv",
                "correlation_analysis.log"
            ]

            logger.info("输出文件:")
            for filename in output_files:
                filepath = os.path.join(args.output, filename)
                if os.path.exists(filepath):
                    logger.info(f"  - {filepath}")

            return 0
        else:
            logger.error("❌ 相关性分析失败")
            return 1

    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
