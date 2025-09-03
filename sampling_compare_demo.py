#!/usr/bin/env python
"""
采样效果比较分析器 - 兼容性包装器
功能：比较智能采样算法与随机/均匀采样的效果
已重构：核心逻辑已迁移到 src.analysis.sampling_comparison 模块
此文件仅作为向后兼容的包装器
"""

import warnings
warnings.filterwarnings('ignore')

# 导入新的模块
from src.analysis.sampling_comparison import analyse_sampling_compare


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='采样效果比较分析器 demo')
    parser.add_argument('--workers', type=int, default=-1, help='并行进程数，默认1（串行）')
    parser.add_argument('--result_dir', type=str, default=None, help='分析结果目录，可选')
    args = parser.parse_args()
    analyse_sampling_compare(result_dir=args.result_dir, workers=args.workers)
