#!/usr/bin/env python
"""
采样效果比较分析器模块
功能：比较智能采样算法与随机/均匀采样的效果
"""

from .sampling_comparison_analyser import SamplingComparisonAnalyser, analyse_sampling_compare

__all__ = ['SamplingComparisonAnalyser', 'analyse_sampling_compare']
