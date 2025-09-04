#!/usr/bin/env python
"""
脚本名: trajectory_analyser.py
功能: ABACUS 轨迹分析器
==================================================

功能：
-----
批量分析 ABACUS 分子动力学模拟生成的 STRU 文件序列，
基于原子间距离向量的标准化分布特性及推荐指标，
评估构象多样性。

核心指标（更新版）：
------------------
1. ConfVol: 构象空间体积（核心多样性指标）
2. ANND: 平均最近邻距离（平均最近邻距离）
3. MPD: 平均成对距离（平均成对距离）
4. PCA explained variance ratios: 主成分方差贡献率

所有输出均为无量纲标准化指标，适用于跨体系比较。

输入结构：
---------
当前目录下包含多个体系文件夹：
    struct_mol_<ID>_conf_<N>_T<T>K/
    └── OUT.ABACUS/
        └── STRU/
            ├── STRU_MD_0
            ├── STRU_MD_1
            └── ...

输出结构：
---------
analysis_results/
├── struct_mol_<ID>/
│   ├── standardized_metrics_per_frame_<system>.csv    ← 单帧指标 + 采样结果
│   ├── standardized_metrics_<system>.json             ← 聚合指标（无每帧）
│   └── ...（合并分析同理）
└── standardized_distribution_summary.csv               ← 所有体系汇总对比

使用方式：
----------
python trajectory_analyser.py [--include_h] [--max_workers N] [--sample_ratio R] [--sample_count N]

依赖：
------
numpy, scipy, stru_parser.py
"""

import logging

import numpy as np
from scipy.spatial.distance import cdist, pdist

# Level 3: 引入结构指标统一模块（供外部使用）
try:  # 软依赖，避免循环导入风险
    from ..core.system_analyser import RMSDCalculator
    kabsch_align = RMSDCalculator.kabsch_align
    iterative_mean_structure = RMSDCalculator.iterative_mean_structure
    compute_rmsf = RMSDCalculator.compute_rmsf
except Exception:  # noqa: BLE001 - 宽松捕获，仅降级功能
    kabsch_align = None  # type: ignore
    iterative_mean_structure = None  # type: ignore
    compute_rmsf = None  # type: ignore

# 导入解析器模块

# 注意：日志配置现在由主程序管理，模块只获取logger
logger = logging.getLogger(__name__)


# --- 核心数据结构 ---

# --- 采样函数 ---
# 说明：原本此处存在 greedy_max_avg_distance 的本地实现，现已统一迁移/整合至 `core.sampler.GreedyMaxDistanceSampler`。
# 如需最大距离贪婪采样，请在上层调用处使用：
# from ..core.sampler import GreedyMaxDistanceSampler
# selected_indices, _, _ = GreedyMaxDistanceSampler.select_frames(points, k)


# --- 核心分析逻辑 ---
# 说明：RMSD/RMSF/Kabsch/迭代均值结构函数已迁移至 core.system_analyser.RMSDCalculator
# 保留兼容导入（文件顶部的 try 导入），避免重复实现。
