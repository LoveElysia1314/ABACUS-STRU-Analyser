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
    calculate_rmsd_series = RMSDCalculator.compute_rmsd_series
    compute_rmsf = RMSDCalculator.compute_rmsf
except Exception:  # noqa: BLE001 - 宽松捕获，仅降级功能
    kabsch_align = None  # type: ignore
    iterative_mean_structure = None  # type: ignore
    calculate_rmsd_series = None  # type: ignore
    compute_rmsf = None  # type: ignore

# 导入解析器模块

# 注意：日志配置现在由主程序管理，模块只获取logger
logger = logging.getLogger(__name__)


# --- 核心数据结构 ---
class FrameMetrics:
    """单帧的标准化分析指标（更新为新指标）"""

    def __init__(self, frame_id, ANND, MPD):
        self.frame_id = frame_id
        self.ANND = ANND  # 平均最近邻距离
        self.MPD = MPD  # 平均成对距离


class DistributionMetrics:
    """整体分布的标准化分析指标（更新版）"""

    def __init__(self, ConfVol, nAPD, ANND, MPD, num_frames, dimension, pca_explained_variance_ratio=None):
        self.ConfVol = ConfVol  # 标准化构象空间体积
        self.nAPD = nAPD  # 标准化平均成对距离
        self.ANND = ANND  # 平均最近邻距离
        self.MPD = MPD  # 平均成对距离
        self.num_frames = num_frames
        self.dimension = dimension  # 距离向量维度
        self.pca_explained_variance_ratio = pca_explained_variance_ratio or []  # PCA方差贡献率


class AnalysisResult:
    """单次分析的结果封装"""

    def __init__(
        self, frame_metrics_list, distribution_metrics, frame_numbers, molecular_formula
    ):
        self.frame_metrics_list = frame_metrics_list
        self.distribution_metrics = distribution_metrics
        self.frame_numbers = frame_numbers
        self.molecular_formula = molecular_formula


class SummaryInfo:
    """用于生成汇总表的信息"""

    def __init__(
        self,
        type_,
        dir_name,
        mol_id,
        conf,
        T,
        ConfVol,
        nAPD,
        ANND,
        num_frames,
        dimension,
        pca_explained_variance_ratio=None,
    ):
        self.type = type_
        self.dir_name = dir_name
        self.mol_id = mol_id
        self.conf = conf
        self.T = T
        self.ConfVol = ConfVol
        self.nAPD = nAPD
        self.ANND = ANND
        self.num_frames = num_frames
        self.dimension = dimension
        self.pca_explained_variance_ratio = pca_explained_variance_ratio or []


# --- 采样函数 ---
# 说明：原本此处存在 greedy_max_avg_distance 的本地实现，现已统一迁移/整合至 `core.sampler.GreedyMaxDistanceSampler`。
# 如需最大距离贪婪采样，请在上层调用处使用：
# from ..core.sampler import GreedyMaxDistanceSampler
# selected_indices, _, _ = GreedyMaxDistanceSampler.select_frames(points, k)


# --- 核心分析逻辑 ---
# 说明：RMSD/RMSF/Kabsch/迭代均值结构函数已迁移至 core.system_analyser.RMSDCalculator
# 保留兼容导入（文件顶部的 try 导入），避免重复实现。
