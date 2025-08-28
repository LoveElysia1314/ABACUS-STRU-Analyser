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
2. MinD: 最小间距（最小距离）
3. ANND: 平均最近邻距离（平均最近邻距离）
4. MPD: 平均成对距离（平均成对距离）
5. PCA explained variance ratios: 主成分方差贡献率

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

# 导入解析器模块

# 注意：日志配置现在由主程序管理，模块只获取logger
logger = logging.getLogger(__name__)


# --- 核心数据结构 ---
class FrameMetrics:
    """单帧的标准化分析指标（更新为新指标）"""

    def __init__(self, frame_id, MinD, ANND, MPD):
        self.frame_id = frame_id
        self.MinD = MinD  # 最小间距
        self.ANND = ANND  # 平均最近邻距离
        self.MPD = MPD  # 平均成对距离


class DistributionMetrics:
    """整体分布的标准化分析指标（更新版）"""

    def __init__(self, ConfVol, nAPD, MinD, ANND, MPD, num_frames, dimension, pca_explained_variance_ratio=None):
        self.ConfVol = ConfVol  # 标准化构象空间体积
        self.nAPD = nAPD  # 标准化平均成对距离
        self.MinD = MinD  # 最小间距
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
        MinD,
        ANND,
        avg_MinD,
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
        self.MinD = MinD
        self.ANND = ANND
        self.avg_MinD = avg_MinD
        self.num_frames = num_frames
        self.dimension = dimension
        self.pca_explained_variance_ratio = pca_explained_variance_ratio or []


# --- 采样函数 ---
def greedy_max_avg_distance(points, k, frame_MinD_values, num_runs=10, seed=42):
    n, d = points.shape
    if k >= n:
        return points, np.arange(n)

    best_sum = -np.inf
    best_indices = None

    for run in range(num_runs):
        np.random.seed(seed + run)
        MinD_array = np.array(frame_MinD_values)
        min_idx = np.argmin(MinD_array)
        max_idx = np.argmax(MinD_array)
        if min_idx == max_idx:
            idxs = np.random.choice(n, 2, replace=False).tolist()
        else:
            idxs = [min_idx, max_idx]

        selected = set(idxs)
        dists_to_S = np.zeros(n)
        for i in range(n):
            if i not in selected:
                dists_to_S[i] = np.sum(cdist(points[list(selected)], points[i : i + 1]))

        for _ in range(2, k):
            candidates = [i for i in range(n) if i not in selected]
            if not candidates:
                break
            next_idx = candidates[np.argmax(dists_to_S[candidates])]
            selected.add(next_idx)

            new_dists = cdist(points[next_idx : next_idx + 1], points).flatten()
            for i in range(n):
                if i not in selected:
                    dists_to_S[i] += new_dists[i]
                else:
                    dists_to_S[i] = -np.inf

        selected_list = list(selected)
        if len(selected_list) >= 2:
            subset_points = points[selected_list]
            pairwise_dists = cdist(subset_points, subset_points)
            dist_sum = np.sum(pairwise_dists) / 2

            if dist_sum > best_sum:
                best_sum = dist_sum
                best_indices = np.array(selected_list)

    if best_indices is None:
        best_indices = np.arange(min(k, n))

    return points[best_indices], best_indices


# --- 核心分析逻辑 ---
def estimate_mean_distance(vectors):
    N = len(vectors)
    if N <= 1:
        return 0.0
    try:
        pairwise_distances = pdist(vectors, metric="euclidean")
        return float(np.mean(pairwise_distances))
    except Exception as e:
        logger.error(f"pdist 计算失败: {e}")
        return 0.0


def calculate_MinD(distance_vectors):
    """计算最小间距 (Minimum Distance)"""
    if len(distance_vectors) <= 1:
        return 0.0
    try:
        pairwise_distances = pdist(distance_vectors, metric="euclidean")
        return float(np.min(pairwise_distances)) if len(pairwise_distances) > 0 else 0.0
    except Exception as e:
        logger.error(f"MinD 计算失败: {e}")
        return 0.0


def calculate_ANND(distance_vectors):
    """计算平均最近邻距离 (Average Nearest Neighbor Distance)"""
    if len(distance_vectors) <= 1:
        return 0.0
    try:
        # 计算所有点对距离矩阵
        distance_matrix = cdist(distance_vectors, distance_vectors, metric="euclidean")
        # 将对角线（自身距离）设为无穷大
        np.fill_diagonal(distance_matrix, np.inf)
        # 找到每行的最小距离（最近邻距离）
        nearest_neighbor_distances = np.min(distance_matrix, axis=1)
        # 返回平均值
        return float(np.mean(nearest_neighbor_distances))
    except Exception as e:
        logger.error(f"ANND 计算失败: {e}")
        return 0.0


def calculate_dRMSF(distance_vectors):
    if len(distance_vectors) <= 1:
        return 0.0
    variances = np.var(distance_vectors, axis=0)
    mean_variance = np.mean(variances)
    return float(np.sqrt(mean_variance))


def calculate_MeanCV(distance_vectors):
    if len(distance_vectors) <= 1:
        return 0.0
    np.mean(distance_vectors, axis=0)
