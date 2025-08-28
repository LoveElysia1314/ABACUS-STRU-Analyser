#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本名: trajectory_analyzer.py
功能: ABACUS 轨迹分析器
==================================================

功能：
-----
批量分析 ABACUS 分子动力学模拟生成的 STRU 文件序列，
基于原子间距离向量的标准化分布特性及推荐指标，
评估构象多样性。

核心指标（精简版）：
------------------
1. ConfVol: 构象空间体积（核心多样性指标）
2. nRMSF: 标准化距离均方根波动（整体波动强度）
3. MCV: 平均变异系数（标准化相对波动）
4. nLdRMS: 每帧到平均结构的距离RMS（单帧指标）

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
python trajectory_analyzer.py [--include_h] [--max_workers N] [--sample_ratio R] [--sample_count N]

依赖：
------
numpy, scipy, stru_parser.py
"""

import os
import glob
import numpy as np
import re
import argparse
from scipy.spatial.distance import pdist, cdist
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import csv
import json
import logging

# 导入解析器模块
from abacus_analyzer.io import stru_parser

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 核心数据结构 ---
class FrameMetrics:
    """单帧的标准化分析指标（仅保留必要指标）"""
    def __init__(self, frame_id, norm_local_dRMS):
        self.frame_id = frame_id
        self.norm_local_dRMS = norm_local_dRMS  # 标准化局部距离RMS

class DistributionMetrics:
    """整体分布的标准化分析指标（精简版）"""
    def __init__(self, ConfVol, nAPD, nRMSF, MCV, num_frames, dimension):
        self.ConfVol = ConfVol    # 标准化构象空间体积
        self.nAPD = nAPD          # 标准化平均成对距离
        self.nRMSF = nRMSF        # 标准化距离均方根波动
        self.MCV = MCV            # 平均变异系数
        self.num_frames = num_frames
        self.dimension = dimension  # 距离向量维度

class AnalysisResult:
    """单次分析的结果封装"""
    def __init__(self, frame_metrics_list, distribution_metrics, frame_numbers, molecular_formula):
        self.frame_metrics_list = frame_metrics_list
        self.distribution_metrics = distribution_metrics
        self.frame_numbers = frame_numbers
        self.molecular_formula = molecular_formula

class SummaryInfo:
    """用于生成汇总表的信息"""
    def __init__(self, type_, dir_name, mol_id, conf, T,
                 ConfVol, nAPD, nRMSF, MCV,
                 avg_nLdRMS,
                 num_frames, dimension):
        self.type = type_
        self.dir_name = dir_name
        self.mol_id = mol_id
        self.conf = conf
        self.T = T
        self.ConfVol = ConfVol
        self.nAPD = nAPD
        self.nRMSF = nRMSF
        self.MCV = MCV
        self.avg_nLdRMS = avg_nLdRMS
        self.num_frames = num_frames
        self.dimension = dimension

# --- 采样函数 ---
def greedy_max_avg_distance(points, k, frame_nLdRMS_values, num_runs=10, seed=42):
    n, d = points.shape
    if k >= n:
        return points, np.arange(n)

    best_sum = -np.inf
    best_indices = None

    for run in range(num_runs):
        np.random.seed(seed + run)
        nLdRMS_array = np.array(frame_nLdRMS_values)
        min_idx = np.argmin(nLdRMS_array)
        max_idx = np.argmax(nLdRMS_array)
        if min_idx == max_idx:
            idxs = np.random.choice(n, 2, replace=False).tolist()
        else:
            idxs = [min_idx, max_idx]
        
        selected = set(idxs)
        dists_to_S = np.zeros(n)
        for i in range(n):
            if i not in selected:
                dists_to_S[i] = np.sum(cdist(points[list(selected)], points[i:i+1]))

        for _ in range(2, k):
            candidates = [i for i in range(n) if i not in selected]
            if not candidates:
                break
            next_idx = candidates[np.argmax(dists_to_S[candidates])]
            selected.add(next_idx)

            new_dists = cdist(points[next_idx:next_idx+1], points).flatten()
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
        pairwise_distances = pdist(vectors, metric='euclidean')
        return float(np.mean(pairwise_distances))
    except Exception as e:
        logger.error(f"pdist 计算失败: {e}")
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
    means = np.mean(distance_vectors, axis=0)
