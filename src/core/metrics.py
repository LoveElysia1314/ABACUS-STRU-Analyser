#!/usr/bin/env python

import os
from typing import Dict

import numpy as np
from scipy.spatial.distance import pdist

from ..utils import Constants, MathUtils, ValidationUtils


class MetricCalculator:
    @staticmethod
    def calculate_distance_vectors(positions: np.ndarray) -> np.ndarray:
        # positions is expected to be a 2D numpy array; be robust if it's None or too short
        if (
            ValidationUtils.is_empty(positions)
            or getattr(positions, "shape", [0])[0] < 2
        ):
            return np.array([])
        raw_vectors = pdist(positions)
        norm = np.linalg.norm(raw_vectors)
        return raw_vectors / norm if norm > Constants.EPSILON else raw_vectors

    @staticmethod
    def compute_all_metrics(vector_matrix: np.ndarray) -> Dict[str, object]:
        # Accept either lists or numpy arrays: use .size when available for correctness
        if ValidationUtils.is_empty(vector_matrix):
            return {
                "global_mean": 0.0,
                "MinD": 0.0,
                "ANND": 0.0,
            }

        # At this point vector_matrix is non-empty. Compute means directly.
        global_mean = np.mean(vector_matrix)

        # 计算最小间距 (MinD)
        MinD = MetricCalculator._calculate_MinD(vector_matrix)

        # 计算平均最近邻距离 (ANND)
        ANND = MetricCalculator._calculate_ANND(vector_matrix)

        # 计算平均成对距离 (MPD)
        MPD = MetricCalculator.estimate_mean_distance(vector_matrix)

        return {
            "global_mean": global_mean,
            "MinD": MinD,
            "ANND": ANND,
            "MPD": MPD,
        }

    @staticmethod
    def _calculate_MinD(vector_matrix: np.ndarray) -> float:
        """计算最小间距 (Minimum Distance)"""
        if ValidationUtils.is_empty(vector_matrix) or len(vector_matrix) <= 1:
            return 0.0
        try:
            pairwise_distances = pdist(vector_matrix, metric="euclidean")
            return float(np.min(pairwise_distances)) if len(pairwise_distances) > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_ANND(vector_matrix: np.ndarray) -> float:
        """计算平均最近邻距离 (Average Nearest Neighbor Distance)"""
        if ValidationUtils.is_empty(vector_matrix) or len(vector_matrix) <= 1:
            return 0.0
        try:
            from scipy.spatial.distance import cdist
            # 计算所有点对距离矩阵
            distance_matrix = cdist(vector_matrix, vector_matrix, metric="euclidean")
            # 将对角线（自身距离）设为无穷大
            np.fill_diagonal(distance_matrix, np.inf)
            # 找到每行的最小距离（最近邻距离）
            nearest_neighbor_distances = np.min(distance_matrix, axis=1)
            # 返回平均值
            return float(np.mean(nearest_neighbor_distances))
        except Exception:
            return 0.0



    @staticmethod
    def estimate_mean_distance(vectors: np.ndarray) -> float:
        """估计向量矩阵的平均成对距离（从trajectory_analyzer迁移）"""
        if ValidationUtils.is_empty(vectors) or len(vectors) <= 1:
            return 0.0
        try:
            pairwise_distances = pdist(vectors, metric="euclidean")
            return float(np.mean(pairwise_distances))
        except Exception:
            return 0.0

    @staticmethod
    def calculate_dRMSF(distance_vectors: np.ndarray) -> float:
        """计算距离均方根波动（从trajectory_analyzer迁移）"""
        if ValidationUtils.is_empty(distance_vectors) or len(distance_vectors) <= 1:
            return 0.0
        variances = np.var(distance_vectors, axis=0)
        mean_variance = np.mean(variances)
        return float(np.sqrt(mean_variance))

    @staticmethod
    def calculate_MeanCV(distance_vectors: np.ndarray) -> float:
        """计算平均变异系数（从trajectory_analyzer迁移）"""
        if ValidationUtils.is_empty(distance_vectors) or len(distance_vectors) <= 1:
            return 0.0
        means = np.mean(distance_vectors, axis=0)
        stds = np.std(distance_vectors, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cvs = np.where(means > Constants.EPSILON, stds / means, 0.0)
        return float(np.mean(cvs))


class TrajectoryMetrics:
    def __init__(
        self,
        system_name: str,
        mol_id: str,
        conf: str,
        temperature: str,
        system_path: str = "",
    ):
        self.system_name = system_name
        self.mol_id = mol_id
        self.conf = conf
        self.temperature = temperature
        self.system_path = system_path  # 添加系统路径
        self.num_frames = 0
        self.dimension = 0
        self.MinD = 0.0
        self.ANND = 0.0
        self.MPD = 0.0
        self.MinD_sampled = 0.0
        self.ANND_sampled = 0.0
        self.MPD_sampled = 0.0
        self.sampled_frames = []
        # PCA相关字段
        self.pca_variance_ratio = 0.0  # PCA目标方差贡献率
        self.pca_explained_variance_ratio = []  # 各主成分方差贡献率
        self.pca_cumulative_variance_ratio = 0.0  # 累积方差贡献率

    @property
    def out_abacus_path(self) -> str:
        """返回OUT.ABACUS文件夹的路径"""
        if self.system_path:
            return os.path.join(self.system_path, "OUT.ABACUS")
        return ""

    def set_original_metrics(self, metrics_data: Dict[str, float]):
        self.MinD = metrics_data["MinD"]
        self.ANND = metrics_data["ANND"]
        self.MPD = metrics_data["MPD"]

    def set_sampled_metrics(self, metrics_data: Dict[str, float]):
        self.MinD_sampled = metrics_data["MinD"]
        self.ANND_sampled = metrics_data["ANND"]
        self.MPD_sampled = metrics_data["MPD"]

    def get_ratio_metrics(self) -> Dict[str, float]:
        ratios = {}
        if self.MinD > Constants.EPSILON:
            ratios["MinD_ratio"] = self.MinD_sampled / self.MinD
        else:
            ratios["MinD_ratio"] = 1.0

        if self.ANND > Constants.EPSILON:
            ratios["ANND_ratio"] = self.ANND_sampled / self.ANND
        else:
            ratios["ANND_ratio"] = 1.0

        if self.MPD > Constants.EPSILON:
            ratios["MPD_ratio"] = self.MPD_sampled / self.MPD
        else:
            ratios["MPD_ratio"] = 1.0

        return ratios
