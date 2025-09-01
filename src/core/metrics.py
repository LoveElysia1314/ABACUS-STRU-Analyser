#!/usr/bin/env python

import os
from typing import Dict

import numpy as np
from scipy.spatial.distance import pdist, cdist

from ..utils import Constants, ValidationUtils
from ..utils.metrics_utils import compute_basic_distance_metrics  # 统一距离指标


class MetricCalculator:
    @staticmethod
    def calculate_distance_vectors(positions: np.ndarray) -> np.ndarray:
        # positions is expected to be a 2D numpy array; be robust if it's None or too short
        if (
            ValidationUtils.is_empty(positions)
            or getattr(positions, "shape", [0])[0] < 2
        ):
            return np.array([])
        # 移除L2归一化步骤，直接返回原始距离向量
        return pdist(positions)

    @staticmethod
    def compute_all_metrics(vector_matrix: np.ndarray) -> Dict[str, object]:
        """统一计算距离相关指标。

        已重构为调用 metrics_utils.compute_basic_distance_metrics 以消除重复。
        保持原返回结构向后兼容。
        """
        if ValidationUtils.is_empty(vector_matrix) or getattr(vector_matrix, "shape", [0])[0] < 2:
            return {"global_mean": 0.0, "ANND": 0.0, "MPD": 0.0}

        global_mean = float(np.mean(vector_matrix))
        basic = compute_basic_distance_metrics(vector_matrix)
        return {
            "global_mean": global_mean,
            "ANND": basic.ANND,
            "MPD": basic.MPD,
        }



    @staticmethod
    def _calculate_ANND(vector_matrix: np.ndarray) -> float:  # deprecated
        """(Deprecated) 保留旧接口，内部委托统一工具。"""
        basic = compute_basic_distance_metrics(vector_matrix)
        return 0.0 if np.isnan(basic.ANND) else float(basic.ANND)



    @staticmethod
    def estimate_mean_distance(vectors: np.ndarray) -> float:  # deprecated
        """(Deprecated) 平均成对距离，委托统一工具 MPD。"""
        basic = compute_basic_distance_metrics(vectors)
        return 0.0 if np.isnan(basic.MPD) else float(basic.MPD)

    @staticmethod
    def calculate_dRMSF(distance_vectors: np.ndarray) -> float:
        """计算距离均方根波动（从trajectory_analyser迁移）"""
        if ValidationUtils.is_empty(distance_vectors) or len(distance_vectors) <= 1:
            return 0.0
        variances = np.var(distance_vectors, axis=0)
        mean_variance = np.mean(variances)
        return float(np.sqrt(mean_variance))

    @staticmethod
    def calculate_MeanCV(distance_vectors: np.ndarray) -> float:
        """计算平均变异系数（从trajectory_analyser迁移）"""
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
        self.ANND = 0.0
        self.MPD = 0.0
        self.sampled_frames = []
        # RMSD相关字段
        self.rmsd_mean = 0.0  # RMSD均值（总体指标）
        self.rmsd_per_frame = []  # 每帧RMSD（单帧指标）
        # PCA相关字段
        self.pca_variance_ratio = 0.0  # PCA目标方差贡献率
        self.pca_explained_variance_ratio = []  # 各主成分方差贡献率
        self.pca_cumulative_variance_ratio = 0.0  # 累积方差贡献率
        self.pca_components = 0  # PCA主成分数量
        # 综合向量相关字段
        self.comprehensive_dimension = 0  # 综合向量维度
        self.energy_available = False  # 是否有能量数据
        # 平均构象坐标（用于后续构象统一）
        self.mean_structure = None  # 平均构象坐标，numpy数组形状为(n_atoms, 3)
        # Level 4: 多样性 / 分布相似性扩展字段（可选输出）
        self.coverage_ratio = None
        self.energy_range = None
        self.js_divergence = None

    def set_diversity_metrics(self, diversity_obj):  # diversity_obj: DiversityMetrics
        try:
            self.coverage_ratio = float(diversity_obj.coverage_ratio)
            self.energy_range = float(diversity_obj.energy_range)
        except Exception:
            pass

    def set_distribution_similarity(self, sim_obj):  # sim_obj: DistributionSimilarity
        try:
            self.js_divergence = float(sim_obj.js_divergence)
        except Exception:
            pass

    @property
    def out_abacus_path(self) -> str:
        """返回OUT.ABACUS文件夹的路径"""
        if self.system_path:
            return os.path.join(self.system_path, "OUT.ABACUS")
        return ""

    def set_original_metrics(self, metrics_data: Dict[str, float]):
        self.ANND = metrics_data["ANND"]
        self.MPD = metrics_data["MPD"]

    def set_sampled_metrics(self, metrics_data: Dict[str, float]):
        """Deprecated placeholder for backward compatibility."""
        return None

    def get_ratio_metrics(self) -> Dict[str, float]:
        """Deprecated: ratio metrics removed."""
        return {}
