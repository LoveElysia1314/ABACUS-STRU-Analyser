#!/usr/bin/env python

import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from ..io.stru_parser import StrUParser
from ..utils import ValidationUtils
from .metrics import MetricCalculator, TrajectoryMetrics
from .sampler import PowerMeanSampler
from ..utils.data_utils import ErrorHandler


class PCAReducer:
    """PCA降维处理器类，支持按累计方差贡献率降维"""

    def __init__(self, pca_variance_ratio: float = 0.90):
        self.pca_variance_ratio = pca_variance_ratio

    def apply_pca_reduction(self, vector_matrix: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """应用PCA降维到指定累计方差贡献率"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("scikit-learn is required for PCA functionality. Please install it with: pip install scikit-learn")

        try:
            n_features = vector_matrix.shape[1]
            # n_components为float时，PCA自动选择满足累计方差贡献率的主成分数
            pca = PCA(n_components=self.pca_variance_ratio, random_state=42)
            reduced = pca.fit_transform(vector_matrix)
            return reduced, pca
        except Exception as e:
            logger = logging.getLogger(__name__)
            ErrorHandler.log_detailed_error(
                logger, e, "PCA降维过程中出错",
                additional_info={
                    "输入矩阵形状": vector_matrix.shape if hasattr(vector_matrix, 'shape') else '未知',
                    "PCA累计方差贡献率": self.pca_variance_ratio,
                    "矩阵类型": type(vector_matrix).__name__
                }
            )

    def extract_pca_components(self, reduced_matrix: np.ndarray, frames: List) -> List[dict]:
        """提取每帧的PCA分量数据用于保存"""
        pca_components_data = []
        n_components = reduced_matrix.shape[1] if reduced_matrix.ndim == 2 else 0
        for i, frame in enumerate(frames):
            if i < len(reduced_matrix):
                components = reduced_matrix[i]
                pca_item = {
                    'system': frame.system_name if hasattr(frame, 'system_name') else '',
                    'frame': frame.frame_id
                }
                # 动态生成PC列
                for pc_num in range(1, n_components + 1):
                    pc_key = f'PC{pc_num}'
                    pca_item[pc_key] = components[pc_num - 1] if pc_num <= len(components) else 0.0
                pca_components_data.append(pca_item)
        return pca_components_data

class SystemAnalyzer:
    def __init__(
        self,
        include_hydrogen: bool = False,
        sample_ratio: float = 0.05,
        power_p: float = 0.5,
    pca_variance_ratio: float = 0.90,
    ):
        self.include_hydrogen = include_hydrogen
        self.sample_ratio = sample_ratio
        self.power_p = power_p
        self.pca_variance_ratio = pca_variance_ratio
        self.parser = StrUParser(exclude_hydrogen=not include_hydrogen)
        self.pca_reducer = PCAReducer(pca_variance_ratio)
        self.logger = logging.getLogger(__name__)

    def analyze_system(self, system_dir: str) -> Optional[Tuple]:
        system_info = self._extract_system_info(system_dir)
        if not system_info:
            return None
        system_name, mol_id, conf, temperature = system_info
        stru_dir = os.path.join(system_dir, "OUT.ABACUS", "STRU")
        if not os.path.exists(stru_dir):
            self.logger.warning(f"STRU目录不存在: {stru_dir}")
            return None
        frames = self.parser.parse_trajectory(stru_dir)
        if ValidationUtils.is_empty(frames):
            self.logger.warning(f"未找到有效轨迹数据: {system_dir}")
            return None
        metrics = TrajectoryMetrics(system_name, mol_id, conf, temperature, system_dir)
        metrics.num_frames = len(frames)
        distance_vectors = []
        for frame in frames:
            dist_vec = MetricCalculator.calculate_distance_vectors(frame.positions)
            # treat None or empty numpy arrays as empty
            if ValidationUtils.is_empty(dist_vec):
                continue
            distance_vectors.append(dist_vec)
            frame.distance_vector = dist_vec
        if ValidationUtils.is_empty(distance_vectors):
            self.logger.warning(f"无法计算距离向量: {system_dir}")
            return None
        min_dim = min(len(vec) for vec in distance_vectors)
        # Only keep vectors that have at least min_dim elements, and truncate them to min_dim
        vector_matrix = [
            vec[:min_dim]
            for vec in distance_vectors
            if not ValidationUtils.is_empty(vec) and len(vec) >= min_dim
        ]
        if ValidationUtils.is_empty(vector_matrix):
            self.logger.warning(f"距离向量维度不一致: {system_dir}")
            return None

        vector_matrix = np.array(vector_matrix)
        metrics.dimension = min_dim
        
        # 应用PCA降维
        reduced_matrix, pca_model = self.pca_reducer.apply_pca_reduction(vector_matrix)
        # 设置PCA相关字段
        # 记录本次用于降维的目标累计方差贡献率（按体系保存）
        metrics.pca_variance_ratio = float(self.pca_variance_ratio)
        if pca_model is not None:
            metrics.pca_components = pca_model.n_components_
            metrics.pca_explained_variance_ratio = pca_model.explained_variance_ratio_.tolist()
            metrics.pca_cumulative_variance_ratio = float(np.sum(pca_model.explained_variance_ratio_))
        else:
            metrics.pca_components = 0
            metrics.pca_explained_variance_ratio = []
            metrics.pca_cumulative_variance_ratio = 0.0
        # 在PCA空间中计算指标
        original_metrics = MetricCalculator.compute_all_metrics(reduced_matrix)
        metrics.set_original_metrics(original_metrics)
        # 提取PCA分量用于保存
        pca_components_data = self.pca_reducer.extract_pca_components(reduced_matrix, frames)
        k = max(2, int(round(self.sample_ratio * metrics.num_frames)))
        swap_count, improve_ratio = 0, 0.0
        if k < metrics.num_frames:
            # 使用新的采样策略：基于MinD和ANND优化
            sampled_indices, swap_count, improve_ratio = PowerMeanSampler.select_frames(
                reduced_matrix, k, p=self.power_p
            )
            metrics.sampled_frames = [frames[i].frame_id for i in sampled_indices]
            sampled_vectors = reduced_matrix[sampled_indices]
            sampled_metrics = MetricCalculator.compute_all_metrics(sampled_vectors)
            metrics.set_sampled_metrics(sampled_metrics)
        else:
            metrics.sampled_frames = [f.frame_id for f in frames]
            metrics.set_sampled_metrics(original_metrics)
        return metrics, frames, swap_count, improve_ratio, pca_components_data, pca_model

    def _extract_system_info(
        self, system_dir: str
    ) -> Optional[Tuple[str, str, str, str]]:
        system_name = os.path.basename(system_dir)
        match = re.match(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name)
        if not match:
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"目录名格式不正确: {system_name}")
            return None
        mol_id, conf, temperature = match.groups()
        return system_name, mol_id, conf, temperature


class BatchAnalyzer:
    def __init__(self, analyzer: SystemAnalyzer):
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)

    def analyze_systems(self, system_paths: List[str]) -> List[TrajectoryMetrics]:
        successful_metrics = []
        failed_count = 0
        for i, system_path in enumerate(system_paths):
            try:
                result = self.analyzer.analyze_system(system_path)
                if result:
                    metrics, _, _, _, _ = result
                    successful_metrics.append(metrics)
                    self.logger.info(
                        f"分析完成 ({i+1}/{len(system_paths)}): {metrics.system_name}"
                    )
                else:
                    failed_count += 1
                    self.logger.warning(
                        f"分析失败 ({i+1}/{len(system_paths)}): {system_path}"
                    )
            except Exception as e:
                failed_count += 1
                self.logger.error(
                    f"分析出错 ({i+1}/{len(system_paths)}): {system_path} - {str(e)}"
                )
        self.logger.info(
            f"批量分析完成: 成功 {len(successful_metrics)}, 失败 {failed_count}"
        )
        return successful_metrics

    def analyze_with_details(self, system_paths: List[str]) -> List[Tuple]:
        successful_results = []
        failed_count = 0
        for i, system_path in enumerate(system_paths):
            try:
                result = self.analyzer.analyze_system(system_path)
                if result:
                    successful_results.append(result)
                    metrics = result[0]
                    self.logger.info(
                        f"分析完成 ({i+1}/{len(system_paths)}): {metrics.system_name}"
                    )
                else:
                    failed_count += 1
                    self.logger.warning(
                        f"分析失败 ({i+1}/{len(system_paths)}): {system_path}"
                    )
            except Exception as e:
                failed_count += 1
                self.logger.error(
                    f"分析出错 ({i+1}/{len(system_paths)}): {system_path} - {str(e)}"
                )
        self.logger.info(
            f"批量分析完成: 成功 {len(successful_results)}, 失败 {failed_count}"
        )
        return successful_results
