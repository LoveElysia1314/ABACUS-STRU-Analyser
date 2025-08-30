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


class RMSDCalculator:
    """经典RMSD计算器，实现Kabsch算法和迭代对齐"""

    @staticmethod
    def center_coordinates(coords: np.ndarray) -> np.ndarray:
        """将坐标平移到质心"""
        centroid = np.mean(coords, axis=0)
        return coords - centroid

    @staticmethod
    def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
        """使用Kabsch算法将mobile坐标对齐到target坐标"""
        # 确保输入是numpy数组
        mobile = np.array(mobile, dtype=float)
        target = np.array(target, dtype=float)

        # 中心化坐标
        mobile_centered = RMSDCalculator.center_coordinates(mobile)
        target_centered = RMSDCalculator.center_coordinates(target)

        # 计算协方差矩阵
        covariance = np.dot(mobile_centered.T, target_centered)

        # SVD分解
        V, S, Wt = np.linalg.svd(covariance)

        # 检查是否需要反射
        d = np.sign(np.linalg.det(np.dot(V, Wt)))

        # 构造旋转矩阵
        rotation = np.dot(V, np.dot(np.diag([1, 1, d]), Wt))

        # 应用旋转和平移
        aligned = np.dot(mobile_centered, rotation.T)

        return aligned

    @staticmethod
    def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """计算两个坐标集之间的RMSD"""
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    @staticmethod
    def iterative_alignment(frames: List, max_iterations: int = 10, tolerance: float = 1e-5) -> Tuple[np.ndarray, List[float]]:
        """迭代对齐算法计算均值结构和每帧RMSD"""
        if not frames:
            return np.array([]), []

        # 初始化参考结构（第一帧）
        ref_coords = frames[0].positions.copy()
        all_coords = [frame.positions.copy() for frame in frames]

        # 迭代对齐
        for iteration in range(max_iterations):
            # 对齐所有帧到当前参考
            aligned_coords = []
            for coords in all_coords:
                aligned = RMSDCalculator.kabsch_align(coords, ref_coords)
                aligned_coords.append(aligned)

            # 计算新的均值结构
            new_ref = np.mean(aligned_coords, axis=0)

            # 检查收敛
            if np.allclose(new_ref, ref_coords, atol=tolerance):
                break

            ref_coords = new_ref

        # 计算每帧RMSD（相对于最终均值结构）
        rmsds = []
        for aligned in aligned_coords:
            rmsd = RMSDCalculator.calculate_rmsd(aligned, ref_coords)
            rmsds.append(rmsd)

        return ref_coords, rmsds


class PCAReducer:
    """PCA降维处理器类，支持按累计方差贡献率降维"""

    def __init__(self, pca_variance_ratio: float = 0.90):
        self.pca_variance_ratio = pca_variance_ratio

    def apply_pca_reduction(self, vector_matrix: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """应用PCA降维到指定累计方差贡献率，并对每个维度除以总方差进行标准化"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("scikit-learn is required for PCA functionality. Please install it with: pip install scikit-learn")

        try:
            n_features = vector_matrix.shape[1]
            # 使用相关系数矩阵PCA，不使用白化
            pca = PCA(n_components=self.pca_variance_ratio, whiten=False, random_state=42)
            reduced = pca.fit_transform(vector_matrix)

            # 对每个主成分除以总方差进行标准化
            # 计算所有主成分的总方差
            total_variance = np.sum(pca.explained_variance_)

            if total_variance > 0:
                # 对每个主成分进行标准化
                for i in range(reduced.shape[1]):
                    pc_variance = pca.explained_variance_[i]
                    if pc_variance > 0:
                        # 除以总方差的平方根，保持尺度的一致性
                        scale_factor = np.sqrt(total_variance)
                        reduced[:, i] /= scale_factor

                        # 记录标准化信息
                        logger = logging.getLogger(__name__)
                        logger.debug(f"PC{i+1} 标准化: 方差={pc_variance:.6f}, "
                                   f"缩放因子={scale_factor:.6f}, "
                                   f"标准化后方差={np.var(reduced[:, i]):.6f}")

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

class SystemAnalyser:
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

    def analyse_system(self, system_dir: str) -> Optional[Tuple]:
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

        # 计算经典RMSD（消除平移和旋转影响）
        try:
            mean_structure, rmsd_per_frame = RMSDCalculator.iterative_alignment(frames)
            if len(rmsd_per_frame) > 0:
                metrics.rmsd_mean = float(np.mean(rmsd_per_frame))
                metrics.rmsd_per_frame = [float(r) for r in rmsd_per_frame]
                self.logger.info(f"RMSD计算完成: 均值={metrics.rmsd_mean:.4f}, 帧数={len(rmsd_per_frame)}")
            else:
                metrics.rmsd_mean = 0.0
                metrics.rmsd_per_frame = []
                self.logger.warning("RMSD计算失败: 无有效帧数据")
        except Exception as e:
            self.logger.warning(f"RMSD计算出错: {e}")
            metrics.rmsd_mean = 0.0
            metrics.rmsd_per_frame = []

        # 构建包含能量、力和PCA分量的综合向量
        comprehensive_matrix = self.build_comprehensive_vectors(frames, reduced_matrix)

        # 设置综合向量相关字段
        metrics.comprehensive_dimension = comprehensive_matrix.shape[1] if comprehensive_matrix.size > 0 else 0
        metrics.energy_available = any(frame.energy is not None for frame in frames)

        # 在综合向量空间中计算指标
        original_metrics = MetricCalculator.compute_all_metrics(comprehensive_matrix)
        metrics.set_original_metrics(original_metrics)

        # 提取PCA分量用于保存（保持原有逻辑）
        pca_components_data = self.pca_reducer.extract_pca_components(reduced_matrix, frames)

        k = max(2, int(round(self.sample_ratio * metrics.num_frames)))
        swap_count, improve_ratio = 0, 0.0
        if k < metrics.num_frames:
            # 使用新的采样策略：基于MinD和ANND优化（使用综合向量）
            sampled_indices, swap_count, improve_ratio = PowerMeanSampler.select_frames(
                comprehensive_matrix, k, p=self.power_p
            )
            metrics.sampled_frames = [frames[i].frame_id for i in sampled_indices]
            sampled_vectors = comprehensive_matrix[sampled_indices]
            sampled_metrics = MetricCalculator.compute_all_metrics(sampled_vectors)
            metrics.set_sampled_metrics(sampled_metrics)

            # 计算采样后帧的RMSD均值
            if len(metrics.rmsd_per_frame) > 0 and len(sampled_indices) > 0:
                sampled_rmsd_values = [metrics.rmsd_per_frame[i] for i in sampled_indices if i < len(metrics.rmsd_per_frame)]
                if sampled_rmsd_values:
                    metrics.rmsd_mean_sampled = float(np.mean(sampled_rmsd_values))
                    self.logger.info(f"采样后RMSD计算完成: 均值={metrics.rmsd_mean_sampled:.4f}, 采样帧数={len(sampled_rmsd_values)}")
                else:
                    metrics.rmsd_mean_sampled = 0.0
            else:
                metrics.rmsd_mean_sampled = 0.0
        else:
            metrics.sampled_frames = [f.frame_id for f in frames]
            metrics.set_sampled_metrics(original_metrics)
            # 如果没有采样，采样后RMSD均值等于总体RMSD均值
            metrics.rmsd_mean_sampled = metrics.rmsd_mean
        return metrics, frames, swap_count, improve_ratio, pca_components_data, pca_model, metrics.rmsd_per_frame

    def build_comprehensive_vectors(self, frames: List, reduced_matrix: np.ndarray) -> np.ndarray:
        """构建包含能量和PCA分量的综合向量，并进行标准化和组间缩放"""
        comprehensive_vectors = []

        # 收集所有有效帧的能量数据
        energies = []
        valid_frames = []

        for i, frame in enumerate(frames):
            if i < len(reduced_matrix) and frame.energy is not None:
                energies.append(frame.energy)
                valid_frames.append(i)

        if not valid_frames:
            self.logger.warning("没有找到包含能量数据的帧，使用原始PCA向量")
            return reduced_matrix

        # 对能量进行标准化
        energies = np.array(energies)
        energy_mean = np.mean(energies)
        energy_std = np.std(energies)
        if energy_std > 0:
            energies_standardized = (energies - energy_mean) / energy_std
        else:
            energies_standardized = energies - energy_mean

        # 构建综合向量
        for idx, frame_idx in enumerate(valid_frames):
            frame = frames[frame_idx]
            # 将标准化能量赋值给FrameData对象
            frame.energy_standardized = float(energies_standardized[idx])
            
            pca_components = reduced_matrix[frame_idx]

            # 组合向量：能量、PCA分量
            combined_vector = np.concatenate([
                [energies_standardized[idx]],  # 能量
                pca_components                 # PCA分量
            ])

            # 组间缩放：分别除以维度的平方根
            # 能量维度为1，PCA维度为pca_components.shape[0]
            energy_dim = 1
            pca_dim = len(pca_components)

            combined_vector[0] /= np.sqrt(energy_dim)  # 能量缩放
            for i in range(pca_dim):
                combined_vector[1 + i] /= np.sqrt(pca_dim)  # PCA分量缩放

            comprehensive_vectors.append(combined_vector)

        comprehensive_matrix = np.array(comprehensive_vectors)
        self.logger.info(f"构建了 {len(comprehensive_vectors)} 个综合向量，维度: {comprehensive_matrix.shape[1]}")
        return comprehensive_matrix

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


class BatchAnalyser:
    def __init__(self, analyser: SystemAnalyser):
        self.analyser = analyser
        self.logger = logging.getLogger(__name__)

    def analyse_systems(self, system_paths: List[str]) -> List[TrajectoryMetrics]:
        successful_metrics = []
        failed_count = 0
        for i, system_path in enumerate(system_paths):
            try:
                result = self.analyser.analyse_system(system_path)
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

    def analyse_with_details(self, system_paths: List[str]) -> List[Tuple]:
        successful_results = []
        failed_count = 0
        for i, system_path in enumerate(system_paths):
            try:
                result = self.analyser.analyse_system(system_path)
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
