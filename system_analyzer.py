#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
系统分析器模块：system_analyzer.py
功能：分析ABACUS系统轨迹，集成解析、计算和采样功能
"""

import os
import re
import glob
import logging
from typing import Optional, Tuple, List

from stru_parser import StrUParser, FrameData
from metrics import MetricCalculator, TrajectoryMetrics
from sampler import PowerMeanSampler


class SystemAnalyzer:
    """系统分析器 - 负责分析单个ABACUS系统"""
    
    def __init__(self, include_hydrogen: bool = False, 
                 sample_ratio: float = 0.05, 
                 power_p: float = 0.5):
        """
        初始化系统分析器
        
        Args:
            include_hydrogen: 是否包含氢原子
            sample_ratio: 采样比例
            power_p: 幂平均参数
        """
        self.include_hydrogen = include_hydrogen
        self.sample_ratio = sample_ratio
        self.power_p = power_p
        self.parser = StrUParser(exclude_hydrogen=not include_hydrogen)
        self.logger = logging.getLogger(__name__)
    
    def analyze_system(self, system_dir: str) -> Optional[Tuple]:
        """
        分析单个系统目录
        
        Args:
            system_dir: 系统目录路径
            
        Returns:
            (metrics, frames, frame_nLdRMS, swap_count, improve_ratio) 或 None
        """
        # 提取系统信息
        system_info = self._extract_system_info(system_dir)
        if not system_info:
            return None
        
        system_name, mol_id, conf, temperature = system_info
        
        # 查找STRU文件
        stru_dir = os.path.join(system_dir, 'OUT.ABACUS', 'STRU')
        if not os.path.exists(stru_dir):
            self.logger.warning(f"STRU目录不存在: {stru_dir}")
            return None
        
        # 解析轨迹
        frames = self.parser.parse_trajectory(stru_dir)
        if not frames:
            self.logger.warning(f"未找到有效的轨迹数据: {system_dir}")
            return None
        
        # 创建指标对象
        metrics = TrajectoryMetrics(system_name, mol_id, conf, temperature)
        metrics.num_frames = len(frames)
        
        # 计算距离向量矩阵
        distance_vectors = []
        for frame in frames:
            dist_vec = MetricCalculator.calculate_distance_vectors(frame.positions)
            if len(dist_vec) == 0:
                continue
            distance_vectors.append(dist_vec)
            frame.distance_vector = dist_vec
        
        if not distance_vectors:
            self.logger.warning(f"无法计算距离向量: {system_dir}")
            return None
        
        # 统一向量维度
        min_dim = min(len(vec) for vec in distance_vectors)
        vector_matrix = [vec[:min_dim] for vec in distance_vectors]
        vector_matrix = [vec for vec in vector_matrix if len(vec) == min_dim]
        
        if not vector_matrix:
            self.logger.warning(f"距离向量维度不一致: {system_dir}")
            return None
        
        import numpy as np
        vector_matrix = np.array(vector_matrix)
        metrics.dimension = min_dim
        
        # 计算原始指标
        original_metrics = MetricCalculator.compute_all_metrics(vector_matrix)
        metrics.set_original_metrics(original_metrics)
        frame_nLdRMS = original_metrics['frame_nLdRMS']
        
        # 执行采样
        k = max(2, int(round(self.sample_ratio * metrics.num_frames)))
        swap_count, improve_ratio = 0, 0.0
        
        if k < metrics.num_frames:
            # 需要采样
            sampled_indices, swap_count, improve_ratio = PowerMeanSampler.select_frames(
                vector_matrix, k, frame_nLdRMS, p=self.power_p
            )
            metrics.sampled_frames = [frames[i].frame_id for i in sampled_indices]
            
            # 计算采样后指标
            sampled_vectors = vector_matrix[sampled_indices]
            sampled_metrics = MetricCalculator.compute_all_metrics(sampled_vectors)
            metrics.set_sampled_metrics(sampled_metrics)
        else:
            # 全部采样
            metrics.sampled_frames = [f.frame_id for f in frames]
            metrics.set_sampled_metrics(original_metrics)
        
        return metrics, frames, frame_nLdRMS, swap_count, improve_ratio
    
    def _extract_system_info(self, system_dir: str) -> Optional[Tuple[str, str, str, str]]:
        """
        从目录名提取系统信息
        
        Args:
            system_dir: 系统目录路径
            
        Returns:
            (system_name, mol_id, conf, temperature) 或 None
        """
        system_name = os.path.basename(system_dir)
        match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', system_name)
        
        if not match:
            self.logger.warning(f"目录名格式不正确: {system_name}")
            return None
        
        mol_id, conf, temperature = match.groups()
        return system_name, mol_id, conf, temperature


class BatchAnalyzer:
    """批量分析器 - 处理多个系统的分析"""
    
    def __init__(self, analyzer: SystemAnalyzer):
        """
        初始化批量分析器
        
        Args:
            analyzer: 系统分析器实例
        """
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
    
    def analyze_systems(self, system_paths: List[str]) -> List[TrajectoryMetrics]:
        """
        批量分析系统
        
        Args:
            system_paths: 系统路径列表
            
        Returns:
            成功分析的系统指标列表
        """
        successful_metrics = []
        failed_count = 0
        
        for i, system_path in enumerate(system_paths):
            try:
                result = self.analyzer.analyze_system(system_path)
                if result:
                    metrics, _, _, _, _ = result
                    successful_metrics.append(metrics)
                    self.logger.info(f"分析完成 ({i+1}/{len(system_paths)}): {metrics.system_name}")
                else:
                    failed_count += 1
                    self.logger.warning(f"分析失败 ({i+1}/{len(system_paths)}): {system_path}")
            except Exception as e:
                failed_count += 1
                self.logger.error(f"分析出错 ({i+1}/{len(system_paths)}): {system_path} - {str(e)}")
        
        self.logger.info(f"批量分析完成: 成功 {len(successful_metrics)}, 失败 {failed_count}")
        return successful_metrics
    
    def analyze_with_details(self, system_paths: List[str]) -> List[Tuple]:
        """
        批量分析系统并返回详细信息
        
        Args:
            system_paths: 系统路径列表
            
        Returns:
            成功分析的详细结果列表
        """
        successful_results = []
        failed_count = 0
        
        for i, system_path in enumerate(system_paths):
            try:
                result = self.analyzer.analyze_system(system_path)
                if result:
                    successful_results.append(result)
                    metrics = result[0]
                    self.logger.info(f"分析完成 ({i+1}/{len(system_paths)}): {metrics.system_name}")
                else:
                    failed_count += 1
                    self.logger.warning(f"分析失败 ({i+1}/{len(system_paths)}): {system_path}")
            except Exception as e:
                failed_count += 1
                self.logger.error(f"分析出错 ({i+1}/{len(system_paths)}): {system_path} - {str(e)}")
        
        self.logger.info(f"批量分析完成: 成功 {len(successful_results)}, 失败 {failed_count}")
        return successful_results
