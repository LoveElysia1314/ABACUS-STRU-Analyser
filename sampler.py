#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
采样器模块：sampler.py
功能：实现基于幂平均距离的智能采样算法
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional
from utils import MathUtils


class PowerMeanSampler:
    """基于幂平均距离的智能采样器"""
    
    @staticmethod
    def select_frames(points: np.ndarray, k: int, 
                     nLdRMS_values: Optional[np.ndarray] = None, 
                     p: float = 0.5) -> Tuple[List[int], int, float]:
        """
        使用幂平均距离进行帧采样
        
        Args:
            points: 距离向量矩阵 (N_frames, N_distances)
            k: 采样数量
            nLdRMS_values: 每帧的nLdRMS值，用于初始种子选择
            p: 幂平均参数 (0:几何平均, 1:算术平均, -1:调和平均)
            
        Returns:
            (selected_indices, swap_count, improvement_ratio)
        """
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0
        
        # 初始种子选择
        selected = PowerMeanSampler._initialize_seeds(points, k, nLdRMS_values)
        
        # 增量选择剩余点
        selected = PowerMeanSampler._incremental_selection(points, selected, k, p)
        
        # 交换优化
        selected, swap_count, improve_ratio = PowerMeanSampler._swap_optimization(
            points, selected, p
        )
        
        return selected, swap_count, improve_ratio
    
    @staticmethod
    def _initialize_seeds(points: np.ndarray, k: int, 
                         nLdRMS_values: Optional[np.ndarray] = None) -> List[int]:
        """初始化种子点"""
        n = len(points)
        
        # 第一个种子：nLdRMS最大的点，否则选择第一个点
        if nLdRMS_values is not None and len(nLdRMS_values) == n:
            first_seed = np.argmax(nLdRMS_values)
        else:
            first_seed = 0
        
        selected = [first_seed]
        
        # 第二个种子：距离第一个种子最远的点
        if k > 1 and n > 1:
            dists = cdist(points, points[[first_seed]])
            second_seed = np.argmax(dists)
            if second_seed != first_seed:
                selected.append(second_seed)
        
        return selected
    
    @staticmethod
    def _incremental_selection(points: np.ndarray, selected: List[int], 
                              k: int, p: float) -> List[int]:
        """增量选择剩余点"""
        n = len(points)
        remaining = set(range(n)) - set(selected)
        
        while len(selected) < k and remaining:
            current_points = points[selected]
            candidate_indices = list(remaining)
            candidate_points = points[candidate_indices]
            
            # 计算候选点到已选点集的距离
            dists = cdist(candidate_points, current_points)
            dists = np.maximum(dists, 1e-12)  # 避免零距离
            
            # 对每个候选点计算到已选点集的幂平均距离
            agg_scores = []
            for i in range(len(candidate_indices)):
                candidate_dists = dists[i, :]
                agg_scores.append(MathUtils.power_mean(candidate_dists, p))
            
            # 选择得分最高的候选点
            best_idx = np.argmax(agg_scores)
            selected.append(candidate_indices[best_idx])
            remaining.remove(selected[-1])
        
        return selected
    
    @staticmethod
    def _swap_optimization(points: np.ndarray, selected: List[int], 
                          p: float) -> Tuple[List[int], int, float]:
        """交换优化已选择的点集"""
        if len(selected) < 2:
            return selected, 0, 0.0
        
        n = len(points)
        selected = selected.copy()  # 避免修改原始列表
        
        # 计算初始目标函数值
        selected_points = points[selected]
        sel_dists = cdist(selected_points, selected_points)
        np.fill_diagonal(sel_dists, np.inf)
        triu_idx = np.triu_indices_from(sel_dists, k=1)
        pair_dists = sel_dists[triu_idx]
        initial_obj = MathUtils.power_mean(pair_dists, p)
        
        current_obj = initial_obj
        swap_count = 0
        not_selected = list(set(range(n)) - set(selected))
        improved = True
        
        # 迭代交换优化
        while improved:
            improved = False
            best_improvement = 0.0
            best_swap = None
            
            # 尝试所有可能的交换
            for i_idx, i in enumerate(selected):
                current_point = points[i:i+1]
                other_indices = [s for j, s in enumerate(selected) if j != i_idx]
                other_points = points[other_indices]
                
                if not other_points.size:
                    continue
                
                # 计算当前点的贡献
                dist_i = cdist(current_point, other_points).flatten()
                old_contrib = MathUtils.power_mean(dist_i, p)
                
                # 尝试替换为未选择的点
                for j in not_selected:
                    dist_j = cdist(points[j:j+1], other_points).flatten()
                    new_contrib = MathUtils.power_mean(dist_j, p)
                    improvement = new_contrib - old_contrib
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i_idx, j)
            
            # 执行最佳交换
            if best_swap and best_improvement > 1e-12:
                i_idx, j = best_swap
                old_point = selected[i_idx]
                selected[i_idx] = j
                
                # 更新未选择列表
                not_selected.remove(j)
                not_selected.append(old_point)
                
                current_obj += best_improvement
                swap_count += 1
                improved = True
        
        # 计算改善比率
        improve_ratio = (current_obj - initial_obj) / initial_obj if initial_obj > 0 else 0.0
        
        return selected, swap_count, improve_ratio


class SamplingStrategy:
    """采样策略枚举和工厂"""
    
    POWER_MEAN = "power_mean"
    RANDOM = "random"
    UNIFORM = "uniform"
    
    @staticmethod
    def create_sampler(strategy: str = POWER_MEAN, **kwargs):
        """创建采样器实例"""
        if strategy == SamplingStrategy.POWER_MEAN:
            return PowerMeanSampler()
        elif strategy == SamplingStrategy.RANDOM:
            return RandomSampler()
        elif strategy == SamplingStrategy.UNIFORM:
            return UniformSampler()
        else:
            raise ValueError(f"未知的采样策略: {strategy}")


class RandomSampler:
    """随机采样器（用于对比测试）"""
    
    @staticmethod
    def select_frames(points: np.ndarray, k: int, **kwargs) -> Tuple[List[int], int, float]:
        """随机选择帧"""
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0
        
        selected = np.random.choice(n, k, replace=False).tolist()
        return selected, 0, 0.0


class UniformSampler:
    """均匀采样器（等间隔采样）"""
    
    @staticmethod
    def select_frames(points: np.ndarray, k: int, **kwargs) -> Tuple[List[int], int, float]:
        """均匀间隔选择帧"""
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0
        
        # 计算等间隔的索引
        indices = np.linspace(0, n-1, k, dtype=int)
        selected = list(indices)
        return selected, 0, 0.0
