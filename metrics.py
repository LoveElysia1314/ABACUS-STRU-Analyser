#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
指标计算模块：metrics.py
功能：计算分子动力学轨迹的各种统计指标
"""

import numpy as np
from scipy.spatial.distance import pdist
from typing import Dict, List
from utils import Constants, MathUtils


class MetricCalculator:
    """指标计算器 - 计算轨迹分析的核心指标"""
    
    @staticmethod
    def calculate_distance_vectors(positions: np.ndarray) -> np.ndarray:
        """
        计算标准化距离向量
        
        Args:
            positions: 原子坐标矩阵 (N_atoms, 3)
            
        Returns:
            标准化的距离向量
        """
        if len(positions) < 2:
            return np.array([])
        
        raw_vectors = pdist(positions)
        norm = np.linalg.norm(raw_vectors)
        return raw_vectors / norm if norm > Constants.EPSILON else raw_vectors
    
    @staticmethod
    def compute_all_metrics(vector_matrix: np.ndarray) -> Dict[str, float]:
        """
        计算所有关键指标
        
        Args:
            vector_matrix: 距离向量矩阵 (N_frames, N_distances)
            
        Returns:
            包含所有指标的字典
        """
        if len(vector_matrix) == 0:
            return {
                'global_mean': 0.0, 
                'nRMSF': 0.0, 
                'MCV': 0.0, 
                'frame_nLdRMS': [], 
                'avg_nLdRMS': 0.0
            }
        
        global_mean = np.mean(vector_matrix) if vector_matrix.size > 0 else 0.0
        
        # nRMSF 计算（归一化均方根涨落）
        nRMSF = MetricCalculator._calculate_nRMSF(vector_matrix, global_mean)
        
        # MCV 计算（平均变异系数）
        MCV = MetricCalculator._calculate_MCV(vector_matrix)
        
        # nLdRMS 计算（归一化局域距离RMS）
        frame_nLdRMS = MetricCalculator._calculate_frame_nLdRMS(vector_matrix, global_mean)
        avg_nLdRMS = np.mean(frame_nLdRMS) if len(frame_nLdRMS) > 0 else 0.0
        
        return {
            'global_mean': global_mean,
            'nRMSF': nRMSF,
            'MCV': MCV,
            'frame_nLdRMS': frame_nLdRMS,
            'avg_nLdRMS': avg_nLdRMS
        }
    
    @staticmethod
    def _calculate_nRMSF(vector_matrix: np.ndarray, global_mean: float) -> float:
        """计算归一化均方根涨落"""
        variances = np.var(vector_matrix, axis=0)
        nRMSF = MathUtils.safe_sqrt(np.mean(variances)) / (global_mean + Constants.EPSILON)
        return nRMSF
    
    @staticmethod
    def _calculate_MCV(vector_matrix: np.ndarray) -> float:
        """计算平均变异系数"""
        means = np.mean(vector_matrix, axis=0)
        stds = np.std(vector_matrix, axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cvs = np.where(means > Constants.EPSILON, stds / means, 0.0)
        
        return np.mean(cvs)
    
    @staticmethod
    def _calculate_frame_nLdRMS(vector_matrix: np.ndarray, global_mean: float) -> np.ndarray:
        """计算每帧的归一化局域距离RMS"""
        mean_vector = np.mean(vector_matrix, axis=0)
        diff = vector_matrix - mean_vector
        frame_nLdRMS = MathUtils.safe_sqrt(np.mean(diff ** 2, axis=1)) / (global_mean + Constants.EPSILON)
        return frame_nLdRMS


class TrajectoryMetrics:
    """轨迹指标数据类"""
    
    def __init__(self, system_name: str, mol_id: str, conf: str, temperature: str):
        self.system_name = system_name
        self.mol_id = mol_id
        self.conf = conf
        self.temperature = temperature
        
        # 基本信息
        self.num_frames = 0
        self.dimension = 0
        
        # 原始指标
        self.nRMSF = 0.0
        self.MCV = 0.0
        self.avg_nLdRMS = 0.0
        
        # 采样后指标
        self.nRMSF_sampled = 0.0
        self.MCV_sampled = 0.0
        self.avg_nLdRMS_sampled = 0.0
        
        # 采样信息
        self.sampled_frames = []
    
    def set_original_metrics(self, metrics_data: Dict[str, float]):
        """设置原始指标"""
        self.nRMSF = metrics_data['nRMSF']
        self.MCV = metrics_data['MCV']
        self.avg_nLdRMS = metrics_data['avg_nLdRMS']
    
    def set_sampled_metrics(self, metrics_data: Dict[str, float]):
        """设置采样后指标"""
        self.nRMSF_sampled = metrics_data['nRMSF']
        self.MCV_sampled = metrics_data['MCV']
        self.avg_nLdRMS_sampled = metrics_data['avg_nLdRMS']
    
    def get_ratio_metrics(self) -> Dict[str, float]:
        """获取采样前后比率"""
        return {
            'nRMSF_ratio': self.nRMSF_sampled / (self.nRMSF + Constants.EPSILON),
            'MCV_ratio': self.MCV_sampled / (self.MCV + Constants.EPSILON),
            'avg_nLdRMS_ratio': self.avg_nLdRMS_sampled / (self.avg_nLdRMS + Constants.EPSILON)
        }
