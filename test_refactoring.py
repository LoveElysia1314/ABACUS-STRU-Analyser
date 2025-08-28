#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重构后功能验证测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.utils import ValidationUtils
from src.core.metrics import MetricCalculator
from src.core.sampler import SamplingStrategy, PowerMeanSampler, GreedyMaxDistanceSampler

def test_validation_utils():
    """测试统一的空检查函数"""
    print("Testing ValidationUtils.is_empty()...")
    
    # 测试 None
    assert ValidationUtils.is_empty(None) == True
    
    # 测试空列表
    assert ValidationUtils.is_empty([]) == True
    assert ValidationUtils.is_empty([1, 2, 3]) == False
    
    # 测试空 numpy 数组
    assert ValidationUtils.is_empty(np.array([])) == True
    assert ValidationUtils.is_empty(np.array([1, 2, 3])) == False
    
    # 测试二维空数组
    assert ValidationUtils.is_empty(np.array([]).reshape(0, 2)) == True
    assert ValidationUtils.is_empty(np.array([[1, 2], [3, 4]])) == False
    
    print("✓ ValidationUtils.is_empty() tests passed")

def test_metric_calculator():
    """测试 MetricCalculator 的新增函数"""
    print("Testing MetricCalculator extensions...")
    
    # 测试空输入
    empty_result = MetricCalculator.estimate_mean_distance(np.array([]))
    assert empty_result == 0.0
    
    empty_result = MetricCalculator.calculate_dRMSF(np.array([]))
    assert empty_result == 0.0
    
    empty_result = MetricCalculator.calculate_MeanCV(np.array([]))
    assert empty_result == 0.0
    
    # 测试正常输入
    test_vectors = np.random.rand(5, 3)
    
    mean_dist = MetricCalculator.estimate_mean_distance(test_vectors)
    assert isinstance(mean_dist, float) and mean_dist >= 0
    
    dRMSF = MetricCalculator.calculate_dRMSF(test_vectors)
    assert isinstance(dRMSF, float) and dRMSF >= 0
    
    meanCV = MetricCalculator.calculate_MeanCV(test_vectors)
    assert isinstance(meanCV, float) and meanCV >= 0
    
    print("✓ MetricCalculator extension tests passed")

def test_sampling_strategies():
    """测试采样器统一接口"""
    print("Testing sampling strategies...")
    
    # 生成测试数据
    test_points = np.random.rand(10, 5)
    k = 5
    
    # 测试 PowerMeanSampler
    power_sampler = SamplingStrategy.create_sampler(SamplingStrategy.POWER_MEAN)
    selected, swaps, ratio = power_sampler.select_frames(test_points, k)
    assert len(selected) == k
    assert all(0 <= idx < 10 for idx in selected)
    
    # 测试 GreedyMaxDistanceSampler
    greedy_sampler = SamplingStrategy.create_sampler(SamplingStrategy.GREEDY_MAX_DISTANCE)
    selected, swaps, ratio = greedy_sampler.select_frames(test_points, k)
    assert len(selected) == k
    assert all(0 <= idx < 10 for idx in selected)
    
    # 测试边界条件
    selected, _, _ = power_sampler.select_frames(test_points, 15)  # k > n
    assert len(selected) == 10  # 应该返回所有点
    
    print("✓ Sampling strategy tests passed")

def test_compute_all_metrics_with_is_empty():
    """测试 compute_all_metrics 使用新的 is_empty 检查"""
    print("Testing compute_all_metrics with is_empty...")
    
    # 测试空矩阵
    empty_metrics = MetricCalculator.compute_all_metrics(np.array([]))
    expected_keys = {'global_mean', 'MinD', 'ANND', 'MPD'}
    assert set(empty_metrics.keys()) == expected_keys
    assert empty_metrics['global_mean'] == 0.0
    
    # 测试正常矩阵
    test_matrix = np.random.rand(8, 10)
    metrics = MetricCalculator.compute_all_metrics(test_matrix)
    assert set(metrics.keys()) == expected_keys
    assert metrics['global_mean'] > 0
    
    print("✓ compute_all_metrics with is_empty tests passed")

if __name__ == "__main__":
    print("Running refactoring validation tests...\n")
    
    test_validation_utils()
    test_metric_calculator()
    test_sampling_strategies()
    test_compute_all_metrics_with_is_empty()
    
    print("\n🎉 All refactoring tests passed! The refactored code is working correctly.")
