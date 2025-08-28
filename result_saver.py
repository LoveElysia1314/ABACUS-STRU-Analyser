#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果保存器模块：result_saver.py
功能：保存分析结果到CSV文件
"""

import os
import csv
from typing import List
import numpy as np

from metrics import TrajectoryMetrics
from stru_parser import FrameData
from utils import FileUtils, Constants


class ResultSaver:
    """结果保存器 - 负责保存各种分析结果"""
    
    @staticmethod
    def save_frame_metrics(output_dir: str, system_name: str, 
                          frames: List[FrameData], 
                          frame_nLdRMS_values: np.ndarray, 
                          sampled_frames: List[int]) -> None:
        """
        保存每帧指标到单体分析结果目录
        
        Args:
            output_dir: 输出根目录
            system_name: 系统名称
            frames: 帧数据列表
            frame_nLdRMS_values: 每帧的nLdRMS值
            sampled_frames: 采样的帧列表
        """
        # 创建单体分析结果目录
        single_analysis_dir = os.path.join(output_dir, "single_analysis_results")
        FileUtils.ensure_dir(single_analysis_dir)
        
        csv_path = os.path.join(single_analysis_dir, f"frame_metrics_{system_name}.csv")
        
        with open(csv_path, 'w', newline='', encoding=Constants.DEFAULT_ENCODING) as f:
            writer = csv.writer(f)
            writer.writerow(['Frame_ID', 'nLdRMS', 'Selected'])
            
            sampled_set = set(sampled_frames)
            for i, frame in enumerate(frames):
                selected = 1 if frame.frame_id in sampled_set else 0
                nLdRMS = frame_nLdRMS_values[i] if i < len(frame_nLdRMS_values) else 0.0
                writer.writerow([frame.frame_id, f"{nLdRMS:.6f}", selected])
    
    @staticmethod
    def save_system_summary(output_dir: str, all_metrics: List[TrajectoryMetrics]) -> None:
        """
        保存系统指标汇总
        
        Args:
            output_dir: 输出根目录
            all_metrics: 所有系统指标列表
        """
        # 创建综合分析结果目录
        combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
        FileUtils.ensure_dir(combined_analysis_dir)
        
        csv_path = os.path.join(combined_analysis_dir, "system_metrics_summary.csv")
        
        with open(csv_path, 'w', newline='', encoding=Constants.DEFAULT_ENCODING) as f:
            writer = csv.writer(f)
            writer.writerow([
                'System', 'Molecule_ID', 'Configuration', 'Temperature(K)',
                'Num_Frames', 'Dimension', 
                'nRMSF', 'MCV', 'avg_nLdRMS',
                'nRMSF_sampled', 'MCV_sampled', 'avg_nLdRMS_sampled',
                'nRMSF_ratio', 'MCV_ratio', 'avg_nLdRMS_ratio'
            ])
            
            for m in all_metrics:
                ratios = m.get_ratio_metrics()
                writer.writerow([
                    m.system_name, m.mol_id, m.conf, m.temperature,
                    m.num_frames, m.dimension,
                    f"{m.nRMSF:.6f}", f"{m.MCV:.6f}", f"{m.avg_nLdRMS:.6f}",
                    f"{m.nRMSF_sampled:.6f}", f"{m.MCV_sampled:.6f}", f"{m.avg_nLdRMS_sampled:.6f}",
                    f"{ratios['nRMSF_ratio']:.6f}",
                    f"{ratios['MCV_ratio']:.6f}",
                    f"{ratios['avg_nLdRMS_ratio']:.6f}"
                ])
    
    @staticmethod
    def save_sampling_records(output_dir: str, all_metrics: List[TrajectoryMetrics]) -> None:
        """
        保存采样记录
        
        Args:
            output_dir: 输出根目录
            all_metrics: 所有系统指标列表
        """
        # 创建综合分析结果目录
        combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
        FileUtils.ensure_dir(combined_analysis_dir)
        
        csv_path = os.path.join(combined_analysis_dir, "sampling_records.csv")
        
        with open(csv_path, 'w', newline='', encoding=Constants.DEFAULT_ENCODING) as f:
            writer = csv.writer(f)
            writer.writerow(['System', 'Sampled_Frames'])
            
            for m in all_metrics:
                writer.writerow([
                    m.system_name,
                    ';'.join(map(str, sorted(m.sampled_frames)))
                ])
    
    @staticmethod
    def save_analysis_statistics(output_dir: str, 
                                analysis_stats: dict,
                                sampling_stats: dict) -> None:
        """
        保存分析统计信息
        
        Args:
            output_dir: 输出根目录
            analysis_stats: 分析统计信息
            sampling_stats: 采样统计信息
        """
        combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
        FileUtils.ensure_dir(combined_analysis_dir)
        
        stats_path = os.path.join(combined_analysis_dir, "analysis_statistics.csv")
        
        with open(stats_path, 'w', newline='', encoding=Constants.DEFAULT_ENCODING) as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            
            # 分析统计
            writer.writerow(['Total_Systems', analysis_stats.get('total_systems', 0)])
            writer.writerow(['Successful_Systems', analysis_stats.get('successful_systems', 0)])
            writer.writerow(['Failed_Systems', analysis_stats.get('failed_systems', 0)])
            writer.writerow(['Total_Frames', analysis_stats.get('total_frames', 0)])
            writer.writerow(['Average_Frames_Per_System', analysis_stats.get('avg_frames_per_system', 0)])
            
            # 采样统计
            if sampling_stats:
                writer.writerow(['Average_Swap_Count', sampling_stats.get('avg_swap_count', 0)])
                writer.writerow(['Average_Improvement_Ratio', sampling_stats.get('avg_improvement_ratio', 0)])
                writer.writerow(['Total_Swap_Count', sampling_stats.get('total_swap_count', 0)])
    
    @staticmethod
    def save_all_results(output_dir: str, analysis_results: List[tuple]) -> None:
        """
        保存所有分析结果
        
        Args:
            output_dir: 输出根目录
            analysis_results: 分析结果列表，每个元素为(metrics, frames, frame_nLdRMS, swap_count, improve_ratio)
        """
        all_metrics = []
        swap_counts = []
        improve_ratios = []
        total_frames = 0
        
        # 处理每个结果
        for result in analysis_results:
            metrics, frames, frame_nLdRMS, swap_count, improve_ratio = result
            
            # 收集指标
            all_metrics.append(metrics)
            swap_counts.append(swap_count)
            improve_ratios.append(improve_ratio)
            total_frames += len(frames)
            
            # 保存每帧指标
            ResultSaver.save_frame_metrics(
                output_dir, metrics.system_name, 
                frames, frame_nLdRMS, metrics.sampled_frames
            )
        
        # 保存汇总结果
        if all_metrics:
            ResultSaver.save_system_summary(output_dir, all_metrics)
            ResultSaver.save_sampling_records(output_dir, all_metrics)
            
            # 计算统计信息
            analysis_stats = {
                'total_systems': len(all_metrics),
                'successful_systems': len(all_metrics),
                'failed_systems': 0,  # 这里只有成功的结果
                'total_frames': total_frames,
                'avg_frames_per_system': total_frames / len(all_metrics) if all_metrics else 0
            }
            
            sampling_stats = {
                'avg_swap_count': np.mean(swap_counts) if swap_counts else 0,
                'avg_improvement_ratio': np.mean(improve_ratios) if improve_ratios else 0,
                'total_swap_count': sum(swap_counts) if swap_counts else 0
            }
            
            ResultSaver.save_analysis_statistics(output_dir, analysis_stats, sampling_stats)


class LegacyResultSaver:
    """传统结果保存器 - 保持与原有代码的兼容性"""
    
    @staticmethod
    def save_combined_summary(output_root_dir: str, all_metrics) -> None:
        """兼容性方法：保存综合指标汇总"""
        # 转换为新格式
        if all_metrics and hasattr(all_metrics[0], 'system_name'):
            # 新格式
            ResultSaver.save_system_summary(output_root_dir, all_metrics)
        else:
            # 旧格式，需要转换
            converted_metrics = []
            for m in all_metrics:
                if hasattr(m, 'system_name'):
                    converted_metrics.append(m)
            if converted_metrics:
                ResultSaver.save_system_summary(output_root_dir, converted_metrics)
    
    @staticmethod
    def save_sampled_frames(output_root_dir: str, all_metrics) -> None:
        """兼容性方法：保存采样帧记录"""
        if all_metrics and hasattr(all_metrics[0], 'system_name'):
            ResultSaver.save_sampling_records(output_root_dir, all_metrics)
    
    @staticmethod
    def save_frame_metrics(output_root_dir: str, system_name: str, 
                          frames, frame_nLdRMS_values, sampled_frames) -> None:
        """兼容性方法：保存每帧指标"""
        ResultSaver.save_frame_metrics(
            output_root_dir, system_name, 
            frames, frame_nLdRMS_values, sampled_frames
        )
