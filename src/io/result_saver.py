#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
from typing import List
import numpy as np

from ..core.metrics import TrajectoryMetrics
from ..io.stru_parser import FrameData
from ..utils import FileUtils, Constants


class ResultSaver:
	@staticmethod
	def save_frame_metrics(output_dir: str, system_name: str, 
						  frames: List[FrameData], 
						  frame_nLdRMS_values: np.ndarray, 
						  sampled_frames: List[int]) -> None:
		single_analysis_dir = os.path.join(output_dir, "single_analysis_results")
		FileUtils.ensure_dir(single_analysis_dir)
		csv_path = os.path.join(single_analysis_dir, f"frame_metrics_{system_name}.csv")
		with open(csv_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['Frame_ID', 'nLdRMS', 'Selected'])
			sampled_set = set(sampled_frames)
			for i, frame in enumerate(frames):
				selected = 1 if frame.frame_id in sampled_set else 0
				nLdRMS = frame_nLdRMS_values[i] if i < len(frame_nLdRMS_values) else 0.0
				writer.writerow([frame.frame_id, f"{nLdRMS:.6f}", selected])

	@staticmethod
	def save_system_summary(output_dir: str, all_metrics: List[TrajectoryMetrics]) -> None:
		combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
		FileUtils.ensure_dir(combined_analysis_dir)
		csv_path = os.path.join(combined_analysis_dir, "system_metrics_summary.csv")
		
		# 按分子序号、构象、温度排序
		sorted_metrics = sorted(all_metrics, key=lambda m: (int(m.mol_id), int(m.conf), int(m.temperature)))
		
		with open(csv_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow([
				'System', 'Molecule_ID', 'Configuration', 'Temperature(K)',
				'Num_Frames', 'Dimension', 
				'nRMSF', 'MCV', 'avg_nLdRMS',
				'nRMSF_sampled', 'MCV_sampled', 'avg_nLdRMS_sampled',
				'nRMSF_ratio', 'MCV_ratio', 'avg_nLdRMS_ratio'
			])
			for m in sorted_metrics:
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
	def save_system_summary_incremental(output_dir: str, new_metrics: List[TrajectoryMetrics]) -> None:
		"""增量模式下保存系统汇总，合并新旧结果"""
		combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
		FileUtils.ensure_dir(combined_analysis_dir)
		csv_path = os.path.join(combined_analysis_dir, "system_metrics_summary.csv")
		
		# 读取已有数据
		existing_systems = set()
		existing_rows = []
		headers = [
			'System', 'Molecule_ID', 'Configuration', 'Temperature(K)',
			'Num_Frames', 'Dimension', 
			'nRMSF', 'MCV', 'avg_nLdRMS',
			'nRMSF_sampled', 'MCV_sampled', 'avg_nLdRMS_sampled',
			'nRMSF_ratio', 'MCV_ratio', 'avg_nLdRMS_ratio'
		]
		
		if os.path.exists(csv_path):
			try:
				with open(csv_path, 'r', encoding='utf-8') as f:
					reader = csv.DictReader(f)
					for row in reader:
						existing_systems.add(row['System'])
						existing_rows.append([row.get(h, '') for h in headers])
			except Exception:
				pass  # 如果读取失败，就重新创建文件
		
		# 合并所有数据并排序
		all_rows = []
		
		# 添加已有数据（跳过已在新数据中的系统）
		new_system_names = {m.system_name for m in new_metrics}
		for row in existing_rows:
			if row[0] not in new_system_names:  # System列
				all_rows.append(row)
		
		# 添加新数据
		for m in new_metrics:
			ratios = m.get_ratio_metrics()
			all_rows.append([
				m.system_name, m.mol_id, m.conf, m.temperature,
				m.num_frames, m.dimension,
				f"{m.nRMSF:.6f}", f"{m.MCV:.6f}", f"{m.avg_nLdRMS:.6f}",
				f"{m.nRMSF_sampled:.6f}", f"{m.MCV_sampled:.6f}", f"{m.avg_nLdRMS_sampled:.6f}",
				f"{ratios['nRMSF_ratio']:.6f}",
				f"{ratios['MCV_ratio']:.6f}",
				f"{ratios['avg_nLdRMS_ratio']:.6f}"
			])
		
		# 按分子序号、构象、温度排序
		def sort_key(row):
			try:
				# 从System列解析分子ID、构象、温度
				system_name = row[0]
				import re
				match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', system_name)
				if match:
					mol_id, conf, temp = match.groups()
					return (int(mol_id), int(conf), int(temp))
			except:
				pass
			return (9999, 9999, 9999)  # 排序到最后
		
		all_rows.sort(key=sort_key)
		
		# 写入排序后的数据
		with open(csv_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(headers)
			writer.writerows(all_rows)

	@staticmethod
	def save_sampling_records(output_dir: str, all_metrics: List[TrajectoryMetrics]) -> None:
		combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
		FileUtils.ensure_dir(combined_analysis_dir)
		csv_path = os.path.join(combined_analysis_dir, "sampling_records.csv")
		
		# 按分子序号、构象、温度排序
		sorted_metrics = sorted(all_metrics, key=lambda m: (int(m.mol_id), int(m.conf), int(m.temperature)))
		
		with open(csv_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['System', 'OUT_ABACUS_Path', 'Sampled_Frames'])
			for m in sorted_metrics:
				writer.writerow([
					m.system_name,
					m.out_abacus_path,
					';'.join(map(str, sorted(m.sampled_frames)))
				])

	@staticmethod
	def save_sampling_records_incremental(output_dir: str, new_metrics: List[TrajectoryMetrics]) -> None:
		"""增量模式下保存采样记录，合并新旧结果"""
		combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
		FileUtils.ensure_dir(combined_analysis_dir)
		csv_path = os.path.join(combined_analysis_dir, "sampling_records.csv")
		
		# 读取已有数据
		existing_rows = []
		headers = ['System', 'OUT_ABACUS_Path', 'Sampled_Frames']
		
		if os.path.exists(csv_path):
			try:
				with open(csv_path, 'r', encoding='utf-8') as f:
					reader = csv.DictReader(f)
					# 检查是否有OUT_ABACUS_Path列（兼容旧格式）
					if 'OUT_ABACUS_Path' not in reader.fieldnames:
						headers = ['System', 'Sampled_Frames']  # 使用旧格式
					for row in reader:
						existing_rows.append([row.get(h, '') for h in headers])
			except Exception:
				pass
		
		# 合并所有数据并排序
		all_rows = []
		
		# 添加已有数据（跳过已在新数据中的系统）
		new_system_names = {m.system_name for m in new_metrics}
		for row in existing_rows:
			if row[0] not in new_system_names:  # System列
				# 如果是旧格式，补充OUT_ABACUS_Path列
				if len(row) == 2 and len(headers) == 3:
					row = [row[0], '', row[1]]
				all_rows.append(row)
		
		# 添加新数据
		for m in new_metrics:
			all_rows.append([
				m.system_name,
				m.out_abacus_path,
				';'.join(map(str, sorted(m.sampled_frames)))
			])
		
		# 按分子序号、构象、温度排序
		def sort_key(row):
			try:
				system_name = row[0]
				import re
				match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', system_name)
				if match:
					mol_id, conf, temp = match.groups()
					return (int(mol_id), int(conf), int(temp))
			except:
				pass
			return (9999, 9999, 9999)
		
		all_rows.sort(key=sort_key)
		
		# 写入排序后的数据（使用新格式）
		with open(csv_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['System', 'OUT_ABACUS_Path', 'Sampled_Frames'])
			writer.writerows(all_rows)

	@staticmethod
	def save_analysis_statistics(output_dir: str, 
								analysis_stats: dict,
								sampling_stats: dict) -> None:
		combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
		FileUtils.ensure_dir(combined_analysis_dir)
		stats_path = os.path.join(combined_analysis_dir, "analysis_statistics.csv")
		with open(stats_path, 'w', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(['Metric', 'Value'])
			writer.writerow(['Total_Systems', analysis_stats.get('total_systems', 0)])
			writer.writerow(['Successful_Systems', analysis_stats.get('successful_systems', 0)])
			writer.writerow(['Failed_Systems', analysis_stats.get('failed_systems', 0)])
			writer.writerow(['Total_Frames', analysis_stats.get('total_frames', 0)])
			writer.writerow(['Average_Frames_Per_System', analysis_stats.get('avg_frames_per_system', 0)])
			if sampling_stats:
				writer.writerow(['Average_Swap_Count', sampling_stats.get('avg_swap_count', 0)])
				writer.writerow(['Average_Improvement_Ratio', sampling_stats.get('avg_improvement_ratio', 0)])
				writer.writerow(['Total_Swap_Count', sampling_stats.get('total_swap_count', 0)])

	@staticmethod
	def save_all_results(output_dir: str, analysis_results: List[tuple]) -> None:
		all_metrics = []
		swap_counts = []
		improve_ratios = []
		total_frames = 0
		for result in analysis_results:
			metrics, frames, frame_nLdRMS, swap_count, improve_ratio = result
			all_metrics.append(metrics)
			swap_counts.append(swap_count)
			improve_ratios.append(improve_ratio)
			total_frames += len(frames)
			ResultSaver.save_frame_metrics(
				output_dir, metrics.system_name, 
				frames, frame_nLdRMS, metrics.sampled_frames
			)
		if all_metrics:
			ResultSaver.save_system_summary(output_dir, all_metrics)
			ResultSaver.save_sampling_records(output_dir, all_metrics)
			analysis_stats = {
				'total_systems': len(all_metrics),
				'successful_systems': len(all_metrics),
				'failed_systems': 0,
				'total_frames': total_frames,
				'avg_frames_per_system': total_frames / len(all_metrics) if all_metrics else 0
			}
			sampling_stats = {
				'avg_swap_count': np.mean(swap_counts) if swap_counts else 0,
				'avg_improvement_ratio': np.mean(improve_ratios) if improve_ratios else 0,
				'total_swap_count': sum(swap_counts) if swap_counts else 0
			}
			ResultSaver.save_analysis_statistics(output_dir, analysis_stats, sampling_stats)

	@staticmethod
	def save_all_results_incremental(output_dir: str, analysis_results: List[tuple]) -> None:
		"""增量模式下保存所有分析结果，与已有结果合并"""
		all_metrics = []
		swap_counts = []
		improve_ratios = []
		total_frames = 0
		
		for result in analysis_results:
			metrics, frames, frame_nLdRMS, swap_count, improve_ratio = result
			all_metrics.append(metrics)
			swap_counts.append(swap_count)
			improve_ratios.append(improve_ratio)
			total_frames += len(frames)
			
			# 保存单个系统的详细结果
			ResultSaver.save_frame_metrics(
				output_dir, metrics.system_name, 
				frames, frame_nLdRMS, metrics.sampled_frames
			)
		
		if all_metrics:
			# 使用增量保存方法
			ResultSaver.save_system_summary_incremental(output_dir, all_metrics)
			ResultSaver.save_sampling_records_incremental(output_dir, all_metrics)
			
			# 统计信息需要重新计算（包含所有系统）
			analysis_stats = {
				'total_systems': len(all_metrics),
				'successful_systems': len(all_metrics),
				'failed_systems': 0,
				'total_frames': total_frames,
				'avg_frames_per_system': total_frames / len(all_metrics) if all_metrics else 0
			}
			sampling_stats = {
				'avg_swap_count': np.mean(swap_counts) if swap_counts else 0,
				'avg_improvement_ratio': np.mean(improve_ratios) if improve_ratios else 0,
				'total_swap_count': sum(swap_counts) if swap_counts else 0
			}
			ResultSaver.save_analysis_statistics(output_dir, analysis_stats, sampling_stats)

	@staticmethod
	def save_summary_only_incremental(output_dir: str, analysis_results: List[tuple]) -> None:
		"""增量模式下仅保存汇总结果，跳过单个系统的详细frame_metrics文件"""
		all_metrics = []
		swap_counts = []
		improve_ratios = []
		total_frames = 0
		
		for result in analysis_results:
			metrics, frames, frame_nLdRMS, swap_count, improve_ratio = result
			all_metrics.append(metrics)
			swap_counts.append(swap_count)
			improve_ratios.append(improve_ratio)
			total_frames += len(frames)
			# 跳过 save_frame_metrics 调用
		
		if all_metrics:
			# 使用增量保存方法
			ResultSaver.save_system_summary_incremental(output_dir, all_metrics)
			ResultSaver.save_sampling_records_incremental(output_dir, all_metrics)
			
			analysis_stats = {
				'total_systems': len(all_metrics),
				'successful_systems': len(all_metrics),
				'failed_systems': 0,
				'total_frames': total_frames,
				'avg_frames_per_system': total_frames / len(all_metrics) if all_metrics else 0
			}
			sampling_stats = {
				'avg_swap_count': np.mean(swap_counts) if swap_counts else 0,
				'avg_improvement_ratio': np.mean(improve_ratios) if improve_ratios else 0,
				'total_swap_count': sum(swap_counts) if swap_counts else 0
			}
			ResultSaver.save_analysis_statistics(output_dir, analysis_stats, sampling_stats)

	@staticmethod
	def save_summary_only(output_dir: str, analysis_results: List[tuple]) -> None:
		"""仅保存汇总结果，跳过单个系统的详细frame_metrics文件"""
		all_metrics = []
		swap_counts = []
		improve_ratios = []
		total_frames = 0
		
		for result in analysis_results:
			metrics, frames, frame_nLdRMS, swap_count, improve_ratio = result
			all_metrics.append(metrics)
			swap_counts.append(swap_count)
			improve_ratios.append(improve_ratio)
			total_frames += len(frames)
			# 跳过 save_frame_metrics 调用
		
		if all_metrics:
			ResultSaver.save_system_summary(output_dir, all_metrics)
			ResultSaver.save_sampling_records(output_dir, all_metrics)
			analysis_stats = {
				'total_systems': len(all_metrics),
				'successful_systems': len(all_metrics),
				'failed_systems': 0,
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
	@staticmethod
	def save_combined_summary(output_root_dir: str, all_metrics) -> None:
		if all_metrics and hasattr(all_metrics[0], 'system_name'):
			ResultSaver.save_system_summary(output_root_dir, all_metrics)
		else:
			converted_metrics = []
			for m in all_metrics:
				if hasattr(m, 'system_name'):
					converted_metrics.append(m)
			if converted_metrics:
				ResultSaver.save_system_summary(output_root_dir, converted_metrics)

	@staticmethod
	def save_sampled_frames(output_root_dir: str, all_metrics) -> None:
		if all_metrics and hasattr(all_metrics[0], 'system_name'):
			ResultSaver.save_sampling_records(output_root_dir, all_metrics)

	@staticmethod
	def save_frame_metrics(output_root_dir: str, system_name: str, 
						  frames, frame_nLdRMS_values, sampled_frames) -> None:
		ResultSaver.save_frame_metrics(
			output_root_dir, system_name, 
			frames, frame_nLdRMS_values, sampled_frames
		)
