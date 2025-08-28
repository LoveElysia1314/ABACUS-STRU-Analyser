#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AnalysisTarget:
	system_path: str
	mol_id: str
	conf: str
	temperature: str
	stru_files: List[str]
	creation_time: float
	status: str = "pending"
	source_hash: str = ""  # 源文件哈希，用于检测源数据变更

	@property
	def system_name(self) -> str:
		return f"struct_mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"

	@property
	def system_key(self) -> str:
		return f"mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"


class PathManager:
	def __init__(self, output_dir: str = "analysis_results"):
		self.base_output_dir = output_dir
		self.output_dir = output_dir  # 将在 set_output_dir_for_params 中更新
		self.targets_file = None  # 将在 set_output_dir_for_params 中设置
		self.summary_file = None
		self.logger = logging.getLogger(__name__)
		self.targets: List[AnalysisTarget] = []
		self.mol_groups: Dict[str, List[AnalysisTarget]] = {}
		self.current_params_hash: str = ""

	def _calculate_params_hash(self, params: Dict[str, Any]) -> str:
		"""计算分析参数的哈希值，仅包含会影响计算结果的关键参数

		注意：skip_single_results 不应影响计算结果，因此不包含在哈希中
		"""
		try:
			key_params = {
				'include_h': params.get('include_h', False),
				'sample_ratio': params.get('sample_ratio', 0.05),
				'power_p': params.get('power_p', -0.5)
			}
			params_str = json.dumps(key_params, sort_keys=True, separators=(',', ':'))
			return hashlib.sha256(params_str.encode('utf-8')).hexdigest()
		except Exception as e:
			self.logger.debug(f"计算参数哈希失败: {str(e)}")
			return ""

	def _calculate_source_hash(self, system_path: str) -> str:
		"""
		计算源文件目录的哈希值，用于检测源数据完整性。
		优化策略：只对帧数最大的 STRU 文件的元信息计算哈希，减少 IO。
		"""
		try:
			stru_dir = os.path.join(system_path, "OUT.ABACUS", "STRU")
			if not os.path.exists(stru_dir):
				return ""
			max_frame_num = -1
			max_frame_file = None
			for filename in os.listdir(stru_dir):
				if filename.startswith('STRU_MD_') and not filename.endswith('.swp'):
					try:
						frame_num = int(filename.split('_')[-1])
						if frame_num > max_frame_num:
							max_frame_num = frame_num
							max_frame_file = filename
					except (ValueError, IndexError):
						continue
			if max_frame_file is None:
				return ""
			max_file_path = os.path.join(stru_dir, max_frame_file)
			stat = os.stat(max_file_path)
			combined_info = f"{max_frame_file}:{max_frame_num}:{stat.st_size}:{int(stat.st_mtime)}"
			return hashlib.sha256(combined_info.encode('utf-8')).hexdigest()
		except Exception as e:
			self.logger.debug(f"计算源文件哈希失败 {system_path}: {str(e)}")
			return ""

	def set_output_dir_for_params(self, analysis_params: Dict[str, Any]) -> str:
		"""根据分析参数设置输出目录，实现参数隔离"""
		params_hash = self._calculate_params_hash(analysis_params)
		self.current_params_hash = params_hash
		
		# 生成基于参数的目录名（简洁版）
		key_params = {
			'include_h': analysis_params.get('include_h', False),
			'sample_ratio': analysis_params.get('sample_ratio', 0.05),
			'power_p': analysis_params.get('power_p', -0.5)
			# 注意：skip_single_results 不影响计算结果，不包含在目录名中
		}
		
		# 生成简洁的目录名
		parts = []
		for key, value in sorted(key_params.items()):
			if isinstance(value, bool):
				parts.append(f"{key}_{str(value)}")
			elif isinstance(value, float):
				parts.append(f"{key}_{value:g}")  # 去除不必要的小数位
			else:
				parts.append(f"{key}_{value}")
		
		dir_name = "run_" + "_".join(parts)
		self.output_dir = os.path.join(self.base_output_dir, dir_name)
		
		# 更新相关文件路径
		self.targets_file = os.path.join(self.output_dir, "analysis_targets.json")
		self.summary_file = os.path.join(self.output_dir, "path_summary.json")
		
		# 确保目录存在
		os.makedirs(self.output_dir, exist_ok=True)
		
		self.logger.info(f"设置参数专用输出目录: {self.output_dir}")
		return self.output_dir

	def check_existing_complete_results(self) -> bool:
		"""检查当前参数目录是否已有完整的分析结果"""
		if not self.output_dir or not os.path.exists(self.output_dir):
			return False
		
		# 检查关键结果文件
		summary_file = os.path.join(self.output_dir, "combined_analysis_results", "system_metrics_summary.csv")
		targets_file = self.targets_file
		
		if not (os.path.exists(summary_file) and os.path.exists(targets_file)):
			self.logger.debug("关键结果文件不存在，需要重新计算")
			return False
		
		# 检查 analysis_targets.json 中的完成状态
		try:
			with open(targets_file, 'r', encoding='utf-8') as f:
				data = json.load(f)
			
			# 检查是否所有系统都已完成
			total_systems = 0
			completed_systems = 0
			
			for mol_data in data.get("molecules", {}).values():
				for system_data in mol_data.get("systems", {}).values():
					total_systems += 1
					if system_data.get("status") == "completed":
						completed_systems += 1
			
			if total_systems == 0:
				self.logger.debug("目标文件中没有系统记录")
				return False
			
			completion_rate = completed_systems / total_systems
			if completion_rate == 1.0:
				self.logger.info(f"发现完整的分析结果: {completed_systems}/{total_systems} 系统已完成")
				return True
			else:
				self.logger.info(f"发现部分完成的分析: {completed_systems}/{total_systems} 系统已完成")
				return False
				
		except Exception as e:
			self.logger.warning(f"检查现有结果时出错: {str(e)}")
			return False

	def load_from_discovery(self, mol_systems: Dict[str, List[str]], preserve_existing: bool = False) -> None:
		"""从发现结果加载分析目标
		
		Args:
			mol_systems: 发现的分子系统字典
			preserve_existing: 是否保留已有的状态信息
		"""
		# 如果需要保留现有状态，先建立系统路径到状态的映射
		existing_status = {}
		if preserve_existing and self.targets:
			for target in self.targets:
				existing_status[target.system_path] = target.status
			self.logger.info(f"保留 {len(existing_status)} 个已有系统的状态信息")
		
		self.targets.clear()
		self.mol_groups.clear()
		total_targets = 0
		processed_molecules = 0
		
		for mol_id, system_paths in mol_systems.items():
			processed_molecules += 1
			if processed_molecules % 50 == 0:
				self.logger.info(f"处理分子进度: {processed_molecules}/{len(mol_systems)}")
			
			targets_for_mol = []
			for system_path in system_paths:
				basename = os.path.basename(system_path)
				import re
				match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', basename)
				if not match:
					self.logger.warning(f"无法解析系统路径: {system_path}")
					continue
				mol_id_parsed, conf, temp = match.groups()
				stru_dir = os.path.join(system_path, "OUT.ABACUS", "STRU")
				stru_files = glob.glob(os.path.join(stru_dir, "STRU_MD_*"))
				if not stru_files:
					self.logger.warning(f"系统 {basename} 没有找到 STRU_MD 文件")
					continue
				try:
					creation_time = os.path.getctime(system_path)
				except OSError:
					creation_time = 0.0
				
				# 使用已有状态或默认为pending
				status = existing_status.get(system_path, "pending")
				
				# 计算源文件哈希
				source_hash = self._calculate_source_hash(system_path)
				
				target = AnalysisTarget(
					system_path=system_path,
					mol_id=mol_id_parsed,
					conf=conf,
					temperature=temp,
					stru_files=stru_files,
					creation_time=creation_time,
					status=status,
					source_hash=source_hash
				)
				self.targets.append(target)
				targets_for_mol.append(target)
				total_targets += 1
			if targets_for_mol:
				self.mol_groups[mol_id] = targets_for_mol
		
		self.logger.info(f"PathManager loaded {len(self.mol_groups)} molecules with {total_targets} targets")
		if preserve_existing:
			preserved_completed = len([t for t in self.targets if t.status == "completed"])
			self.logger.info(f"保留了 {preserved_completed} 个已完成系统的状态")
	
	def deduplicate_targets(self) -> None:
		"""去除重复的分析目标，基于系统名称(system_key)保留修改时间最晚的"""
		if not self.targets:
			return
		
		# 按system_key分组
		system_groups = {}
		for target in self.targets:
			key = target.system_key  # mol_id_conf_id_T_tempK
			if key not in system_groups:
				system_groups[key] = []
			system_groups[key].append(target)
		
		# 去重：每组保留修改时间最晚的
		deduplicated_targets = []
		duplicates_removed = 0
		
		for key, targets_group in system_groups.items():
			if len(targets_group) == 1:
				deduplicated_targets.extend(targets_group)
			else:
				# 多个重复，选择修改时间最晚的
				latest_target = max(targets_group, key=lambda t: t.creation_time)
				deduplicated_targets.append(latest_target)
				duplicates_removed += len(targets_group) - 1
				
				# 记录被移除的重复项
				removed_paths = [t.system_path for t in targets_group if t != latest_target]
				self.logger.info(f"去重: 系统 {key} 保留 {latest_target.system_path}")
				for removed_path in removed_paths:
					self.logger.info(f"  移除重复: {removed_path}")
		
		# 更新targets和mol_groups
		self.targets = deduplicated_targets
		self._rebuild_mol_groups()
		
		if duplicates_removed > 0:
			self.logger.info(f"去重完成: 移除 {duplicates_removed} 个重复目标，保留 {len(self.targets)} 个")
	
	def _rebuild_mol_groups(self) -> None:
		"""重建分子分组"""
		self.mol_groups.clear()
		for target in self.targets:
			mol_id = target.mol_id
			if mol_id not in self.mol_groups:
				self.mol_groups[mol_id] = []
			self.mol_groups[mol_id].append(target)


	def load_targets(self) -> bool:
		if not os.path.exists(self.targets_file):
			return False
		try:
			with open(self.targets_file, 'r', encoding='utf-8') as f:
				targets_data = json.load(f)
			self.targets.clear()
			self.mol_groups.clear()
			for target_dict in targets_data:
				target_dict.pop('created_at', None)
				target = AnalysisTarget(**target_dict)
				self.targets.append(target)
				mol_id = target.mol_id
				if mol_id not in self.mol_groups:
					self.mol_groups[mol_id] = []
				self.mol_groups[mol_id].append(target)
			self.logger.info(f"Loaded {len(self.targets)} analysis targets from file")
			return True
		except Exception as e:
			self.logger.error(f"Failed to load targets: {str(e)}")
			return False

	def get_targets_by_molecule(self, mol_id: str) -> List[AnalysisTarget]:
		return self.mol_groups.get(mol_id, [])

	def get_all_targets(self) -> List[AnalysisTarget]:
		return self.targets.copy()

	def get_targets_by_status(self, status: str) -> List[AnalysisTarget]:
		return [target for target in self.targets if target.status == status]

	def update_target_status(self, system_path: str, status: str) -> bool:
		"""更新目标状态"""
		for target in self.targets:
			if target.system_path == system_path:
				target.status = status
				return True
		return False

	def get_progress_summary(self) -> Dict[str, int]:
		summary = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
		for target in self.targets:
			summary[target.status] += 1
		summary["total"] = len(self.targets)
		return summary

	def save_analysis_targets(self, analysis_params: Dict[str, Any] = None) -> str:
		"""保存分析目标到 analysis_targets.json，包含详细的哈希和状态信息"""
		if not self.targets_file:
			raise ValueError("输出目录未设置，请先调用 set_output_dir_for_params")
		
		targets_file = self.targets_file
		temp_file = targets_file + ".tmp"
		
		try:
			# 计算参数哈希
			params_hash = ""
			if analysis_params:
				params_hash = self._calculate_params_hash(analysis_params)
			
			# 生成JSON结构
			# 检查 single_analysis_results 是否存在并统计文件数，用以在 metadata 中记录
			single_dir = os.path.join(self.output_dir, "single_analysis_results")
			single_present = False
			single_count = 0
			try:
				if os.path.exists(single_dir) and os.path.isdir(single_dir):
					files = glob.glob(os.path.join(single_dir, "frame_metrics_*.csv"))
					if files:
						single_present = True
						single_count = len(files)
			except Exception:
				single_present = False
				single_count = 0

			analysis_data = {
				"metadata": {
					"generated_at": datetime.now().isoformat(),
					"generator": "ABACUS-STRU-Analyser",
					"version": "1.0",
					"params_hash": params_hash,
					"analysis_params": analysis_params or {},
					"output_directory": self.output_dir,
					"single_frame_results_present": single_present,
					"single_frame_results_count": single_count
				},
				"summary": {
					"total_molecules": len(self.mol_groups),
					"total_systems": len(self.targets),
					"status_counts": self.get_progress_summary()
				},
				"molecules": {}
			}
			
			# 按分子组织数据
			for mol_id, targets in self.mol_groups.items():
				mol_data = {
					"molecule_id": mol_id,
					"system_count": len(targets),
					"systems": {}
				}
				
				for target in targets:
					# 简化输出：移除每个系统的 params_hash，只在 metadata 保存一次
					system_data = {
						"system_path": target.system_path,
						"stru_files_count": len(target.stru_files),
						"status": target.status,
						"source_hash": target.source_hash
					}
					mol_data["systems"][target.system_name] = system_data
				
				analysis_data["molecules"][mol_id] = mol_data
			
			# 原子性写入：先写临时文件，再重命名
			with open(temp_file, 'w', encoding='utf-8') as f:
				json.dump(analysis_data, f, indent=2, ensure_ascii=False)
			
			# 原子性重命名
			os.replace(temp_file, targets_file)
			
			self.logger.info(f"Analysis targets saved to: {targets_file}")
			self.logger.info(f"Summary: {len(self.mol_groups)} molecules, {len(self.targets)} systems")
			self.logger.info(f"Params hash: {params_hash}")
			
			return targets_file
			
		except Exception as e:
			self.logger.error(f"Failed to save analysis targets: {str(e)}")
			# 清理临时文件
			if os.path.exists(temp_file):
				try:
					os.remove(temp_file)
				except:
					pass
			raise

	def load_analysis_targets(self) -> bool:
		"""从新格式的 analysis_targets.json 加载分析目标"""
		targets_file = os.path.join(self.output_dir, "analysis_targets.json")
		if not os.path.exists(targets_file):
			return False
		
		try:
			with open(targets_file, 'r', encoding='utf-8') as f:
				analysis_data = json.load(f)
			
			# 检查格式版本
			if "metadata" not in analysis_data or "molecules" not in analysis_data:
				self.logger.warning("分析目标文件格式不兼容，尝试旧格式加载")
				return self.load_targets()  # 回退到旧格式
			
			self.targets.clear()
			self.mol_groups.clear()
			
			# 提取参数哈希用于后续比较
			self.loaded_params_hash = analysis_data.get("metadata", {}).get("params_hash", "")
			
			# 加载分子和系统数据
			for mol_id, mol_data in analysis_data["molecules"].items():
				targets_for_mol = []
				for system_name, system_data in mol_data["systems"].items():
					# 兼容性：若 JSON 中缺失 mol_id/conf/temperature/creation_time，使用安全默认值
					target = AnalysisTarget(
						system_path=system_data.get("system_path", ""),
						mol_id=system_data.get("mol_id", mol_id),
						conf=system_data.get("conf", "0"),
						temperature=system_data.get("temperature", "0"),
						stru_files=[],  # 将在 load_from_discovery 中重新填充
						creation_time=system_data.get("creation_time", 0.0),
						status=system_data.get("status", "pending"),
						source_hash=system_data.get("source_hash", "")
					)
					self.targets.append(target)
					targets_for_mol.append(target)
				
				if targets_for_mol:
					self.mol_groups[mol_id] = targets_for_mol
			
			self.logger.info(f"Loaded {len(self.targets)} analysis targets from new format")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to load analysis targets: {str(e)}")
			return False

	def check_params_compatibility(self, current_params: Dict[str, Any]) -> bool:
		"""检查当前参数是否与加载的参数兼容"""
		if not hasattr(self, 'loaded_params_hash') or not self.loaded_params_hash:
			return False
		
		current_params_hash = self._calculate_params_hash(current_params)
		compatible = current_params_hash == self.loaded_params_hash
		
		if not compatible:
			self.logger.warning(f"参数不兼容 - 当前: {current_params_hash}, 加载的: {self.loaded_params_hash}")
		else:
			self.logger.info(f"参数兼容 - 哈希: {current_params_hash}")
		
		return compatible

	def check_existing_results(self, skip_single_results: bool = False) -> None:
		"""检查已有的分析结果，更新系统状态以实现增量计算"""
		if not self.targets:
			return
		
		updated_count = 0
		hash_mismatch_count = 0
		
		# 检查汇总结果文件是否存在
		summary_file = os.path.join(self.output_dir, "combined_analysis_results", "system_metrics_summary.csv")
		existing_systems = set()
		
		if os.path.exists(summary_file):
			try:
				import csv
				with open(summary_file, 'r', encoding='utf-8') as f:
					reader = csv.DictReader(f)
					for row in reader:
						if 'System' in row:
							existing_systems.add(row['System'])
			except Exception as e:
				self.logger.warning(f"读取汇总结果文件失败: {str(e)}")
		
		# 检查每个目标
		for target in self.targets:
			if target.status in ["completed", "failed"]:
				continue  # 已标记为完成或失败，跳过检查
			
			system_name = target.system_name
			system_path = target.system_path
			
			# 检查汇总结果中是否有该系统
			has_summary = system_name in existing_systems
			
			# 如果不跳过单个结果，还需要检查frame_metrics文件
			has_frame_metrics = True
			if not skip_single_results:
				frame_metrics_file = os.path.join(
					self.output_dir, "single_analysis_results", 
					f"frame_metrics_{system_name}.csv"
				)
				has_frame_metrics = os.path.exists(frame_metrics_file)
			
			# 检查源文件哈希是否匹配（从当前目录的targets文件）
			hash_matches = True
			if os.path.exists(self.targets_file):
				try:
					with open(self.targets_file, 'r', encoding='utf-8') as f:
						targets_data = json.load(f)
						# 查找匹配的系统记录
						for target_dict in targets_data.get("molecules", {}).values():
							for sys_data in target_dict.get("systems", {}).values():
								if sys_data.get("system_path") == system_path:
									old_hash = sys_data.get("source_hash", "")
									new_hash = target.source_hash
									if old_hash and new_hash and old_hash != new_hash:
										hash_matches = False
										hash_mismatch_count += 1
										self.logger.warning(f"检测到源文件变更: {system_name} (哈希不匹配)")
									break
				except Exception as e:
					self.logger.warning(f"读取历史哈希信息失败: {str(e)}")
			
			# 如果同时满足条件且哈希匹配，标记为已完成
			if has_summary and has_frame_metrics and hash_matches:
				target.status = "completed"
				updated_count += 1
				self.logger.info(f"检测到已完成的分析结果: {system_name}")
			elif has_summary and has_frame_metrics and not hash_matches:
				# 有结果但源文件已变更，重置为待处理
				target.status = "pending"
				self.logger.info(f"源文件已变更，重新分析: {system_name}")
		
		if updated_count > 0:
			self.logger.info(f"增量计算: 发现 {updated_count} 个已完成的系统，将跳过重新计算")
		if hash_mismatch_count > 0:
			self.logger.warning(f"源文件变更检测: {hash_mismatch_count} 个系统需要重新分析")
		if updated_count == 0 and hash_mismatch_count == 0:
			self.logger.info("增量计算: 未发现已完成的分析结果，将进行完整计算")
