#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABACUS分析器 - 重构版本
统一的分析入口，支持模块化任务执行

主要功能模块：
1. 采样算法 (sampling) - 默认执行
2. 体系分析 (analysis) - 单帧分析和合并分析
4. 采样效果对比 (comparison)
5. DeepMD数据格式转换 (deepmd)

特性：
- 彻底并行化，避免全局文件解析
- 智能续算逻辑
- 任务模块化管理
"""

import os
import sys
import time
import argparse
import logging
import multiprocessing as mp
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib
import json
import datetime

# 导入自定义模块
from src.utils.logmanager import LoggerManager
from src.io.path_manager import PathManager, lightweight_discover_systems
from src.core.system_analyser import SystemAnalyser
from src.io.result_saver import ResultSaver
from src.utils.common import ErrorHandler, FileUtils
from src.core.process_scheduler import ProcessScheduler, ProcessAnalysisTask


class TaskType(Enum):
    """任务类型枚举"""
    SAMPLING = "sampling"           # 采样算法
    ANALYSIS = "analysis"           # 体系分析（单帧+合并）
    COMPARISON = "comparison"       # 采样效果对比
    DEEPMD = "deepmd"              # DeepMD数据格式转换


@dataclass
class TaskConfig:
    """任务配置类"""
    # 核心参数
    sample_ratio: float = 0.1
    power_p: float = -0.5
    pca_variance_ratio: float = 0.90
    
    # 运行配置
    workers: int = -1
    output_dir: str = 'analysis_results'
    search_paths: List[str] = None
    include_project: bool = False
    force_recompute: bool = False
    
    # 任务控制
    enabled_tasks: Set[TaskType] = None
    scheduler: str = 'process'  # process / thread
    
    def __post_init__(self):
        if self.search_paths is None:
            self.search_paths = []
        if self.enabled_tasks is None:
            self.enabled_tasks = {TaskType.SAMPLING}  # 默认只执行采样


@dataclass
class SystemRecord:
    """体系记录"""
    system_path: str
    system_name: str
    mol_id: str
    source_hash: str
    selected_files: List[str]
    
    @classmethod
    def from_lightweight_record(cls, lwt_record) -> 'SystemRecord':
        """从LightweightSystemRecord创建SystemRecord"""
        return cls(
            system_path=lwt_record.system_path,
            system_name=lwt_record.system_name,
            mol_id=lwt_record.mol_id,
            source_hash=lwt_record.source_hash,
            selected_files=lwt_record.selected_files or []
        )


class ResumeChecker:
    """续算检查器"""
    
    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        self.cache = {}
    
    def can_skip_sampling(self, system_record: SystemRecord) -> Tuple[bool, Optional[List[int]]]:
        """检查是否可以跳过采样"""
        try:
            targets_file = os.path.join(self.output_dir, 'analysis_targets.json')
            if not os.path.exists(targets_file):
                return False, None
            
            with open(targets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 在molecules中查找系统
            for mol_id, mol_data in data.get('molecules', {}).items():
                systems = mol_data.get('systems', {})
                if system_record.system_name in systems:
                    system_info = systems[system_record.system_name]
                    # 检查source_hash匹配
                    if system_info.get('source_hash') == system_record.source_hash:
                        sampled_frames_str = system_info.get('sampled_frames')
                        if sampled_frames_str:
                            try:
                                # 解析字符串格式的采样帧
                                import ast
                                sampled_frames = ast.literal_eval(sampled_frames_str)
                                if isinstance(sampled_frames, list) and sampled_frames:
                                    self.logger.debug(f"采样续算: {system_record.system_name}")
                                    return True, sampled_frames
                            except (ValueError, SyntaxError) as e:
                                self.logger.warning(f"解析采样帧失败 {system_record.system_name}: {e}")
            
            return False, None
        except Exception as e:
            self.logger.warning(f"检查采样续算失败: {e}")
            return False, None
    
    def can_skip_analysis(self, system_record: SystemRecord) -> bool:
        """检查是否可以跳过体系分析"""
        try:
            # 检查单帧分析文件
            single_results_dir = os.path.join(self.output_dir, 'single_analysis_results')
            pattern = f"frame_metrics_{system_record.system_name}_*.csv"
            import glob
            single_files = glob.glob(os.path.join(single_results_dir, pattern))
            
            if not single_files:
                return False
            
            
            
            
            
            return False
        except Exception as e:
            self.logger.warning(f"检查体系分析续算失败: {e}")
            return False
    
    def can_skip_deepmd(self, system_record: SystemRecord) -> bool:
        """检查是否可以跳过DeepMD转换"""
        try:
            deepmd_dir = os.path.join(self.output_dir, 'deepmd_npy_per_system', system_record.system_name)
            
            # 检查目录存在且有内容（npy文件或完成标记）
            if os.path.exists(deepmd_dir):
                # 检查是否有npy文件
                import glob
                npy_files = glob.glob(os.path.join(deepmd_dir, '*.npy'))
                if npy_files:
                    self.logger.debug(f"DeepMD转换续算: {system_record.system_name} (npy文件)")
                    return True
                
                # 检查完成标记文件
                marker_file = os.path.join(deepmd_dir, '.deepmd_export_complete')
                if os.path.exists(marker_file):
                    self.logger.debug(f"DeepMD转换续算: {system_record.system_name} (标记文件)")
                    return True
                
                # 检查目录非空
                if os.listdir(deepmd_dir):
                    self.logger.debug(f"DeepMD转换续算: {system_record.system_name} (非空目录)")
                    return True
            
            return False
        except Exception as e:
            self.logger.warning(f"检查DeepMD转换续算失败: {e}")
            return False


class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, config: TaskConfig, logger: logging.Logger, log_queue: mp.Queue = None):
        self.config = config
        self.logger = logger
        self.log_queue = log_queue
        self.output_dir = None
        self.resume_checker = None
    
    def setup_output_dir(self) -> str:
        """设置输出目录"""
        current_analysis_params = {
            'sample_ratio': self.config.sample_ratio,
            'power_p': self.config.power_p,
            'pca_variance_ratio': self.config.pca_variance_ratio
        }
        
        path_manager = PathManager(self.config.output_dir)
        self.output_dir = path_manager.set_output_dir_for_params(current_analysis_params)
        self.resume_checker = ResumeChecker(self.output_dir, self.logger)
        
        if self.logger:
            self.logger.info(f"输出目录: {self.output_dir}")
        return self.output_dir
    
    def discover_systems(self) -> List[SystemRecord]:
        """发现系统并构建记录"""
        search_paths = self.config.search_paths or [os.path.abspath(os.path.join(os.getcwd(), '..'))]
        
        if self.logger:
            self.logger.info(f"搜索路径: {search_paths}")
        
        t_start = time.time()
        raw_records = lightweight_discover_systems(search_paths, include_project=self.config.include_project)
        
        if not raw_records:
            raise RuntimeError("未发现任何ABACUS系统")
        
        # 转换为SystemRecord
        system_records = []
        for rec in raw_records:
            system_records.append(SystemRecord.from_lightweight_record(rec))
        
        if self.logger:
            self.logger.info(f"发现 {len(system_records)} 个系统，耗时 {time.time()-t_start:.1f}s")
        return system_records
    
    def execute_sampling_task(self, system_records: List[SystemRecord]) -> List[Tuple]:
        """执行采样任务"""
        if TaskType.SAMPLING not in self.config.enabled_tasks:
            return []
        
        if self.logger:
            self.logger.info("=== 执行采样任务 ===")
        
        # 过滤需要处理的系统
        pending_records = []
        reuse_count = 0
        
        for record in system_records:
            if not self.config.force_recompute:
                can_skip, sampled_frames = self.resume_checker.can_skip_sampling(record)
                if can_skip:
                    reuse_count += 1
                    continue
            pending_records.append(record)
        
        if self.logger:
            self.logger.info(f"采样任务: 待处理 {len(pending_records)}, 复用 {reuse_count}")
        
        if not pending_records:
            return []
        
        # 执行采样
        return self._run_parallel_sampling(pending_records)
    
    def execute_analysis_task(self, system_records: List[SystemRecord]) -> List[Tuple]:
        """执行体系分析任务"""
        if TaskType.ANALYSIS not in self.config.enabled_tasks:
            return []
        
        if self.logger:
            self.logger.info("=== 执行体系分析任务 ===")
        
        # 过滤需要处理的系统
        pending_records = []
        skip_count = 0
        
        for record in system_records:
            if not self.config.force_recompute:
                if self.resume_checker.can_skip_analysis(record):
                    skip_count += 1
                    continue
            pending_records.append(record)
        
        if self.logger:
            self.logger.info(f"体系分析任务: 待处理 {len(pending_records)}, 跳过 {skip_count}")
        
        if not pending_records:
            return []
        
        # 执行体系分析
        return self._run_parallel_analysis(pending_records)
    
    
    def execute_comparison_task(self) -> bool:
        """执行采样效果对比任务"""
        if TaskType.COMPARISON not in self.config.enabled_tasks:
            return True
        
        if self.logger:
            self.logger.info("=== 执行采样效果对比任务 ===")
        
        try:
            from src.analysis.sampling_comparison import analyse_sampling_compare
            
            # 检查是否已存在结果（支持续算）
            comparison_files = [
                os.path.join(self.output_dir, "combined_analysis_results", "sampling_methods_comparison.csv"),
                os.path.join(self.output_dir, "combined_analysis_results", "sampling_compare_enhanced.csv")
            ]
            
            if not self.config.force_recompute:
                existing_files = [f for f in comparison_files if os.path.exists(f)]
                if existing_files:
                    if self.logger:
                        self.logger.info(f"采样效果对比结果已存在，跳过: {existing_files}")
                    return True
            
            analyse_sampling_compare(result_dir=self.output_dir)
            if self.logger:
                self.logger.info("采样效果对比完成")
            return True
        
        except ImportError:
            if self.logger:
                self.logger.warning("采样效果对比模块不可用，跳过")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"采样效果对比失败: {e}")
            return False
    
    def execute_deepmd_task(self, system_records: List[SystemRecord]) -> bool:
        """执行DeepMD数据格式转换任务"""
        if TaskType.DEEPMD not in self.config.enabled_tasks:
            return True
        
        if self.logger:
            self.logger.info("=== 执行DeepMD数据格式转换任务 ===")
        
        # 从analysis_targets.json读取体系信息和采样列表
        target_systems = self._load_deepmd_targets()
        if not target_systems:
            if self.logger:
                self.logger.warning("未找到analysis_targets.json或无有效采样数据，DeepMD转换跳过")
            return False
        
        # 过滤需要处理的系统（续算逻辑）
        pending_systems = []
        skip_count = 0
        
        for system_info in target_systems:
            if not self.config.force_recompute:
                # 创建临时的SystemRecord用于续算检查
                temp_record = SystemRecord(
                    system_path=system_info['system_path'],
                    system_name=system_info['system_name'],
                    mol_id=system_info['mol_id'],
                    source_hash=system_info['source_hash'],
                    selected_files=[]
                )
                if self.resume_checker.can_skip_deepmd(temp_record):
                    skip_count += 1
                    continue
            pending_systems.append(system_info)
        
        if self.logger:
            self.logger.info(f"DeepMD转换任务: 待处理 {len(pending_systems)}, 跳过 {skip_count}")
        
        if not pending_systems:
            return True
        
        # 执行转换
        return self._run_parallel_deepmd_conversion_from_targets(pending_systems)
    
    def _load_deepmd_targets(self) -> List[Dict]:
        """从analysis_targets.json加载DeepMD转换目标"""
        try:
            targets_file = os.path.join(self.output_dir, 'analysis_targets.json')
            if not os.path.exists(targets_file):
                return []
            
            with open(targets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            target_systems = []
            for mol_id, mol_data in data.get('molecules', {}).items():
                for system_name, system_info in mol_data.get('systems', {}).items():
                    # 检查是否有采样帧
                    sampled_frames_str = system_info.get('sampled_frames')
                    if not sampled_frames_str:
                        continue
                    
                    try:
                        # 解析采样帧
                        import ast
                        sampled_frames = ast.literal_eval(sampled_frames_str)
                        if not isinstance(sampled_frames, list) or not sampled_frames:
                            continue
                    except (ValueError, SyntaxError):
                        continue
                    
                    target_systems.append({
                        'system_path': system_info['system_path'],
                        'system_name': system_name,
                        'mol_id': mol_id,
                        'source_hash': system_info['source_hash'],
                        'sampled_frames': sampled_frames
                    })
            
            if self.logger:
                self.logger.info(f"从analysis_targets.json加载 {len(target_systems)} 个体系")
            return target_systems
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"加载DeepMD目标失败: {e}")
            return []
    
    def _run_parallel_deepmd_conversion_from_targets(self, target_systems: List[Dict]) -> bool:
        """从analysis_targets.json并行执行DeepMD转换"""
        from src.io.result_saver import ResultSaver
        from src.io.stru_parser import StrUParser
        success_count = 0
        
        for system_info in target_systems:
            try:
                system_path = system_info['system_path']
                system_name = system_info['system_name']
                sampled_frames = system_info['sampled_frames']
                
                if self.logger:
                    self.logger.info(f"处理DeepMD转换: {system_name}")
                
                # 解析STRU目录获得frames
                parser = StrUParser()
                stru_dir = os.path.join(system_path, "OUT.ABACUS", "STRU")
                frames = parser.parse_trajectory(stru_dir)
                if not frames:
                    if self.logger:
                        self.logger.warning(f"DeepMD转换跳过 {system_name}: STRU解析无帧")
                    continue
                
                # 调用导出函数
                deepmd_dir = os.path.join(self.output_dir, 'deepmd_npy_per_system')
                ResultSaver.export_sampled_frames_per_system(
                    frames=frames,
                    sampled_frame_ids=sampled_frames,
                    system_path=system_path,
                    output_root=deepmd_dir,
                    system_name=system_name,
                    logger=self.logger,
                    force=self.config.force_recompute
                )
                success_count += 1
                if self.logger:
                    self.logger.info(f"DeepMD转换完成: {system_name}")
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"DeepMD转换失败 {system_info['system_name']}: {e}")
        
        if self.logger:
            self.logger.info(f"DeepMD转换任务完成: 成功 {success_count}/{len(target_systems)}")
        return success_count > 0
    
    def _run_parallel_sampling(self, records: List[SystemRecord]) -> List[Tuple]:
        """并行执行采样"""
        workers = self._determine_workers()
        
        if self.config.scheduler == 'process':
            return self._run_process_sampling(records, workers)
        else:
            return self._run_thread_sampling(records, workers)
        """并行执行体系分析"""
        workers = self._determine_workers()
        
        if self.config.scheduler == 'process':
            return self._run_process_analysis(records, workers)
        else:
            return self._run_thread_analysis(records, workers)
    
    def _run_process_sampling(self, records: List[SystemRecord], workers: int) -> List[Tuple]:
        """使用进程池执行采样"""
        analyser_params = {
            'sample_ratio': self.config.sample_ratio,
            'power_p': self.config.power_p,
            'pca_variance_ratio': self.config.pca_variance_ratio
        }
        
        scheduler = ProcessScheduler(max_workers=workers, analyser_params=analyser_params, log_queue=self.log_queue)
        
        # 添加任务
        for record in records:
            # 查找可复用的采样帧
            can_skip, pre_sampled = self.resume_checker.can_skip_sampling(record)
            
            scheduler.add_task(ProcessAnalysisTask(
                system_path=record.system_path,
                system_name=record.system_name,
                pre_sampled_frames=pre_sampled if can_skip else None,
                pre_stru_files=record.selected_files,
                sampling_only=True  # 仅采样模式
            ))
        
        if self.logger:
            self.logger.info(f"进程池采样任务: {len(records)} 个系统")
        raw_results = scheduler.run()
        
        # 过滤有效结果
        results = [r for r in raw_results if r]
        
        # 保存结果
        for result in results:
            try:
                ResultSaver.save_single_system(self.output_dir, result, sampling_only=True)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"保存采样结果失败: {e}")
        
        # 保存采样信息到analysis_targets.json
        if results:
            self._save_sampling_targets(results, records)
        
        return results
    
    def _run_process_analysis(self, records: List[SystemRecord], workers: int) -> List[Tuple]:
        """使用进程池执行体系分析"""
        analyser_params = {
            'sample_ratio': self.config.sample_ratio,
            'power_p': self.config.power_p,
            'pca_variance_ratio': self.config.pca_variance_ratio
        }
        
        scheduler = ProcessScheduler(max_workers=workers, analyser_params=analyser_params, log_queue=self.log_queue)
        
        # 添加任务
        for record in records:
            # 查找可复用的采样帧
            can_skip, pre_sampled = self.resume_checker.can_skip_sampling(record)
            
            scheduler.add_task(ProcessAnalysisTask(
                system_path=record.system_path,
                system_name=record.system_name,
                pre_sampled_frames=pre_sampled if can_skip else None,
                pre_stru_files=record.selected_files,
                sampling_only=False  # 完整分析模式
            ))
        
        if self.logger:
            self.logger.info(f"进程池分析任务: {len(records)} 个系统")
        raw_results = scheduler.run()
        
        # 过滤有效结果
        results = [r for r in raw_results if r]
        
        # 保存结果
        for result in results:
            try:
                ResultSaver.save_single_system(self.output_dir, result, sampling_only=False)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"保存分析结果失败: {e}")
        
        # 保存分析信息到analysis_targets.json
        if results:
            self._save_analysis_targets(results, records)
        
        return results
    
    def _run_parallel_deepmd_conversion(self, records: List[SystemRecord]) -> bool:
        """并行执行DeepMD转换，优先从analysis_targets.json读取采样帧步数，失败则自动采样，然后导出npy"""
        from src.io.result_saver import ResultSaver
        from src.io.stru_parser import StrUParser
        success_count = 0
        for record in records:
            try:
                # 1. 优先读取采样帧步数
                can_skip, sampled_frames = self.resume_checker.can_skip_sampling(record)
                if not sampled_frames:
                    # 2. 若无采样帧则自动执行采样算法
                    if self.logger:
                        self.logger.info(f"{record.system_name} 未找到采样帧，自动执行采样算法...")
                    sampling_result = self._run_parallel_sampling([record])
                    if sampling_result and len(sampling_result) > 0 and len(sampling_result[0]) > 1:
                        frames = sampling_result[0][1]
                        # 采样算法返回的帧对象中，sampled_frames为帧的frame_id
                        sampled_frames = [f.frame_id for f in frames if hasattr(f, 'selected') and f.selected]
                    else:
                        if self.logger:
                            self.logger.warning(f"DeepMD转换跳过 {record.system_name}: 自动采样失败")
                        continue
                # 3. 解析STRU目录获得frames
                parser = StrUParser()
                stru_dir = os.path.join(record.system_path, "OUT.ABACUS", "STRU")
                frames = parser.parse_trajectory(stru_dir)
                if not frames:
                    if self.logger:
                        self.logger.warning(f"DeepMD转换跳过 {record.system_name}: STRU解析无帧")
                    continue
                # 4. 调用导出函数
                deepmd_dir = os.path.join(self.output_dir, 'deepmd_npy_per_system')
                ResultSaver.export_sampled_frames_per_system(
                    frames=frames,
                    sampled_frame_ids=sampled_frames,
                    system_path=record.system_path,
                    output_root=deepmd_dir,
                    system_name=record.system_name,
                    logger=self.logger,
                    force=self.config.force_recompute
                )
                success_count += 1
                if self.logger:
                    self.logger.info(f"DeepMD转换完成: {record.system_name}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"DeepMD转换失败 {record.system_name}: {e}")
        if self.logger:
            self.logger.info(f"DeepMD转换任务完成: 成功 {success_count}/{len(records)}")
    def _run_thread_sampling(self, records: List[SystemRecord], workers: int) -> List[Tuple]:
        """使用线程池执行采样（简化实现）"""
        if self.logger:
            self.logger.warning("线程池模式暂未完全实现，回退到进程池")
        return self._run_process_sampling(records, workers)
    
    def _run_thread_analysis(self, records: List[SystemRecord], workers: int) -> List[Tuple]:
        """使用线程池执行分析（简化实现）"""
        if self.logger:
            self.logger.warning("线程池模式暂未完全实现，回退到进程池")
        return self._run_process_analysis(records, workers)
    
    def _determine_workers(self) -> int:
        """确定工作进程数"""
        if self.config.workers == -1:
            try:
                workers = int(os.environ.get('SLURM_CPUS_PER_TASK',
                           os.environ.get('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())))
            except (ValueError, TypeError):
                workers = max(1, mp.cpu_count())
        else:
            workers = max(1, self.config.workers)
        return workers
    
    def _save_sampling_targets(self, results: List[Tuple], records: List[SystemRecord]) -> None:
        """保存采样结果到analysis_targets.json"""
        try:
            targets_file = os.path.join(self.output_dir, 'analysis_targets.json')
            
            # 构建records的查找映射
            record_map = {rec.system_name: rec for rec in records}
            
            # 构建targets数据结构
            metadata = {
                "generated_at": datetime.datetime.now().isoformat(),
                "generator": "ABACUS-STRU-Analyser",
                "version": "2.0",
                "analysis_params": {
                    "sample_ratio": self.config.sample_ratio,
                    "power_p": self.config.power_p,
                    "pca_variance_ratio": self.config.pca_variance_ratio
                },
                "output_directory": self.output_dir,
                "md_dumpfreq": 10
            }
            
            molecules = {}
            for result in results:
                if len(result) < 2:
                    continue
                
                metrics = result[0]
                system_name = getattr(metrics, 'system_name', 'unknown')
                sampled_frames = getattr(metrics, 'sampled_frames', [])
                system_path = getattr(metrics, 'system_path', '')
                
                # 从记录中获取更多信息
                record = record_map.get(system_name)
                source_hash = record.source_hash if record else "unknown"
                mol_id = record.mol_id if record else "unknown"
                
                if mol_id != "unknown":
                    if mol_id not in molecules:
                        molecules[mol_id] = {
                            "molecule_id": mol_id,
                            "system_count": 0,
                            "systems": {}
                        }
                    
                    molecules[mol_id]["system_count"] += 1
                    molecules[mol_id]["systems"][system_name] = {
                        "system_path": system_path,
                        "stru_files_count": len(sampled_frames) if sampled_frames else 0,
                        "source_hash": source_hash,
                        "sampled_frames": str(sampled_frames)  # 保持字符串格式兼容
                    }
            
            targets_data = {
                "metadata": metadata,
                "summary": {
                    "total_molecules": len(molecules),
                    "total_systems": sum(mol["system_count"] for mol in molecules.values())
                },
                "molecules": molecules
            }
            
            with open(targets_file, 'w', encoding='utf-8') as f:
                json.dump(targets_data, f, ensure_ascii=False, indent=2)
            
            if self.logger:
                self.logger.info(f"已保存采样目标: {targets_file}")
        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"保存采样目标失败: {e}")
    
    def _save_analysis_targets(self, results: List[Tuple], records: List[SystemRecord]) -> None:
        """保存分析结果到analysis_targets.json（与采样目标格式相同）"""
        self._save_sampling_targets(results, records)


class ABACUSAnalyser:
    """ABACUS分析器主类"""
    
    def __init__(self):
        self.logger = None
        self.log_queue = None
        self.log_listener = None
    
    def setup_logging(self, output_dir: str) -> None:
        """设置日志系统"""
        self.log_queue, self.log_listener = LoggerManager.create_multiprocess_logging_setup(
            output_dir=output_dir,
            log_filename="main.log",
            when="D",
            backup_count=14
        )
        
        self.log_listener.start()
        
        self.logger = LoggerManager.setup_worker_logger(
            name=__name__,
            queue=self.log_queue,
            level=logging.INFO,
            add_console=True
        )
    
    def cleanup_logging(self) -> None:
        """清理日志系统"""
        if self.log_listener:
            try:
                LoggerManager.stop_listener(self.log_listener)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"停止日志监听器出错: {e}")
    
    def run(self, config: TaskConfig) -> None:
        """运行分析流程"""
        start_time = time.time()
        
        try:
            # 先设置临时输出目录以获取日志路径
            current_analysis_params = {
                'sample_ratio': config.sample_ratio,
                'power_p': config.power_p,
                'pca_variance_ratio': config.pca_variance_ratio
            }
            
            from src.io.path_manager import PathManager
            path_manager = PathManager(config.output_dir)
            output_dir = path_manager.set_output_dir_for_params(current_analysis_params)
            
            # 设置日志
            self.setup_logging(output_dir)
            
            # 创建任务执行器
            executor = TaskExecutor(config, self.logger, self.log_queue)
            executor.output_dir = output_dir
            
            # 创建任务执行器
            executor = TaskExecutor(config, self.logger, self.log_queue)
            executor.output_dir = output_dir
            executor.resume_checker = ResumeChecker(output_dir, self.logger)
            
            # 记录启动信息
            self._log_startup_info(config)
            
            # 发现系统
            system_records = executor.discover_systems()
            
            # 执行任务
            results = {}
            
            # 1. 采样任务（默认执行）
            sampling_results = executor.execute_sampling_task(system_records)
            results['sampling'] = sampling_results
            
            # 2. 体系分析任务
            analysis_results = executor.execute_analysis_task(system_records)
            results['analysis'] = analysis_results
            
            # 4. 采样效果对比任务
            comparison_success = executor.execute_comparison_task()
            results['comparison'] = comparison_success
            
            # 5. DeepMD转换任务
            deepmd_success = executor.execute_deepmd_task(system_records)
            results['deepmd'] = deepmd_success
            
            # 输出结果统计
            self._log_final_statistics(results, start_time, output_dir, config)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"分析流程执行失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            else:
                print(f"分析流程执行失败: {e}", file=sys.stderr)
        finally:
            self.cleanup_logging()
    
    def _log_startup_info(self, config: TaskConfig) -> None:
        """记录启动信息"""
        enabled_tasks = [task.value for task in config.enabled_tasks]
        
        self.logger.info("=" * 60)
        self.logger.info("ABACUS分析器启动 - 重构版本")
        self.logger.info(f"启用任务: {', '.join(enabled_tasks)}")
        self.logger.info(f"采样参数: ratio={config.sample_ratio}, p={config.power_p}, var={config.pca_variance_ratio}")
        self.logger.info(f"并行配置: scheduler={config.scheduler}, workers={config.workers}")
        self.logger.info(f"强制重算: {config.force_recompute}")
        self.logger.info("=" * 60)
    
    def _log_final_statistics(self, results: Dict, start_time: float, output_dir: str, config: TaskConfig) -> None:
        """记录最终统计信息"""
        elapsed = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("分析完成统计:")
        
        # 只显示启用的任务的结果
        if TaskType.SAMPLING in config.enabled_tasks and 'sampling' in results and results['sampling']:
            self.logger.info(f"  采样任务: 处理 {len(results['sampling'])} 个系统")
        
        if TaskType.ANALYSIS in config.enabled_tasks and 'analysis' in results and results['analysis']:
            self.logger.info(f"  体系分析: 处理 {len(results['analysis'])} 个系统")
        
        
        if TaskType.COMPARISON in config.enabled_tasks and 'comparison' in results:
            status = "完成" if results['comparison'] else "失败"
            self.logger.info(f"  采样效果对比: {status}")
        
        if TaskType.DEEPMD in config.enabled_tasks and 'deepmd' in results:
            status = "完成" if results['deepmd'] else "失败"
            self.logger.info(f"  DeepMD转换: {status}")
        
        self.logger.info(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info("=" * 60)


def parse_arguments() -> TaskConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ABACUS分析器 - 重构版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
任务模块:
  sampling     - 采样算法 (默认启用)
  analysis     - 体系分析 (单帧分析+合并分析)
  comparison   - 采样效果对比
  deepmd       - DeepMD数据格式转换

使用示例:
  # 仅执行采样 (默认)
  python abacus_analyser.py
  
  # 执行采样和体系分析
  python abacus_analyser.py --tasks sampling analysis
  
  # 执行所有任务
  
  # 指定搜索路径和参数
  python abacus_analyser.py --tasks sampling analysis -s /path/to/data -r 0.05 -w 32
        """
    )
    
    # 核心参数
    parser.add_argument('-r', '--sample_ratio', type=float, default=0.1, 
                       help='采样比例 (默认: 0.1)')
    parser.add_argument('-p', '--power_p', type=float, default=-0.5, 
                       help='幂平均距离的p值 (默认: -0.5)')
    parser.add_argument('-v', '--pca_variance_ratio', type=float, default=0.90, 
                       help='PCA降维累计方差贡献率 (默认: 0.90)')
    
    # 运行配置
    parser.add_argument('-w', '--workers', type=int, default=-1, 
                       help='并行工作进程数 (默认: -1 自动)')
    parser.add_argument('-o', '--output_dir', type=str, default='analysis_results', 
                       help='输出根目录 (默认: analysis_results)')
    parser.add_argument('-s', '--search_path', nargs='*', default=None, 
                       help='递归搜索路径')
    parser.add_argument('--include_project', action='store_true', 
                       help='允许搜索项目自身目录')
    parser.add_argument('-f', '--force_recompute', action='store_true', 
                       help='强制重新计算，忽略续算')
    
    # 任务控制
    parser.add_argument('--tasks', nargs='*', 
                       choices=['sampling', 'analysis', 'comparison', 'deepmd'],
                       default=['sampling'],
                       help='要执行的任务列表 (默认: sampling)')
    parser.add_argument('--scheduler', choices=['process', 'thread'], default='process',
                       help='调度器类型 (默认: process)')
    
    args = parser.parse_args()
    
    # 转换任务列表
    enabled_tasks = set()
    for task_name in args.tasks:
        enabled_tasks.add(TaskType(task_name))
    
    return TaskConfig(
        sample_ratio=args.sample_ratio,
        power_p=args.power_p,
        pca_variance_ratio=args.pca_variance_ratio,
        workers=args.workers,
        output_dir=args.output_dir,
        search_paths=args.search_path or [],
        include_project=args.include_project,
        force_recompute=args.force_recompute,
        enabled_tasks=enabled_tasks,
        scheduler=args.scheduler
    )


def main():
    """主函数"""
    config = parse_arguments()
    analyser = ABACUSAnalyser()
    analyser.run(config)


if __name__ == "__main__":
    main()
