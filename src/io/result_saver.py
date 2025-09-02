#!/usr/bin/env python

import csv
import logging
import os
from typing import Optional, List, Dict, Tuple, Sequence, Iterable
import datetime
import json
import re

import numpy as np

from ..core.metrics import TrajectoryMetrics
from ..io.stru_parser import FrameData
from ..utils import FileUtils
from ..utils.common import ErrorHandler
from ..core.metrics import (
    SYSTEM_SUMMARY_HEADERS as REGISTRY_SYSTEM_SUMMARY_HEADERS,
    build_summary_row as build_registry_summary_row,
    SCHEMA_VERSION as SUMMARY_SCHEMA_VERSION,
)


class ResultSaver:
    """结果保存器类，负责保存分析结果到CSV文件"""

    # CSV Headers constants
    # Level 4+ 列顺序（进一步语义分组 & 类型聚类）
    # 分组顺序：
    # 1) 基础标识 & 条件 -> 2) 规模/维度 -> 3) 核心结构距离指标 -> 4) 多样性/覆盖/能量 ->
    # 5) PCA 概览 (数量 -> 方差占比 -> 累积 -> 明细数组) -> 6) 分布/采样相似性
    # 注意：原第7组 Mean_Structure_Coordinates 已拆分为独立 JSON 文件导出，列中移除（PR1）。
    # 统一来源：core.metrics.SYSTEM_SUMMARY_HEADERS
    SYSTEM_SUMMARY_HEADERS = REGISTRY_SYSTEM_SUMMARY_HEADERS
    SYSTEM_SUMMARY_SCHEMA_VERSION = SUMMARY_SCHEMA_VERSION

    @staticmethod
    def load_progress(output_dir: str) -> Dict[str, any]:
        """加载进度信息，用于断点续算
        
        Returns:
            包含已处理系统列表和其他进度信息的字典
        """
        progress_path = os.path.join(output_dir, "combined_analysis_results", "progress.json")
        if not os.path.exists(progress_path):
            return {'processed_systems': [], 'last_updated': None}
            
        try:
            import json
            with open(progress_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"加载进度文件失败: {e}")
            return {'processed_systems': [], 'last_updated': None}

    @staticmethod
    def _format_metric_row(metrics: TrajectoryMetrics) -> List[str]:
        """使用统一 registry 构建行。"""
        return build_registry_summary_row(metrics)

    @staticmethod
    def export_mean_structure(output_dir: str, metrics: TrajectoryMetrics, force_update: bool = False) -> None:
        """导出单体系平均结构到独立 JSON 文件。

        输出路径: <run_dir>/mean_structures/mean_structure_<system>.json
        内容包含: 版本号, 导出时间, 系统基础信息, frame 统计, 维度, 均值结构 shape, 实际坐标数据。
        若均值结构为空/None 则跳过。
        
        Args:
            force_update: 如果为True，强制更新已存在的文件
        """
        logger = logging.getLogger(__name__)
        try:
            if metrics.mean_structure is None or getattr(metrics.mean_structure, 'size', 0) == 0:
                return
                
            import json
            mean_dir = os.path.join(output_dir, 'mean_structures')
            FileUtils.ensure_dir(mean_dir)
            filepath = os.path.join(mean_dir, f"mean_structure_{metrics.system_name}.json")
            
            # 检查文件是否已存在且无需强制更新
            if not force_update and os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        # 如果文件已存在且版本相同，跳过更新
                        if existing_data.get('version') == '1.0' and existing_data.get('num_frames') == metrics.num_frames:
                            logger.debug(f"均值结构文件已存在且最新: {metrics.system_name}")
                            return
                except Exception:
                    # 如果读取失败，继续写入
                    pass
            
            data = {
                "version": "1.0",  # PR1 初始版本
                "exported_at": datetime.datetime.utcnow().isoformat() + "Z",
                "system": metrics.system_name,
                "molecule_id": metrics.mol_id,
                "configuration": metrics.conf,
                "temperature_K": metrics.temperature,
                "num_frames": metrics.num_frames,
                "dimension": metrics.dimension,
                "mean_structure_shape": list(metrics.mean_structure.shape),
                "mean_structure": metrics.mean_structure.tolist(),
                # 新增RMSD缓存支持（续算复用）
                "rmsd_mean": metrics.rmsd_mean,
                "rmsd_per_frame": metrics.rmsd_per_frame,
            }
            # 原子级别字段扩展预留
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # 维护索引文件（追加/更新）
            try:
                index_path = os.path.join(mean_dir, 'index.json')
                index = {}
                if os.path.exists(index_path):
                    try:
                        with open(index_path, 'r', encoding='utf-8') as idx_f:
                            index = json.load(idx_f) or {}
                    except Exception:
                        index = {}
                index[metrics.system_name] = {
                    "file": os.path.basename(filepath),
                    "num_frames": metrics.num_frames,
                    "dimension": metrics.dimension,
                    "shape": list(metrics.mean_structure.shape),
                    "updated_at": data["exported_at"],
                }
                with open(index_path, 'w', encoding='utf-8') as idx_f:
                    json.dump(index, idx_f, ensure_ascii=False, indent=2)
                    idx_f.flush()
                    os.fsync(idx_f.fileno())
            except Exception as ie:
                logger.warning(f"更新均值结构索引失败: {ie}")
                
            logger.debug(f"成功导出均值结构: {metrics.system_name}")
            
        except Exception as e:
            logger.warning(f"导出均值结构失败: {metrics.system_name}: {e}")

    @staticmethod
    def save_results(output_dir: str, analysis_results: List[Tuple]) -> None:
        """(兼容接口) 使用流式逻辑逐体系保存；不再区分增量/完整模式。"""
        logger = logging.getLogger(__name__)
        for result in analysis_results or []:
            try:
                ResultSaver.save_single_system(output_dir, result, sampling_only=False)
            except Exception as e:
                logger.warning(f"批量兼容保存单体系失败(忽略): {e}")
        # 批量调用后统一排序（若文件存在）
        try:
            ResultSaver.reorder_system_summary(output_dir)
        except Exception:
            pass

    # -------------------- Streaming / Per-System Saving Enhancements --------------------
    @staticmethod
    def save_single_system(
        output_dir: str,
        result: Tuple,
        sampling_only: bool = False,
        flush_targets_hook: Optional[callable] = None,
    ) -> None:
        """流式保存单个体系的全部可用结果 (体系完成后立即调用)。

        按当前模式与可用数据自动降级：
          - 完整模式(result 长度>=7)：写 frame_metrics, append system_metrics_summary, mean_structure
          - 仅采样模式/数据不足：只尝试导出均值结构(如果存在) + 采样帧（采样帧主要已在 analysis_targets.json 由调用方刷新）

        Args:
            output_dir: run_* 目录
            result: analyse_system 返回的 tuple
            sampling_only: 是否 sampling_only 模式
            flush_targets_hook: 可选回调，用于调用方在成功后刷新 analysis_targets.json
        """
        logger = logging.getLogger(__name__)
        if not result:
            return
        try:
            metrics = result[0]
            # sampling_only 模式下只返回 (metrics, frames)
            frames = result[1] if len(result) > 1 else []
            # 完整模式下期望 7 元素
            pca_components_data = result[4] if not sampling_only and len(result) > 4 else None
            rmsd_per_frame = result[6] if not sampling_only and len(result) > 6 else None

            # 1) system_metrics_summary.csv 追加写（仅完整模式且有足够数据）
            if not sampling_only and hasattr(metrics, 'system_name'):
                ResultSaver.append_system_summary_rows(output_dir, [metrics])
                ResultSaver.export_mean_structure(output_dir, metrics, force_update=True)
            elif sampling_only:
                # 采样模式：仍尝试导出均值结构（如果 earlier pipeline 填充）
                try:
                    ResultSaver.export_mean_structure(output_dir, metrics, force_update=True)
                except Exception:
                    pass

            # 2) frame_metrics_{system}.csv (仅完整模式)
            if (not sampling_only) and frames and pca_components_data is not None:
                try:
                    sampled_frames = [fid for fid in getattr(metrics, 'sampled_frames', [])]
                    ResultSaver.save_frame_metrics(
                        output_dir=output_dir,
                        system_name=metrics.system_name,
                        frames=frames,
                        sampled_frames=sampled_frames,
                        pca_components_data=pca_components_data,
                        rmsd_per_frame=rmsd_per_frame,
                        incremental=False,
                    )
                except Exception as fe:
                    logger.warning(f"单体系帧指标写入失败 {metrics.system_name}: {fe}")

            # 3) 更新 progress.json
            try:
                ResultSaver._update_progress(output_dir, metrics.system_name)
            except Exception as pe:
                logger.warning(f"progress.json 更新失败 {metrics.system_name}: {pe}")

            # 4) 可选刷新 analysis_targets.json (由 orchestrator 提供的 hook 控制频率)
            if flush_targets_hook:
                try:
                    flush_targets_hook()
                except Exception as he:
                    logger.warning(f"analysis_targets.json 刷新失败: {he}")
        except Exception as e:
            logger.error(f"流式保存体系结果失败: {e}")

    @staticmethod
    def _update_progress(output_dir: str, system_name: str) -> None:
        """将单个 system_name 追加到 progress.json (幂等)"""
        combined_dir = os.path.join(output_dir, "combined_analysis_results")
        FileUtils.ensure_dir(combined_dir)
        progress_path = os.path.join(combined_dir, "progress.json")
        data = {"processed_systems": [], "last_updated": None}
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) or data
            except Exception:
                pass
        if system_name not in data.get('processed_systems', []):
            data['processed_systems'].append(system_name)
        data['last_updated'] = datetime.datetime.utcnow().isoformat() + 'Z'
        tmp = progress_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, progress_path)

    @staticmethod
    def append_system_summary_rows(output_dir: str, metrics_list: Iterable) -> None:
        """通用追加接口：向 system_metrics_summary.csv 追加多个 metrics (无排序)。"""
        if not metrics_list:
            return
        combined_dir = os.path.join(output_dir, 'combined_analysis_results')
        FileUtils.ensure_dir(combined_dir)
        csv_path = os.path.join(combined_dir, 'system_metrics_summary.csv')
        file_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(ResultSaver.SYSTEM_SUMMARY_HEADERS)
                for m in metrics_list:
                    try:
                        row = ResultSaver._format_metric_row(m)
                        writer.writerow(row)
                    except Exception as rexc:
                        logging.getLogger(__name__).warning(f"写入单行失败 {getattr(m,'system_name','?')}: {rexc}")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logging.getLogger(__name__).error(f"追加 system_metrics_summary 失败: {e}")

    @staticmethod
    def reorder_system_summary(output_dir: str) -> None:
        """读取当前 system_metrics_summary.csv 重新排序并原子覆盖。
    仅基于 system_name 中的 mol/conf/温度数值排序。
        """
        combined_dir = os.path.join(output_dir, 'combined_analysis_results')
        csv_path = os.path.join(combined_dir, 'system_metrics_summary.csv')
        if not os.path.exists(csv_path):
            return
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
            if not reader:
                return
            header = reader[0]
            rows = reader[1:]
            def sort_key(row):
                try:
                    system_name = row[0]
                    match = re.match(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name)
                    if match:
                        mol_id, conf, temp = match.groups()
                        return (int(mol_id), int(conf), int(temp))
                except Exception:
                    pass
                return (999999, 999999, 999999)
            rows.sort(key=sort_key)
            tmp = csv_path + '.tmp'
            with open(tmp, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            os.replace(tmp, csv_path)
            logging.getLogger(__name__).info("system_metrics_summary.csv 已重新排序")
        except Exception as e:
            logging.getLogger(__name__).warning(f"system_metrics_summary 排序失败(忽略): {e}")

    @staticmethod
    # 已移除旧的完整/增量系统汇总保存函数，统一使用 append + reorder 机制

    @staticmethod
    def save_frame_metrics(
        output_dir: str,
        system_name: str,
        frames: List[FrameData],
        sampled_frames: List[int],
        pca_components_data: List[Dict] = None,
        rmsd_per_frame: List[float] = None,
        incremental: bool = False,
    ) -> None:
        """Save individual frame metrics to CSV file, with energy/force info if available
        
        Args:
            incremental: If True, append to existing file instead of overwriting
        """
        single_analysis_dir = os.path.join(output_dir, "single_analysis_results")
        FileUtils.ensure_dir(single_analysis_dir)
        csv_path = os.path.join(single_analysis_dir, f"frame_metrics_{system_name}.csv")

        try:
            file_exists = os.path.exists(csv_path)
            write_mode = 'a' if incremental and file_exists else 'w'
            
            with open(csv_path, write_mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # 只有在创建新文件或非增量模式时写入表头
                if not file_exists or not incremental:
                    # 准备表头
                    headers = ["Frame_ID", "Selected"]
                    headers.append("RMSD")  # 基于构象均值的RMSD
                    # 能量补充信息（放在后面）
                    headers.append("Energy(eV)")
                    headers.append("Energy_Standardized")
                    if pca_components_data:
                        # 获取所有可能的PC列
                        max_pc = 0
                        for item in pca_components_data:
                            for key in item.keys():
                                if key.startswith('PC'):
                                    pc_num = int(key[2:])
                                    max_pc = max(max_pc, pc_num)
                        headers.extend([f"PC{i}" for i in range(1, max_pc + 1)])
                    writer.writerow(headers)

                sampled_set = set(sampled_frames)

                # 创建PCA数据查找字典
                pca_lookup = {}
                if pca_components_data:
                    for item in pca_components_data:
                        frame_id = item.get('frame')
                        if frame_id is not None:
                            pca_lookup[frame_id] = item

                for i, frame in enumerate(frames):
                    selected = 1 if frame.frame_id in sampled_set else 0
                    row = [frame.frame_id, selected]
                    # RMSD数据
                    rmsd_value = rmsd_per_frame[i] if rmsd_per_frame and i < len(rmsd_per_frame) else ""
                    row.append(f"{rmsd_value:.6f}" if isinstance(rmsd_value, (int, float)) else rmsd_value)
                    # 能量补充 - 直接使用FrameData中的信息
                    energy = frame.energy if frame.energy is not None else ""
                    energy_standardized = frame.energy_standardized if frame.energy_standardized is not None else ""
                    row.append(energy)
                    row.append(energy_standardized)
                    # 添加PCA分量（放在后面）
                    if pca_components_data and frame.frame_id in pca_lookup:
                        pca_item = pca_lookup[frame.frame_id]
                        for pc_num in range(1, max_pc + 1):
                            pc_key = f'PC{pc_num}'
                            pc_value = pca_item.get(pc_key, 0.0)
                            row.append(f"{pc_value:.6f}")
                    elif pca_components_data:
                        for pc_num in range(1, max_pc + 1):
                            row.append("0.000000")
                    writer.writerow(row)
                    
                # 确保数据写入磁盘
                f.flush()
                os.fsync(f.fileno())
                
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save frame metrics for {system_name}: {e}")
            raise

    # 旧增量与采样记录聚合函数已移除


    # ---- DeepMD Export Functionality (merged from deepmd_exporter.py) ----
    ALL_TYPE_MAP = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]

    @staticmethod
    def _build_frame_id_index(frames: Sequence) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for idx, f in enumerate(frames):
            fid = getattr(f, 'frame_id', None)
            if fid is not None and fid not in mapping:
                mapping[fid] = idx
        return mapping

    @staticmethod
    def export_sampled_frames_per_system(
        frames: Sequence,
        sampled_frame_ids: List[int],
        system_path: str,
        output_root: str,
        system_name: str,
        logger,
        force: bool = False,
    ) -> Optional[str]:
        if not sampled_frame_ids:
            logger.debug(f"[deepmd-export] {system_name} 无采样帧，跳过导出")
            return None
        target_dir = os.path.join(output_root, system_name)
        marker_file = os.path.join(target_dir, 'export.done')
        if os.path.isdir(target_dir) and os.path.exists(marker_file) and not force:
            logger.debug(f"[deepmd-export] {system_name} 已存在且未强制覆盖，跳过")
            return target_dir
        try:
            import dpdata  # type: ignore
            id2idx = ResultSaver._build_frame_id_index(frames)
            subset_indices = [id2idx[fid] for fid in sampled_frame_ids if fid in id2idx]
            if not subset_indices:
                logger.warning(f"[deepmd-export] {system_name} 采样帧索引映射为空，跳过")
                return None
            ls = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ResultSaver.ALL_TYPE_MAP)
            n_total = len(ls)
            valid_subset = [i for i in subset_indices if 0 <= i < n_total]
            if not valid_subset:
                logger.warning(f"[deepmd-export] {system_name} 有效帧子集为空，跳过")
                return None
            sub_ls = ls[valid_subset]
            os.makedirs(target_dir, exist_ok=True)
            sub_ls.to_deepmd_npy(target_dir)
            with open(marker_file, 'w', encoding='utf-8') as f:
                f.write(f"frames={len(valid_subset)}\n")
            logger.info(f"[deepmd-export] 导出 {system_name} deepmd npy 成功，帧数={len(valid_subset)} -> {target_dir}")
            return target_dir
        except Exception as e:
            logger.error(f"[deepmd-export] 导出 {system_name} 失败: {e}")
            return None

    # ---- 批量 DeepMD 导出 (多体系合并) ----
    @staticmethod
    def _dir_non_empty(path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    @staticmethod
    def get_md_parameters(system_path: str, logger):
        out_dir = os.path.join(system_path, 'OUT.ABACUS')
        input_file = os.path.join(system_path, 'INPUT')
        log_file = os.path.join(out_dir, 'running_md.log')
        md_dumpfreq = 1
        total_steps = 0
        try:
            if os.path.exists(input_file):
                with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        ls = line.strip()
                        if not ls or ls.startswith('#'):
                            continue
                        if 'md_dumpfreq' in ls.split('#')[0]:
                            parts = ls.split()
                            for i, p in enumerate(parts):
                                if p.lower() == 'md_dumpfreq' and i + 1 < len(parts):
                                    try:
                                        md_dumpfreq = int(parts[i+1])
                                    except ValueError:
                                        pass
                                    break
                            break
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        ls = line.strip()
                        if not ls or ls.startswith('#'):
                            continue
                        first = ls.split()[0]
                        if first.isdigit():
                            step_val = int(first)
                            if step_val > total_steps:
                                total_steps = step_val
        except Exception as e:
            logger.warning(f"Failed to parse MD parameters: {e}")
        return md_dumpfreq, total_steps

    @staticmethod
    def steps_to_frame_indices(sampled_steps, md_dumpfreq, n_frames, logger):
        if md_dumpfreq <= 0:
            md_dumpfreq = 1
        result = []
        for st in sampled_steps:
            frame_idx = st // md_dumpfreq
            if frame_idx >= n_frames:
                logger.warning(f"Step {st} -> frame index {frame_idx} out of range (max {n_frames-1}), using last frame")
                frame_idx = n_frames - 1
            elif frame_idx < 0:
                logger.warning(f"Step {st} -> negative frame index {frame_idx}, discarding")
                continue
            result.append(frame_idx)
        return result

    @staticmethod
    def export_sampled_frames_to_deepmd(run_dir: str, output_dir: str, split_ratio: List[float] = None,
                                       logger=None, force_reexport: bool = False, seed: int = 42) -> None:
        import json
        import dpdata
        if logger is None:
            from ..utils.logmanager import create_standard_logger
            logger = create_standard_logger(__name__, level=20)
        if ResultSaver._dir_non_empty(output_dir) and not force_reexport:
            logger.info(f"Output directory exists and is not empty, skipping export: {output_dir}")
            logger.info("To re-export, set force_reexport=True or clear the directory")
            return
        logger.info("Starting sampled frames export task")
        targets_path = os.path.join(run_dir, 'analysis_targets.json')
        if not os.path.exists(targets_path):
            logger.error(f"Targets file not found: {targets_path}")
            return
        with open(targets_path, 'r', encoding='utf-8') as f:
            targets = json.load(f)
        ms = dpdata.MultiSystems()
        total_sampled_frames = 0
        for mol in targets['molecules'].values():
            for sys_name, sys_info in mol['systems'].items():
                system_path = sys_info['system_path']
                sampled_frames = json.loads(sys_info['sampled_frames']) if isinstance(sys_info['sampled_frames'], str) else sys_info['sampled_frames']
                if not os.path.exists(system_path):
                    logger.warning(f"System path not found: {system_path}")
                    continue
                try:
                    logger.info(f"Processing {sys_name}, original sampled list length: {len(sampled_frames)}")
                    dd = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ResultSaver.ALL_TYPE_MAP)
                    n_frames = len(dd)
                    if n_frames == 0:
                        logger.warning(f"{sys_name} has no available frames, skipping")
                        continue
                    treat_as_steps = False
                    if sampled_frames:
                        max_val = max(sampled_frames)
                        if max_val >= n_frames:
                            treat_as_steps = True
                    if treat_as_steps:
                        md_dumpfreq, total_steps = ResultSaver.get_md_parameters(system_path, logger)
                        logger.info(f"Detected sampled list as steps, converting using md_dumpfreq={md_dumpfreq}")
                        sampled_frames_conv = ResultSaver.steps_to_frame_indices(sampled_frames, md_dumpfreq, n_frames, logger)
                    else:
                        sampled_frames_conv = sampled_frames
                    if (not treat_as_steps and sampled_frames_conv and min(sampled_frames_conv) >= 1
                            and max(sampled_frames_conv) == n_frames and 0 not in sampled_frames_conv):
                        logger.info(f"Detected possible 1-based frame indices, auto-correcting by -1 ({sys_name})")
                        sampled_frames_conv = [i - 1 for i in sampled_frames_conv]
                    valid_indices = [i for i in sampled_frames_conv if 0 <= i < n_frames]
                    invalid_count = len(sampled_frames_conv) - len(valid_indices)
                    if invalid_count > 0:
                        logger.warning(f"{sys_name} filtered out {invalid_count} out-of-bounds indices (valid frame range 0~{n_frames-1})")
                    if not valid_indices:
                        logger.warning(f"{sys_name} has no valid sampled frames, skipping")
                        continue
                    ordered_indices = list(dict.fromkeys(valid_indices))
                    sub_dd = dd[ordered_indices]
                    ms.append(sub_dd)
                    total_sampled_frames += len(ordered_indices)
                    logger.info(f"Successfully added {sys_name} frames: {len(ordered_indices)} (original list {len(sampled_frames)}, converted {len(sampled_frames_conv)})")
                except Exception as e:
                    logger.error(f"Failed to read {system_path}: {e}")
        logger.info(f"Total sampled frames: {total_sampled_frames}")
        logger.info(f"MultiSystems info: {ms}")
        os.makedirs(output_dir, exist_ok=True)
        ms.to_deepmd_npy(output_dir)
        logger.info(f"Exported DeepMD npy to {output_dir}")
        if split_ratio:
            ResultSaver.split_and_save(ms, output_dir, split_ratio, logger, seed=seed)

    @staticmethod
    def split_and_save(ms, output_dir: str, split_ratio: List[float], logger, seed: int = 42) -> None:
        import dpdata
        import numpy as np
        total_frames = ms.get_nframes()
        logger.info(f"Starting dataset split, total frames: {total_frames}")
        indices = np.arange(total_frames)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        split_points = np.cumsum([int(r * total_frames) for r in split_ratio[:-1]])
        splits = np.split(indices, split_points)
        for i, idx in enumerate(splits):
            sub_ms = dpdata.MultiSystems()
            frame_offset = 0
            for sys in ms.systems:
                sys_frames = len(sys)
                sys_indices = [j for j in idx if frame_offset <= j < frame_offset + sys_frames]
                if sys_indices:
                    local_indices = [j - frame_offset for j in sys_indices]
                    sub_sys = sys[local_indices]
                    sub_ms.append(sub_sys)
                frame_offset += sys_frames
            sub_dir = os.path.join(output_dir, f"split_{i}")
            os.makedirs(sub_dir, exist_ok=True)
            sub_ms.to_deepmd_npy(sub_dir)
            logger.info(f"Saved split subset {sub_dir}, frames: {len(idx)}")
