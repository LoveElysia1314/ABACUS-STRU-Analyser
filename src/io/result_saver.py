#!/usr/bin/env python

import csv
import logging
import os
from typing import Optional, List, Dict, Tuple, Sequence
import datetime

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
    def save_results(output_dir: str, analysis_results: List[Tuple], incremental: bool = False) -> None:
        """保存分析结果，包括系统汇总、单体系详细结果和PCA分量"""
        logger = logging.getLogger(__name__)
        try:
            # 提取数据
            all_metrics = []

            for result in analysis_results:
                metrics, frames, swap_count, improve_ratio, pca_components_data, pca_model, rmsd_per_frame = result
                all_metrics.append(metrics)


            if all_metrics:
                # 保存系统汇总
                if incremental:
                    ResultSaver.save_system_summary_incremental(output_dir, all_metrics)
                    # 续算也强制覆盖均值结构（保持与最新指标一致）
                    for m in all_metrics:
                        ResultSaver.export_mean_structure(output_dir, m, force_update=True)
                else:
                    # 完整保存逻辑 + 导出均值结构 JSON
                    ResultSaver._save_system_summary_complete(output_dir, all_metrics)
                    for m in all_metrics:
                        ResultSaver.export_mean_structure(output_dir, m)

                # 保存单体系详细结果
                for metrics, result in zip(all_metrics, analysis_results):
                    frames = result[1]
                    sampled_frames = [f.frame_id for f in frames if f.frame_id in metrics.sampled_frames]
                    pca_components_data = result[4]  # PCA分量数据
                    rmsd_per_frame = result[6]  # RMSD数据
                    ResultSaver.save_frame_metrics(output_dir, metrics.system_name, frames, sampled_frames, pca_components_data, rmsd_per_frame, incremental=incremental)

                # PCA分量已集成到单体系结果中，无需单独保存

        except Exception as e:
            ErrorHandler.log_detailed_error(
                logger, e, "保存分析结果失败",
                additional_info={
                    "输出目录": output_dir,
                    "结果数量": len(analysis_results) if analysis_results else 0,
                    "增量模式": incremental
                }
            )
            raise

    @staticmethod
    def _save_system_summary_complete(output_dir: str, new_metrics: List[TrajectoryMetrics]) -> None:
        """保存完整的系统汇总（非增量模式）"""
        try:
            combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
            FileUtils.ensure_dir(combined_analysis_dir)
            csv_path = os.path.join(combined_analysis_dir, "system_metrics_summary.csv")

            all_rows = []
            for m in new_metrics:
                row = ResultSaver._format_metric_row(m)
                all_rows.append(row)

            # 排序
            def sort_key(row):
                try:
                    system_name = row[0]
                    import re
                    match = re.match(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name)
                    if match:
                        mol_id, conf, temp = match.groups()
                        return (int(mol_id), int(conf), int(temp))
                except Exception:
                    pass
                return (9999, 9999, 9999)

            all_rows.sort(key=sort_key)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(ResultSaver.SYSTEM_SUMMARY_HEADERS)
                writer.writerows(all_rows)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save complete system summary: {e}")
            raise

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

    @staticmethod
    def save_system_summary_incremental(
        output_dir: str, new_metrics: List[TrajectoryMetrics]
    ) -> None:
        """Save system summary with incremental support - append new rows without rewriting entire file"""
        try:
            combined_analysis_dir = os.path.join(
                output_dir, "combined_analysis_results"
            )
            FileUtils.ensure_dir(combined_analysis_dir)
            csv_path = os.path.join(combined_analysis_dir, "system_metrics_summary.csv")
            progress_path = os.path.join(combined_analysis_dir, "progress.json")

            # 读取已处理的系统
            processed_systems = set()
            if os.path.exists(progress_path):
                try:
                    import json
                    with open(progress_path, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                        processed_systems = set(progress_data.get('processed_systems', []))
                except Exception:
                    processed_systems = set()

            # 过滤出新的系统
            new_system_metrics = [m for m in new_metrics if m.system_name not in processed_systems]
            
            if not new_system_metrics:
                logger = logging.getLogger(__name__)
                logger.info("所有系统已处理完毕，无需更新")
                return

            # 检查文件是否存在
            file_exists = os.path.exists(csv_path)
            write_mode = 'a' if file_exists else 'w'

            with open(csv_path, write_mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # 只有在创建新文件时写入表头
                if not file_exists:
                    writer.writerow(ResultSaver.SYSTEM_SUMMARY_HEADERS)

                # 追加新数据
                for m in new_system_metrics:
                    row = ResultSaver._format_metric_row(m)
                    writer.writerow(row)
                    
                # 确保数据写入磁盘
                f.flush()
                os.fsync(f.fileno())

            # 更新进度文件
            processed_systems.update(m.system_name for m in new_system_metrics)
            progress_data = {
                'processed_systems': list(processed_systems),
                'last_updated': str(datetime.datetime.utcnow().isoformat() + "Z")
            }
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

            logger = logging.getLogger(__name__)
            logger.info(f"增量保存了 {len(new_system_metrics)} 个新系统的汇总数据")

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save system summary incrementally: {e}")
            raise

    @staticmethod
    def save_sampling_records(output_dir: str) -> None:
        """保存所有体系的采样记录到CSV文件

        从single_analysis_results目录中读取各个系统的frame_metrics_*.csv文件，
        提取被选择的帧号（Selected=1），并保存到combined_analysis_results目录中。
        """
        logger = logging.getLogger(__name__)

        try:
            # 确定输入和输出路径
            single_analysis_dir = os.path.join(output_dir, "single_analysis_results")
            combined_analysis_dir = os.path.join(output_dir, "combined_analysis_results")
            FileUtils.ensure_dir(combined_analysis_dir)
            
            # 采样记录保存到output_dir目录（combined_analysis_results的父目录）
            sampling_records_path = os.path.join(output_dir, "sampling_records.csv")

            # 检查single_analysis_results目录是否存在
            if not os.path.exists(single_analysis_dir):
                logger.warning(f"single_analysis_results目录不存在: {single_analysis_dir}")
                return

            # 查找所有frame_metrics_*.csv文件
            frame_metrics_files = []
            for filename in os.listdir(single_analysis_dir):
                if filename.startswith("frame_metrics_") and filename.endswith(".csv"):
                    frame_metrics_files.append(os.path.join(single_analysis_dir, filename))

            if not frame_metrics_files:
                logger.warning(f"在{single_analysis_dir}中未找到任何frame_metrics_*.csv文件")
                return


            # 读取 system_name 到 system_path 的映射表
            import json
            system_map_path = os.path.join(output_dir, "system_paths_map.json")
            system_name_to_path = {}
            if os.path.exists(system_map_path):
                try:
                    with open(system_map_path, "r", encoding="utf-8") as f:
                        system_name_to_path = json.load(f)
                except Exception as e:
                    logger.warning(f"读取 system_paths_map.json 失败: {e}")

            all_sampling_records = []
            for csv_file_path in frame_metrics_files:
                try:
                    filename = os.path.basename(csv_file_path)
                    system_name = filename.replace("frame_metrics_", "").replace(".csv", "")
                    # 优先用映射表
                    system_path = system_name_to_path.get(system_name, "")
                    sampled_frames = []
                    with open(csv_file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        # 读取CSV数据
                        csv_content = "\n".join(lines)
                        reader = csv.DictReader(csv_content.split('\n'))
                        for row in reader:
                            if not row or not any(row.values()):
                                continue
                            selected = row.get("Selected", "0").strip()
                            if selected == "1":
                                frame_id = row.get("Frame_ID", "").strip()
                                if frame_id:
                                    try:
                                        sampled_frames.append(int(frame_id))
                                    except ValueError:
                                        logger.warning(f"无效的Frame_ID: {frame_id} 在文件 {filename}")
                    sampled_frames.sort()
                    if sampled_frames:
                        sampled_frames_str = f"[{','.join(map(str, sampled_frames))}]"
                    else:
                        sampled_frames_str = "[]"
                    all_sampling_records.append([system_name, system_path, sampled_frames_str])
                except Exception as e:
                    logger.error(f"处理文件 {csv_file_path} 时出错: {str(e)}")
                    continue

            if not all_sampling_records:
                logger.warning("未找到任何有效的采样记录")
                return

            # 按照system_metrics_summary的排序逻辑对记录进行排序
            def sort_key(record):
                try:
                    system_name = record[0]
                    import re
                    match = re.match(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name)
                    if match:
                        mol_id, conf, temp = match.groups()
                        return (int(mol_id), int(conf), int(temp))
                except Exception:
                    pass
                return (9999, 9999, 9999)

            all_sampling_records.sort(key=sort_key)

            # 保存到CSV文件
            with open(sampling_records_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(ResultSaver.SAMPLING_RECORDS_HEADERS)
                writer.writerows(all_sampling_records)

            logger.info(f"采样记录已保存到: {sampling_records_path}")
            logger.info(f"共处理了 {len(all_sampling_records)} 个系统")

        except Exception as e:
            ErrorHandler.log_detailed_error(
                logger, e, "保存采样记录失败",
                additional_info={"输出目录": output_dir}
            )
            raise

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
        """
        Export sampled frames for a single system to DeepMD format.

        Args:
            frames: Sequence of frame objects
            sampled_frame_ids: List of sampled frame IDs
            system_path: Path to the ABACUS system directory
            output_root: Root output directory
            system_name: Name of the system
            logger: Logger instance
            force: Whether to force re-export

        Returns:
            Path to the exported directory or None if failed
        """
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

            ls = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ALL_TYPE_MAP)
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

    @staticmethod
    def _build_frame_id_index(frames: Sequence) -> Dict[int, int]:
        """Build mapping from frame_id to frame index"""
        mapping: Dict[int, int] = {}
        for idx, f in enumerate(frames):
            fid = getattr(f, 'frame_id', None)
            if fid is not None and fid not in mapping:
                mapping[fid] = idx
        return mapping


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
    """Build mapping from frame_id to frame index"""
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
    """
    Export sampled frames for a single system to DeepMD format.

    Args:
        frames: Sequence of frame objects
        sampled_frame_ids: List of sampled frame IDs
        system_path: Path to the ABACUS system directory
        output_root: Root output directory
        system_name: Name of the system
        logger: Logger instance
        force: Whether to force re-export

    Returns:
        Path to the exported directory or None if failed
    """
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

        ls = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ALL_TYPE_MAP)
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


# ---- Batch DeepMD Export Functionality (merged from sampled_frames_to_deepmd.py) ----

@staticmethod
def _dir_non_empty(path: str) -> bool:
    """Check if directory exists and is not empty"""
    return os.path.isdir(path) and any(os.scandir(path))

@staticmethod
def get_md_parameters(system_path: str, logger):
    """
    Extract md_dumpfreq and total_steps from ABACUS output directory.

    Expected directory structure: <system_path>/OUT.ABACUS/
    1. Read INPUT for md_dumpfreq (default to 1 if not found)
    2. Parse running_md.log for maximum step number
    Returns (md_dumpfreq, total_steps)
    """
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
                    # Log may start with step
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
    """Convert MD steps to frame indices"""
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
    """
    Export sampled frames from analysis_targets.json to DeepMD format with optional splitting.

    Args:
        run_dir: Analysis results directory containing analysis_targets.json
        output_dir: Output directory for DeepMD npy files
        split_ratio: List of split ratios, e.g., [0.8, 0.2]
        logger: Optional logger instance
        force_reexport: Whether to force re-export (ignore existing files)
        seed: Random seed for splitting
    """
    import json
    import dpdata

    if logger is None:
        from ..utils.logmanager import create_standard_logger
        logger = create_standard_logger(__name__, level=20)

    # Skip if directory exists and not forcing
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
                dd = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ALL_TYPE_MAP)
                n_frames = len(dd)

                if n_frames == 0:
                    logger.warning(f"{sys_name} has no available frames, skipping")
                    continue

                # Determine if sampled_frames looks like "steps" rather than "frame indices"
                treat_as_steps = False
                if sampled_frames:
                    max_val = max(sampled_frames)
                    min_val = min(sampled_frames)
                    if max_val >= n_frames:  # Obviously exceeds frame count => treat as steps
                        treat_as_steps = True

                if treat_as_steps:
                    md_dumpfreq, total_steps = ResultSaver.get_md_parameters(system_path, logger)
                    logger.info(f"Detected sampled list as steps, converting using md_dumpfreq={md_dumpfreq} (max step {max_val}, parsed max actual step {total_steps})")
                    sampled_frames_conv = ResultSaver.steps_to_frame_indices(sampled_frames, md_dumpfreq, n_frames, logger)
                else:
                    sampled_frames_conv = sampled_frames

                # Possible 1-based -> 0-based correction (heuristic, only for non-step cases)
                if (not treat_as_steps and sampled_frames_conv and min(sampled_frames_conv) >= 1
                        and max(sampled_frames_conv) == n_frames and 0 not in sampled_frames_conv):
                    logger.info(f"Detected possible 1-based frame indices, auto-correcting by -1 ({sys_name})")
                    sampled_frames_conv = [i - 1 for i in sampled_frames_conv]

                # Filter out-of-bounds indices
                valid_indices = [i for i in sampled_frames_conv if 0 <= i < n_frames]
                invalid_count = len(sampled_frames_conv) - len(valid_indices)

                if invalid_count > 0:
                    logger.warning(f"{sys_name} filtered out {invalid_count} out-of-bounds indices (valid frame range 0~{n_frames-1})")

                if not valid_indices:
                    logger.warning(f"{sys_name} has no valid sampled frames, skipping")
                    continue

                # Remove duplicates while preserving order
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
    """Split and save MultiSystems dataset"""
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
            sys_frames = len(sys)  # Use len() instead of get_nframes()
            sys_indices = [j for j in idx if frame_offset <= j < frame_offset + sys_frames]

            if sys_indices:
                local_indices = [j - frame_offset for j in sys_indices]
                # Correction: use indices instead of subset method
                sub_sys = sys[local_indices]
                sub_ms.append(sub_sys)

            frame_offset += sys_frames

        sub_dir = os.path.join(output_dir, f"split_{i}")
        os.makedirs(sub_dir, exist_ok=True)
        sub_ms.to_deepmd_npy(sub_dir)
        logger.info(f"Saved split subset {sub_dir}, frames: {len(idx)}")
