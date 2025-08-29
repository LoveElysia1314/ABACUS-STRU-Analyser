#!/usr/bin/env python

import csv
import logging
import os
from typing import List, Dict, Tuple

import numpy as np

from ..core.metrics import TrajectoryMetrics
from ..io.stru_parser import FrameData
from ..utils import FileUtils
from ..utils.data_utils import ErrorHandler


class ResultSaver:
    """结果保存器类，负责保存分析结果到CSV文件"""

    # CSV Headers constants
    SYSTEM_SUMMARY_HEADERS = [
        "System",
        "Molecule_ID",
        "Configuration",
        "Temperature(K)",
        "Num_Frames",
        "Dimension",
        "MinD",
        "ANND",
        "MPD",
        "MinD_sampled",
        "ANND_sampled",
        "MPD_sampled",
        "MinD_ratio",
        "ANND_ratio",
        "MPD_ratio",
        "PCA_Variance_Ratio",
        "PCA_Cumulative_Variance_Ratio",
        "PCA_Num_Components_Retained",
        "PCA_Variance_Ratios",  # 各主成分方差贡献率（JSON格式）
    ]

    SAMPLING_RECORDS_HEADERS = ["System", "System_Path", "Sampled_Frames"]  # Sampled_Frames格式: [1,5,10,15,20]

    @staticmethod
    def save_results(output_dir: str, analysis_results: List[Tuple], incremental: bool = False) -> None:
        """保存分析结果，包括系统汇总、单体系详细结果和PCA分量"""
        logger = logging.getLogger(__name__)
        try:
            # 提取数据
            all_metrics = []

            for result in analysis_results:
                if len(result) >= 6:  # 包含PCA数据的完整结果
                    metrics, frames, swap_count, improve_ratio, pca_components_data, pca_model = result
                    all_metrics.append(metrics)
                elif len(result) >= 4:  # 兼容旧格式
                    metrics, frames, swap_count, improve_ratio = result
                    all_metrics.append(metrics)

            if all_metrics:
                # 保存系统汇总
                if incremental:
                    ResultSaver.save_system_summary_incremental(output_dir, all_metrics)
                else:
                    # 重新实现完整保存逻辑
                    ResultSaver._save_system_summary_complete(output_dir, all_metrics)

                # 保存单体系详细结果
                for metrics, result in zip(all_metrics, analysis_results):
                    if len(result) >= 6:  # 包含PCA数据的完整结果
                        frames = result[1]
                        sampled_frames = [f.frame_id for f in frames if f.frame_id in metrics.sampled_frames]
                        pca_components_data = result[4]  # PCA分量数据
                        ResultSaver.save_frame_metrics(output_dir, metrics.system_name, frames, sampled_frames, pca_components_data, metrics.system_path)
                    elif len(result) >= 4:  # 兼容旧格式
                        frames = result[1]
                        sampled_frames = [f.frame_id for f in frames if f.frame_id in metrics.sampled_frames]
                        ResultSaver.save_frame_metrics(output_dir, metrics.system_name, frames, sampled_frames, system_path=metrics.system_path)

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
                ratios = m.get_ratio_metrics()
                row = [
                    m.system_name,
                    m.mol_id,
                    m.conf,
                    m.temperature,
                    m.num_frames,
                    m.dimension,
                    f"{m.MinD:.6f}",
                    f"{m.ANND:.6f}",
                    f"{m.MPD:.6f}",
                    f"{m.MinD_sampled:.6f}",
                    f"{m.ANND_sampled:.6f}",
                    f"{m.MPD_sampled:.6f}",
                    f"{ratios['MinD_ratio']:.6f}",
                    f"{ratios['ANND_ratio']:.6f}",
                    f"{ratios['MPD_ratio']:.6f}",
                ]

                # 添加PCA方差贡献率信息
                import json
                pca_variance_ratios_str = json.dumps(m.pca_explained_variance_ratio, ensure_ascii=False)

                # 插入保留主成分数量，使列顺序与 SYSTEM_SUMMARY_HEADERS 保持一致
                row.extend([
                    f"{m.pca_variance_ratio:.6f}",
                    f"{m.pca_cumulative_variance_ratio:.6f}",
                    f"{int(m.pca_components)}",
                    pca_variance_ratios_str,
                ])

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
        system_path: str = "",
    ) -> None:
        """Save individual frame metrics to CSV file, with energy/force info if available"""
        single_analysis_dir = os.path.join(output_dir, "single_analysis_results")
        FileUtils.ensure_dir(single_analysis_dir)
        csv_path = os.path.join(single_analysis_dir, f"frame_metrics_{system_name}.csv")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 准备表头
                headers = ["Frame_ID", "Selected"]
                # 能量补充信息（放在前面）
                headers.append("Energy(eV)")
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
                    # 能量补充 - 直接使用FrameData中的信息
                    energy = frame.energy if frame.energy is not None else ""
                    row.append(energy)
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
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save frame metrics for {system_name}: {e}")
            raise

    @staticmethod
    def save_system_summary_incremental(
        output_dir: str, new_metrics: List[TrajectoryMetrics]
    ) -> None:
        """Save system summary with incremental support"""
        try:
            combined_analysis_dir = os.path.join(
                output_dir, "combined_analysis_results"
            )
            FileUtils.ensure_dir(combined_analysis_dir)
            csv_path = os.path.join(combined_analysis_dir, "system_metrics_summary.csv")

            all_rows = []

            # Read existing data if file exists
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        new_system_names = {m.system_name for m in new_metrics}
                        for row in reader:
                            if row["System"] not in new_system_names:
                                all_rows.append(
                                    [
                                        row.get(h, "")
                                        for h in ResultSaver.SYSTEM_SUMMARY_HEADERS
                                    ]
                                )
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to read existing CSV file {csv_path}: {e}")

            # Add new data
            for m in new_metrics:
                ratios = m.get_ratio_metrics()
                # 添加PCA方差贡献率信息
                import json
                pca_variance_ratios_str = json.dumps(m.pca_explained_variance_ratio, ensure_ascii=False)

                # Ensure ordering matches SYSTEM_SUMMARY_HEADERS
                all_rows.append(
                    [
                        m.system_name,
                        m.mol_id,
                        m.conf,
                        m.temperature,
                        m.num_frames,
                        m.dimension,
                        f"{m.MinD:.6f}",
                        f"{m.ANND:.6f}",
                        f"{m.MPD:.6f}",
                        f"{m.MinD_sampled:.6f}",
                        f"{m.ANND_sampled:.6f}",
                        f"{m.MPD_sampled:.6f}",
                        f"{ratios['MinD_ratio']:.6f}",
                        f"{ratios['ANND_ratio']:.6f}",
                        f"{ratios['MPD_ratio']:.6f}",
                        f"{m.pca_variance_ratio:.6f}",
                        f"{m.pca_cumulative_variance_ratio:.6f}",
                        f"{int(m.pca_components)}",
                        pca_variance_ratios_str,
                    ]
                )

            # Sort and write data
            def sort_key(row):
                try:
                    system_name = row[0]
                    import re

                    match = re.match(
                        r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K", system_name
                    )
                    if match:
                        mol_id, conf, temp = match.groups()
                        return (int(mol_id), int(conf), int(temp))
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to parse system name {system_name}: {e}")
                return (9999, 9999, 9999)

            all_rows.sort(key=sort_key)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(ResultSaver.SYSTEM_SUMMARY_HEADERS)
                writer.writerows(all_rows)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save system summary: {e}")
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
