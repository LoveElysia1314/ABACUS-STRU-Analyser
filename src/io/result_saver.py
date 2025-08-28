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

    SAMPLING_RECORDS_HEADERS = ["System", "OUT_ABACUS_Path", "Sampled_Frames"]

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
                        ResultSaver.save_frame_metrics(output_dir, metrics.system_name, frames, sampled_frames, pca_components_data)
                    elif len(result) >= 4:  # 兼容旧格式
                        frames = result[1]
                        sampled_frames = [f.frame_id for f in frames if f.frame_id in metrics.sampled_frames]
                        ResultSaver.save_frame_metrics(output_dir, metrics.system_name, frames, sampled_frames)

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
    ) -> None:
        """Save individual frame metrics to CSV file"""
        single_analysis_dir = os.path.join(output_dir, "single_analysis_results")
        FileUtils.ensure_dir(single_analysis_dir)
        csv_path = os.path.join(single_analysis_dir, f"frame_metrics_{system_name}.csv")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # 准备表头
                headers = ["Frame_ID", "Selected"]
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

                    # 添加PCA分量
                    if pca_components_data and frame.frame_id in pca_lookup:
                        pca_item = pca_lookup[frame.frame_id]
                        for pc_num in range(1, max_pc + 1):
                            pc_key = f'PC{pc_num}'
                            pc_value = pca_item.get(pc_key, 0.0)
                            row.append(f"{pc_value:.6f}")
                    elif pca_components_data:
                        # 如果没有PCA数据，填充空值
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
