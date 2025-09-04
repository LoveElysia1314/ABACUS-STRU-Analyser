#!/usr/bin/env python
"""
脚本名: trajectory_analyser.py
功能: ABACUS 轨迹分析器
==================================================

功能：
-----
批量分析 ABACUS 分子动力学模拟生成的 STRU 文件序列，
基于原子间距离向量的标准化分布特性及推荐指标，
评估构象多样性。

核心指标（更新版）：
------------------
1. ConfVol: 构象空间体积（核心多样性指标）
2. ANND: 平均最近邻距离（平均最近邻距离）
3. MPD: 平均成对距离（平均成对距离）
4. PCA explained variance ratios: 主成分方差贡献率

所有输出均为无量纲标准化指标，适用于跨体系比较。

输入结构：
---------
当前目录下包含多个体系文件夹：
    struct_mol_<ID>_conf_<N>_T<T>K/
    └── OUT.ABACUS/
        └── STRU/
            ├── STRU_MD_0
            ├── STRU_MD_1
            └── ...

输出结构：
---------
analysis_results/
├── struct_mol_<ID>/
│   ├── standardized_metrics_per_frame_<system>.csv    ← 单帧指标 + 采样结果
│   ├── standardized_metrics_<system>.json             ← 聚合指标（无每帧）
│   └── ...（合并分析同理）
└── standardized_distribution_summary.csv               ← 所有体系汇总对比

使用方式：
----------
python trajectory_analyser.py [--include_h] [--max_workers N] [--sample_ratio R] [--sample_count N]

依赖：
------
numpy, scipy, stru_parser.py
"""

import logging

import numpy as np
from scipy.spatial.distance import cdist, pdist

# Level 3: 引入结构指标统一模块（供外部使用）
try:  # 软依赖，避免循环导入风险
    from ..core.system_analyser import RMSDCalculator
    kabsch_align = RMSDCalculator.kabsch_align
    iterative_mean_structure = RMSDCalculator.iterative_mean_structure
    compute_rmsf = RMSDCalculator.compute_rmsf
except Exception:  # noqa: BLE001 - 宽松捕获，仅降级功能
    kabsch_align = None  # type: ignore
    iterative_mean_structure = None  # type: ignore
    compute_rmsf = None  # type: ignore

# 导入解析器模块

# 注意：日志配置现在由主程序管理，模块只获取logger
logger = logging.getLogger(__name__)


# --- 核心数据结构 ---

# --- 采样函数 ---
# 说明：原本此处存在 greedy_max_avg_distance 的本地实现，现已统一迁移/整合至 `core.sampler.GreedyMaxDistanceSampler`。
# 如需最大距离贪婪采样，请在上层调用处使用：
# from ..core.sampler import GreedyMaxDistanceSampler
# selected_indices, _, _ = GreedyMaxDistanceSampler.select_frames(points, k)


# --- 核心分析逻辑 ---
# 说明：RMSD/RMSF/Kabsch/迭代均值结构函数已迁移至 core.system_analyser.RMSDCalculator
# 保留兼容导入（文件顶部的 try 导入），避免重复实现。


# ---- Force Energy Parser (merged from force_energy_parser.py) ----

import os
import re
import csv
from typing import List, Tuple, Dict


def parse_running_md_log(log_path: str) -> Tuple[Dict[int, float], Dict[int, List[Tuple[str, float, float, float]]]]:
    """Parse ABACUS running_md.log file to extract energies and forces.

    Args:
        log_path: Path to the running_md.log file

    Returns:
        Tuple of (energies_dict, forces_dict) where:
        - energies_dict: frame_id -> energy_value
        - forces_dict: frame_id -> list of (atom_label, fx, fy, fz)
    """
    energies = {}
    forces = {}
    current_frame = None
    current_forces = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析帧号
            if 'STEP OF MOLECULAR DYNAMICS' in line:
                match = re.search(r'STEP OF MOLECULAR DYNAMICS\s*:\s*(\d+)', line)
                if match:
                    current_frame = int(match.group(1))
                    current_forces = []
            # 解析能量
            elif 'final etot' in line and current_frame is not None:
                match = re.search(r'final etot is ([\-\d\.Ee]+) eV', line)
                if match:
                    energies[current_frame] = float(match.group(1))
            # 解析力
            elif 'TOTAL-FORCE' in line and current_frame is not None:
                current_forces = []
                next(f)  # 跳过分隔线
                for force_line in f:
                    if '-' * 10 in force_line or 'TOTAL-STRESS' in force_line:
                        break
                    parts = force_line.split()
                    if len(parts) == 4:
                        atom_label = parts[0]
                        fx, fy, fz = map(float, parts[1:])
                        current_forces.append((atom_label, fx, fy, fz))
                if current_forces:
                    forces[current_frame] = current_forces
    return energies, forces


def save_forces_to_csv(energies: Dict[int, float], forces: Dict[int, List[Tuple[str, float, float, float]]], output_dir: str):
    """Save parsed energies and forces to CSV files.

    Args:
        energies: Dictionary of frame_id -> energy
        forces: Dictionary of frame_id -> list of force tuples
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    for frame_id, energy in energies.items():
        csv_path = os.path.join(output_dir, f'frame_{frame_id}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['atom', 'fx(eV/Angstrom)', 'fy(eV/Angstrom)', 'fz(eV/Angstrom)'])
            if frame_id in forces:
                for atom in forces[frame_id]:
                    writer.writerow([atom[0], atom[1], atom[2], atom[3]])
            writer.writerow([])
            writer.writerow(['final_etot(eV)', energy])


def main():
    """Main function for standalone execution."""
    log_file = os.path.join('OUT.ABACUS', 'running_md.log')
    output_dir = os.path.join('analysis_results', 'single_force_results')
    energies, forces = parse_running_md_log(log_file)
    save_forces_to_csv(energies, forces, output_dir)
    print(f'已将每帧能量和力保存到 {output_dir} 文件夹下的单帧csv文件中。')


if __name__ == '__main__':
    main()
