"""
根据 analysis_targets.json，自动读取每个体系的 system_path 和 sampled_frames，
从 OUT.ABACUS 目录提取采样帧，导出 deepmd/npy 数据集，并支持拆分。
用法：
    python abacus_sampled_frames_to_deepmd.py --run_dir <分析结果目录> --output_dir <输出npy目录> [--split_ratio 0.8 0.2]

依赖：dpdata, numpy, pandas, tqdm
"""
import os
import argparse
import json
import dpdata
from tqdm import tqdm
import numpy as np
from pathlib import Path
from src.logging import create_standard_logger

def parse_args():
    parser = argparse.ArgumentParser(description="根据analysis_targets.json采样帧导出deepmd数据集并可拆分")
    parser.add_argument('--run_dir', type=str, default='analysis_results/run_r0.05_p-0.5_v0.9', help='分析结果目录，包含analysis_targets.json')
    parser.add_argument('--output_dir', type=str, default='output_npy', help='输出deepmd npy目录')
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[0.8, 0.2], help='拆分比例，如0.8 0.2')
    return parser.parse_args()

ALL_TYPE_MAP = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", 
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

def export_sampled_frames_to_deepmd(run_dir, output_dir, split_ratio=None, logger=None):
    """
    从 analysis_targets.json 导出采样帧为 deepmd npy，并可拆分。
    run_dir: 分析结果目录（含 analysis_targets.json）
    output_dir: 输出 deepmd npy 目录
    split_ratio: 拆分比例列表，如 [0.8, 0.2]
    logger: 可选日志器
    """
    if logger is None:
        logger = create_standard_logger(__name__, level=20)
    logger.info("开始处理采样帧导出任务")
    targets_path = os.path.join(run_dir, 'analysis_targets.json')
    if not os.path.exists(targets_path):
        logger.error(f"未找到目标文件: {targets_path}")
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
                logger.warning(f"未找到系统路径: {system_path}")
                continue
            try:
                logger.info(f"正在处理 {sys_name}，采样帧数: {len(sampled_frames)}")
                dd = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ALL_TYPE_MAP)
                sub_dd = dd.subset(sampled_frames)
                ms.append(sub_dd)
                total_sampled_frames += len(sampled_frames)
                logger.info(f"成功添加 {sys_name} 采样帧")
            except Exception as e:
                logger.error(f"读取 {system_path} 失败: {e}")
    logger.info(f"总采样帧数: {total_sampled_frames}")
    logger.info(f"MultiSystems 信息: {ms}")
    os.makedirs(output_dir, exist_ok=True)
    ms.to_deepmd_npy(output_dir)
    logger.info(f"已导出deepmd npy至 {output_dir}")
    if split_ratio:
        split_and_save(ms, output_dir, split_ratio, logger)

def split_and_save(ms, output_dir, split_ratio, logger):
    total_frames = ms.get_nframes()
    logger.info(f"开始拆分数据集，总帧数: {total_frames}")
    indices = np.arange(total_frames)
    np.random.shuffle(indices)
    split_points = np.cumsum([int(r * total_frames) for r in split_ratio[:-1]])
    splits = np.split(indices, split_points)
    for i, idx in enumerate(splits):
        sub_ms = dpdata.MultiSystems()
        frame_offset = 0
        for sys in ms.systems:
            sys_frames = sys.get_nframes()
            sys_indices = [j for j in idx if frame_offset <= j < frame_offset + sys_frames]
            if sys_indices:
                local_indices = [j - frame_offset for j in sys_indices]
                sub_sys = sys.subset(local_indices)
                sub_ms.append(sub_sys)
            frame_offset += sys_frames
        sub_dir = os.path.join(output_dir, f"split_{i}")
        os.makedirs(sub_dir, exist_ok=True)
        sub_ms.to_deepmd_npy(sub_dir)
        logger.info(f"已保存拆分子集 {sub_dir}，帧数: {len(idx)}")

def main():
    args = parse_args()
    export_sampled_frames_to_deepmd(args.run_dir, args.output_dir, args.split_ratio)

if __name__ == "__main__":
    main()
