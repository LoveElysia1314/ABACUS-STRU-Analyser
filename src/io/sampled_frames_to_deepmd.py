#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 analysis_targets.json 自动读取每个体系的 system_path 与 sampled_frames（可能是帧索引或 MD 步数），
从 ABACUS 输出目录提取采样帧，导出 deepmd/npy 数据集，可按比例拆分。

依赖（本文件直接使用）：dpdata, numpy
"""
import os
import argparse
import json
import dpdata
import numpy as np
from src.logmanager import create_standard_logger


def _dir_non_empty(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))

def parse_args():
    parser = argparse.ArgumentParser(description="根据analysis_targets.json采样帧导出deepmd数据集并可拆分")
    parser.add_argument('--run_dir', type=str, default='analysis_results/run_r0.1_p-0.5_v0.9', help='分析结果目录，包含analysis_targets.json')
    parser.add_argument('--output_dir', type=str, default='analysis_results/run_r0.1_p-0.5_v0.9/deepmd_npy', help='输出deepmd npy目录')
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[0.8, 0.2], help='拆分比例，如0.8 0.2')
    parser.add_argument('--force', action='store_true', help='强制重新导出（忽略已有目录内容）')
    parser.add_argument('--seed', type=int, default=42, help='随机拆分种子，保证可复现')
    return parser.parse_args()

ALL_TYPE_MAP = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", 
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

def get_md_parameters(system_path: str, logger):
    """从 ABACUS 输出目录中提取 md_dumpfreq 与估计的总步数。

    期望目录结构: <system_path>/OUT.ABACUS/
    1. 读取 INPUT 中的 md_dumpfreq（若缺省则为1）
    2. 解析 running_md.log 的第一列（步数）取最大值作为 total_steps
    返回 (md_dumpfreq, total_steps)
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
                    # 运行日志可能以 step 开头
                    first = ls.split()[0]
                    if first.isdigit():
                        step_val = int(first)
                        if step_val > total_steps:
                            total_steps = step_val
    except Exception as e:
        logger.warning(f"解析 MD 参数失败: {e}")
    return md_dumpfreq, total_steps


def steps_to_frame_indices(sampled_steps, md_dumpfreq, n_frames, logger):
    if md_dumpfreq <= 0:
        md_dumpfreq = 1
    result = []
    for st in sampled_steps:
        frame_idx = st // md_dumpfreq
        if frame_idx >= n_frames:
            logger.warning(f"步数 {st} -> 帧索引 {frame_idx} 超出范围(最大 {n_frames-1})，使用最后一帧")
            frame_idx = n_frames - 1
        elif frame_idx < 0:
            logger.warning(f"步数 {st} -> 负帧索引 {frame_idx}，丢弃")
            continue
        result.append(frame_idx)
    return result


def export_sampled_frames_to_deepmd(run_dir, output_dir, split_ratio=None, logger=None, force_reexport=False, seed: int = 42):
    """
    从 analysis_targets.json 导出采样帧为 deepmd npy，并可拆分。
    run_dir: 分析结果目录（含 analysis_targets.json）
    output_dir: 输出 deepmd npy 目录
    split_ratio: 拆分比例列表，如 [0.8, 0.2]
    logger: 可选日志器
    force_reexport: 是否强制重新导出（忽略已存在的文件）
    """
    if logger is None:
        logger = create_standard_logger(__name__, level=20)
    
    # 仅目录非空且未强制时跳过
    if _dir_non_empty(output_dir) and not force_reexport:
        logger.info(f"输出目录已存在且非空，跳过导出: {output_dir}")
        logger.info("如需重新导出，请设置 force_reexport=True 或清空该目录")
        return
    
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
                logger.info(f"正在处理 {sys_name}，原始采样列表长度: {len(sampled_frames)}")
                dd = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ALL_TYPE_MAP)
                n_frames = len(dd)
                if n_frames == 0:
                    logger.warning(f"{sys_name} 无可用帧，跳过")
                    continue

                # 判断采样列表是否看似为“步数”而非“帧索引”
                treat_as_steps = False
                if sampled_frames:
                    max_val = max(sampled_frames)
                    min_val = min(sampled_frames)
                    if max_val >= n_frames:  # 明显超过帧数上限 => 视为步数
                        treat_as_steps = True
                if treat_as_steps:
                    md_dumpfreq, total_steps = get_md_parameters(system_path, logger)
                    logger.info(f"检测到采样列表疑似为步数，将按 md_dumpfreq={md_dumpfreq} 转换帧索引 (最大步 {max_val}，解析到最大实际步 {total_steps})")
                    sampled_frames_conv = steps_to_frame_indices(sampled_frames, md_dumpfreq, n_frames, logger)
                else:
                    sampled_frames_conv = sampled_frames

                # 可能的 1-based -> 0-based 纠正（启发式，仅在非步数情况下考虑）
                if (not treat_as_steps and sampled_frames_conv and min(sampled_frames_conv) >= 1 
                        and max(sampled_frames_conv) == n_frames and 0 not in sampled_frames_conv):
                    logger.info(f"检测到可能的1-based帧索引，自动减1处理 ({sys_name})")
                    sampled_frames_conv = [i - 1 for i in sampled_frames_conv]

                # 过滤越界索引
                valid_indices = [i for i in sampled_frames_conv if 0 <= i < n_frames]
                invalid_count = len(sampled_frames_conv) - len(valid_indices)
                if invalid_count > 0:
                    logger.warning(f"{sys_name} 过滤掉 {invalid_count} 个越界索引 (有效帧范围 0~{n_frames-1})")
                if not valid_indices:
                    logger.warning(f"{sys_name} 无有效采样帧，跳过")
                    continue
                # 去重保持顺序
                ordered_indices = list(dict.fromkeys(valid_indices))
                sub_dd = dd[ordered_indices]
                ms.append(sub_dd)
                total_sampled_frames += len(ordered_indices)
                logger.info(f"成功添加 {sys_name} 帧: {len(ordered_indices)} (原列表 {len(sampled_frames)}，转换后 {len(sampled_frames_conv)})")
            except Exception as e:
                logger.error(f"读取 {system_path} 失败: {e}")
    logger.info(f"总采样帧数: {total_sampled_frames}")
    logger.info(f"MultiSystems 信息: {ms}")
    os.makedirs(output_dir, exist_ok=True)
    ms.to_deepmd_npy(output_dir)
    logger.info(f"已导出deepmd npy至 {output_dir}")
    if split_ratio:
        split_and_save(ms, output_dir, split_ratio, logger, seed=seed)

def split_and_save(ms, output_dir, split_ratio, logger, seed: int = 42):
    total_frames = ms.get_nframes()
    logger.info(f"开始拆分数据集，总帧数: {total_frames}")
    indices = np.arange(total_frames)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_points = np.cumsum([int(r * total_frames) for r in split_ratio[:-1]])
    splits = np.split(indices, split_points)
    for i, idx in enumerate(splits):
        sub_ms = dpdata.MultiSystems()
        frame_offset = 0
        for sys in ms.systems:
            sys_frames = len(sys)  # 使用len()而不是get_nframes()
            sys_indices = [j for j in idx if frame_offset <= j < frame_offset + sys_frames]
            if sys_indices:
                local_indices = [j - frame_offset for j in sys_indices]
                # 修正：使用索引而不是subset方法
                sub_sys = sys[local_indices]
                sub_ms.append(sub_sys)
            frame_offset += sys_frames
        sub_dir = os.path.join(output_dir, f"split_{i}")
        os.makedirs(sub_dir, exist_ok=True)
        sub_ms.to_deepmd_npy(sub_dir)
        logger.info(f"已保存拆分子集 {sub_dir}，帧数: {len(idx)}")

def main():
    args = parse_args()
    export_sampled_frames_to_deepmd(
        args.run_dir,
        args.output_dir,
        args.split_ratio,
        force_reexport=args.force,
        seed=args.seed
    )

if __name__ == "__main__":
    main()