#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按体系粒度导出采样帧为 deepmd npy 的辅助工具。

以系统粒度（system_name）输出：
<output_root>/<system_name>/ （包含 dpdata 标准的 *.npy 与 type.raw 等）

幂等：存在 export.done 且未 force 时跳过。
"""
from __future__ import annotations
import os
from typing import List, Sequence, Dict
import dpdata  # type: ignore

ALL_TYPE_MAP = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]


def _build_frame_id_index(frames: Sequence) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for idx, f in enumerate(frames):
        fid = getattr(f, 'frame_id', None)
        if fid is not None and fid not in mapping:
            mapping[fid] = idx
    return mapping


def export_sampled_frames_per_system(
    frames: Sequence,
    sampled_frame_ids: List[int],
    system_path: str,
    output_root: str,
    system_name: str,
    logger,
    force: bool = False,
) -> str | None:
    if not sampled_frame_ids:
        logger.debug(f"[deepmd-export] {system_name} 无采样帧，跳过导出")
        return None
    target_dir = os.path.join(output_root, system_name)
    marker_file = os.path.join(target_dir, 'export.done')
    if os.path.isdir(target_dir) and os.path.exists(marker_file) and not force:
        logger.debug(f"[deepmd-export] {system_name} 已存在且未强制覆盖，跳过")
        return target_dir
    try:
        id2idx = _build_frame_id_index(frames)
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

__all__ = ['export_sampled_frames_per_system']
