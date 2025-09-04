def parse_sampled_frames(val):
    """统一解析 analysis_targets.json 中的 sampled_frames 字段，支持字符串或列表。"""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val or []
#!/usr/bin/env python

import logging
import os
from typing import Optional, List, Dict, Tuple, Sequence
import json
from ..io.stru_parser import FrameData
from ..utils import FileUtils
from .path_manager import NEW_DIR_SINGLE, LEGACY_DIR_SINGLE, NEW_FRAME_PREFIX, LEGACY_FRAME_PREFIX


class ResultSaver:
    """结果保存器类，负责保存分析结果到CSV文件"""


    @staticmethod
    def should_skip_analysis(output_dir: str, system_name: str) -> bool:
        """检查是否应该跳过体系分析
        
        Args:
            output_dir: 输出目录
            system_name: 体系名称
            
        Returns:
            如果single_analysis中存在对应文件且analysis_targets.json存在，则返回True
        """
        # 新/旧目录兼容
        single_dir_candidates = [os.path.join(output_dir, NEW_DIR_SINGLE),
                                 os.path.join(output_dir, LEGACY_DIR_SINGLE)]
        single_analysis_dir = None
        for c in single_dir_candidates:
            if os.path.isdir(c):
                single_analysis_dir = c
                break
        if single_analysis_dir is None:
            return False
        # 新旧命名兼容
        frame_metrics_file = os.path.join(single_analysis_dir, f"{NEW_FRAME_PREFIX}{system_name}.csv")
        if not os.path.exists(frame_metrics_file):
            legacy_try = os.path.join(single_analysis_dir, f"{LEGACY_FRAME_PREFIX}{system_name}.csv")
            if os.path.exists(legacy_try):
                frame_metrics_file = legacy_try
        
        # 检查analysis_targets.json文件
        targets_file = os.path.join(output_dir, "analysis_targets.json")
        
        # 如果两个文件都存在，则跳过分析
        return os.path.exists(frame_metrics_file) and os.path.exists(targets_file)

    # -------------------- 新增: 体系状态判定与直接DeepMD导出 --------------------
    @staticmethod
    def classify_system_status(
        output_dir: str,
        system_name: str,
        sampling_meta: Optional[Dict[str, any]],
        deepmd_root: Optional[str] = None,
    ) -> str:
        """判定体系当前状态以决定后续动作

        状态定义（优先级）：
          ALL_DONE: 采样列表 + 单帧指标 + deepmd 导出全部存在
          NEED_EXPORT_ONLY: 采样列表 + 单帧指标存在，但 deepmd 缺失 → 仅执行 deepmd 导出
          NEED_ANALYSIS_WITH_REUSED_SAMPLING: 仅存在采样列表 (单帧指标缺失) → 复用采样直接做完整分析
          NEED_FULL_ANALYSIS: 以上条件都不满足 → 全流程重新分析

        Args:
            output_dir: run_* 实际输出目录
            system_name: 体系名称
            sampling_meta: analysis_targets.json 中该体系的元数据 (含 sampled_frames, source_hash 等)
            deepmd_root: deepmd_npy_per_system 目录

        Returns:
            字符串标识的状态
        """
        single_analysis_dir = os.path.join(output_dir, NEW_DIR_SINGLE)
        frame_metrics_file = os.path.join(single_analysis_dir, f"{NEW_FRAME_PREFIX}{system_name}.csv")
        if not os.path.exists(frame_metrics_file):
            legacy_try = os.path.join(single_analysis_dir, f"{LEGACY_FRAME_PREFIX}{system_name}.csv")
            if os.path.exists(legacy_try):
                frame_metrics_file = legacy_try
        metrics_exists = os.path.exists(frame_metrics_file)
        has_sampling_list = bool(sampling_meta and parse_sampled_frames(sampling_meta.get('sampled_frames')))

        deepmd_root = deepmd_root or os.path.join(output_dir, 'deepmd_npy_per_system')
        deepmd_done = os.path.exists(os.path.join(deepmd_root, system_name, 'export.done'))

        if has_sampling_list and metrics_exists and deepmd_done:
            return 'ALL_DONE'
        if has_sampling_list and metrics_exists and not deepmd_done:
            return 'NEED_EXPORT_ONLY'
        if has_sampling_list and not metrics_exists:
            return 'NEED_ANALYSIS_WITH_REUSED_SAMPLING'
        return 'NEED_FULL_ANALYSIS'

    @staticmethod
    def classify_system_status_sampling_only(
        output_dir: str,
        system_name: str,
        sampling_meta: Optional[Dict[str, any]],
    ) -> str:
        """判定体系在仅采样模式下的状态（不考虑DeepMD）
        
        状态定义：
          SAMPLING_DONE: 采样列表存在 → 跳过采样
          NEED_SAMPLING: 采样列表不存在 → 需要采样
        
        Args:
            output_dir: run_* 实际输出目录
            system_name: 体系名称
            sampling_meta: analysis_targets.json 中该体系的元数据
        
        Returns:
            字符串标识的状态
        """
        has_sampling_list = bool(sampling_meta and parse_sampled_frames(sampling_meta.get('sampled_frames')))
        if has_sampling_list:
            return 'SAMPLING_DONE'
        return 'NEED_SAMPLING'

    @staticmethod
    def export_sampled_frames_direct(
        system_path: str,
        sampled_frame_ids: List[int],
        output_root: str,
        system_name: str,
        logger: Optional[logging.Logger] = None,
        force: bool = False,
    ) -> Optional[str]:
        """无需重建 FrameData / 不依赖分析结果，直接用 dpdata 导出指定帧

        用于: 已有采样列表 & 单帧指标，只缺 deepmd 导出时的快速补齐。
        """
        logger = logger or logging.getLogger(__name__)
        if not sampled_frame_ids:
            logger.warning(f"[deepmd-direct] {system_name} 采样帧列表为空，跳过")
            return None
        target_dir = os.path.join(output_root, system_name)
        marker_file = os.path.join(target_dir, 'export.done')
        if os.path.isdir(target_dir) and os.path.exists(marker_file) and not force:
            logger.info(f"[deepmd-direct] {system_name} 已存在导出，跳过 (force=False)")
            return target_dir
        try:
            import dpdata  # type: ignore
            ls = dpdata.LabeledSystem(system_path, fmt="abacus/lcao/md", type_map=ResultSaver.ALL_TYPE_MAP)
            n_total = len(ls)
            valid_indices = [i for i in sampled_frame_ids if 0 <= i < n_total]
            if not valid_indices:
                logger.warning(f"[deepmd-direct] {system_name} 无有效帧索引，跳过")
                return None
            sub_ls = ls[valid_indices]
            os.makedirs(target_dir, exist_ok=True)
            sub_ls.to_deepmd_npy(target_dir)
            with open(marker_file, 'w', encoding='utf-8') as f:
                f.write(f"frames={len(valid_indices)}\n")
            logger.info(f"[deepmd-direct] 导出 {system_name} 成功，帧数={len(valid_indices)} -> {target_dir}")
            return target_dir
        except Exception as e:  # noqa
            logger.error(f"[deepmd-direct] 导出 {system_name} 失败: {e}")
            return None



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
          - 完整模式(result 长度>=7)：写 frame_metrics
          - 仅采样模式/数据不足：只写采样帧信息（采样帧主要已在 analysis_targets.json 由调用方刷新）

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

            # 1) frame_{system}.csv (仅完整模式)
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

            # 4) 可选刷新 analysis_targets.json (由 orchestrator 提供的 hook 控制频率)
            if flush_targets_hook:
                try:
                    flush_targets_hook()
                except Exception as he:
                    logger.warning(f"analysis_targets.json 刷新失败: {he}")
        except Exception as e:
            logger.error(f"流式保存体系结果失败: {e}")

    @staticmethod
    def save_frame_metrics(
        output_dir: str,
        system_name: str,
        frames: List[FrameData],
        sampled_frames: List[int],
        pca_components_data: List[Dict] = None,
        rmsd_per_frame: List[float] = None,
    ) -> None:
        """Save individual frame metrics to CSV file, with energy/force info if available
        """
        single_analysis_dir = os.path.join(output_dir, NEW_DIR_SINGLE)
        FileUtils.ensure_dir(single_analysis_dir)
        csv_path = os.path.join(single_analysis_dir, f"{NEW_FRAME_PREFIX}{system_name}.csv")

        headers = ["Frame_ID", "Selected", "RMSD", "Energy(eV)", "Energy_Standardized"]
        max_pc = 0
        if pca_components_data:
            for item in pca_components_data:
                for key in item.keys():
                    if key.startswith('PC'):
                        pc_num = int(key[2:])
                        max_pc = max(max_pc, pc_num)
            headers.extend([f"PC{i}" for i in range(1, max_pc + 1)])

        sampled_set = set(sampled_frames)
        pca_lookup = {}
        if pca_components_data:
            for item in pca_components_data:
                frame_id = item.get('frame')
                if frame_id is not None:
                    pca_lookup[frame_id] = item

        rows = []
        for i, frame in enumerate(frames):
            selected = 1 if frame.frame_id in sampled_set else 0
            row = [frame.frame_id, selected]
            rmsd_value = rmsd_per_frame[i] if rmsd_per_frame and i < len(rmsd_per_frame) else ""
            row.append(f"{rmsd_value:.6f}" if isinstance(rmsd_value, (int, float)) else rmsd_value)
            energy = frame.energy if frame.energy is not None else ""
            energy_standardized = frame.energy_standardized if frame.energy_standardized is not None else ""
            row.append(energy)
            row.append(energy_standardized)
            if pca_components_data and frame.frame_id in pca_lookup:
                pca_item = pca_lookup[frame.frame_id]
                for pc_num in range(1, max_pc + 1):
                    pc_key = f'PC{pc_num}'
                    pc_value = pca_item.get(pc_key, 0.0)
                    row.append(f"{pc_value:.6f}")
            elif pca_components_data:
                for pc_num in range(1, max_pc + 1):
                    row.append("0.000000")
            rows.append(row)

        FileUtils.safe_write_csv(csv_path, rows, headers=headers, encoding='utf-8-sig')

    # 旧增量与采样记录聚合函数已移除


    # ---- DeepMD Export Functionality (merged from deepmd_exporter.py) ----
    ALL_TYPE_MAP = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
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
                sampled_frames = parse_sampled_frames(sys_info['sampled_frames'])
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
