"""
并行任务准备与结果处理工具
适用于主流程批量任务的通用模板
"""
import os
from typing import List, Dict, Any, Tuple

def prepare_sampling_only_tasks(records, config, path_manager, ResultSaver, logger=None):
    """
    为仅采样步骤准备任务（不包含DeepMD导出检查）
    返回: (tasks, stats_dict)
    """
    from src.core.process_scheduler import ProcessAnalysisTask
    tasks = []
    reused = 0
    skipped = 0
    targets_meta = {}
    targets_file = path_manager.targets_file
    if targets_file and os.path.exists(targets_file):
        try:
            import json
            with open(targets_file, 'r', encoding='utf-8') as tf:
                targets_json = json.load(tf)
            for mol in targets_json.get('molecules', {}).values():
                for s_name, s_info in mol.get('systems', {}).items():
                    targets_meta[s_name] = s_info
        except Exception as e:
            if logger:
                logger.warning(f"读取 targets 元数据失败(忽略): {e}")
    
    for rec in records:
        if config.force_recompute:
            status = 'FORCE_RECOMPUTE'
        else:
            sampling_meta = targets_meta.get(rec.system_name)
            # 仅采样模式：只检查采样结果，不检查DeepMD
            status = ResultSaver.classify_system_status_sampling_only(
                path_manager.output_dir,
                rec.system_name,
                sampling_meta=sampling_meta
            )
        
        if status == 'SAMPLING_DONE':
            skipped += 1
            if logger:
                logger.info(f"{rec.system_name} 采样已完成，跳过")
            continue
        
        # 检查采样复用
        pre = None
        meta = getattr(rec, 'reuse_meta', None) or {}
        if hasattr(rec, 'system_name') and hasattr(rec, 'source_hash'):
            meta = getattr(rec, 'reuse_meta', None) or {}
            if meta and meta.get('source_hash') == getattr(rec, 'source_hash', None) and meta.get('sampled_frames'):
                pre = meta.get('sampled_frames')
                reused += 1
        
        tasks.append(ProcessAnalysisTask(
            system_path=rec.system_path,
            system_name=rec.system_name,
            pre_sampled_frames=pre,
            pre_stru_files=getattr(rec, 'selected_files', None),
        ))
    
    stats = dict(
        total=len(records),
        tasks=len(tasks),
        reused=reused,
        skipped=skipped,
        deepmd_only=0,  # 仅采样模式中没有DeepMD导出
        reuse_sampling_analysis=0
    )
    return tasks, stats


def prepare_parallel_tasks(records, config, path_manager, ResultSaver, logger=None):
    """
    统一判定任务状态、构造任务对象、统计跳过/复用等信息。
    返回: (tasks, stats_dict)
    """
    from src.core.process_scheduler import ProcessAnalysisTask
    tasks = []
    reused = 0
    skipped = 0
    deepmd_only = 0
    reuse_sampling_analysis = 0
    targets_meta = {}
    targets_file = path_manager.targets_file
    if targets_file and os.path.exists(targets_file):
        try:
            import json
            with open(targets_file, 'r', encoding='utf-8') as tf:
                targets_json = json.load(tf)
            for mol in targets_json.get('molecules', {}).values():
                for s_name, s_info in mol.get('systems', {}).items():
                    targets_meta[s_name] = s_info
        except Exception as e:
            if logger:
                logger.warning(f"读取 targets 元数据失败(忽略): {e}")
    deepmd_root = os.path.join(path_manager.output_dir, 'deepmd_npy_per_system')
    for rec in records:
        if config.force_recompute:
            status = 'FORCE_RECOMPUTE'
        else:
            sampling_meta = targets_meta.get(rec.system_name)
            status = ResultSaver.classify_system_status(
                path_manager.output_dir,
                rec.system_name,
                sampling_meta=sampling_meta,
                deepmd_root=deepmd_root
            )
        if status == 'ALL_DONE':
            skipped += 1
            if logger:
                logger.info(f"{rec.system_name} 已完成(采样+指标+deepmd)，跳过")
            continue
        if status == 'NEED_EXPORT_ONLY':
            sampled_frames = sampling_meta.get('sampled_frames') if sampling_meta else []
            try:
                ResultSaver.export_sampled_frames_direct(
                    system_path=rec.system_path,
                    sampled_frame_ids=sampled_frames,
                    output_root=deepmd_root,
                    system_name=rec.system_name,
                    logger=logger,
                    force=False
                )
                deepmd_only += 1
                if logger:
                    logger.info(f"{rec.system_name} 仅缺 deepmd，已补齐导出，跳过分析")
                continue
            except Exception as e:
                if logger:
                    logger.warning(f"{rec.system_name} deepmd 快速导出失败，转入完整分析: {e}")
        pre = None
        meta = getattr(rec, 'reuse_meta', None) or {}
        if hasattr(rec, 'system_name') and hasattr(rec, 'source_hash'):
            meta = getattr(rec, 'reuse_meta', None) or {}
            if meta and meta.get('source_hash') == getattr(rec, 'source_hash', None) and meta.get('sampled_frames'):
                pre = meta.get('sampled_frames')
                reused += 1
        if status == 'NEED_ANALYSIS_WITH_REUSED_SAMPLING' and pre:
            reuse_sampling_analysis += 1
        tasks.append(ProcessAnalysisTask(
            system_path=rec.system_path,
            system_name=rec.system_name,
            pre_sampled_frames=pre,
            pre_stru_files=getattr(rec, 'selected_files', None),
        ))
    stats = dict(
        total=len(records),
        tasks=len(tasks),
        reused=reused,
        skipped=skipped,
        deepmd_only=deepmd_only,
        reuse_sampling_analysis=reuse_sampling_analysis
    )
    return tasks, stats


def process_sampling_only_results(results, actual_output_dir, config, logger, ResultSaver):
    """
    处理仅采样的结果（不包含DeepMD导出）
    """
    if not results:
        logger.info("没有采样结果需要保存")
        return
        
    logger.info(f"开始保存 {len(results)} 个体系的采样结果...")
    for i, res in enumerate(results):
        try:
            if len(res) >= 2:
                metrics = res[0]
                system_name = getattr(metrics, 'system_name', 'unknown')
                # 只保存采样结果，不导出DeepMD
                ResultSaver.save_single_system(actual_output_dir, res, sampling_only=True)
                if (i + 1) % 5 == 0 or (i + 1) == len(results):
                    logger.info(f"已保存 {i + 1}/{len(results)} 个体系的采样结果")
        except Exception as e:
            logger.warning(f"保存体系采样结果失败(忽略): {e}")
    logger.info(f"所有采样结果保存完成")


def process_parallel_results(results, actual_output_dir, config, logger, ResultSaver):
    """
    统一处理并行结果，包括保存、导出、进度日志等。
    """
    logger.info(f"开始保存 {len(results)} 个体系的分析结果...")
    for i, res in enumerate(results):
        try:
            if len(res) >= 2:
                metrics = res[0]
                frames = res[1]
                system_name = getattr(metrics, 'system_name', 'unknown')
                system_path = getattr(metrics, 'system_path', '')
                ResultSaver.save_single_system(actual_output_dir, res, sampling_only=(config.mode.value=="sampling_only"))
                out_root = os.path.join(actual_output_dir, 'deepmd_npy_per_system')
                ResultSaver.export_sampled_frames_per_system(
                    frames=frames,
                    sampled_frame_ids=getattr(metrics, 'sampled_frames', []) or [],
                    system_path=system_path,
                    output_root=out_root,
                    system_name=system_name,
                    logger=logger,
                    force=False
                )
                if (i + 1) % 5 == 0 or (i + 1) == len(results):
                    logger.info(f"已保存 {i + 1}/{len(results)} 个体系的结果")
        except Exception as e:
            logger.warning(f"保存体系结果失败(忽略): {e}")
    logger.info(f"所有分析结果和DeepMD数据保存完成")
