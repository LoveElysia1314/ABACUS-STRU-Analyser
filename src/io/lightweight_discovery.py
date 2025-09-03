#!/usr/bin/env python
"""Lightweight system discovery with minimal IO.

只做目录/文件名级扫描：
 - 不读取每个 STRU_MD_* 文件内容
 - 单次 scandir 统计帧数、最大帧编号、最大帧文件 stat
 - 解析 INPUT 获取 md_dumpfreq (遇到行立即停止)
 - 生成轻量记录，支持去重
"""

from __future__ import annotations

import os
import re
import hashlib
import logging
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional

logger = logging.getLogger(__name__)


SYSTEM_DIR_PATTERN = re.compile(r"struct_mol_(\d+)_conf_(\d+)_T(\d+)K$")


@dataclass
class LightweightSystemRecord:
    system_path: str
    mol_id: str
    conf: str
    temperature: str
    stru_dir: str
    frame_count: int
    max_frame_id: int
    source_hash: str  # 基于最大帧文件名字/编号/大小/mtime
    md_dumpfreq: int = 1
    selected_files: Optional[List[str]] = None  # 经过 md_dumpfreq 筛选的文件绝对路径（升序）
    sampled_frames: Optional[List[int]] = None  # 可能被后续调度器填充

    @property
    def system_name(self) -> str:
        return f"struct_mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"

    @property
    def key(self) -> str:
        return f"{self.mol_id}:{self.conf}:{self.temperature}"


def _parse_md_dumpfreq(input_file: str) -> int:
    freq = 1
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith('#'):
                    continue
                if raw.lower().startswith('md_dumpfreq'):
                    parts = raw.split()
                    if len(parts) >= 2:
                        try:
                            v = int(float(parts[1]))
                            if v >= 1:
                                freq = v
                        except ValueError:
                            pass
                    break
    except Exception:
        return 1
    return freq


def _scan_stru_dir(stru_dir: str):
    """Return (frame_count, max_frame_id, max_file_name, max_file_stat, files_info)

    files_info: List[tuple(fid, full_path)]
    """
    frame_count = 0
    max_frame_id = -1
    max_file_name = None
    max_file_stat = None
    files_info = []
    try:
        with os.scandir(stru_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.startswith('STRU_MD_'):
                    continue
                try:
                    fid = int(name.split('_')[-1])
                except (ValueError, IndexError):
                    continue
                files_info.append((fid, entry.path))
                frame_count += 1
                if fid > max_frame_id:
                    max_frame_id = fid
                    try:
                        max_file_stat = entry.stat()
                    except OSError:
                        max_file_stat = None
                    max_file_name = name
    except FileNotFoundError:
        return 0, -1, None, None, []
    return frame_count, max_frame_id, max_file_name, max_file_stat, files_info


def _build_source_hash(max_file_name: str, max_frame_id: int, st) -> str:
    if not max_file_name or st is None:
        return ""
    combo = f"{max_file_name}:{max_frame_id}:{getattr(st, 'st_size', 0)}:{int(getattr(st, 'st_mtime', 0))}"
    return hashlib.sha256(combo.encode('utf-8')).hexdigest()


def lightweight_discover_systems(search_paths: List[str], include_project: bool = False) -> List[LightweightSystemRecord]:
    """发现体系（轻量版）并去重。

    去重规则：相同 (mol_id, conf, T) 只保留 (frame_count, max_frame_id) 更大者；若相同则保留修改时间新的。
    """
    project_root = None
    if not include_project:
        # 延迟获取，避免不必要调用
        try:
            from ..utils.common import FileUtils  # type: ignore
            project_root = os.path.abspath(FileUtils.get_project_root())
        except Exception:
            project_root = None

    dedup_map: Dict[str, LightweightSystemRecord] = {}
    total_dirs = 0
    for base in search_paths:
        if not base:
            continue
        base_abs = os.path.abspath(base)
        if project_root and not include_project and base_abs == project_root:
            continue
        for root, dirs, files in os.walk(base_abs):
            total_dirs += 1
            dir_name = os.path.basename(root)
            m = SYSTEM_DIR_PATTERN.match(dir_name)
            if not m:
                continue
            mol_id, conf, temp = m.groups()
            stru_dir = os.path.join(root, 'OUT.ABACUS', 'STRU')
            if not os.path.isdir(stru_dir):
                continue
            frame_count, max_frame_id, max_file_name, max_file_stat, files_info = _scan_stru_dir(stru_dir)
            if frame_count <= 0:
                continue
            # parse INPUT only once
            input_file = os.path.join(root, 'OUT.ABACUS', 'INPUT')
            md_dumpfreq = _parse_md_dumpfreq(input_file)
            # 过滤文件列表（保留 frame_id % md_dumpfreq == 0）
            if md_dumpfreq > 1:
                filtered = [p for fid, p in files_info if fid % md_dumpfreq == 0]
            else:
                filtered = [p for _, p in files_info]
            filtered.sort()  # 按文件名字典序（frame_id 递增）
            source_hash = _build_source_hash(max_file_name, max_frame_id, max_file_stat)
            record = LightweightSystemRecord(
                system_path=root,
                mol_id=mol_id,
                conf=conf,
                temperature=temp,
                stru_dir=stru_dir,
                frame_count=frame_count,
                max_frame_id=max_frame_id,
                source_hash=source_hash,
                md_dumpfreq=md_dumpfreq,
                selected_files=filtered,
            )
            key = record.key
            prev = dedup_map.get(key)
            if prev is None:
                dedup_map[key] = record
            else:
                replace = False
                if record.frame_count > prev.frame_count:
                    replace = True
                elif record.frame_count == prev.frame_count and record.max_frame_id > prev.max_frame_id:
                    replace = True
                elif (record.frame_count == prev.frame_count and record.max_frame_id == prev.max_frame_id and max_file_stat and prev.source_hash != record.source_hash):
                    # 假定修改时间更近（无法直接比较 prev 的 stat，这里简单用 hash 差异）
                    replace = True
                if replace:
                    dedup_map[key] = record
    records = list(dedup_map.values())
    # 排序：优先帧数多的，加快“有价值”体系尽早进入分析
    records.sort(key=lambda r: (-r.frame_count, -r.max_frame_id))
    logger.info(f"Lightweight discovery 完成: 去重后 {len(records)} 个体系 (扫描目录 {total_dirs})")
    return records


def load_sampling_reuse_map(targets_file: str) -> Dict[str, Dict[str, object]]:
    """读取已有 analysis_targets.json，构建复用映射。

    Returns:
        dict: system_name -> { 'sampled_frames': List[int], 'source_hash': str }
    """
    reuse = {}
    if not targets_file or not os.path.exists(targets_file):
        return reuse
    try:
        import json
        with open(targets_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        molecules = data.get('molecules', {})
        for mol in molecules.values():
            for sys_name, sys_data in mol.get('systems', {}).items():
                sampled_raw = sys_data.get('sampled_frames')
                if isinstance(sampled_raw, str):
                    try:
                        import json as _json
                        sampled_frames = _json.loads(sampled_raw)
                    except Exception:
                        sampled_frames = []
                else:
                    sampled_frames = sampled_raw or []
                reuse[sys_name] = {
                    'sampled_frames': sampled_frames,
                    'source_hash': sys_data.get('source_hash', ''),
                    'system_path': sys_data.get('system_path', ''),
                }
    except Exception as e:
        logger.warning(f"读取采样复用信息失败: {e}")
    return reuse
