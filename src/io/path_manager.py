#!/usr/bin/env python

import glob
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.common import FileUtils


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


def lightweight_discover_systems(search_paths: List[str], include_project: bool = False) -> List[LightweightSystemRecord]:
    """只做目录级扫描，匹配所有 struct_*_conf_*_T*K/OUT.ABACUS/STRU 目录，
    对于重复体系，选择创建时间最晚的。其余分析交由主流程完成。"""
    dedup_map: Dict[str, LightweightSystemRecord] = {}
    stru_dirs = []
    total_dirs = 0
    sampling_ratio = 0.1
    n_workers = 20
    # 分层扫描：先找体系目录，再判断STRU
    for base in search_paths:
        if not base:
            continue
        base_abs = os.path.abspath(base)
        pattern = os.path.join(base_abs, 'struct_mol_*_conf_*_T*K')
        for sys_dir in glob.glob(pattern):
            stru_dir = os.path.join(sys_dir, 'OUT.ABACUS', 'STRU')
            if os.path.isdir(stru_dir):
                stru_dirs.append(stru_dir)
    total_dirs = len(stru_dirs)

    def get_record(stru_dir):
        parent = os.path.dirname(os.path.dirname(stru_dir))
        dir_name = os.path.basename(parent)
        m = SYSTEM_DIR_PATTERN.match(dir_name)
        if not m:
            return None
        mol_id, conf, temp = m.groups()
        try:
            ctime = os.path.getctime(stru_dir)
        except Exception:
            ctime = 0
        
        # 计算source_hash
        source_hash = ""
        try:
            # 使用PathManager的source_hash计算方法
            pm = PathManager()
            source_hash = pm._calculate_source_hash(parent)
        except Exception:
            pass
        
        record = LightweightSystemRecord(
            system_path=parent,
            mol_id=mol_id,
            conf=conf,
            temperature=temp,
            stru_dir=stru_dir,
            frame_count=0,
            max_frame_id=0,
            source_hash=source_hash,
            md_dumpfreq=1,
            selected_files=None,
            sampled_frames=None
        )
        record.ctime = ctime
        return (f"{mol_id}:{conf}:{temp}", record)

    # 并行获取ctime和生成record
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(get_record, d) for d in stru_dirs]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            key, record = res
            prev = dedup_map.get(key)
            if prev is None or record.ctime > getattr(prev, 'ctime', 0):
                dedup_map[key] = record

    records = list(dedup_map.values())
    records.sort(key=lambda r: -getattr(r, 'ctime', 0))
    logger = logging.getLogger(__name__)
    logger.info(f"Lightweight discovery 完成: 去重后 {len(records)} 个体系 (扫描目录 {total_dirs})")
    return records



@dataclass
class AnalysisTarget:
    system_path: str
    mol_id: str
    conf: str
    temperature: str
    stru_files: List[str]
    creation_time: float
    source_hash: str = ""  # 源文件哈希，用于检测源数据变更
    sampled_frames: List[int] = None  # 采样帧编号列表
    reuse_sampling: bool = False  # 是否复用既有采样结果（跳过采样算法）
    md_dumpfreq: int = 1  # MD dump frequency from INPUT file

    def __post_init__(self):
        if self.sampled_frames is None:
            self.sampled_frames = []

    @property
    def system_name(self) -> str:
        return f"struct_mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"

    @property
    def system_key(self) -> str:
        return f"mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"


class PathManager:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(FileUtils.get_project_root(), "analysis_results")
        self.base_output_dir = output_dir
        self.output_dir = output_dir  # 将在 set_output_dir_for_params 中更新
        self.targets_file = None  # 将在 set_output_dir_for_params 中设置
        self.summary_file = None
        self.logger = logging.getLogger(__name__)
        self.targets: List[AnalysisTarget] = []
        self.mol_groups: Dict[str, List[AnalysisTarget]] = {}
        self.current_params_hash: str = ""

    def _calculate_params_hash(self, params: Dict[str, Any]) -> str:
        """计算分析参数的哈希值，只包含影响计算结果的参数"""
        try:
            key_params = {
                "sample_ratio": params.get("sample_ratio", 0.1),
                "power_p": params.get("power_p", -0.5),
                "pca_variance_ratio": params.get("pca_variance_ratio", 0.90),
            }
            params_str = json.dumps(key_params, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(params_str.encode("utf-8")).hexdigest()
        except Exception as e:
            self.logger.debug(f"计算参数哈希失败: {str(e)}")
            return ""

    def _calculate_source_hash(self, system_path: str) -> str:
        """计算源文件目录的哈希值，用于检测源数据完整性
        优化策略：只对帧数最大的STRU文件计算哈希，大幅减少计算量
        """
        try:
            # STRU文件实际在OUT.ABACUS/STRU文件夹中
            stru_dir = os.path.join(system_path, "OUT.ABACUS", "STRU")
            if not os.path.exists(stru_dir):
                return ""

            # 找到帧数最大的STRU文件
            max_frame_num = -1
            max_frame_file = None

            for filename in os.listdir(stru_dir):
                if filename.startswith("STRU_MD_") and not filename.endswith(".swp"):
                    try:
                        # 从文件名提取帧数，例如 STRU_MD_1000 -> 1000
                        frame_num = int(filename.split("_")[-1])
                        if frame_num > max_frame_num:
                            max_frame_num = frame_num
                            max_frame_file = filename
                    except (ValueError, IndexError):
                        continue

            if max_frame_file is None:
                return ""

            # 计算最大帧数文件的哈希
            max_file_path = os.path.join(stru_dir, max_frame_file)
            stat = os.stat(max_file_path)

            # 组合信息：文件名 + 帧数 + 大小 + 修改时间
            combined_info = (
                f"{max_frame_file}:{max_frame_num}:{stat.st_size}:{int(stat.st_mtime)}"
            )
            return hashlib.sha256(combined_info.encode("utf-8")).hexdigest()

        except Exception as e:
            self.logger.debug(f"计算源文件哈希失败 {system_path}: {str(e)}")
            return ""

    def set_output_dir_for_params(self, analysis_params: Dict[str, Any]) -> str:
        """根据分析参数设置输出目录，实现参数隔离"""
        params_hash = self._calculate_params_hash(analysis_params)
        self.current_params_hash = params_hash

        # 生成基于参数的目录名（简洁版）
        # include_h 固定为 False，不作为目录命名的一部分
        key_params = {
            "sample_ratio": analysis_params.get("sample_ratio", 0.1),
            "power_p": analysis_params.get("power_p", -0.5),
            "pca_variance_ratio": analysis_params.get("pca_variance_ratio", 0.90),
        }

        # 生成基于短选项的简洁目录名（使用短选项字母，不使用大小写区分）
        # 对应关系：sample_ratio -> r, power_p -> p, pca_variance_ratio -> v
        short_map = {
            "sample_ratio": "r",
            "power_p": "p",
            "pca_variance_ratio": "v",
        }

        # 保持固定顺序：r, p, v
        ordered_keys = ["sample_ratio", "power_p", "pca_variance_ratio"]
        parts = []
        for key in ordered_keys:
            value = key_params.get(key)
            short = short_map.get(key, key)
            if isinstance(value, bool):
                # 布尔值用 1/0 表示
                parts.append(f"{short}{int(value)}")
            elif isinstance(value, float):
                # 使用通用格式去除不必要的小数位
                formatted = f"{value:g}"
                parts.append(f"{short}{formatted}")
            else:
                parts.append(f"{short}{value}")

        dir_name = "run_" + "_".join(parts)
        self.output_dir = os.path.join(self.base_output_dir, dir_name)

        # 更新相关文件路径
        self.targets_file = os.path.join(self.output_dir, "analysis_targets.json")
        self.summary_file = os.path.join(self.output_dir, "path_summary.json")

        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info(f"设置参数专用输出目录: {self.output_dir}")
        return self.output_dir

    def check_existing_complete_results(self) -> bool:
        """检查当前参数目录是否已有完整的分析结果"""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return False

        # 检查关键结果文件
        summary_file = os.path.join(
            self.output_dir, "combined_analysis_results", "system_metrics_summary.csv"
        )
        targets_file = self.targets_file

        if not (os.path.exists(summary_file) and os.path.exists(targets_file)):
            self.logger.debug("关键结果文件不存在，需要重新计算")
            return False

        # 检查 analysis_targets.json 中的完成状态
        try:
            with open(targets_file, encoding="utf-8") as f:
                data = json.load(f)

            # 检查是否所有系统都已完成
            total_systems = 0

            for mol_data in data.get("molecules", {}).values():
                for system_data in mol_data.get("systems", {}).values():
                    total_systems += 1

            if total_systems == 0:
                self.logger.debug("目标文件中没有系统记录")
                return False

            # 由于不再使用 status 字段，总是返回 False 表示需要重新计算
            self.logger.info(
                f"发现 {total_systems} 个系统记录，将重新计算所有输出"
            )
            return False

        except Exception as e:
            self.logger.warning(f"检查现有结果时出错: {str(e)}")
            return False

    def load_from_discovery(
        self, search_paths: List[str], preserve_existing: bool = False
    ) -> None:
        """从发现结果加载分析目标

        Args:
                search_paths: 搜索路径列表
                preserve_existing: 是否保留已有的状态信息
        """
        # 使用高效的轻量发现
        records = lightweight_discover_systems(search_paths)

        self.targets.clear()
        self.mol_groups.clear()
        total_targets = 0

        for record in records:
            # 获取STRU文件列表
            stru_files = glob.glob(os.path.join(record.stru_dir, "STRU_MD_*"))
            if not stru_files:
                self.logger.warning(f"系统 {record.system_name} 没有找到 STRU_MD 文件")
                continue

            # 应用md_dumpfreq筛选
            md_dumpfreq = 1  # 默认值
            try:
                from .stru_parser import StrUParser
                parser = StrUParser()
                input_file = os.path.join(record.system_path, "OUT.ABACUS", "INPUT")
                md_dumpfreq = parser.parse_md_dumpfreq(input_file)
                original_count = len(stru_files)
                stru_files = parser.select_frames_by_md_dumpfreq(stru_files, md_dumpfreq)
                if not stru_files:
                    self.logger.warning(
                        f"系统 {record.system_name} 依据 md_dumpfreq={md_dumpfreq} 过滤后无可用帧 (原始 {original_count} 帧)"
                    )
                    continue
                if len(stru_files) != original_count:
                    self.logger.info(
                        f"系统 {record.system_name}: 按 md_dumpfreq={md_dumpfreq} 筛选 {len(stru_files)}/{original_count} 帧 (包含0帧)"
                    )
            except Exception as e:
                self.logger.warning(f"系统 {record.system_name} 应用 md_dumpfreq 筛选失败，使用所有帧: {e}")

            # 计算源文件哈希
            source_hash = self._calculate_source_hash(record.system_path)

            target = AnalysisTarget(
                system_path=record.system_path,
                mol_id=record.mol_id,
                conf=record.conf,
                temperature=record.temperature,
                stru_files=stru_files,
                creation_time=getattr(record, 'ctime', 0),
                source_hash=source_hash,
                md_dumpfreq=md_dumpfreq,
            )
            self.targets.append(target)
            total_targets += 1

        # 按mol_id分组
        self._rebuild_mol_groups()

        self.logger.info(
            f"PathManager loaded {len(self.mol_groups)} molecules with {total_targets} targets"
        )

    def deduplicate_targets(self) -> None:
        """去除重复的分析目标，基于系统名称(system_key)保留修改时间最晚的"""
        if not self.targets:
            return

        # 按system_key分组
        system_groups = {}
        for target in self.targets:
            key = target.system_key  # mol_id_conf_id_T_tempK
            if key not in system_groups:
                system_groups[key] = []
            system_groups[key].append(target)

        # 去重：每组保留修改时间最晚的
        deduplicated_targets = []
        duplicates_removed = 0

        for key, targets_group in system_groups.items():
            if len(targets_group) == 1:
                deduplicated_targets.extend(targets_group)
            else:
                # 多个重复，选择修改时间最晚的
                latest_target = max(targets_group, key=lambda t: t.creation_time)
                deduplicated_targets.append(latest_target)
                duplicates_removed += len(targets_group) - 1

                # 记录被移除的重复项
                removed_paths = [
                    t.system_path for t in targets_group if t != latest_target
                ]
                self.logger.info(f"去重: 系统 {key} 保留 {latest_target.system_path}")
                for removed_path in removed_paths:
                    self.logger.info(f"  移除重复: {removed_path}")

        # 更新targets和mol_groups
        self.targets = deduplicated_targets
        self._rebuild_mol_groups()

        if duplicates_removed > 0:
            self.logger.info(
                f"去重完成: 移除 {duplicates_removed} 个重复目标，保留 {len(self.targets)} 个"
            )

    def _rebuild_mol_groups(self) -> None:
        """重建分子分组"""
        self.mol_groups.clear()
        for target in self.targets:
            mol_id = target.mol_id
            if mol_id not in self.mol_groups:
                self.mol_groups[mol_id] = []
            self.mol_groups[mol_id].append(target)

    def load_targets(self) -> bool:
        if not os.path.exists(self.targets_file):
            return False
        try:
            with open(self.targets_file, encoding="utf-8") as f:
                targets_data = json.load(f)
            self.targets.clear()
            self.mol_groups.clear()
            for target_dict in targets_data:
                target_dict.pop("created_at", None)
                target = AnalysisTarget(**target_dict)
                self.targets.append(target)
                mol_id = target.mol_id
                if mol_id not in self.mol_groups:
                    self.mol_groups[mol_id] = []
                self.mol_groups[mol_id].append(target)
            self.logger.info(f"Loaded {len(self.targets)} analysis targets from file")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load targets: {str(e)}")
            return False

    def get_targets_by_molecule(self, mol_id: str) -> List[AnalysisTarget]:
        return self.mol_groups.get(mol_id, [])

    def get_all_targets(self) -> List[AnalysisTarget]:
        return self.targets.copy()

    def save_analysis_targets(self, analysis_params: Dict[str, Any] = None) -> str:
        """保存分析目标到 analysis_targets.json，包含详细的哈希和状态信息"""
        if not self.targets_file:
            raise ValueError("输出目录未设置，请先调用 set_output_dir_for_params")

        targets_file = self.targets_file
        temp_file = targets_file + ".tmp"

        try:
            # 计算参数哈希
            params_hash = ""
            if analysis_params:
                params_hash = self._calculate_params_hash(analysis_params)

            # 统计全局md_dumpfreq（如有多个体系，取最大值/最小值/列表？此处取所有target的md_dumpfreq，若无则为None）
            md_dumpfreqs = [getattr(t, 'md_dumpfreq', None) for t in self.targets]
            # 只保留有效int
            md_dumpfreqs = [v for v in md_dumpfreqs if isinstance(v, int) and v > 0]
            # 若所有体系md_dumpfreq一致，则写单值，否则写列表
            if md_dumpfreqs:
                if all(v == md_dumpfreqs[0] for v in md_dumpfreqs):
                    md_dumpfreq_meta = md_dumpfreqs[0]
                else:
                    md_dumpfreq_meta = md_dumpfreqs
            else:
                md_dumpfreq_meta = None

            # 生成JSON结构
            analysis_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generator": "ABACUS-STRU-Analyser",
                    "version": "1.0",
                    "params_hash": params_hash,
                    "analysis_params": analysis_params or {},
                    "output_directory": self.output_dir,
                    "md_dumpfreq": md_dumpfreq_meta,
                },
                "summary": {
                    "total_molecules": len(self.mol_groups),
                    "total_systems": len(self.targets),
                },
                "molecules": {},
            }

            # 按分子组织数据
            for mol_id, targets in self.mol_groups.items():
                mol_data = {
                    "molecule_id": mol_id,
                    "system_count": len(targets),
                    "systems": {},
                }

                for target in targets:
                    # 简化输出：移除每个系统的 params_hash，只在 metadata 保存一次
                    # 将sampled_frames转换为紧凑格式，避免多行显示
                    sampled_frames_compact = None
                    if target.sampled_frames:
                        sampled_frames_compact = json.dumps(target.sampled_frames, separators=(',', ':'))

                    system_data = {
                        "system_path": target.system_path,
                        "stru_files_count": len(target.stru_files),
                        "source_hash": target.source_hash,
                    }
                    # 始终添加sampled_frames字段，确保采样列表被输出到JSON中
                    if sampled_frames_compact is not None:
                        system_data["sampled_frames"] = sampled_frames_compact
                    else:
                        # 即使为空也添加该字段，避免字段缺失
                        system_data["sampled_frames"] = "[]"
                    mol_data["systems"][target.system_name] = system_data

                analysis_data["molecules"][mol_id] = mol_data

            # 原子性写入：先写临时文件，再重命名
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)

            # 原子性重命名
            os.replace(temp_file, targets_file)

            self.logger.info(f"Analysis targets saved to: {targets_file}")
            self.logger.info(
                f"Summary: {len(self.mol_groups)} molecules, {len(self.targets)} systems"
            )
            self.logger.info(f"Params hash: {params_hash}")

            return targets_file

        except Exception as e:
            self.logger.error(f"Failed to save analysis targets: {str(e)}")
            # 清理临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to remove temporary file {temp_file}: {e}"
                    )
            raise

    def load_analysis_targets(self) -> bool:
        """从新格式的 analysis_targets.json 加载分析目标"""
        targets_file = os.path.join(self.output_dir, "analysis_targets.json")
        if not os.path.exists(targets_file):
            return False

        try:
            with open(targets_file, encoding="utf-8") as f:
                analysis_data = json.load(f)

            # 检查格式版本
            if "metadata" not in analysis_data or "molecules" not in analysis_data:
                self.logger.warning("分析目标文件格式不兼容，尝试旧格式加载")
                return self.load_targets()  # 回退到旧格式

            self.targets.clear()
            self.mol_groups.clear()

            # 提取参数哈希用于后续比较
            self.loaded_params_hash = analysis_data.get("metadata", {}).get(
                "params_hash", ""
            )

            # 加载分子和系统数据
            for mol_id, mol_data in analysis_data["molecules"].items():
                targets_for_mol = []
                for system_name, system_data in mol_data["systems"].items():
                    # 从system_name解析mol_id、conf、temperature
                    # 格式: struct_mol_{mol_id}_conf_{conf}_T{temperature}K
                    try:
                        parts = system_name.split('_')
                        if len(parts) >= 6 and parts[0] == 'struct' and parts[1] == 'mol' and parts[3] == 'conf':
                            parsed_mol_id = parts[2]
                            parsed_conf = parts[4]
                            temp_part = parts[5]  # T{temperature}K
                            if temp_part.startswith('T') and temp_part.endswith('K'):
                                parsed_temperature = temp_part[1:-1]  # 移除T和K
                            else:
                                parsed_temperature = "0"
                        else:
                            # 回退到旧的解析方式
                            parsed_mol_id = system_data.get("mol_id", mol_id)
                            parsed_conf = system_data.get("conf", "0")
                            parsed_temperature = system_data.get("temperature", "0")
                    except Exception:
                        # 解析失败，使用默认值
                        parsed_mol_id = system_data.get("mol_id", mol_id)
                        parsed_conf = system_data.get("conf", "0")
                        parsed_temperature = system_data.get("temperature", "0")

                    # 兼容性：若 JSON 中缺失某些字段，使用解析出的值或安全默认值
                    # 处理sampled_frames：可能是字符串格式（紧凑模式）或列表格式
                    sampled_frames_data = system_data.get("sampled_frames")
                    if sampled_frames_data is None:
                        sampled_frames_data = []
                    elif isinstance(sampled_frames_data, str):
                        try:
                            sampled_frames_data = json.loads(sampled_frames_data)
                        except (json.JSONDecodeError, TypeError):
                            sampled_frames_data = []

                    target = AnalysisTarget(
                        system_path=system_data.get("system_path", ""),
                        mol_id=parsed_mol_id,
                        conf=parsed_conf,
                        temperature=parsed_temperature,
                        stru_files=[],  # 将在 load_from_discovery 中重新填充
                        creation_time=system_data.get("creation_time", 0.0),
                        source_hash=system_data.get("source_hash", ""),
                        sampled_frames=sampled_frames_data,
                        md_dumpfreq=system_data.get("md_dumpfreq", 1),  # 从JSON中读取或使用默认值
                    )
                    self.targets.append(target)
                    targets_for_mol.append(target)

                if targets_for_mol:
                    self.mol_groups[mol_id] = targets_for_mol

            self.logger.info(
                f"Loaded {len(self.targets)} analysis targets from new format"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load analysis targets: {str(e)}")
            return False

    def check_params_compatibility(self, current_params: Dict[str, Any]) -> bool:
        """检查当前参数是否与加载的参数兼容"""
        if not hasattr(self, "loaded_params_hash") or not self.loaded_params_hash:
            return False

        current_params_hash = self._calculate_params_hash(current_params)
        compatible = current_params_hash == self.loaded_params_hash

        if not compatible:
            self.logger.warning(
                f"参数不兼容 - 当前: {current_params_hash}, 加载的: {self.loaded_params_hash}"
            )
        else:
            self.logger.info(f"参数兼容 - 哈希: {current_params_hash}")

        return compatible

    def check_existing_results(self) -> None:
        """检查已有的分析结果，验证哈希匹配性"""
        if not self.targets:
            return

        hash_mismatch_count = 0

        # 检查每个目标的源文件哈希是否匹配
        for target in self.targets:
            system_name = target.system_name
            system_path = target.system_path

            # 检查源文件哈希是否匹配（从当前目录的targets文件）
            if os.path.exists(self.targets_file):
                try:
                    with open(self.targets_file, encoding="utf-8") as f:
                        targets_data = json.load(f)
                        # 查找匹配的系统记录
                        for target_dict in targets_data.get("molecules", {}).values():
                            for sys_data in target_dict.get("systems", {}).values():
                                if sys_data.get("system_path") == system_path:
                                    old_hash = sys_data.get("source_hash", "")
                                    new_hash = target.source_hash
                                    if old_hash and new_hash and old_hash != new_hash:
                                        hash_mismatch_count += 1
                                        self.logger.warning(
                                            f"检测到源文件变更: {system_name} (哈希不匹配)"
                                        )
                                    break
                except Exception as e:
                    self.logger.warning(f"读取历史哈希信息失败: {str(e)}")

        if hash_mismatch_count > 0:
            self.logger.warning(
                f"源文件变更检测: {hash_mismatch_count} 个系统需要重新分析"
            )
        else:
            self.logger.info("所有系统源文件哈希匹配，无需重新采样")

    # 新增：仅基于 analysis_targets.json + 源数据哈希 判定是否复用采样结果；不跳过其它计算
    def determine_sampling_reuse(self) -> dict:
        """确定哪些系统可以复用采样结果。

        复用条件:
          1) analysis_targets.json 存在且包含系统记录
          2) 记录中含 sampled_frames 且非空
          3) 记录中的 source_hash == 当前重新计算的 source_hash

        满足后:
          - target.sampled_frames 赋值为历史采样帧
          - target.reuse_sampling = True
          - 所有下游输出都重新计算

        Returns:
          dict[system_name] -> List[int]  (可复用的采样帧列表)
        """
        reuse_map = {}
        if not self.targets or not self.targets_file or not os.path.exists(self.targets_file):
            return reuse_map
        try:
            with open(self.targets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            molecules = data.get('molecules', {})
            # 构建两个索引：system_path 与 system_name，防止路径层级或大小写差异导致失配
            path_index = {}
            name_index = {}
            for mol in molecules.values():
                for sys_name, sys_data in mol.get('systems', {}).items():
                    rec = (
                        sys_data.get('sampled_frames'),
                        sys_data.get('source_hash','')
                    )
                    sys_path = sys_data.get('system_path','')
                    if sys_path:
                        path_index[os.path.normpath(sys_path).lower()] = rec
                    name_index[sys_name] = rec
            for target in self.targets:
                rec = path_index.get(os.path.normpath(target.system_path).lower())
                if rec is None:
                    rec = name_index.get(target.system_name)
                if rec is None:
                    continue
                sampled_raw, old_hash = rec
                # 解析紧凑JSON字符串
                if isinstance(sampled_raw, str):
                    try:
                        sampled_frames = json.loads(sampled_raw)
                    except Exception:
                        sampled_frames = []
                else:
                    sampled_frames = sampled_raw or []
                if (sampled_frames and old_hash and old_hash == target.source_hash):
                    target.sampled_frames = sampled_frames
                    target.reuse_sampling = True
                    reuse_map[target.system_name] = sampled_frames
            if reuse_map:
                self.logger.info(f"采样复用: {len(reuse_map)} 个系统将跳过采样算法，仅更新其余输出")
            else:
                self.logger.info("采样复用: 未发现可复用的采样结果")
        except Exception as e:
            self.logger.warning(f"解析已有采样结果失败: {e}")
        return reuse_map

    def load_sampled_frames_from_csv(self) -> None:
        """从single_analysis_results中的CSV文件加载采样帧信息"""
        if not self.output_dir:
            self.logger.warning("输出目录未设置，无法加载采样帧信息")
            return

        single_results_dir = os.path.join(self.output_dir, "single_analysis_results")
        if not os.path.exists(single_results_dir):
            self.logger.info("single_analysis_results目录不存在，跳过采样帧信息加载")
            return

        import csv

        loaded_count = 0
        for target in self.targets:
            # 从系统路径提取系统名称
            system_path = target.system_path
            system_dir_name = os.path.basename(system_path)
            csv_filename = f"frame_metrics_{system_dir_name}.csv"
            csv_path = os.path.join(single_results_dir, csv_filename)

            # 如果找不到匹配的CSV文件，尝试使用target的system_name
            if not os.path.exists(csv_path):
                csv_filename = f"frame_metrics_{target.system_name}.csv"
                csv_path = os.path.join(single_results_dir, csv_filename)

            if os.path.exists(csv_path):
                try:
                    sampled_frames = []
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        # 跳过注释行（以#开头的行）
                        lines = f.readlines()
                        data_start = 0
                        for i, line in enumerate(lines):
                            if not line.strip().startswith('#'):
                                data_start = i
                                break

                        # 读取CSV数据
                        csv_content = ''.join(lines[data_start:])
                        reader = csv.DictReader(csv_content.splitlines())

                        for row in reader:
                            frame_id = int(row['Frame_ID'])
                            selected = int(row['Selected'])
                            if selected == 1:
                                sampled_frames.append(frame_id)

                    # 更新目标的采样帧信息
                    target.sampled_frames = sampled_frames
                    loaded_count += 1
                    self.logger.debug(f"加载采样帧信息: {target.system_name} ({len(sampled_frames)} 帧)")

                except Exception as e:
                    self.logger.warning(f"读取采样帧信息失败 {target.system_name}: {str(e)}")
            else:
                self.logger.debug(f"采样帧CSV文件不存在: {csv_path}")

        if loaded_count > 0:
            self.logger.info(f"成功加载 {loaded_count} 个系统的采样帧信息")
        else:
            self.logger.info("未找到任何采样帧信息文件")

    def update_sampled_frames_from_results(self, analysis_results: List[tuple]) -> None:
        """从分析结果中直接更新采样帧信息到targets"""
        if not analysis_results:
            return
        
        # 建立 system_name -> sampled_frames 的映射
        sampled_frames_map = {}
        for result in analysis_results:
            if len(result) >= 2:
                metrics = result[0]
                system_name = getattr(metrics, 'system_name', 'unknown')
                sampled_frames = getattr(metrics, 'sampled_frames', [])
                sampled_frames_map[system_name] = sampled_frames
                
        # 同步采样帧到PathManager.targets
        updated_count = 0
        for target in self.targets:
            if target.system_name in sampled_frames_map:
                target.sampled_frames = sampled_frames_map[target.system_name]
                updated_count += 1
                self.logger.debug(f"更新采样帧: {target.system_name} ({len(target.sampled_frames)} 帧)")
                
        self.logger.info(f"从分析结果同步采样帧信息: {updated_count} 个系统")


def load_sampling_reuse_map(targets_file: str) -> Dict[str, Dict[str, object]]:
    """读取已有 analysis_targets.json，构建复用映射。

    Returns:
        dict: system_name -> { 'sampled_frames': List[int], 'source_hash': str }
    """
    reuse = {}
    if not targets_file or not os.path.exists(targets_file):
        return reuse
    try:
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
        logger = logging.getLogger('src.io.path_manager')
        logger.warning(f"读取采样复用信息失败: {e}")
    return reuse
