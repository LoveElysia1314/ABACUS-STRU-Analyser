#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
路径管理器：管理ABACUS分析项目的文件路径和系统信息
"""

import os
import json
import glob
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AnalysisTarget:
    """分析目标信息"""
    system_path: str
    mol_id: str
    conf: str
    temperature: str
    stru_files: List[str]
    creation_time: float
    status: str = "pending"  # pending, processing, completed, failed
    
    @property
    def system_name(self) -> str:
        """获取系统名称"""
        return f"struct_mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"
    
    @property
    def system_key(self) -> str:
        """获取系统唯一标识"""
        return f"mol_{self.mol_id}_conf_{self.conf}_T{self.temperature}K"

class PathManager:
    """路径管理器：负责管理和存储分析路径信息"""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = output_dir
        self.targets_file = os.path.join(output_dir, "analysis_targets.json")
        self.summary_file = os.path.join(output_dir, "path_summary.json")
        self.logger = logging.getLogger(__name__)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 分析目标列表
        self.targets: List[AnalysisTarget] = []
        self.mol_groups: Dict[str, List[AnalysisTarget]] = {}
    
    def load_from_discovery(self, mol_systems: Dict[str, List[str]]) -> None:
        """从目录发现结果加载分析目标"""
        self.targets.clear()
        self.mol_groups.clear()
        
        total_targets = 0
        for mol_id, system_paths in mol_systems.items():
            targets_for_mol = []
            
            for system_path in system_paths:
                # 解析系统信息
                basename = os.path.basename(system_path)
                import re
                match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', basename)
                if not match:
                    self.logger.warning(f"无法解析系统路径: {system_path}")
                    continue
                
                mol_id_parsed, conf, temp = match.groups()
                
                # 查找STRU文件
                stru_dir = os.path.join(system_path, "OUT.ABACUS", "STRU")
                stru_files = glob.glob(os.path.join(stru_dir, "STRU_MD_*"))
                
                if not stru_files:
                    self.logger.warning(f"系统 {basename} 没有找到STRU_MD文件")
                    continue
                
                # 获取创建时间
                try:
                    creation_time = os.path.getctime(system_path)
                except OSError:
                    creation_time = 0.0
                
                target = AnalysisTarget(
                    system_path=system_path,
                    mol_id=mol_id_parsed,
                    conf=conf,
                    temperature=temp,
                    stru_files=stru_files,
                    creation_time=creation_time
                )
                
                self.targets.append(target)
                targets_for_mol.append(target)
                total_targets += 1
            
            if targets_for_mol:
                self.mol_groups[mol_id] = targets_for_mol
        
        self.logger.info(f"路径管理器已加载 {len(self.mol_groups)} 个分子的 {total_targets} 个分析目标")
    
    def save_targets(self) -> None:
        """保存分析目标到文件"""
        targets_data = []
        for target in self.targets:
            target_dict = asdict(target)
            # 添加时间戳
            target_dict['created_at'] = datetime.fromtimestamp(target.creation_time).isoformat()
            targets_data.append(target_dict)
        
        with open(self.targets_file, 'w', encoding='utf-8') as f:
            json.dump(targets_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"分析目标已保存到: {self.targets_file}")
    
    def save_summary(self) -> None:
        """保存路径摘要信息"""
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_molecules": len(self.mol_groups),
            "total_systems": len(self.targets),
            "molecules": {}
        }
        
        for mol_id, targets in self.mol_groups.items():
            mol_summary = {
                "count": len(targets),
                "systems": [],
                "status_counts": {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
            }
            
            for target in targets:
                mol_summary["systems"].append({
                    "name": target.system_name,
                    "path": target.system_path,
                    "stru_files_count": len(target.stru_files),
                    "status": target.status
                })
                mol_summary["status_counts"][target.status] += 1
            
            summary["molecules"][mol_id] = mol_summary
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"路径摘要已保存到: {self.summary_file}")
    
    def load_targets(self) -> bool:
        """从文件加载分析目标"""
        if not os.path.exists(self.targets_file):
            return False
        
        try:
            with open(self.targets_file, 'r', encoding='utf-8') as f:
                targets_data = json.load(f)
            
            self.targets.clear()
            self.mol_groups.clear()
            
            for target_dict in targets_data:
                # 移除额外的时间戳字段
                target_dict.pop('created_at', None)
                target = AnalysisTarget(**target_dict)
                self.targets.append(target)
                
                # 重建分子分组
                mol_id = target.mol_id
                if mol_id not in self.mol_groups:
                    self.mol_groups[mol_id] = []
                self.mol_groups[mol_id].append(target)
            
            self.logger.info(f"已从文件加载 {len(self.targets)} 个分析目标")
            return True
        
        except Exception as e:
            self.logger.error(f"加载分析目标失败: {str(e)}")
            return False
    
    def get_targets_by_molecule(self, mol_id: str) -> List[AnalysisTarget]:
        """获取指定分子的分析目标"""
        return self.mol_groups.get(mol_id, [])
    
    def get_all_targets(self) -> List[AnalysisTarget]:
        """获取所有分析目标"""
        return self.targets.copy()
    
    def get_targets_by_status(self, status: str) -> List[AnalysisTarget]:
        """按状态获取分析目标"""
        return [target for target in self.targets if target.status == status]
    
    def update_target_status(self, system_path: str, status: str) -> bool:
        """更新分析目标状态"""
        for target in self.targets:
            if target.system_path == system_path:
                target.status = status
                return True
        return False
    
    def get_progress_summary(self) -> Dict[str, int]:
        """获取进度摘要"""
        summary = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
        for target in self.targets:
            summary[target.status] += 1
        summary["total"] = len(self.targets)
        return summary
    
    def export_target_paths(self, output_file: str = None) -> str:
        """导出分析目标路径列表"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "target_paths.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# ABACUS分析目标路径列表\n")
            f.write(f"# 生成时间: {datetime.now().isoformat()}\n")
            f.write(f"# 总计: {len(self.mol_groups)} 个分子, {len(self.targets)} 个体系\n\n")
            
            for mol_id, targets in self.mol_groups.items():
                f.write(f"## 分子 {mol_id} ({len(targets)} 个体系)\n")
                for target in targets:
                    f.write(f"{target.system_path}\n")
                    f.write(f"  STRU文件数: {len(target.stru_files)}\n")
                    f.write(f"  状态: {target.status}\n")
                f.write("\n")
        
        self.logger.info(f"目标路径列表已导出到: {output_file}")
        return output_file
    
    def validate_targets(self) -> Tuple[int, int]:
        """验证分析目标的有效性"""
        valid_count = 0
        invalid_count = 0
        
        for target in self.targets:
            try:
                # 检查系统路径是否存在
                if not os.path.exists(target.system_path):
                    self.logger.warning(f"系统路径不存在: {target.system_path}")
                    invalid_count += 1
                    continue
                
                # 检查STRU文件是否存在
                valid_stru = 0
                for stru_file in target.stru_files:
                    if os.path.exists(stru_file):
                        valid_stru += 1
                
                if valid_stru == 0:
                    self.logger.warning(f"系统 {target.system_name} 没有有效的STRU文件")
                    invalid_count += 1
                else:
                    valid_count += 1
            
            except Exception as e:
                self.logger.error(f"验证目标 {target.system_name} 时出错: {str(e)}")
                invalid_count += 1
        
        self.logger.info(f"目标验证完成: {valid_count} 个有效, {invalid_count} 个无效")
        return valid_count, invalid_count
