#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STRU文件解析器模块：stru_parser.py
功能: 解析ABACUS分子动力学模拟生成的STRU文件
"""

import os
import re
import glob
import numpy as np
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FrameData:
    """单帧数据结构"""
    frame_id: int
    positions: np.ndarray
    elements: List[str]
    distance_vector: Optional[np.ndarray] = None


class StrUParser:
    """ABACUS STRU文件解析器"""
    
    def __init__(self, exclude_hydrogen: bool = True):
        """
        初始化解析器
        
        Args:
            exclude_hydrogen: 是否排除氢原子
        """
        self.exclude_hydrogen = exclude_hydrogen
        self.logger = logging.getLogger(__name__)
    
    def parse_file(self, stru_file: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        解析单个STRU文件
        
        Args:
            stru_file: STRU文件路径
            
        Returns:
            (positions, elements) 或 None
        """
        try:
            with open(stru_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            self.logger.warning(f"无法读取文件 {stru_file}: {e}")
            return None

        try:
            return self._parse_lines(lines)
        except Exception as e:
            self.logger.warning(f"解析文件 {stru_file} 时出错: {e}")
            return None
    
    def _parse_lines(self, lines: List[str]) -> Optional[Tuple[np.ndarray, List[str]]]:
        """解析STRU文件内容"""
        lattice_constant = 1.0
        positions = []
        elements = []
        current_element = None
        element_atoms_count = 0
        element_atoms_collected = 0
        section = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # 识别节段
            if "LATTICE_CONSTANT" in line:
                section = "LATTICE_CONSTANT"
                continue
            elif "LATTICE_VECTORS" in line:
                section = "LATTICE_VECTORS"
                continue
            elif "ATOMIC_SPECIES" in line:
                section = "ATOMIC_SPECIES"
                continue
            elif "ATOMIC_POSITIONS" in line:
                section = "ATOMIC_POSITIONS"
                continue
                
            # 解析各节段内容
            if section == "LATTICE_CONSTANT":
                lattice_constant = self._parse_lattice_constant(line)
            elif section == "ATOMIC_POSITIONS":
                result = self._parse_atomic_positions_line(
                    line, current_element, element_atoms_count, element_atoms_collected,
                    positions, elements, lattice_constant
                )
                if result:
                    current_element, element_atoms_count, element_atoms_collected = result

        if not positions:
            return None
            
        return np.array(positions), elements
    
    def _parse_lattice_constant(self, line: str) -> float:
        """解析晶格常数"""
        try:
            return float(re.split(r'\s+', line)[0])
        except (ValueError, IndexError):
            return 1.0
    
    def _parse_atomic_positions_line(self, line: str, current_element: str, 
                                   element_atoms_count: int, element_atoms_collected: int,
                                   positions: List, elements: List, 
                                   lattice_constant: float) -> Optional[Tuple]:
        """解析原子位置节段的一行"""
        # 元素声明行
        if re.match(r'^[A-Za-z]{1,2}\s*#', line):
            parts = re.split(r'\s+', line)
            current_element = parts[0]
            element_atoms_count = 0
            element_atoms_collected = 0
            return current_element, element_atoms_count, element_atoms_collected

        # 原子数量行
        if current_element and "number of atoms" in line:
            try:
                element_atoms_count = int(re.split(r'\s+', line)[0])
            except (ValueError, IndexError):
                element_atoms_count = 0
            return current_element, element_atoms_count, element_atoms_collected

        # 原子坐标行
        if (current_element and element_atoms_count > 0 and 
            element_atoms_collected < element_atoms_count):
            
            # 如果排除氢原子
            if self.exclude_hydrogen and current_element.upper() in ("H", "HYDROGEN"):
                element_atoms_collected += 1
                return current_element, element_atoms_count, element_atoms_collected
            
            try:
                parts = re.split(r'\s+', line)
                if len(parts) < 3:
                    return current_element, element_atoms_count, element_atoms_collected
                
                coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                positions.append(np.array(coords) * lattice_constant)
                elements.append(current_element)
                element_atoms_collected += 1
                
            except (ValueError, IndexError):
                pass
            
            return current_element, element_atoms_count, element_atoms_collected
        
        return None
    
    def parse_trajectory(self, stru_dir: str) -> List[FrameData]:
        """
        解析整个轨迹目录
        
        Args:
            stru_dir: STRU文件目录路径
            
        Returns:
            FrameData列表，按frame_id排序
        """
        stru_files = glob.glob(os.path.join(stru_dir, 'STRU_MD_*'))
        if not stru_files:
            self.logger.warning(f"在 {stru_dir} 中未找到STRU_MD_*文件")
            return []
        
        frames = []
        for stru_file in stru_files:
            # 提取帧ID
            match = re.search(r'STRU_MD_(\d+)', os.path.basename(stru_file))
            if not match:
                continue
                
            frame_id = int(match.group(1))
            
            # 解析文件
            result = self.parse_file(stru_file)
            if result is None:
                continue
                
            positions, elements = result
            if len(positions) < 2:  # 至少需要2个原子
                continue
                
            frames.append(FrameData(frame_id, positions, elements))
        
        # 按frame_id排序
        frames.sort(key=lambda x: x.frame_id)
        self.logger.info(f"解析轨迹完成: {len(frames)} 帧有效数据")
        
        return frames


def parse_abacus_stru(stru_file: str, exclude_hydrogen: bool = True) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    兼容性函数：解析单个ABACUS STRU文件
    
    Args:
        stru_file: STRU文件路径
        exclude_hydrogen: 是否排除氢原子
        
    Returns:
        (positions, elements) 或 None
    """
    parser = StrUParser(exclude_hydrogen=exclude_hydrogen)
    return parser.parse_file(stru_file)