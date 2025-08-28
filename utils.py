#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具模块: utils.py
功能: ABACUS STRU 分析器通用工具函数
==================================================

包含的工具类和函数：
-----------------
1. LoggerManager: 统一的日志管理器
2. FileUtils: 文件操作工具
3. DataUtils: 数据处理工具
4. MathUtils: 数学计算工具

作者：LoveElysia1314
版本：v1.0
日期：2025年8月16日
"""

import os
import sys
import glob
import logging
import csv
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd


class LoggerManager:
    """统一的日志管理器"""
    
    @staticmethod
    def create_logger(name: str, level: int = logging.INFO, 
                     add_console: bool = True, 
                     log_file: Optional[str] = None,
                     log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
                     date_format: str = '%H:%M:%S') -> logging.Logger:
        """
        创建标准化的日志记录器
        
        Args:
            name: logger名称
            level: 日志级别
            add_console: 是否添加控制台输出
            log_file: 日志文件路径（可选）
            log_format: 日志格式
            date_format: 时间格式
            
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 清除现有处理器避免重复
        logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(log_format, datefmt=date_format)
        
        # 添加控制台处理器
        if add_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 添加文件处理器
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.propagate = False  # 防止传播到根logger
        return logger
    
    @staticmethod
    def add_file_handler(logger: logging.Logger, log_file: str,
                        log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
                        date_format: str = '%H:%M:%S') -> logging.FileHandler:
        """
        为现有logger添加文件处理器
        
        Args:
            logger: 现有的logger
            log_file: 日志文件路径
            log_format: 日志格式
            date_format: 时间格式
            
        Returns:
            创建的文件处理器
        """
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return file_handler
    
    @staticmethod
    def remove_handler(logger: logging.Logger, handler: logging.Handler) -> None:
        """安全地移除和关闭处理器"""
        if handler in logger.handlers:
            logger.removeHandler(handler)
        handler.close()


class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def ensure_dir(path: str) -> None:
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def find_files(pattern: str, search_dir: str = ".", recursive: bool = True) -> List[str]:
        """
        查找匹配模式的文件
        
        Args:
            pattern: 文件名模式（支持通配符）
            search_dir: 搜索目录
            recursive: 是否递归搜索
            
        Returns:
            匹配的文件路径列表
        """
        if recursive:
            full_pattern = os.path.join(search_dir, "**", pattern)
            return glob.glob(full_pattern, recursive=True)
        else:
            full_pattern = os.path.join(search_dir, pattern)
            return glob.glob(full_pattern)
    
    @staticmethod
    def find_file_prioritized(filename: str, search_dir: str = ".", 
                             priority_subdirs: Optional[List[str]] = None) -> Optional[str]:
        """
        按优先级查找文件
        
        Args:
            filename: 要查找的文件名
            search_dir: 搜索目录
            priority_subdirs: 优先搜索的子目录列表
            
        Returns:
            找到的文件路径，如果未找到则返回None
        """
        # 在当前目录查找
        current_path = os.path.join(search_dir, filename)
        if os.path.exists(current_path):
            return current_path
        
        # 在优先子目录中查找
        if priority_subdirs:
            for subdir in priority_subdirs:
                priority_path = os.path.join(search_dir, subdir, filename)
                if os.path.exists(priority_path):
                    return priority_path
        
        # 递归查找
        files = FileUtils.find_files(filename, search_dir, recursive=True)
        return files[0] if files else None
    
    @staticmethod
    def safe_write_csv(filepath: str, data: List[List[Any]], 
                      headers: Optional[List[str]] = None,
                      encoding: str = 'utf-8-sig') -> bool:
        """
        安全地写入CSV文件
        
        Args:
            filepath: 文件路径
            data: 数据行列表
            headers: 表头（可选）
            encoding: 编码格式
            
        Returns:
            是否写入成功
        """
        try:
            FileUtils.ensure_dir(os.path.dirname(filepath))
            with open(filepath, 'w', newline='', encoding=encoding) as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            return True
        except Exception:
            return False


class DataUtils:
    """数据处理工具类"""
    
    @staticmethod
    def to_python_types(data: Any) -> Any:
        """
        将numpy类型转换为Python原生类型
        
        Args:
            data: 输入数据（可以是单个值、列表或序列）
            
        Returns:
            转换后的Python原生类型数据
        """
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            # 处理序列类型
            result = []
            for item in data:
                if hasattr(item, 'item'):  # numpy类型有item()方法
                    result.append(item.item())
                elif isinstance(item, (np.integer, np.floating, np.complexfloating)):
                    result.append(item.item())
                elif str(item).isdigit():
                    result.append(int(item))
                else:
                    result.append(item)
            return result
        else:
            # 处理单个值
            if hasattr(data, 'item'):
                return data.item()
            elif isinstance(data, (np.integer, np.floating, np.complexfloating)):
                return data.item()
            elif str(data).isdigit():
                return int(data)
            else:
                return data
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全除法，避免除零错误"""
        return numerator / denominator if abs(denominator) > 1e-12 else default
    
    @staticmethod
    def format_number(value: float, precision: int = 6) -> str:
        """格式化数字为字符串"""
        return f"{value:.{precision}f}"
    
    @staticmethod
    def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        """
        检查DataFrame是否包含必需的列
        
        Args:
            df: DataFrame
            required_columns: 必需的列名列表
            
        Returns:
            缺失的列名列表
        """
        return [col for col in required_columns if col not in df.columns]
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        清理DataFrame，移除指定列中包含NaN的行
        
        Args:
            df: DataFrame
            columns: 要检查的列名列表
            
        Returns:
            清理后的DataFrame
        """
        return df[columns].dropna()


class MathUtils:
    """数学计算工具类"""
    
    @staticmethod
    def safe_sqrt(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """安全开平方根"""
        if isinstance(value, np.ndarray):
            return np.sqrt(np.maximum(0, value))
        else:
            return np.sqrt(max(0, value))
    
    @staticmethod
    def safe_log(value: float, base: Optional[float] = None) -> float:
        """安全对数计算"""
        value = max(1e-12, value)
        if base is None:
            return np.log(value)
        else:
            return np.log(value) / np.log(base)
    
    @staticmethod
    def normalize_vector(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        归一化向量
        
        Args:
            vector: 输入向量
            epsilon: 小数阈值
            
        Returns:
            归一化后的向量
        """
        norm = np.linalg.norm(vector)
        return vector / norm if norm > epsilon else vector
    
    @staticmethod
    def power_mean(arr: np.ndarray, p: float) -> float:
        """
        计算幂平均
        
        Args:
            arr: 输入数组
            p: 幂指数
            
        Returns:
            幂平均值
        """
        arr = np.asarray(arr)
        arr = np.maximum(arr, 1e-12)
        
        if p == 0:  # 几何平均
            return np.exp(np.mean(np.log(arr)))
        elif p == 1:  # 算术平均
            return np.mean(arr)
        elif p == -1:  # 调和平均
            return len(arr) / np.sum(1.0 / arr)
        else:  # 幂平均
            return (np.mean(arr ** p)) ** (1.0 / p)
    
    @staticmethod
    def calculate_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
        """计算均方根误差"""
        return np.sqrt(np.mean((predicted - actual) ** 2))
    
    @staticmethod
    def calculate_correlation_strength(abs_r: float) -> str:
        """获取相关性强度解释"""
        if abs_r < 0.3:
            return "弱相关"
        elif abs_r < 0.7:
            return "中等相关"
        else:
            return "强相关"
    
    @staticmethod
    def calculate_effect_size_interpretation(eta_squared: float) -> str:
        """获取效应量解释"""
        if eta_squared < 0.01:
            return "无效应"
        elif eta_squared < 0.06:
            return "小效应"
        elif eta_squared < 0.14:
            return "中等效应"
        else:
            return "大效应"


class ValidationUtils:
    """数据验证工具类"""
    
    @staticmethod
    def validate_file_exists(filepath: str) -> bool:
        """验证文件是否存在"""
        return os.path.exists(filepath) and os.path.isfile(filepath)
    
    @staticmethod
    def validate_sample_size(data: Union[pd.DataFrame, np.ndarray, List], 
                           min_size: int = 2) -> bool:
        """验证样本量是否足够"""
        return len(data) >= min_size
    
    @staticmethod
    def validate_numeric_data(data: Union[pd.Series, np.ndarray, List]) -> bool:
        """验证数据是否为数值型"""
        try:
            np.asarray(data, dtype=float)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_groups_for_anova(groups: List[np.ndarray], min_group_size: int = 2) -> bool:
        """验证方差分析的组数据是否有效"""
        if len(groups) < 2:
            return False
        return all(len(group) >= min_group_size for group in groups)


# 常用常量
class Constants:
    """常用常量定义"""
    
    # 数值常量
    EPSILON = 1e-12
    SMALL_NUMBER = 1e-6
    
    # 统计显著性水平
    ALPHA_0_05 = 0.05
    ALPHA_0_01 = 0.01
    ALPHA_0_001 = 0.001
    
    # 相关性强度阈值
    WEAK_CORRELATION = 0.3
    MODERATE_CORRELATION = 0.7
    
    # 效应量阈值
    SMALL_EFFECT = 0.01
    MEDIUM_EFFECT = 0.06
    LARGE_EFFECT = 0.14
    
    # 默认文件编码
    DEFAULT_ENCODING = 'utf-8-sig'
    
    # 默认日志格式
    DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%H:%M:%S'


# 便捷函数
def create_standard_logger(name: str, output_dir: Optional[str] = None, 
                          log_filename: str = "analysis.log") -> logging.Logger:
    """
    创建标准的分析日志记录器
    
    Args:
        name: logger名称
        output_dir: 输出目录（如果提供则创建日志文件）
        log_filename: 日志文件名
        
    Returns:
        配置好的日志记录器
    """
    log_file = None
    if output_dir:
        log_file = os.path.join(output_dir, log_filename)
    
    return LoggerManager.create_logger(
        name=name,
        add_console=True,
        log_file=log_file,
        log_format=Constants.DEFAULT_LOG_FORMAT,
        date_format=Constants.DEFAULT_DATE_FORMAT
    )


def safe_float_format(value: float, precision: int = 6) -> str:
    """安全格式化浮点数"""
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return "0.000000"


def safe_int_convert(value: Any) -> int:
    """安全转换为整数"""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


class DirectoryDiscovery:
    """目录发现和管理工具"""
    
    @staticmethod
    def find_abacus_systems(search_path: str = None) -> Dict[str, List[str]]:
        """
        递归查找ABACUS系统目录
        
        Args:
            search_path: 搜索根路径，默认为当前目录的父目录
            
        Returns:
            Dict[mol_id, List[system_paths]]: 按分子ID分组的系统路径
        """
        import re
        from collections import defaultdict
        
        if search_path is None:
            search_path = os.path.dirname(os.getcwd())
        
        logger = logging.getLogger(__name__)
        logger.info(f"开始递归搜索目录: {search_path}")
        
        # 发现所有候选系统
        systems = DirectoryDiscovery._discover_candidates(search_path)
        if not systems:
            logger.warning("未找到符合格式的系统目录")
            return {}
        
        # 按分子归类并去重
        grouped = DirectoryDiscovery._group_and_deduplicate(systems, logger)
        
        # 验证并返回最终结果
        validated = DirectoryDiscovery._validate_systems(grouped, logger)
        
        logger.info(f"目录分析完成: 发现 {len(validated)} 个分子")
        for mol_id, systems in validated.items():
            logger.info(f"  分子 {mol_id}: {len(systems)} 个体系")
        
        return validated
    
    @staticmethod
    def _discover_candidates(search_path: str) -> List[Dict]:
        """发现所有候选系统目录"""
        import re
        
        systems = []
        pattern = re.compile(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K$')
        
        for root, dirs, files in os.walk(search_path):
            for dir_name in dirs:
                match = pattern.match(dir_name)
                if match:
                    mol_id, conf, temp = match.groups()
                    dir_path = os.path.join(root, dir_name)
                    
                    # 检查STRU目录和文件
                    stru_dir = os.path.join(dir_path, "OUT.ABACUS", "STRU")
                    if os.path.exists(stru_dir):
                        stru_files = glob.glob(os.path.join(stru_dir, "STRU_MD_*"))
                        if stru_files:
                            try:
                                creation_time = os.path.getctime(dir_path)
                            except OSError:
                                creation_time = 0.0
                            
                            systems.append({
                                'path': dir_path,
                                'mol_id': mol_id,
                                'conf': conf,
                                'temperature': temp,
                                'creation_time': creation_time,
                                'stru_files': stru_files,
                                'system_key': f"mol_{mol_id}_conf_{conf}_T{temp}K"
                            })
        
        return systems
    
    @staticmethod
    def _group_and_deduplicate(systems: List[Dict], logger) -> Dict[str, List[Dict]]:
        """按分子归类并处理重复系统"""
        from collections import defaultdict
        
        # 按系统标识分组
        system_groups = defaultdict(list)
        for system in systems:
            system_groups[system['system_key']].append(system)
        
        # 去重：选择创建时间最晚的
        deduplicated = {}
        duplicate_count = 0
        
        for system_key, candidates in system_groups.items():
            if len(candidates) > 1:
                selected = max(candidates, key=lambda x: x['creation_time'])
                duplicate_count += len(candidates) - 1
                logger.info(f"发现重复体系 {system_key}: {len(candidates)} 个，选择最新的")
            else:
                selected = candidates[0]
            
            # 按分子ID归类
            mol_id = selected['mol_id']
            if mol_id not in deduplicated:
                deduplicated[mol_id] = []
            deduplicated[mol_id].append(selected)
        
        if duplicate_count > 0:
            logger.info(f"去重完成: 处理了 {duplicate_count} 个重复体系")
        
        return deduplicated
    
    @staticmethod
    def _validate_systems(grouped_systems: Dict[str, List[Dict]], logger) -> Dict[str, List[str]]:
        """验证系统并返回路径列表"""
        validated = {}
        invalid_count = 0
        
        for mol_id, systems in grouped_systems.items():
            valid_paths = []
            for system in systems:
                try:
                    # 验证STRU文件可读性
                    if system['stru_files']:
                        test_file = system['stru_files'][0]
                        with open(test_file, 'r', encoding='utf-8') as f:
                            content = f.read(1000)
                            if 'LATTICE_CONSTANT' in content or 'LATTICE_VECTORS' in content:
                                valid_paths.append(system['path'])
                            else:
                                logger.warning(f"系统 {system['system_key']} 的STRU文件格式异常")
                                invalid_count += 1
                    else:
                        logger.warning(f"系统 {system['system_key']} 没有找到STRU_MD文件")
                        invalid_count += 1
                except Exception as e:
                    logger.warning(f"验证系统 {system['system_key']} 时出错: {str(e)}")
                    invalid_count += 1
            
            if valid_paths:
                validated[mol_id] = valid_paths
        
        if invalid_count > 0:
            logger.warning(f"验证完成: {invalid_count} 个系统被排除")
        
        return validated
