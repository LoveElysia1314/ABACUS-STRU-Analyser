#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本名: correlation_analyzer.py
功能: ABACUS STRU 轨迹分析相关性分析器
==================================================

功能特性：
---------
✨ 专业的统计分析模块，用于分析 ABACUS STRU 轨迹数据
🔬 科学严谨的统计方法，确保结果可靠性
📊 智能的样本量检查和质量控制
🛡️ 稳健的分析策略，避免小样本偏误

使用方式：
---------
1. 独立脚本运行：分析指定的 CSV 文件
2. 模块调用：集成到主程序分析流程

输入要求：
---------
- system_metrics_summary.csv 文件
- 包含分子ID、构象、温度和各项指标

输出结果：
---------
- parameter_analysis_results.csv：全局分析详细结果
- parameter_analysis_summary.csv：全局分析汇总
- correlation_analysis.log：分析日志

核心特性：
---------
- 全局温度相关性分析（大样本，高可靠性）
- 全局构象效应分析（跨所有分子和温度）
- 完整的统计检验和效应量评估
- 简化的输出格式，聚焦全局结果

作者：LoveElysia1314
版本：v3.0
日期：2025年8月16日
更新：重构为独立模块，增强统计稳健性
"""

import os
import sys
import csv
import argparse
import logging
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
from scipy import stats

# 导入工具模块
from utils import (
    LoggerManager, FileUtils, DataUtils, MathUtils, ValidationUtils, 
    Constants, create_standard_logger
)


class CorrelationAnalyzer:
    """相关性分析器类"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化相关性分析器
        
        Args:
            logger: 日志记录器，如果为None则创建默认日志记录器
        """
        self.logger = logger if logger is not None else self._create_default_logger()
        
        # 定义分析指标
        self.indicators = [
            'nRMSF', 'MCV', 'avg_nLdRMS', 
            'nRMSF_sampled', 'MCV_sampled', 'avg_nLdRMS_sampled'
        ]
    
    def _create_default_logger(self) -> logging.Logger:
        """创建默认日志记录器"""
        return LoggerManager.create_logger(
            name='CorrelationAnalyzer',
            level=logging.INFO,
            add_console=True,
            log_format=Constants.DEFAULT_LOG_FORMAT,
            date_format=Constants.DEFAULT_DATE_FORMAT
        )
    
    def _to_python_list(self, seq):
        """将序列中的numpy类型转为Python原生类型"""
        return DataUtils.to_python_types(seq)
    
    def analyze_correlations(self, csv_file_path: str, output_dir: str) -> bool:
        """
        分析初始构象、温度与各指标的相关性
        
        Args:
            csv_file_path: system_metrics_summary.csv 文件路径
            output_dir: 输出目录路径
            
        Returns:
            bool: 分析是否成功完成
        """
        try:
            # 确保输出目录存在
            FileUtils.ensure_dir(output_dir)
            
            # 如果使用外部logger，为相关性分析创建额外的文件处理器
            file_handler = None
            if hasattr(self.logger, 'name') and self.logger.name != 'CorrelationAnalyzer':
                # 使用外部logger时，添加文件记录到analysis_results目录
                analysis_results_dir = os.path.join(os.getcwd(), "analysis_results")
                os.makedirs(analysis_results_dir, exist_ok=True)
                log_file = os.path.join(analysis_results_dir, "correlation_analysis.log")
                file_handler = LoggerManager.add_file_handler(
                    self.logger, log_file,
                    Constants.DEFAULT_LOG_FORMAT, 
                    Constants.DEFAULT_DATE_FORMAT,
                    encoding='utf-8'
                )
            
            # 检查输入文件是否存在
            if not ValidationUtils.validate_file_exists(csv_file_path):
                self.logger.error(f"输入文件不存在: {csv_file_path}")
                return False
            
            # 读取CSV数据
            df = pd.read_csv(csv_file_path)
            self.logger.info(f"成功读取数据文件: {csv_file_path}")
            self.logger.info("开始进行相关性分析（遵循单一变量原则）...")
            
            # 检查必要的列是否存在
            required_columns = ['Configuration', 'Temperature(K)'] + self.indicators
            missing_columns = DataUtils.check_required_columns(df, required_columns)
            if missing_columns:
                self.logger.error(f"CSV文件缺少必要的列: {missing_columns}")
                return False
            
            # 检查是否有Molecule_ID列，如果没有则添加警告
            if 'Molecule_ID' not in df.columns:
                self.logger.warning("未找到Molecule_ID列，将假设所有数据来自同一分子")
                df['Molecule_ID'] = 'Unknown'
            
            # 获取所有分子、构象和温度（保持原始类型，避免强制转为int导致重复或混乱）
            molecules = sorted(df['Molecule_ID'].unique())
            configs = sorted(df['Configuration'].unique())
            temperatures = sorted(df['Temperature(K)'].unique())
            
            self.logger.info(f"发现 {len(molecules)} 个分子、{len(configs)} 种构象编号和 {len(temperatures)} 种温度")
            self.logger.info(f"分子: {self._to_python_list(molecules)}")
            self.logger.info(f"构象编号: {self._to_python_list(configs)}")
            self.logger.info(f"温度: {self._to_python_list(temperatures)}K")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 执行全局分析（移除分子级分析）
            global_temp_results = self._analyze_global_temperature_correlations(df)
            global_config_results = self._analyze_global_configuration_effects(df)
            
            # 保存结果（传入空列表替代分子级结果）
            self._save_results([], [], global_temp_results, global_config_results, output_dir)
            self.logger.info(f"相关性分析结果已保存到 {output_dir}")
            
            # 输出总结（传入空列表替代分子级结果）
            self._log_summary([], [], global_temp_results, global_config_results)
            
            # 清理添加的文件处理器
            if file_handler:
                LoggerManager.remove_handler(self.logger, file_handler)
            
            return True
            
        except Exception as e:
            self.logger.error(f"相关性分析出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 清理添加的文件处理器
            if 'file_handler' in locals() and file_handler:
                LoggerManager.remove_handler(self.logger, file_handler)
            
            return False
    
    # 以下分子级分析方法已不再使用，保留仅供参考
    def _analyze_temperature_correlations_unused(self, df: pd.DataFrame, molecules: List) -> List[Dict]:
        """分析温度与各指标的相关性（以分子为单位控制构象变量）"""
        temp_results = []
        
        # 统计有效的分子-构象组合
        valid_combinations = 0
        total_combinations = 0
        insufficient_data_count = 0
        
        for molecule in molecules:
            # 获取该分子的所有数据
            mol_df = df[df['Molecule_ID'] == molecule]
            configs_in_mol = sorted(mol_df['Configuration'].unique())
            
            for config in configs_in_mol:
                total_combinations += 1
                # 在单一分子的单一构象下分析温度影响
                mol_config_df = mol_df[mol_df['Configuration'] == config]
                
                # 检查该分子构象下是否有足够的温度变化
                unique_temps = mol_config_df['Temperature(K)'].unique()
                if len(unique_temps) < 2:
                    insufficient_data_count += 1
                    continue
                
                # 样本量太小（通常<5），统计意义有限，但仍记录用于汇总
                valid_combinations += 1
                
                for indicator in self.indicators:
                    # 检查数据有效性
                    valid_data = mol_config_df[['Temperature(K)', indicator]].dropna()
                    if len(valid_data) < 2:
                        continue
                    
                    # Pearson相关系数（线性相关）
                    pearson_r, pearson_p = stats.pearsonr(
                        valid_data['Temperature(K)'], valid_data[indicator]
                    )
                    # Spearman相关系数（秩相关）
                    spearman_r, spearman_p = stats.spearmanr(
                        valid_data['Temperature(K)'], valid_data[indicator]
                    )
                    
                    temp_results.append({
                        'Molecule_ID': molecule,
                        'Configuration': config,
                        'Variable': 'Temperature',
                        'Indicator': indicator,
                        'Sample_Size': len(valid_data),
                        'Temperature_Range': f"{min(unique_temps)}-{max(unique_temps)}K",
                        'Pearson_r': pearson_r,
                        'Pearson_p': pearson_p,
                        'Spearman_r': spearman_r,
                        'Spearman_p': spearman_p,
                        'Significance': 'Yes' if pearson_p < 0.05 else 'No'
                    })
        
        return temp_results
    
    def _analyze_configuration_effects_unused(self, df: pd.DataFrame, molecules: List) -> List[Dict]:
        """分析构象与各指标的关系（以分子为单位控制温度变量）"""
        config_results = []
        
        # 统计有效的分子-温度组合
        valid_combinations = 0
        total_combinations = 0
        insufficient_data_count = 0
        
        for molecule in molecules:
            # 获取该分子的所有数据
            mol_df = df[df['Molecule_ID'] == molecule]
            temperatures_in_mol = sorted(mol_df['Temperature(K)'].unique())
            
            for temp in temperatures_in_mol:
                total_combinations += 1
                # 在单一分子的单一温度下分析构象影响
                mol_temp_df = mol_df[mol_df['Temperature(K)'] == temp]
                
                # 检查该分子温度下是否有多种构象
                available_configs = mol_temp_df['Configuration'].unique()
                if len(available_configs) < 2:
                    insufficient_data_count += 1
                    continue
                
                # 样本量太小，统计意义有限，但仍记录用于汇总
                valid_combinations += 1
                
                for indicator in self.indicators:
                    # 检查数据有效性
                    valid_mol_temp_df = mol_temp_df[[indicator, 'Configuration']].dropna()
                    if len(valid_mol_temp_df) < 2:
                        continue
                    
                    # 准备各构象组的数据
                    groups = []
                    config_labels = []
                    for config in available_configs:
                        group_data = valid_mol_temp_df[valid_mol_temp_df['Configuration'] == config][indicator].values
                        if len(group_data) > 0:
                            groups.append(group_data)
                            config_labels.append(config)
                    
                    # 确保至少有一个有效组，且每组至少有2个样本
                    if len(groups) < 1:
                        continue
                    
                    # 检查每组的样本量，每组至少需要2个样本才能进行ANOVA
                    if any(len(group) < 2 for group in groups):
                        continue
                    
                    # 执行单因素方差分析
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # 计算Eta平方（效应量）
                    indicator_data = valid_mol_temp_df[indicator]
                    ss_total = np.sum((indicator_data - indicator_data.mean()) ** 2)
                    ss_between = sum([
                        len(group) * (np.mean(group) - indicator_data.mean()) ** 2 
                        for group in groups if len(group) > 0
                    ])
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    # 计算各组的描述性统计
                    group_stats = {}
                    for i, config in enumerate(config_labels):
                        group_data = groups[i]
                        if len(group_data) > 0:
                            group_stats[f'Config_{config}_mean'] = np.mean(group_data)
                            group_stats[f'Config_{config}_std'] = np.std(group_data)
                            group_stats[f'Config_{config}_n'] = len(group_data)
                    
                    config_results.append({
                        'Molecule_ID': molecule,
                        'Temperature': temp,
                        'Variable': 'Configuration',
                        'Indicator': indicator,
                        'Configurations': sorted(config_labels),
                        'F_statistic': f_stat,
                        'P_value': p_value,
                        'Eta_squared': eta_squared,
                        'Significance': 'Yes' if p_value < 0.05 else 'No',
                        **group_stats
                    })
        
        return config_results
    
    def _analyze_global_temperature_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """分析全局温度相关性（跨所有分子和构象）"""
        global_temp_results = []
        
        # 检查是否有足够的温度变化
        unique_temps = df['Temperature(K)'].unique()
        if not ValidationUtils.validate_sample_size(unique_temps, min_size=2):
            return global_temp_results
        
        for indicator in self.indicators:
            # 检查数据有效性
            valid_data = DataUtils.clean_dataframe(df, ['Temperature(K)', indicator])
            if not ValidationUtils.validate_sample_size(valid_data, min_size=2):
                continue
            
            # Pearson相关系数（线性相关）
            pearson_r, pearson_p = stats.pearsonr(
                valid_data['Temperature(K)'], valid_data[indicator]
            )
            # Spearman相关系数（秩相关）
            spearman_r, spearman_p = stats.spearmanr(
                valid_data['Temperature(K)'], valid_data[indicator]
            )
            
            global_temp_results.append({
                'Analysis_Type': 'Global_Temperature',
                'Variable': 'Temperature',
                'Indicator': indicator,
                'Sample_Size': len(valid_data),
                'Temperature_Range': f"{min(unique_temps)}-{max(unique_temps)}K",
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'Significance': 'Yes' if pearson_p < 0.05 else 'No'
            })
        
        return global_temp_results
    
    def _analyze_global_configuration_effects(self, df: pd.DataFrame) -> List[Dict]:
        """分析全局构象效应（跨所有分子和温度）"""
        global_config_results = []
        
        # 检查是否有足够的构象变化
        unique_configs = df['Configuration'].unique()
        if not ValidationUtils.validate_sample_size(unique_configs, min_size=2):
            return global_config_results
        
        for indicator in self.indicators:
            # 检查数据有效性
            valid_data = DataUtils.clean_dataframe(df, ['Configuration', indicator])
            if not ValidationUtils.validate_sample_size(valid_data, min_size=2):
                continue
            
            # 准备各构象组的数据
            groups = []
            config_labels = []
            for config in unique_configs:
                group_data = valid_data[valid_data['Configuration'] == config][indicator].values
                if len(group_data) > 0:
                    groups.append(group_data)
                    config_labels.append(config)
            
            # 确保至少有两个有效组
            if not ValidationUtils.validate_sample_size(groups, min_size=2):
                continue
            
            # 检查每组的样本量，每组至少需要2个样本才能进行ANOVA
            if not ValidationUtils.validate_groups_for_anova(groups, min_group_size=2):
                continue
            
            # 执行单因素方差分析
            f_stat, p_value = stats.f_oneway(*groups)
            
            # 计算Eta平方（效应量）
            indicator_data = valid_data[indicator]
            ss_total = np.sum((indicator_data - indicator_data.mean()) ** 2)
            ss_between = sum([
                len(group) * (np.mean(group) - indicator_data.mean()) ** 2 
                for group in groups if len(group) > 0
            ])
            eta_squared = DataUtils.safe_divide(ss_between, ss_total, default=0.0)
            
            # 计算各组的描述性统计
            group_stats = {}
            for i, config in enumerate(config_labels):
                group_data = groups[i]
                if len(group_data) > 0:
                    group_stats[f'Config_{config}_mean'] = np.mean(group_data)
                    group_stats[f'Config_{config}_std'] = np.std(group_data)
                    group_stats[f'Config_{config}_n'] = len(group_data)
            
            global_config_results.append({
                'Analysis_Type': 'Global_Configuration',
                'Variable': 'Configuration',
                'Indicator': indicator,
                'Sample_Size': len(valid_data),
                'Configurations': sorted(config_labels),
                'F_statistic': f_stat,
                'P_value': p_value,
                'Eta_squared': eta_squared,
                'Significance': 'Yes' if p_value < 0.05 else 'No',
                **group_stats
            })
        
        return global_config_results
    
    def _get_correlation_strength(self, abs_r: float) -> str:
        """获取相关性强度解释"""
        return MathUtils.calculate_correlation_strength(abs_r)
    
    def _get_effect_size_interpretation(self, eta_squared: float) -> str:
        """获取效应量解释"""
        return MathUtils.calculate_effect_size_interpretation(eta_squared)
    
    def _save_results(self, temp_results: List[Dict], config_results: List[Dict], global_temp_results: List[Dict], global_config_results: List[Dict], output_dir: str) -> None:
        """保存分析结果到CSV文件（仅保存全局分析结果）"""
        # 主要结果：仅保存全局相关性分析，移除重复字段简化格式
        main_csv_path = os.path.join(output_dir, "parameter_analysis_results.csv")
        main_data = []
        
        # 保存全局温度相关性结果
        for result in global_temp_results:
            main_data.append([
                'Temperature_Correlation', result['Indicator'],
                DataUtils.format_number(result['Pearson_r']), 
                DataUtils.format_number(result['Pearson_p']),
                DataUtils.format_number(abs(result['Pearson_r'])),
                result['Significance'],
                self._get_correlation_strength(abs(result['Pearson_r'])),
                result['Sample_Size'],
                f"Spearman_r={result['Spearman_r']:.3f}; Range={result['Temperature_Range']}"
            ])
        
        # 保存全局构象效应结果
        for result in global_config_results:
            configs_str = ','.join(map(str, result['Configurations']))
            main_data.append([
                'Configuration_Effect', result['Indicator'],
                DataUtils.format_number(result['F_statistic']), 
                DataUtils.format_number(result['P_value']),
                DataUtils.format_number(result['Eta_squared']),
                result['Significance'],
                self._get_effect_size_interpretation(result['Eta_squared']),
                result['Sample_Size'],
                f"Configs=[{configs_str}]"
            ])
        
        # 写入主要结果文件
        FileUtils.safe_write_csv(
            main_csv_path, main_data,
            headers=[
                'Analysis_Type', 'Indicator', 'Statistic_Value', 'P_value', 
                'Effect_Size', 'Significance', 'Interpretation', 'Sample_Size', 'Additional_Info'
            ],
            encoding='utf-8'
        )
        
        # 汇总表：仅保存全局分析汇总
        summary_csv_path = os.path.join(output_dir, "parameter_analysis_summary.csv")
        summary_data = []
        
        # 全局温度相关性汇总
        for result in global_temp_results:
            summary_data.append([
                'Temperature_Correlation', result['Indicator'],
                result['Significance'],
                f"{abs(result['Pearson_r']):.3f}",
                f"r={result['Pearson_r']:.3f}, p={result['Pearson_p']:.3f}"
            ])
        
        # 全局构象效应汇总
        for result in global_config_results:
            summary_data.append([
                'Configuration_Effect', result['Indicator'],
                result['Significance'],
                f"{result['Eta_squared']:.3f}",
                f"F={result['F_statistic']:.3f}, p={result['P_value']:.3f}, η²={result['Eta_squared']:.3f}"
            ])
        
        # 写入汇总文件
        FileUtils.safe_write_csv(
            summary_csv_path, summary_data,
            headers=[
                'Analysis_Type', 'Indicator', 'Significance', 
                'Effect_Size', 'Statistic_Info'
            ],
            encoding='utf-8'
        )
    
    def _log_summary(self, temp_results: List[Dict], config_results: List[Dict], global_temp_results: List[Dict], global_config_results: List[Dict]) -> None:
        """输出分析总结"""
        self.logger.info("=" * 50)
        self.logger.info("相关性分析总结:")
        
        # 数据概览信息（仅使用全局数据）
        total_samples = global_temp_results[0]['Sample_Size'] if global_temp_results else 0
        temp_range = global_temp_results[0]['Temperature_Range'] if global_temp_results else "未知"
        configs = global_config_results[0]['Configurations'] if global_config_results else []
        
        self.logger.info(f"数据概览: {total_samples}个样本, 温度范围{temp_range}, 构象{self._to_python_list(sorted(configs))}")
        
        # 全局温度相关性分析
        if global_temp_results:
            significant_global = [r for r in global_temp_results if r['Significance'] == 'Yes']
            self.logger.info(f"全局温度相关性: {len(significant_global)}/{len(global_temp_results)}个指标显著相关")
            
            for result in global_temp_results:
                significance_mark = "***" if result['Significance'] == 'Yes' else ""
                corr_strength = self._get_correlation_strength(abs(result['Pearson_r']))
                self.logger.info(f"  {result['Indicator']}: r={result['Pearson_r']:.3f} (p={result['Pearson_p']:.3f}) {significance_mark} - {corr_strength}")
            
            if significant_global:
                strongest_global = max(significant_global, key=lambda x: abs(x['Pearson_r']))
                self.logger.info(f"  最强相关: {strongest_global['Indicator']} (r={strongest_global['Pearson_r']:.3f})")
        
        # 全局构象效应分析
        if global_config_results:
            significant_global_config = [r for r in global_config_results if r['Significance'] == 'Yes']
            self.logger.info(f"全局构象效应: {len(significant_global_config)}/{len(global_config_results)}个指标显著")
            
            for result in global_config_results:
                significance_mark = "***" if result['Significance'] == 'Yes' else ""
                effect_strength = self._get_effect_size_interpretation(result['Eta_squared'])
                self.logger.info(f"  {result['Indicator']}: F={result['F_statistic']:.3f} (p={result['P_value']:.3f}), η²={result['Eta_squared']:.3f} {significance_mark} - {effect_strength}")
            
            if significant_global_config:
                strongest_config = max(significant_global_config, key=lambda x: x['Eta_squared'])
                self.logger.info(f"  最强效应: {strongest_config['Indicator']} (η²={strongest_config['Eta_squared']:.3f})")
        
        self.logger.info("=" * 50)


def setup_file_logger(output_dir: str) -> logging.Logger:
    """设置文件日志记录器"""
    # 日志输出到analysis_results目录
    analysis_results_dir = os.path.join(os.getcwd(), "analysis_results")
    os.makedirs(analysis_results_dir, exist_ok=True)
    
    logger = logging.getLogger('CorrelationAnalyzer')
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 文件处理器 - 使用更合适的日志文件名
    log_file = os.path.join(analysis_results_dir, "correlation_analysis.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')

    # 控制台处理器，强制UTF-8编码
    try:
        console_handler = logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
    except Exception:
        # 某些环境下sys.stdout.fileno()不可用，退回默认
        console_handler = logging.StreamHandler(sys.stdout)

    # 格式设置
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def find_system_metrics_csv(search_dir: str = ".") -> Optional[str]:
    """
    在指定目录及其子目录中查找 system_metrics_summary.csv 文件
    
    Args:
        search_dir: 搜索目录，默认为当前目录
        
    Returns:
        str: 找到的CSV文件路径，如果未找到则返回None
    """
    return FileUtils.find_file_prioritized(
        filename="system_metrics_summary.csv",
        search_dir=search_dir,
        priority_subdirs=["combined_analysis_results"]
    )


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description='ABACUS STRU 轨迹分析相关性分析器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 自动查找当前目录下的system_metrics_summary.csv并分析（默认启用日志文件）
  python correlation_analyzer.py
  
  # 指定输入文件
  python correlation_analyzer.py -i analysis_results/combined_analysis_results/system_metrics_summary.csv
  
  # 指定输入文件和输出目录
  python correlation_analyzer.py -i data.csv -o combined_results
  
  # 禁用日志文件，仅输出到控制台
  python correlation_analyzer.py --no-log-file
        """
    )
    
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        help='输入的system_metrics_summary.csv文件路径（如不指定则自动查找）'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='combined_analysis_results',
        help='输出目录路径（默认: combined_analysis_results）'
    )
    parser.add_argument(
        '--no-log-file', 
        action='store_true',
        help='禁用日志文件输出，仅输出到控制台'
    )
    
    args = parser.parse_args()
    
    # 确定输入文件路径
    if args.input:
        csv_file_path = args.input
        if not os.path.exists(csv_file_path):
            print(f"错误: 指定的输入文件不存在: {csv_file_path}")
            sys.exit(1)
    else:
        # 自动查找
        csv_file_path = find_system_metrics_csv()
        if csv_file_path is None:
            print("错误: 未找到 system_metrics_summary.csv 文件")
            print("请使用 -i 参数指定输入文件路径")
            sys.exit(1)
        else:
            print(f"自动找到输入文件: {csv_file_path}")
    
    # 设置日志 - 默认启用文件日志
    if args.no_log_file:
        logger = None  # 仅使用默认控制台日志
    else:
        logger = setup_file_logger(args.output)
        # 记录独立运行的日志到analysis_results目录，确保UTF-8编码
        logger.info("已启用文件日志记录 (UTF-8 编码, 输出到 analysis_results 目录)")
    
    # 创建分析器并执行分析
    analyzer = CorrelationAnalyzer(logger=logger)
    
    print(f"开始分析文件: {csv_file_path}")
    print(f"输出目录: {args.output}")
    if not args.no_log_file:
        print(f"日志文件: analysis_results/correlation_analysis.log")
    
    success = analyzer.analyze_correlations(csv_file_path, args.output)
    
    if success:
        print(f"\n分析完成！结果已保存到: {args.output}")
        print("输出文件:")
        print(f"  - {os.path.join(args.output, 'parameter_analysis_results.csv')}")
        print(f"  - {os.path.join(args.output, 'parameter_analysis_summary.csv')}")
        if not args.no_log_file:
            print(f"  - analysis_results/correlation_analysis.log")
    else:
        print("分析失败，请检查输入文件格式和内容")
        sys.exit(1)


if __name__ == "__main__":
    main()
