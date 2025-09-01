#!/usr/bin/env python
"""
脚本名: correlation_analyser.py
功能: ABACUS STRU 轨迹分析相关性分析器
==================================================

功能特性：
---------
✨ 专业的统            self.logger.info(f"分子: {DataUtils.to_python_types(molecules)}")
            self.logger.info(f"构象编号: {DataUtils.to_python_types(configs)}")
            self.logger.info(f"温度: {DataUtils.to_python_types(temperatures)}K")模块，用于分析 ABACUS STRU 轨迹数据
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
- parameter_analysis_results.csv：全局分析结果（整合数值和可读信息）
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

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# 导入工具模块
from ..utils import (
    Constants,
    DataUtils,
    FileUtils,
    LoggerManager,
    MathUtils,
    ValidationUtils,
)


class CorrelationAnalyser:
    """相关性分析器类"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化相关性分析器

        Args:
            logger: 日志记录器，如果为None则创建默认日志记录器
        """
        self.logger = logger if logger is not None else self._create_default_logger()

        # 定义分析指标（原参数和采样后参数）
        self.indicators = [
            "RMSD_Mean",
            "MinD",
            "ANND",
            "MPD",
        ]

    def _create_default_logger(self) -> logging.Logger:
        """创建默认日志记录器"""
        return LoggerManager.create_logger(
            name="CorrelationAnalyser",
            level=logging.INFO,
            add_console=True,
            log_format=Constants.DEFAULT_LOG_FORMAT,
            date_format=Constants.DEFAULT_DATE_FORMAT,
        )

    # 补充：统一列表/数组到纯 Python list 的安全转换，避免 AttributeError
    def _to_python_list(self, obj):  # 轻量工具，保持与日志调用兼容
        try:
            import numpy as np  # 局部导入，避免全局依赖

            def _convert(x):
                # 将 numpy 标量、安全类型转为内置类型，保持稳定日志输出
                if isinstance(x, np.generic):
                    return x.item()
                return x

            if isinstance(obj, (list, tuple, set)):
                return [_convert(x) for x in obj]
            if hasattr(obj, 'tolist'):
                v = obj.tolist()
                if isinstance(v, list):
                    return [_convert(x) for x in v]
                return [_convert(v)]
            return [_convert(obj)]
        except Exception:
            return [str(obj)]



    def analyse_correlations(self, csv_file_path: str, output_dir: str) -> bool:
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
            if (
                hasattr(self.logger, "name")
                and self.logger.name != "CorrelationAnalyser"
            ):
                # 使用外部logger时，添加文件记录到analysis_results目录
                analysis_results_dir = os.path.join(FileUtils.get_project_root(), "analysis_results")
                os.makedirs(analysis_results_dir, exist_ok=True)
                log_file = os.path.join(
                    analysis_results_dir, "correlation_analysis.log"
                )
                file_handler = LoggerManager.add_file_handler(
                    self.logger,
                    log_file,
                    Constants.DEFAULT_LOG_FORMAT,
                    Constants.DEFAULT_DATE_FORMAT,
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
            required_columns = ["Configuration", "Temperature(K)"] + self.indicators
            missing_columns = DataUtils.check_required_columns(df, required_columns)
            if missing_columns:
                self.logger.error(f"CSV文件缺少必要的列: {missing_columns}")
                return False

            # 检查是否有Molecule_ID列，如果没有则添加警告
            if "Molecule_ID" not in df.columns:
                self.logger.warning("未找到Molecule_ID列，将假设所有数据来自同一分子")
                df["Molecule_ID"] = "Unknown"

            # 获取所有分子、构象和温度（保持原始类型，避免强制转为int导致重复或混乱）
            molecules = sorted(df["Molecule_ID"].unique())
            configs = sorted(df["Configuration"].unique())
            temperatures = sorted(df["Temperature(K)"].unique())

            self.logger.info(
                f"发现 {len(molecules)} 个分子、{len(configs)} 种构象编号和 {len(temperatures)} 种温度"
            )
            self.logger.info(f"分子: {self._to_python_list(molecules)}")
            self.logger.info(f"构象编号: {self._to_python_list(configs)}")
            self.logger.info(f"温度: {self._to_python_list(temperatures)}K")

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 执行全局分析（移除分子级分析）
            global_temp_results = self._analyse_global_temperature_correlations(df)
            global_config_results = self._analyse_global_configuration_effects(df)

            # 保存结果（传入空列表替代分子级结果）
            self._save_results(
                [], [], global_temp_results, global_config_results, output_dir
            )
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
            if "file_handler" in locals() and file_handler:
                LoggerManager.remove_handler(self.logger, file_handler)

            return False

    def _analyse_global_temperature_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """分析温度相关性（按单变量控制原则：固定分子和构象，分析温度效应）"""
        global_temp_results = []

        # 检查是否有足够的温度变化
        unique_temps = df["Temperature(K)"].unique()
        total_systems = len(df)

        if not ValidationUtils.validate_sample_size(unique_temps, min_size=2):
            self.logger.warning(
                f"温度种类不足({len(unique_temps)}<2)，跳过温度相关性分析"
            )
            return global_temp_results

        # 按(分子,构象)分组，只保留样本数>=2的组
        valid_groups = []
        filtered_groups = []

        for mol_id in df["Molecule_ID"].unique():
            mol_data = df[df["Molecule_ID"] == mol_id]
            for config in mol_data["Configuration"].unique():
                group_data = mol_data[mol_data["Configuration"] == config]
                group_data = DataUtils.clean_dataframe(
                    group_data, ["Temperature(K)"] + self.indicators
                )

                if len(group_data) >= 2:
                    valid_groups.append(
                        {
                            "molecule": mol_id,
                            "config": config,
                            "data": group_data,
                            "size": len(group_data),
                        }
                    )
                else:
                    filtered_groups.append(
                        {"molecule": mol_id, "config": config, "size": len(group_data)}
                    )

        sum(group["size"] for group in valid_groups)
        total_filtered_samples = sum(group["size"] for group in filtered_groups)

        if not valid_groups:
            self.logger.warning("没有找到样本数>=2的(分子,构象)组，跳过温度相关性分析")
            return global_temp_results

        # 记录分组信息
        self.logger.info("温度分析分组情况:")
        self.logger.info(f"  有效组: {len(valid_groups)}个")
        if filtered_groups:
            self.logger.info(f"  过滤组: {len(filtered_groups)}个(组内样本不足2个)")

        for indicator in self.indicators:
            # 合并所有有效组的数据进行相关性分析
            all_temps = []
            all_values = []

            for group in valid_groups:
                group_data = group["data"]
                if indicator in group_data.columns:
                    all_temps.extend(group_data["Temperature(K)"].tolist())
                    all_values.extend(group_data[indicator].tolist())

            if len(all_temps) < 2:
                continue

            # Pearson相关系数（线性相关）
            try:
                pearson_r, pearson_p = stats.pearsonr(all_temps, all_values)
                if np.isnan(pearson_r) or np.isnan(pearson_p):
                    self.logger.warning(f"指标 '{indicator}' 的Pearson相关系数计算结果无效，跳过")
                    continue
            except Exception as e:
                self.logger.warning(f"指标 '{indicator}' 的Pearson相关系数计算失败: {str(e)}，跳过")
                continue

            # Spearman相关系数（秩相关）
            try:
                spearman_r, spearman_p = stats.spearmanr(all_temps, all_values)
                if np.isnan(spearman_r) or np.isnan(spearman_p):
                    self.logger.warning(f"指标 '{indicator}' 的Spearman相关系数计算结果无效，跳过")
                    continue
            except Exception as e:
                self.logger.warning(f"指标 '{indicator}' 的Spearman相关系数计算失败: {str(e)}，跳过")
                continue

            significance = "Yes" if pearson_p < 0.05 else "No"

            global_temp_results.append(
                {
                    "Analysis_Type": "Controlled_Temperature",
                    "Variable": "Temperature",
                    "Indicator": indicator,
                    "Sample_Size": len(all_temps),
                    "Total_Systems": total_systems,
                    "Filtered_Systems": total_filtered_samples,
                    "Valid_Groups": len(valid_groups),
                    "Filtered_Groups": len(filtered_groups),
                    "Temperature_Range": f"{min(unique_temps)}-{max(unique_temps)}K",
                    "Pearson_r": pearson_r,
                    "Pearson_p": pearson_p,
                    "Spearman_r": spearman_r,
                    "Spearman_p": spearman_p,
                    "Significance": significance,
                }
            )

        return global_temp_results

    def _analyse_global_configuration_effects(self, df: pd.DataFrame) -> List[Dict]:
        """分析构象效应（按单变量控制原则：固定分子和温度，分析构象效应）"""
        global_config_results = []

        # 检查是否有足够的构象变化
        unique_configs = df["Configuration"].unique()
        total_systems = len(df)

        if not ValidationUtils.validate_sample_size(unique_configs, min_size=2):
            self.logger.warning(
                f"构象种类不足({len(unique_configs)}<2)，跳过构象效应分析"
            )
            return global_config_results

        # 按(分子,温度)分组，只保留样本数>=2的组
        valid_groups = []
        filtered_groups = []

        for mol_id in df["Molecule_ID"].unique():
            mol_data = df[df["Molecule_ID"] == mol_id]
            for temp in mol_data["Temperature(K)"].unique():
                group_data = mol_data[mol_data["Temperature(K)"] == temp]
                group_data = DataUtils.clean_dataframe(
                    group_data, ["Configuration"] + self.indicators
                )

                if len(group_data) >= 2:
                    valid_groups.append(
                        {
                            "molecule": mol_id,
                            "temperature": temp,
                            "data": group_data,
                            "size": len(group_data),
                        }
                    )
                else:
                    filtered_groups.append(
                        {
                            "molecule": mol_id,
                            "temperature": temp,
                            "size": len(group_data),
                        }
                    )

        sum(group["size"] for group in valid_groups)
        total_filtered_samples = sum(group["size"] for group in filtered_groups)

        if not valid_groups:
            self.logger.warning("没有找到样本数>=2的(分子,温度)组，跳过构象效应分析")
            return global_config_results

        # 记录分组信息
        self.logger.info("构象分析分组情况:")
        self.logger.info(f"  有效组: {len(valid_groups)}个")
        if filtered_groups:
            self.logger.info(f"  过滤组: {len(filtered_groups)}个(组内样本不足2个)")
        for indicator in self.indicators:
            # 准备各构象组的数据（来自所有有效组）
            config_data_dict = {}

            for group in valid_groups:
                group_data = group["data"]
                if indicator in group_data.columns:
                    for _, row in group_data.iterrows():
                        config = row["Configuration"]
                        value = row[indicator]
                        if config not in config_data_dict:
                            config_data_dict[config] = []
                        config_data_dict[config].append(value)

            # 检查是否有足够的构象组进行比较
            if len(config_data_dict) < 2:
                continue

            groups = []
            config_labels = []
            group_sample_counts = []

            for config in sorted(config_data_dict.keys()):
                group_data = np.array(config_data_dict[config])
                if len(group_data) > 0:
                    groups.append(group_data)
                    config_labels.append(config)
                    group_sample_counts.append(len(group_data))

            # 确保至少有两个有效组
            if not ValidationUtils.validate_sample_size(groups, min_size=2):
                continue

            # 检查每组的样本量，每组至少需要2个样本才能进行ANOVA（计算方差）
            valid_groups_for_anova = [group for group in groups if len(group) >= 2]
            if len(valid_groups_for_anova) < 2:
                self.logger.warning(f"指标 '{indicator}' 的有效组不足2个（每组需要至少2个样本），跳过构象效应分析")
                continue

            # 记录哪些构象组被过滤掉
            filtered_configs = []
            for i, config in enumerate(config_labels):
                if i >= len(groups) or len(groups[i]) < 2:
                    filtered_configs.append(config)

            if filtered_configs:
                self.logger.info(f"过滤的构象组（样本量不足）: {filtered_configs}")

            # 执行单因素方差分析
            try:
                f_stat, p_value = stats.f_oneway(*valid_groups_for_anova)

                # 检查结果是否有效
                if np.isnan(f_stat) or np.isnan(p_value):
                    self.logger.warning(f"指标 '{indicator}' 的方差分析结果无效，跳过")
                    continue

            except Exception as e:
                self.logger.warning(f"指标 '{indicator}' 的方差分析执行失败: {str(e)}，跳过")
                continue

            # 计算Eta平方（效应量）
            all_values = np.concatenate(valid_groups_for_anova)
            ss_total = np.sum((all_values - all_values.mean()) ** 2)
            ss_between = sum(
                [
                    len(group) * (np.mean(group) - all_values.mean()) ** 2
                    for group in valid_groups_for_anova
                    if len(group) > 0
                ]
            )
            eta_squared = DataUtils.safe_divide(ss_between, ss_total, default=0.0)

            significance = "Yes" if p_value < 0.05 else "No"

            # 更新有效的构象标签（只包含有足够样本的组）
            valid_config_labels = []
            for i, config in enumerate(config_labels):
                if i < len(groups) and len(groups[i]) >= 2:
                    valid_config_labels.append(config)

            # 计算各组的描述性统计
            group_stats = {}
            for i, config in enumerate(valid_config_labels):
                if i < len(valid_groups_for_anova):
                    group_data = valid_groups_for_anova[i]
                    group_stats[f"Config_{config}_mean"] = np.mean(group_data)
                    group_stats[f"Config_{config}_std"] = np.std(group_data, ddof=1) if len(group_data) > 1 else 0.0
                    group_stats[f"Config_{config}_n"] = len(group_data)

            global_config_results.append(
                {
                    "Analysis_Type": "Controlled_Configuration",
                    "Variable": "Configuration",
                    "Indicator": indicator,
                    "Sample_Size": len(all_values),
                    "Total_Systems": total_systems,
                    "Filtered_Systems": total_filtered_samples,
                    "Valid_Groups": len(valid_groups),
                    "Filtered_Groups": len(filtered_groups),
                    "Configurations": valid_config_labels,
                    "F_statistic": f_stat,
                    "P_value": p_value,
                    "Eta_squared": eta_squared,
                    "Significance": significance,
                    **group_stats,
                }
            )

        return global_config_results

    def _get_confidence_level(self, p_value: float) -> str:
        """获取置信程度评价"""
        if p_value < 0.001:
            return "99.9%置信"
        elif p_value < 0.01:
            return "99%置信"
        elif p_value < 0.05:
            return "95%置信"
        elif p_value < 0.1:
            return "90%置信"
        else:
            return "不显著"

    def _save_results(
        self,
        temp_results: List[Dict],
        config_results: List[Dict],
        global_temp_results: List[Dict],
        global_config_results: List[Dict],
        output_dir: str,
    ) -> None:
        """保存分析结果到统一的CSV文件（结构化、去冗余的输出）

        输出列（规范）：
        Analysis_Type, Indicator, Statistic, P_Value, Effect_Size, Significance, Eval,
        Valid_Samples, Total_Systems, Valid_Groups, Filtered_Systems, Spearman_r,
        Temp_Range, Configs, Notes
        """
        main_csv_path = os.path.join(output_dir, "parameter_analysis_results.csv")
        main_data = []

        # 温度相关性（每行代表一个指标）
        for result in global_temp_results:
            r_val = result.get("Pearson_r")
            p_val = result.get("Pearson_p")
            sample_size = result.get("Sample_Size")
            total = result.get("Total_Systems")
            valid_groups = result.get("Valid_Groups")
            filtered_systems = result.get("Filtered_Systems")
            spearman = result.get("Spearman_r")
            temp_range = result.get("Temperature_Range")

            # 生成可读评价（Eval）并确定显著性（由数值决定）
            eval_text = self._get_temperature_correlation_evaluation(
                r_val if r_val is not None else float("nan"),
                p_val if p_val is not None else float("nan"),
            )

            notes = []
            # 兼容可能的逻辑冲突（CSV 中原先写的 Interpretation）
            if "Interpretation" in result and result.get("Interpretation"):
                notes.append(f"orig_interp={result.get('Interpretation')}")

            main_data.append(
                [
                    "Temp_Corr",
                    result.get("Indicator"),
                    DataUtils.format_number(r_val),
                    DataUtils.format_number(p_val),
                    DataUtils.format_number(abs(r_val) if r_val is not None else None),
                    "Yes" if (p_val is not None and not np.isnan(p_val) and p_val < 0.05) else "No",
                    eval_text,
                    int(sample_size) if sample_size is not None else None,
                    int(total) if total is not None else None,
                    int(valid_groups) if valid_groups is not None else None,
                    int(filtered_systems) if filtered_systems is not None else None,
                    DataUtils.format_number(spearman),
                    temp_range,
                    None,
                    "; ".join(notes) if notes else None,
                ]
            )

        # 构象效应（ANOVA）
        for result in global_config_results:
            f_stat = result.get("F_statistic")
            p_val = result.get("P_value")
            eta_sq = result.get("Eta_squared")
            sample_size = result.get("Sample_Size")
            total = result.get("Total_Systems")
            valid_groups = result.get("Valid_Groups")
            filtered_systems = result.get("Filtered_Systems")
            configs = result.get("Configurations")

            notes = []
            if p_val is None or (isinstance(p_val, float) and np.isnan(p_val)):
                notes.append("ANOVA not computable: insufficient groups/data")

            eval_text = self._get_configuration_effect_evaluation(
                eta_sq if eta_sq is not None else float("nan"),
                p_val if p_val is not None else float("nan"),
            )

            # 将 configs 序列化为字符串以便 CSV 存储
            configs_str = None
            if configs is not None:
                try:
                    configs_str = ",".join(map(str, configs))
                except Exception:
                    configs_str = str(configs)

            main_data.append(
                [
                    "Config_Effect",
                    result.get("Indicator"),
                    DataUtils.format_number(f_stat),
                    DataUtils.format_number(p_val),
                    DataUtils.format_number(eta_sq),
                    "Yes" if (p_val is not None and not np.isnan(p_val) and p_val < 0.05) else "No",
                    eval_text,
                    int(sample_size) if sample_size is not None else None,
                    int(total) if total is not None else None,
                    int(valid_groups) if valid_groups is not None else None,
                    int(filtered_systems) if filtered_systems is not None else None,
                    None,
                    None,
                    configs_str,
                    "; ".join(notes) if notes else None,
                ]
            )

        # 写入结构化CSV
        FileUtils.safe_write_csv(
            main_csv_path,
            main_data,
            headers=[
                "Analysis_Type",
                "Indicator",
                "Statistic",
                "P_Value",
                "Effect_Size",
                "Significance",
                "Eval",
                "Valid_Samples",
                "Total_Systems",
                "Valid_Groups",
                "Filtered_Systems",
                "Spearman_r",
                "Temp_Range",
                "Configs",
                "Notes",
            ],
            encoding="utf-8-sig",
        )

    def _log_summary(
        self,
        temp_results: List[Dict],
        config_results: List[Dict],
        global_temp_results: List[Dict],
        global_config_results: List[Dict],
    ) -> None:
        """输出分析总结"""
        self.logger.info("=" * 60)
        self.logger.info("相关性分析总结")
        self.logger.info("=" * 60)

        # 数据概览信息（合并避免重复）
        if global_temp_results or global_config_results:
            # 从任一结果中提取基本信息
            source_result = global_temp_results[0] if global_temp_results else global_config_results[0]
            total_samples = source_result["Total_Systems"]

            self.logger.info(f"数据概览:")
            self.logger.info(f"   总体系数: {total_samples}")

            if global_temp_results:
                first_temp = global_temp_results[0]
                valid_samples_temp = first_temp["Sample_Size"]
                filtered_samples_temp = first_temp["Filtered_Systems"]
                valid_groups_temp = first_temp["Valid_Groups"]
                temp_range = first_temp["Temperature_Range"]
                self.logger.info(f"   温度分析: {valid_samples_temp}/{total_samples}有效样本 (过滤{filtered_samples_temp})")
                self.logger.info(f"   温度范围: {temp_range}")

            if global_config_results:
                first_config = global_config_results[0]
                valid_samples_config = first_config["Sample_Size"]
                filtered_samples_config = first_config["Filtered_Systems"]
                valid_groups_config = first_config["Valid_Groups"]
                configs = first_config["Configurations"]
                self.logger.info(f"   构象分析: {valid_samples_config}/{total_samples}有效样本 (过滤{filtered_samples_config})")
                self.logger.info(f"   构象类型: {DataUtils.to_python_types(sorted(configs))}")

        # 温度相关性分析结果
        if global_temp_results:
            self.logger.info(f"\n温度相关性分析:")
            significant_count = sum(1 for r in global_temp_results if r["Significance"] == "Yes")
            self.logger.info(f"   显著相关指标: {significant_count}/{len(global_temp_results)}")

            for result in global_temp_results:
                evaluation = self._get_temperature_correlation_evaluation(
                    result["Pearson_r"], result["Pearson_p"]
                )
                self.logger.info(
                    f"   {result['Indicator']:<15} r={result['Pearson_r']:.3f} (p={result['Pearson_p']:.3f}) - {evaluation}"
                )

        # 构象效应分析结果
        if global_config_results:
            self.logger.info(f"\n构象效应分析:")
            significant_count = sum(1 for r in global_config_results if r["Significance"] == "Yes")
            self.logger.info(f"   显著效应指标: {significant_count}/{len(global_config_results)}")

            for result in global_config_results:
                # 处理 NaN 值
                f_stat = result["F_statistic"]
                p_value = result["P_value"]
                eta_sq = result["Eta_squared"]

                evaluation = self._get_configuration_effect_evaluation(eta_sq, p_value)
                if p_value is not None and not (isinstance(p_value, float) and np.isnan(p_value)) and p_value < 0.05:
                    evaluation += f" (η²={eta_sq:.3f})"
                
                if np.isnan(f_stat) or np.isnan(p_value):
                    stat_info = "F=nan (p=nan)"
                else:
                    stat_info = f"F={f_stat:.3f} (p={p_value:.3f})"

                self.logger.info(
                    f"   {result['Indicator']:<15} {stat_info}, η²={eta_sq:.3f} - {evaluation}"
                )

        self.logger.info("=" * 60)

    def _get_temperature_correlation_evaluation(self, r: float, p_value: float) -> str:
        """根据新标准自动评价温度相关性"""
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            return "数据不足，无法计算"
        
        if r is None or (isinstance(r, float) and np.isnan(r)):
            return "相关系数无效，无法计算"
        
        if p_value < 0.05:
            abs_r = abs(r)
            if abs_r >= 0.5:
                strength = "强相关"
            elif abs_r >= 0.3:
                strength = "中等相关"
            elif abs_r >= 0.1:
                strength = "弱相关"
            else:
                strength = "极弱相关"
            confidence = self._get_confidence_level(p_value)
            return f"{strength}, {confidence}置信"
        else:
            return "无相关, 不显著"

    def _get_configuration_effect_evaluation(self, eta_squared: float, p_value: float) -> str:
        """根据新标准自动评价构象效应"""
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            return "数据不足，无法计算"
        
        if p_value < 0.05:
            if eta_squared >= 0.04:
                effect_size = "中等效应"
            elif eta_squared >= 0.01:
                effect_size = "小效应"
            else:
                effect_size = "微弱效应"
            return f"显著, {effect_size}"
        else:
            return "无效应, 不显著"


def setup_file_logger(output_dir: str) -> logging.Logger:
    """设置文件日志记录器"""
    # 使用新的集中式日志管理器
    return LoggerManager.create_analysis_logger(
        name="CorrelationAnalyser",
        output_dir=output_dir,
        log_filename="correlation_analysis.log"
    )


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
        priority_subdirs=["combined_analysis_results"],
    )


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="ABACUS STRU 轨迹分析相关性分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 自动查找当前目录下的system_metrics_summary.csv并分析（默认启用日志文件）
  python correlation_analyser.py

  # 指定输入文件
  python correlation_analyser.py -i analysis_results/combined_analysis_results/system_metrics_summary.csv

  # 指定输入文件和输出目录
  python correlation_analyser.py -i data.csv -o combined_results

  # 禁用日志文件，仅输出到控制台
  python correlation_analyser.py --no-log-file
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="输入的system_metrics_summary.csv文件路径（如不指定则自动查找）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="combined_analysis_results",
        help="输出目录路径（默认: combined_analysis_results）",
    )
    parser.add_argument(
        "--no-log-file", action="store_true", help="禁用日志文件输出，仅输出到控制台"
    )

    args = parser.parse_args()

    # 确定输入文件路径
    if args.input:
        csv_file_path = args.input
        if not os.path.exists(csv_file_path):
            logger = logging.getLogger(__name__)
            logger.error(f"错误: 指定的输入文件不存在: {csv_file_path}")
            sys.exit(1)
    else:
        # 自动查找
        csv_file_path = find_system_metrics_csv()
        if csv_file_path is None:
            logger = logging.getLogger(__name__)
            logger.error("未找到 system_metrics_summary.csv 文件")
            logger.error("请使用 -i 参数指定输入文件路径")
            sys.exit(1)
        else:
            logger = logging.getLogger(__name__)
            logger.info(f"自动找到输入文件: {csv_file_path}")

    # 设置日志 - 默认启用文件日志
    if args.no_log_file:
        logger = None  # 仅使用默认控制台日志
    else:
        # 使用新的集中式日志管理器
        analysis_results_dir = os.path.join(FileUtils.get_project_root(), "analysis_results")
        logger = LoggerManager.create_analysis_logger(
            name="CorrelationAnalyser",
            output_dir=analysis_results_dir,
            log_filename="correlation_analysis.log"
        )
        # 记录独立运行的日志到analysis_results目录，确保UTF-8编码
        logger.info("已启用文件日志记录 (UTF-8 编码, 输出到 analysis_results 目录)")

    # 创建分析器并执行分析
    analyser = CorrelationAnalyser(logger=logger)

    log_runtime = logging.getLogger("CorrelationAnalyserRuntime")
    log_runtime.info(f"开始分析文件: {csv_file_path}")
    log_runtime.info(f"输出目录: {args.output}")
    if not args.no_log_file:
        log_runtime.info("日志文件: analysis_results/correlation_analysis.log")

    success = analyser.analyse_correlations(csv_file_path, args.output)

    if success:
        log_runtime.info(f"分析完成，结果已保存到: {args.output}")
        log_runtime.info("输出文件:")
        log_runtime.info(f"  - {os.path.join(args.output, 'parameter_analysis_results.csv')} (整合数值和可读信息)")
        if not args.no_log_file:
            log_runtime.info("  - analysis_results/correlation_analysis.log")
    else:
        log_runtime.error("分析失败，请检查输入文件格式和内容")
        sys.exit(1)


if __name__ == "__main__":
    main()
