#!/usr/bin/env python
"""
采样效果比较分析器
功能：比较智能采样算法与随机/均匀采样的效果
已重构：核心工具函数已迁移到统一模块，保留业务逻辑
"""

import os
import logging
import numpy as np
import pandas as pd
from glob import glob
from typing import List
import warnings

from ...utils import LoggerManager
from ...core.metrics import MetricsToolkit
from ...core.sampler import SamplingStrategy, calculate_improvement, calculate_significance
from ...core.metrics import get_headers_by_categories
from ...io.stru_parser import StrUParser
from ...core.system_analyser import RMSDCalculator

warnings.filterwarnings('ignore')


class SamplingComparisonAnalyser:
    """采样效果比较分析器类"""

    def __init__(self):
        self.logger = LoggerManager.create_logger(__name__)

    def analyse_sampling_compare(self, result_dir=None):
        """采样效果比较分析主函数"""
        # 自动定位结果目录
        if result_dir is None:
            dirs = sorted(glob('analysis_results/run_*'), reverse=True)
            if not dirs:
                self.logger.warning("未找到分析结果，请先运行主程序。")
                return
            result_dir = dirs[0]

        # 加载系统路径映射
        targets_file = os.path.join(result_dir, 'analysis_targets.json')
        system_paths = {}
        if os.path.exists(targets_file):
            try:
                import json
                with open(targets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for mol_data in data.get('molecules', {}).values():
                        for sys_name, sys_data in mol_data.get('systems', {}).items():
                            system_paths[sys_name] = sys_data.get('system_path', '')
            except Exception as e:
                self.logger.warning(f"加载系统路径映射失败: {e}")

        single_dir = os.path.join(result_dir, 'single_analysis_results')
        files = glob(os.path.join(single_dir, 'frame_metrics_*.csv'))

        if not files:
            self.logger.warning("未找到帧指标文件")
            return

        rows = []
        for f in files:
            row = self._analyze_single_system(f, system_paths)
            if row:
                rows.append(row)

        if rows:
            self._save_comparison_results(rows, result_dir)
            self._create_summary_table(rows, result_dir)

    def _analyze_single_system(self, file_path, system_paths):
        """分析单个系统的数据"""
        try:
            df = pd.read_csv(file_path)
            system = os.path.basename(file_path).replace('frame_metrics_', '').replace('.csv', '')
            system_path = system_paths.get(system, '')

            # 准备数据
            vector_cols = [col for col in df.columns if (col == 'Energy_Standardized' or col.startswith('PC'))]
            vectors = df[vector_cols].values
            selected = df['Selected'] == 1
            k = selected.sum()
            n = len(df)
            sample_ratio = k / n if n > 0 else 0

            if k == 0 or n == 0:
                return None

            # 获取帧索引
            frame_indices = df['Frame_ID'].values
            sampled_indices = frame_indices[selected]

            # 重新计算采样组的RMSD（基于本组mean structure）
            sampled_rmsd = []
            if system_path:
                sampled_rmsd = self._calculate_group_rmsd(system_path, sampled_indices.tolist())
                if len(sampled_rmsd) == 0:
                    self.logger.warning(f"无法计算采样组RMSD，使用原有RMSD数据")
                    sampled_rmsd = pd.to_numeric(df.loc[selected, 'RMSD'], errors='coerce').values if 'RMSD' in df.columns else []
            else:
                self.logger.warning(f"未找到系统路径 {system}，使用原有RMSD数据")
                sampled_rmsd = pd.to_numeric(df.loc[selected, 'RMSD'], errors='coerce').values if 'RMSD' in df.columns else []

            # 采样算法结果
            sampled_metrics = MetricsToolkit.adapt_sampling_metrics(
                vectors[selected], vectors,
                sampled_rmsd if len(sampled_rmsd) > 0 else []
            )

            # 随机采样比较
            rand_results = self._run_random_sampling_comparison(vectors, selected, df, system_path, sampled_indices, k, n)

            # 均匀采样比较
            uniform_metrics = self._run_uniform_sampling_comparison(vectors, df, system_path, k, n)

            # 构建结果行
            return self._build_result_row(
                system, sample_ratio, n, k,
                sampled_metrics, rand_results, uniform_metrics
            )

        except Exception as e:
            self.logger.error(f"分析系统 {file_path} 时出错: {e}")
            return None

    def _run_random_sampling_comparison(self, vectors, selected_mask, df, system_path, sampled_indices, k, n):
        """运行随机采样比较"""
        rand_results = []
        frame_indices = df['Frame_ID'].values

        for _ in range(10):
            idx = np.random.choice(n, k, replace=False)
            sel_vectors = vectors[idx]
            sel_frame_indices = frame_indices[idx]

            # 重新计算随机组的RMSD
            rand_rmsd = []
            if system_path:
                rand_rmsd = self._calculate_group_rmsd(system_path, sel_frame_indices.tolist())
                if len(rand_rmsd) == 0:
                    self.logger.warning(f"无法计算随机组RMSD，使用原有RMSD数据")
                    rand_rmsd = pd.to_numeric(df.iloc[idx]['RMSD'], errors='coerce').values if 'RMSD' in df.columns else []
            else:
                rand_rmsd = pd.to_numeric(df.iloc[idx]['RMSD'], errors='coerce').values if 'RMSD' in df.columns else []

            sel_metrics = MetricsToolkit.adapt_sampling_metrics(
                sel_vectors, vectors,
                rand_rmsd if len(rand_rmsd) > 0 else []
            )
            rand_results.append(sel_metrics)
        return rand_results

    def _run_uniform_sampling_comparison(self, vectors, df, system_path, k, n):
        """运行均匀采样比较"""
        if k == 0:
            return {}

        frame_indices = df['Frame_ID'].values
        idx_uniform = SamplingStrategy.uniform_sample_indices(n, k)
        sel_vectors = vectors[idx_uniform]
        sel_frame_indices = frame_indices[idx_uniform]

        # 重新计算均匀组的RMSD
        uniform_rmsd = []
        if system_path:
            uniform_rmsd = self._calculate_group_rmsd(system_path, sel_frame_indices.tolist())
            if len(uniform_rmsd) == 0:
                self.logger.warning(f"无法计算均匀组RMSD，使用原有RMSD数据")
                uniform_rmsd = pd.to_numeric(df.iloc[idx_uniform]['RMSD'], errors='coerce').values if 'RMSD' in df.columns else []
        else:
            uniform_rmsd = pd.to_numeric(df.iloc[idx_uniform]['RMSD'], errors='coerce').values if 'RMSD' in df.columns else []

        return MetricsToolkit.adapt_sampling_metrics(
            sel_vectors, vectors,
            uniform_rmsd if len(uniform_rmsd) > 0 else []
        )

    def _build_result_row(self, system, sample_ratio, n, k, sampled_metrics, rand_results, uniform_metrics):
        """构建结果行数据"""
        # 收集随机采样统计
        rand_ANND = MetricsToolkit.collect_metric_values(rand_results, 'ANND')
        rand_MPD = MetricsToolkit.collect_metric_values(rand_results, 'MPD')
        rand_Cov = MetricsToolkit.collect_metric_values(rand_results, 'Coverage_Ratio')
        rand_JS = MetricsToolkit.collect_metric_values(rand_results, 'JS_Divergence')
        rand_RMSD = MetricsToolkit.collect_metric_values(rand_results, 'RMSD_Mean')
        rand_EnergyRange = MetricsToolkit.collect_metric_values(rand_results, 'Energy_Range')

        return {
            # 基本信息
            'System': system,
            'Sample_Ratio': sample_ratio,
            'Total_Frames': n,
            'Sampled_Frames': k,

            # 采样算法结果
            'ANND_sampled': sampled_metrics.get('ANND'),
            'MPD_sampled': sampled_metrics.get('MPD'),
            'Coverage_Ratio_sampled': sampled_metrics.get('Coverage_Ratio'),
            'Energy_Range_sampled': sampled_metrics.get('Energy_Range'),
            'JS_Divergence_sampled': sampled_metrics.get('JS_Divergence'),
            'RMSD_Mean_sampled': sampled_metrics.get('RMSD_Mean'),

            # 随机采样统计
            'ANND_random_mean': np.mean(rand_ANND) if rand_ANND else np.nan,
            'ANND_random_std': np.std(rand_ANND, ddof=1) if len(rand_ANND) >= 2 else np.nan,
            'MPD_random_mean': np.mean(rand_MPD) if rand_MPD else np.nan,
            'MPD_random_std': np.std(rand_MPD, ddof=1) if len(rand_MPD) >= 2 else np.nan,
            'Coverage_random_mean': np.mean(rand_Cov) if rand_Cov else np.nan,
            'Energy_Range_random_mean': np.mean(rand_EnergyRange) if rand_EnergyRange else np.nan,
            'JS_random_mean': np.mean(rand_JS) if rand_JS else np.nan,
            'RMSD_random_mean': np.mean(rand_RMSD) if rand_RMSD else np.nan,

            # 均匀采样结果
            'ANND_uniform': uniform_metrics.get('ANND'),
            'MPD_uniform': uniform_metrics.get('MPD'),
            'Coverage_Ratio_uniform': uniform_metrics.get('Coverage_Ratio'),
            'Energy_Range_uniform': uniform_metrics.get('Energy_Range'),
            'JS_Divergence_uniform': uniform_metrics.get('JS_Divergence'),
            'RMSD_Mean_uniform': uniform_metrics.get('RMSD_Mean'),

            # 改进百分比
            'ANND_improvement_pct': calculate_improvement(
                sampled_metrics.get('ANND'),
                np.mean(rand_ANND) if rand_ANND else np.nan
            ),
            'RMSD_improvement_pct': calculate_improvement(
                sampled_metrics.get('RMSD_Mean'),
                np.mean(rand_RMSD) if rand_RMSD else np.nan
            ),

            # 统计显著性
            'ANND_p_value': calculate_significance(sampled_metrics.get('ANND'), rand_ANND),
            'RMSD_p_value': calculate_significance(sampled_metrics.get('RMSD_Mean'), rand_RMSD),

            # 相对于均匀采样的改进
            'ANND_vs_uniform_pct': calculate_improvement(
                sampled_metrics.get('ANND'),
                uniform_metrics.get('ANND')
            ),
            'RMSD_vs_uniform_pct': calculate_improvement(
                sampled_metrics.get('RMSD_Mean'),
                uniform_metrics.get('RMSD_Mean')
            ),
        }

    def _save_comparison_results(self, rows, result_dir):
        """保存比较结果"""
        single_dir = os.path.join(result_dir, 'single_analysis_results')
        os.makedirs(single_dir, exist_ok=True)
        out_path = os.path.join(single_dir, 'sampling_compare_enhanced.csv')

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        self.logger.info(f"增强版采样对比结果已保存到 {out_path}")

    def _create_summary_table(self, rows, result_dir):
        """创建汇总表格"""
        if not rows:
            self.logger.warning("没有数据行，无法创建汇总表格")
            return

        # 定义汇总指标
        ordered = ["RMSD_Mean", "ANND", "MPD", "Coverage_Ratio", "Energy_Range", "JS_Divergence"]
        metrics_to_summarize = []
        for m in ordered:
            metrics_to_summarize.append((
                m,
                f"{m}_sampled",
                f"{m.replace('Coverage_Ratio','Coverage').replace('RMSD_Mean','RMSD').replace('JS_Divergence','JS')}_random_mean",
                f"{m}_uniform"
            ))

        self.logger.info(f"开始处理 {len(rows)} 个系统的数据...")

        # 计算汇总统计
        summary_rows = []
        for row in rows:
            summary_rows.append({
                'System': row.get('System', ''),
                **{k: row.get(k, np.nan) for _, k, r, u in metrics_to_summarize for k in [k, r, u]}
            })

        summary_df_per_system = pd.DataFrame(summary_rows)
        summary_rows_final = []

        for metric_name, sampled_col, random_col, uniform_col in metrics_to_summarize:
            sampled_means = summary_df_per_system[sampled_col].values
            random_means = summary_df_per_system[random_col].values
            uniform_means = summary_df_per_system[uniform_col].values

            # 过滤NaN值
            sampled_means = sampled_means[~np.isnan(sampled_means)]
            random_means = random_means[~np.isnan(random_means)]
            uniform_means = uniform_means[~np.isnan(uniform_means)]

            row = {'Metric': metric_name}
            row.update({
                'Sampled_Mean': np.mean(sampled_means) if len(sampled_means) > 0 else np.nan,
                'Sampled_Std': np.std(sampled_means, ddof=1) if len(sampled_means) >= 2 else np.nan,
                'Random_Mean': np.mean(random_means) if len(random_means) > 0 else np.nan,
                'Random_Std': np.std(random_means, ddof=1) if len(random_means) >= 2 else np.nan,
                'Uniform_Mean': np.mean(uniform_means) if len(uniform_means) > 0 else np.nan,
                'Uniform_Std': np.std(uniform_means, ddof=1) if len(uniform_means) >= 2 else np.nan,
            })
            summary_rows_final.append(row)

        summary_df = pd.DataFrame(summary_rows_final)

        # 保存汇总结果
        combined_dir = os.path.join(result_dir, 'combined_analysis_results')
        os.makedirs(combined_dir, exist_ok=True)
        summary_path = os.path.join(combined_dir, 'sampling_methods_comparison.csv')
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"均值对比汇总表格已保存到 {summary_path}")
        self.logger.info(f"汇总了 {len(rows)} 个系统的数据")

    def _calculate_group_rmsd(self, system_path: str, frame_indices: List[int]) -> np.ndarray:
        """为指定帧索引列表计算基于本组mean structure的RMSD序列

        Args:
            system_path: 系统目录路径
            frame_indices: 帧号列表（Frame_ID）

        Returns:
            RMSD序列数组
        """
        try:
            # 加载原子坐标数据
            stru_dir = os.path.join(system_path, "OUT.ABACUS", "STRU")
            if not os.path.exists(stru_dir):
                self.logger.warning(f"STRU目录不存在: {stru_dir}")
                return np.array([])

            parser = StrUParser(exclude_hydrogen=True)
            all_frames = parser.parse_trajectory(stru_dir)

            if not all_frames:
                self.logger.warning(f"未找到有效轨迹数据: {system_path}")
                return np.array([])

            # 按frame_id排序
            all_frames.sort(key=lambda x: x.frame_id)

            # 将帧号映射到数组索引（帧号 // 10）
            array_indices = []
            for frame_id in frame_indices:
                idx = frame_id // 10
                if idx < len(all_frames):
                    array_indices.append(idx)
                else:
                    self.logger.warning(f"帧号 {frame_id} 对应的数组索引 {idx} 超出范围 (总帧数: {len(all_frames)})")

            if len(array_indices) < 2:
                self.logger.warning(f"有效帧数不足: {len(array_indices)}")
                return np.array([])

            # 提取指定帧的positions
            selected_positions = [all_frames[idx].positions.copy() for idx in array_indices]

            # 计算本组的mean structure
            mean_structure, aligned_positions = RMSDCalculator.iterative_mean_structure(
                selected_positions, max_iter=20, tol=1e-6
            )

            # 计算每帧到mean structure的RMSD
            rmsds = []
            for pos in aligned_positions:
                rmsd = RMSDCalculator.calculate_rmsd(pos, mean_structure)
                rmsds.append(rmsd)

            return np.array(rmsds, dtype=float)

        except Exception as e:
            self.logger.error(f"计算组RMSD时出错: {e}")
            return np.array([])


# 兼容性函数
def analyse_sampling_compare(result_dir=None):
    """兼容性函数，保持向后兼容"""
    analyser = SamplingComparisonAnalyser()
    return analyser.analyse_sampling_compare(result_dir)
