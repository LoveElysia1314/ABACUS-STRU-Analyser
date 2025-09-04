#!/usr/bin/env python
"""
采样效果比较分析器
功能：比较智能采样算法与随机/均匀采样的效果
已重构：核心工具函数已迁移到统一模块，保留业务逻辑
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from typing import List
import warnings

from src.utils import LoggerManager
from src.core.metrics import MetricsToolkit
from src.core.sampler import SamplingStrategy, calculate_improvement
from src.core.system_analyser import RMSDCalculator

warnings.filterwarnings('ignore')


class SamplingComparisonAnalyser:
    """采样效果比较分析器类"""

    def __init__(self):
        self.logger = LoggerManager.create_logger(__name__)

    def analyse_sampling_compare(self, result_dir=None, workers: int = -1, parallel_mode: str = "process"):
        """采样效果比较分析主函数，分类型输出结果"""
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

        # 新旧单体系目录兼容
        single_dir_candidates = [os.path.join(result_dir, 'single_analysis'),
                                 os.path.join(result_dir, 'single_analysis_results')]
        single_dir = None
        for c in single_dir_candidates:
            if os.path.isdir(c):
                single_dir = c
                break
        if single_dir is None:
            self.logger.warning("未找到 single_analysis 目录")
            return
        # 新旧文件命名兼容：frame_*.csv / frame_metrics_*.csv
        files = glob(os.path.join(single_dir, 'frame_*.csv')) + glob(os.path.join(single_dir, 'frame_metrics_*.csv'))

        if not files:
            self.logger.warning("未找到帧指标文件")
            return

        # 新建分类型输出文件夹
        cache_dir = os.path.join(result_dir, 'sampling_comparison')
        os.makedirs(cache_dir, exist_ok=True)

        # 读取缓存csv
        sampled_cache = None
        random_cache = None
        uniform_cache = None
        if os.path.exists(os.path.join(cache_dir, 'sampled.csv')):
            sampled_cache = pd.read_csv(os.path.join(cache_dir, 'sampled.csv'))
        if os.path.exists(os.path.join(cache_dir, 'random.csv')):
            random_cache = pd.read_csv(os.path.join(cache_dir, 'random.csv'))
        if os.path.exists(os.path.join(cache_dir, 'uniform.csv')):
            uniform_cache = pd.read_csv(os.path.join(cache_dir, 'uniform.csv'))


        from src.utils.parallel_utils import run_parallel_tasks
        import functools

        sampled_rows = []
        random_rows = []
        uniform_rows = []
        cache_hit = 0
        cache_miss = 0
        tasks = []
        for f in files:
            base = os.path.basename(f)
            if base.startswith('frame_metrics_'):
                system = base[len('frame_metrics_'):-4]
            elif base.startswith('frame_'):
                system = base[len('frame_'):-4]
            else:
                continue
            sampled_row = None
            random_row = None
            uniform_row = None
            if sampled_cache is not None:
                match = sampled_cache[sampled_cache['System'] == system]
                if not match.empty:
                    sampled_row = match.iloc[0].to_dict()
            if random_cache is not None:
                match = random_cache[random_cache['System'] == system]
                if not match.empty:
                    random_row = match.iloc[0].to_dict()
            if uniform_cache is not None:
                match = uniform_cache[uniform_cache['System'] == system]
                if not match.empty:
                    uniform_row = match.iloc[0].to_dict()

            if sampled_row and random_row and uniform_row:
                sampled_rows.append(sampled_row)
                random_rows.append(random_row)
                uniform_rows.append(uniform_row)
                cache_hit += 1
                continue

            # 未命中缓存，加入并行任务
            tasks.append(f)

        # 并行处理未命中缓存的文件
        if tasks:
            worker = functools.partial(self._analyze_single_system, system_paths=system_paths)
            results = run_parallel_tasks(
                tasks,
                worker_fn=worker,
                workers=workers,
                mode=parallel_mode,
                logger=self.logger,
                desc="采样比较分析"
            )
            for row in results:
                if not row:
                    continue
                cache_miss += 1
                base_info = {
                    'System': row.get('System', ''),
                    'Sample_Ratio': row.get('Sample_Ratio', ''),
                    'Total_Frames': row.get('Total_Frames', ''),
                    'Sampled_Frames': row.get('Sampled_Frames', '')
                }
                sampled_rows.append({**base_info,
                    'ANND': row.get('ANND_sampled'),
                    'MPD': row.get('MPD_sampled'),
                    'Coverage_Ratio': row.get('Coverage_Ratio_sampled'),
                    'Energy_Range': row.get('Energy_Range_sampled'),
                    'JS_Divergence': row.get('JS_Divergence_sampled'),
                    'RMSD_Mean': row.get('RMSD_Mean_sampled')
                })
                random_rows.append({**base_info,
                    'ANND': row.get('ANND_random'),
                    'MPD': row.get('MPD_random'),
                    'Coverage_Ratio': row.get('Coverage_Ratio_random'),
                    'Energy_Range': row.get('Energy_Range_random'),
                    'JS_Divergence': row.get('JS_Divergence_random'),
                    'RMSD_Mean': row.get('RMSD_Mean_random')
                })
                uniform_rows.append({**base_info,
                    'ANND': row.get('ANND_uniform'),
                    'MPD': row.get('MPD_uniform'),
                    'Coverage_Ratio': row.get('Coverage_Ratio_uniform'),
                    'Energy_Range': row.get('Energy_Range_uniform'),
                    'JS_Divergence': row.get('JS_Divergence_uniform'),
                    'RMSD_Mean': row.get('RMSD_Mean_uniform')
                })

        pd.DataFrame(sampled_rows).to_csv(os.path.join(cache_dir, 'sampled.csv'), index=False)
        pd.DataFrame(random_rows).to_csv(os.path.join(cache_dir, 'random.csv'), index=False)
        pd.DataFrame(uniform_rows).to_csv(os.path.join(cache_dir, 'uniform.csv'), index=False)

        total = cache_hit + cache_miss
        if cache_hit == total:
            self.logger.info(f"共{total}个体系命中缓存，全部跳过计算。")
        elif cache_hit > 0:
            self.logger.info(f"{cache_hit}个体系命中缓存，{cache_miss}个体系重新计算。")
        else:
            self.logger.info(f"全部{total}个体系均重新计算。")

        self.logger.info(f"采样效果分类型结果已保存到 {result_dir}")

        if sampled_rows:
            self._create_summary_table([row for row in sampled_rows], result_dir)

    def _analyze_single_system(self, file_path, system_paths):
        """分析单个系统的数据"""
        try:
            df = pd.read_csv(file_path)
            system = os.path.basename(file_path).replace('frame_metrics_', '').replace('.csv', '').replace('frame_', '')
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
            rand_metrics = self._run_random_sampling_comparison(vectors, df, system_path, k, n)

            # 均匀采样比较
            uniform_metrics = self._run_uniform_sampling_comparison(vectors, df, system_path, k, n)

            # 构建结果行
            return self._build_result_row(
                system, sample_ratio, n, k,
                sampled_metrics, rand_metrics, uniform_metrics
            )

        except Exception as e:
            self.logger.error(f"分析系统 {file_path} 时出错: {e}")
            return None

    def _run_random_sampling_comparison(self, vectors, df, system_path, k, n):
        """运行随机采样比较（单次，固定 seed=42），返回单个指标字典

        参数精简：移除未使用的 selected_mask / sampled_indices。
        """
        frame_indices = df['Frame_ID'].values
        rng = np.random.default_rng(42)
        idx = rng.choice(n, k, replace=False)
        sel_vectors = vectors[idx]
        sel_frame_indices = frame_indices[idx]

        # 重新计算随机组的RMSD
        if system_path:
            rand_rmsd = self._calculate_group_rmsd(system_path, sel_frame_indices.tolist())
            if len(rand_rmsd) == 0:
                self.logger.warning("无法计算随机组RMSD，使用原有RMSD数据")
                rand_rmsd = pd.to_numeric(df.iloc[idx]['RMSD'], errors='coerce').values if 'RMSD' in df.columns else []
        else:
            rand_rmsd = pd.to_numeric(df.iloc[idx]['RMSD'], errors='coerce').values if 'RMSD' in df.columns else []

        return MetricsToolkit.adapt_sampling_metrics(
            sel_vectors, vectors,
            rand_rmsd if len(rand_rmsd) > 0 else []
        )

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

    def _build_result_row(self, system, sample_ratio, n, k, sampled_metrics, rand_metrics, uniform_metrics):
        """构建结果行数据"""
        # 随机采样单次结果直接使用 rand_metrics

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
            'ANND_random': rand_metrics.get('ANND'),
            'MPD_random': rand_metrics.get('MPD'),
            'Coverage_Ratio_random': rand_metrics.get('Coverage_Ratio'),
            'Energy_Range_random': rand_metrics.get('Energy_Range'),
            'JS_Divergence_random': rand_metrics.get('JS_Divergence'),
            'RMSD_Mean_random': rand_metrics.get('RMSD_Mean'),

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
                rand_metrics.get('ANND')
            ),
            'RMSD_improvement_pct': calculate_improvement(
                sampled_metrics.get('RMSD_Mean'),
                rand_metrics.get('RMSD_Mean')
            ),

            # 统计显著性
            # 单次随机比较无法进行显著性检验，置 NaN
            'ANND_p_value': np.nan,
            'RMSD_p_value': np.nan,

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

    def _create_summary_table(self, rows, result_dir):
        """创建汇总表格（自动读取三种采样类型缓存并合并）"""
        import pandas as pd
        import numpy as np
        import os
        # 新旧缓存目录兼容
        cache_dir_new = os.path.join(result_dir, 'sampling_comparison')
        cache_dir_legacy = os.path.join(result_dir, 'sampling_comparison_cache')
        if os.path.isdir(cache_dir_new):
            cache_dir = cache_dir_new
        else:
            cache_dir = cache_dir_legacy
        sampled_path = os.path.join(cache_dir, 'sampled.csv')
        random_path = os.path.join(cache_dir, 'random.csv')
        uniform_path = os.path.join(cache_dir, 'uniform.csv')

        if not (os.path.exists(sampled_path) and os.path.exists(random_path) and os.path.exists(uniform_path)):
            self.logger.warning("三种采样类型缓存文件不全，无法生成汇总表格")
            return

        df_sampled = pd.read_csv(sampled_path)
        df_random = pd.read_csv(random_path)
        df_uniform = pd.read_csv(uniform_path)

        # 合并三表，按System字段
        df_merged = df_sampled.merge(df_random, on="System", suffixes=("_sampled", "_random"))
        # uniform后缀需手动加
        df_uniform = df_uniform.add_suffix("_uniform")
        df_uniform = df_uniform.rename(columns={"System_uniform": "System"})
        df_merged = df_merged.merge(df_uniform, on="System", how="left")

        ordered = ["RMSD_Mean", "ANND", "MPD", "Coverage_Ratio", "Energy_Range", "JS_Divergence"]
        metrics_to_summarize = []
        for m in ordered:
            metrics_to_summarize.append((
                m,
                f"{m}_sampled",
                f"{m}_random",
                f"{m}_uniform"
            ))

        self.logger.info(f"开始处理 {len(df_merged)} 个系统的数据...")

        summary_rows_final = []
        for metric_name, sampled_col, random_col, uniform_col in metrics_to_summarize:
            sampled_means = df_merged[sampled_col].values
            random_means = df_merged[random_col].values
            uniform_means = df_merged[uniform_col].values

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

        # 保存汇总结果（新规范：直接写到 run_* 根目录）
        summary_path = os.path.join(result_dir, 'sampling_methods_comparison.csv')
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"采样方法汇总已保存: {summary_path}")
    # 不再写入 combined_analysis_results 目录
        self.logger.info(f"汇总体系数: {len(df_merged)}")

    def _calculate_group_rmsd(self, system_path: str, frame_indices: List[int]) -> np.ndarray:
        return RMSDCalculator.calculate_group_rmsd(system_path, frame_indices, self.logger)


# 兼容性函数
def analyse_sampling_compare(result_dir=None, workers: int = -1, parallel_mode: str = "process"):
    """兼容性函数，保持向后兼容，支持并行参数"""
    analyser = SamplingComparisonAnalyser()
    return analyser.analyse_sampling_compare(result_dir=result_dir, workers=workers, parallel_mode=parallel_mode)
