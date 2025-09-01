import os
import logging
import numpy as np
import pandas as pd
from glob import glob
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import warnings
from src.utils import LoggerManager

warnings.filterwarnings('ignore')

logger = LoggerManager.create_logger(__name__)

from src.utils.metrics_utils import (
    compute_basic_distance_metrics,
    compute_diversity_metrics,
    compute_distribution_similarity,
    summarize_rmsd,
    adapt_sampling_metrics,
)
from src.utils.metrics_registry import (
    get_headers_by_categories,
)

def _wrap_diversity(vectors: np.ndarray):
    m = compute_diversity_metrics(vectors)
    return {
        'coverage_ratio': m.coverage_ratio,
        'pca_variance_ratio': m.pca_variance_ratio,
        'energy_range': m.energy_range,
    }

def _wrap_rmsd(values):
    s = summarize_rmsd(values)
    return {
        'rmsd_mean': s.rmsd_mean,
        'rmsd_std': s.rmsd_std,
        'rmsd_min': s.rmsd_min,
        'rmsd_max': s.rmsd_max,
    }

def _wrap_similarity(sample_vectors: np.ndarray, full_vectors: np.ndarray):
    m = compute_distribution_similarity(sample_vectors, full_vectors)
    return {
        'js_divergence': m.js_divergence,
    }

def uniform_sample_indices(n, k):
    if k >= n:
        return np.arange(n)
    return np.round(np.linspace(0, n-1, k)).astype(int)

def analyse_sampling_compare(result_dir=None):
    # 1. 自动定位结果目录
    if result_dir is None:
        dirs = sorted(glob('analysis_results/run_*'), reverse=True)
        if not dirs:
            logger.warning("未找到分析结果，请先运行主程序。")
            return
        result_dir = dirs[0]
    single_dir = os.path.join(result_dir, 'single_analysis_results')
    files = glob(os.path.join(single_dir, 'frame_metrics_*.csv'))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        system = os.path.basename(f).replace('frame_metrics_', '').replace('.csv', '')
        vector_cols = [col for col in df.columns if (col == 'Energy_Standardized' or col.startswith('PC'))]
        vectors = df[vector_cols].values
        selected = df['Selected'] == 1
        k = selected.sum()
        n = len(df)
        sample_ratio = k / n if n > 0 else 0
        rmsd_data = []
        if 'RMSD' in df.columns:
            rmsd_data = pd.to_numeric(df['RMSD'], errors='coerce').values

        # 采样算法结果
        sampled_metrics = adapt_sampling_metrics(
            vectors[selected],
            vectors,
            rmsd_data[selected] if len(rmsd_data) > 0 else []
        )

        # 随机采样（多次运行）
        rand_results = []
        for _ in range(10):
            if k == 0 or n == 0:
                break
            idx = np.random.choice(n, k, replace=False)
            sel_metrics = adapt_sampling_metrics(vectors[idx], vectors, rmsd_data[idx] if len(rmsd_data) > 0 else [])
            rand_results.append({
                'ANND': sel_metrics.get('ANND'),
                'MPD': sel_metrics.get('MPD'),
                'Coverage_Ratio': sel_metrics.get('Coverage_Ratio'),
                'JS_Divergence': sel_metrics.get('JS_Divergence'),
                'RMSD_Mean': sel_metrics.get('RMSD_Mean'),
                'Energy_Range': sel_metrics.get('Energy_Range'),
            })

        # 均匀采样
        idx_uniform = uniform_sample_indices(n, k) if k > 0 else np.array([], dtype=int)
        uniform_metrics = adapt_sampling_metrics(
            vectors[idx_uniform],
            vectors,
            rmsd_data[idx_uniform] if (k > 0 and len(rmsd_data) > 0) else []
        ) if k > 0 else {}

        # 统计集合
        def _collect(key):
            vals = [r.get(key) for r in rand_results]
            return [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]

        # 随机采样集合统计
    rand_ANND = _collect('ANND')
    rand_MPD = _collect('MPD')
    rand_Cov = _collect('Coverage_Ratio')
    rand_JS = _collect('JS_Divergence')
    rand_RMSD = _collect('RMSD_Mean')
    rand_EnergyRange = _collect('Energy_Range')

    # 计算改进百分比
    def calc_improvement(sample_val, baseline_mean, baseline_std):
        if np.isnan(sample_val) or np.isnan(baseline_mean) or baseline_mean == 0:
            return np.nan
        improvement = (sample_val - baseline_mean) / abs(baseline_mean) * 100
        return improvement

    # 统计显著性检验（t-test）
    def calc_significance(sample_val, baseline_vals):
        if len(baseline_vals) < 2 or np.isnan(sample_val) or np.all(np.isnan(baseline_vals)):
            return np.nan
        # 过滤掉NaN值
        baseline_vals = [v for v in baseline_vals if not np.isnan(v)]
        if len(baseline_vals) < 2:
            return np.nan
        from scipy.stats import ttest_1samp
        try:
            t_stat, p_val = ttest_1samp(baseline_vals, sample_val)
            return p_val
        except:
            return np.nan

    row = {
        # 基本信息
        'System': system,
        'Sample_Ratio': sample_ratio,
        'Total_Frames': n,
        'Sampled_Frames': k,

        # 采样算法结果（统一）
        'ANND_sampled': sampled_metrics.get('ANND'),
        'MPD_sampled': sampled_metrics.get('MPD'),
        'Coverage_Ratio_sampled': sampled_metrics.get('Coverage_Ratio'),
        'Energy_Range_sampled': sampled_metrics.get('Energy_Range'),
        'JS_Divergence_sampled': sampled_metrics.get('JS_Divergence'),
        'RMSD_Mean_sampled': sampled_metrics.get('RMSD_Mean'),

        # 随机采样统计量
        'ANND_random_mean': np.mean(rand_ANND) if rand_ANND else np.nan,
        'ANND_random_std': np.std(rand_ANND, ddof=1) if len(rand_ANND) >= 2 else (np.std(rand_ANND, ddof=0) if len(rand_ANND) == 1 else np.nan),
        'MPD_random_mean': np.mean(rand_MPD) if rand_MPD else np.nan,
        'MPD_random_std': np.std(rand_MPD, ddof=1) if len(rand_MPD) >= 2 else (np.std(rand_MPD, ddof=0) if len(rand_MPD) == 1 else np.nan),
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

        # 改进百分比（相对于随机采样均值）
        'ANND_improvement_pct': calc_improvement(sampled_metrics.get('ANND'), np.mean(rand_ANND) if rand_ANND else np.nan, np.std(rand_ANND) if len(rand_ANND) > 1 else np.nan),
        'RMSD_improvement_pct': calc_improvement(sampled_metrics.get('RMSD_Mean'), np.mean(rand_RMSD) if rand_RMSD else np.nan, np.std(rand_RMSD) if len(rand_RMSD) > 1 else np.nan),

        # 统计显著性（p值）
        'ANND_p_value': calc_significance(sampled_metrics.get('ANND'), rand_ANND),
        'RMSD_p_value': calc_significance(sampled_metrics.get('RMSD_Mean'), rand_RMSD),

        # 相对于均匀采样的改进
        'ANND_vs_uniform_pct': calc_improvement(sampled_metrics.get('ANND'), uniform_metrics.get('ANND'), 0),
        'RMSD_vs_uniform_pct': calc_improvement(sampled_metrics.get('RMSD_Mean'), uniform_metrics.get('RMSD_Mean'), 0),
    }
    rows.append(row)

    out_df = pd.DataFrame(rows)
    
    # 修改输出路径到single_analysis_results文件夹
    single_dir = os.path.join(result_dir, 'single_analysis_results')
    os.makedirs(single_dir, exist_ok=True)
    out_path = os.path.join(single_dir, 'sampling_compare_enhanced.csv')
    out_df.to_csv(out_path, index=False)
    logger.info(f"增强版采样对比结果已保存到 {out_path}")
    
    # 创建均值对比汇总表格
    create_summary_table(rows, result_dir)
    
    logger.info("新增统计量包括：")
    logger.info("- 变异性指标：标准差、最小值、最大值")
    logger.info("- 多样性指标：多样性得分、覆盖率")
    logger.info("- 分布相似性：JS散度、EMD距离")
    logger.info("- RMSD指标：RMSD均值、标准差、最小值、最大值")
    logger.info("- 相对性能：改进百分比、统计显著性")
    logger.info("- 效率指标：采样率、帧数统计")

def create_summary_table(rows, result_dir):
    """创建均值对比汇总表格"""
    if not rows:
        logger.warning("没有数据行，无法创建汇总表格")
        return
    
    # 定义需要汇总的指标
    # 根据注册表动态获取需要 summarization 的指标（排除非 summary 列，如能量范围暂不在采样比较中出现的 random/uniform 版本）
    # 只保留剩余指标
    ordered = ["RMSD_Mean", "ANND", "MPD", "Coverage_Ratio", "Energy_Range", "JS_Divergence"]
    metrics_to_summarize = []
    for m in ordered:
        metrics_to_summarize.append(
            (
                m,
                f"{m}_sampled",
                f"{m.replace('Coverage_Ratio','Coverage').replace('RMSD_Mean','RMSD').replace('JS_Divergence','JS')}_random_mean",
                f"{m}_uniform"
            )
        )
    
    logger.info(f"开始处理 {len(rows)} 个系统的数据...")
    
    # 检查是否有数据
    if len(rows) == 0:
        logger.error("没有系统数据")
        return
    
    # 检查第一行数据的结构
    if rows:
        sample_keys = list(rows[0].keys())
        logger.debug(f"数据行包含的列：{sample_keys}")
    
    # 计算每个系统的采样算法/随机采样/均匀采样的均值
    summary_rows = []
    for row in rows:
        summary_rows.append({
            'System': row.get('System', ''),
            **{k: row.get(k, np.nan) for _, k, r, u in metrics_to_summarize for k in [k, r, u]}
        })
    summary_df_per_system = pd.DataFrame(summary_rows)

    # 针对所有系统，统计每个指标的均值和“系统间标准差”
    summary_rows_final = []
    for metric_name, sampled_col, random_col, uniform_col in metrics_to_summarize:
        sampled_means = summary_df_per_system[sampled_col].values
        random_means = summary_df_per_system[random_col].values
        uniform_means = summary_df_per_system[uniform_col].values

        # 过滤nan
        sampled_means = sampled_means[~np.isnan(sampled_means)]
        random_means = random_means[~np.isnan(random_means)]
        uniform_means = uniform_means[~np.isnan(uniform_means)]

        row = {'Metric': metric_name}
        row.update({
            'Sampled_Mean': np.mean(sampled_means) if len(sampled_means) > 0 else np.nan,
            'Sampled_Std': np.std(sampled_means, ddof=1) if len(sampled_means) >= 2 else (np.std(sampled_means, ddof=0) if len(sampled_means) == 1 else np.nan),
            'Random_Mean': np.mean(random_means) if len(random_means) > 0 else np.nan,
            'Random_Std': np.std(random_means, ddof=1) if len(random_means) >= 2 else (np.std(random_means, ddof=0) if len(random_means) == 1 else np.nan),
            'Uniform_Mean': np.mean(uniform_means) if len(uniform_means) > 0 else np.nan,
            'Uniform_Std': np.std(uniform_means, ddof=1) if len(uniform_means) >= 2 else (np.std(uniform_means, ddof=0) if len(uniform_means) == 1 else np.nan),
        })
        summary_rows_final.append(row)

    summary_df = pd.DataFrame(summary_rows_final)

    # 输出到combined_analysis_results文件夹
    combined_dir = os.path.join(result_dir, 'combined_analysis_results')
    os.makedirs(combined_dir, exist_ok=True)
    summary_path = os.path.join(combined_dir, 'sampling_methods_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"均值对比汇总表格已保存到 {summary_path}")
    logger.info(f"汇总了 {len(rows)} 个系统的数据")

if __name__ == '__main__':
    analyse_sampling_compare()
