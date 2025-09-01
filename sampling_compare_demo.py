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
)

def _wrap_diversity(vectors: np.ndarray):
    m = compute_diversity_metrics(vectors)
    return {
        'diversity_score': m.diversity_score,
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
        'emd_distance': m.emd_distance,
        'mean_distance': m.mean_distance,
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
        basic = compute_basic_distance_metrics(vectors[selected])
        sampled_diversity = _wrap_diversity(vectors[selected])
        sampled_similarity = _wrap_similarity(vectors[selected], vectors)
        sampled_rmsd = _wrap_rmsd(rmsd_data[selected] if len(rmsd_data) > 0 else [])

        # 随机采样（多次运行）
        rand_results = []
        for _ in range(10):
            if k == 0 or n == 0:
                break
            idx = np.random.choice(n, k, replace=False)
            b2 = compute_basic_distance_metrics(vectors[idx])
            div = _wrap_diversity(vectors[idx])
            sim = _wrap_similarity(vectors[idx], vectors)
            rmsd_metrics = _wrap_rmsd(rmsd_data[idx] if len(rmsd_data) > 0 else [])
            rand_results.append({
                'minD': b2.MinD, 'annd': b2.ANND, 'mpd': b2.MPD,
                'diversity_score': div['diversity_score'],
                'coverage_ratio': div['coverage_ratio'],
                'js_divergence': sim['js_divergence'],
                'emd_distance': sim['emd_distance'],
                'rmsd_mean': rmsd_metrics['rmsd_mean']
            })

        # 均匀采样
        idx_uniform = uniform_sample_indices(n, k) if k > 0 else np.array([], dtype=int)
        basic_uniform = compute_basic_distance_metrics(vectors[idx_uniform]) if k > 0 else compute_basic_distance_metrics(np.empty((0,)))
        uniform_diversity = _wrap_diversity(vectors[idx_uniform]) if k > 0 else _wrap_diversity(np.empty((0,)))
        uniform_similarity = _wrap_similarity(vectors[idx_uniform], vectors) if k > 1 else {'js_divergence': np.nan, 'emd_distance': np.nan, 'mean_distance': np.nan}
        uniform_rmsd = _wrap_rmsd(rmsd_data[idx_uniform] if (k > 0 and len(rmsd_data) > 0) else [])

        # 统计集合
        rand_minD = [r['minD'] for r in rand_results if not np.isnan(r['minD'])]
        rand_annd = [r['annd'] for r in rand_results if not np.isnan(r['annd'])]
        rand_mpd = [r['mpd'] for r in rand_results if not np.isnan(r['mpd'])]
        rand_diversity = [r['diversity_score'] for r in rand_results if not np.isnan(r['diversity_score'])]
        rand_coverage = [r['coverage_ratio'] for r in rand_results if not np.isnan(r['coverage_ratio'])]
        rand_js = [r['js_divergence'] for r in rand_results if not np.isnan(r['js_divergence'])]
        rand_emd = [r['emd_distance'] for r in rand_results if not np.isnan(r['emd_distance'])]
        rand_rmsd = [r['rmsd_mean'] for r in rand_results if not np.isnan(r['rmsd_mean'])]

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

            # 采样算法结果
            'MinD_sampled': basic.MinD,
            'ANND_sampled': basic.ANND,
            'MPD_sampled': basic.MPD,
            'Diversity_Score_sampled': sampled_diversity['diversity_score'],
            'Coverage_Ratio_sampled': sampled_diversity['coverage_ratio'],
            'JS_Divergence_sampled': sampled_similarity['js_divergence'],
            'EMD_Distance_sampled': sampled_similarity['emd_distance'],
            'RMSD_Mean_sampled': sampled_rmsd['rmsd_mean'],

            # 随机采样统计量
            'MinD_random_mean': np.mean(rand_minD) if rand_minD else np.nan,
            'MinD_random_std': np.std(rand_minD) if len(rand_minD) > 1 else np.nan,
            'MinD_random_min': np.min(rand_minD) if rand_minD else np.nan,
            'MinD_random_max': np.max(rand_minD) if rand_minD else np.nan,
            'ANND_random_mean': np.mean(rand_annd) if rand_annd else np.nan,
            'ANND_random_std': np.std(rand_annd) if len(rand_annd) > 1 else np.nan,
            'MPD_random_mean': np.mean(rand_mpd) if rand_mpd else np.nan,
            'MPD_random_std': np.std(rand_mpd, ddof=1) if len(rand_mpd) >= 2 else (np.std(rand_mpd, ddof=0) if len(rand_mpd) == 1 else np.nan),
            'Diversity_random_mean': np.mean(rand_diversity) if rand_diversity else np.nan,
            'Coverage_random_mean': np.mean(rand_coverage) if rand_coverage else np.nan,
            'JS_random_mean': np.mean(rand_js) if rand_js else np.nan,
            'EMD_random_mean': np.mean(rand_emd) if rand_emd else np.nan,
            'RMSD_random_mean': np.mean(rand_rmsd) if rand_rmsd else np.nan,

            # 均匀采样结果
            'MinD_uniform': basic_uniform.MinD,
            'ANND_uniform': basic_uniform.ANND,
            'MPD_uniform': basic_uniform.MPD,
            'Diversity_Score_uniform': uniform_diversity['diversity_score'],
            'Coverage_Ratio_uniform': uniform_diversity['coverage_ratio'],
            'JS_Divergence_uniform': uniform_similarity['js_divergence'],
            'EMD_Distance_uniform': uniform_similarity['emd_distance'],
            'RMSD_Mean_uniform': uniform_rmsd['rmsd_mean'],

            # 改进百分比（相对于随机采样均值）
            'MinD_improvement_pct': calc_improvement(basic.MinD, np.mean(rand_minD) if rand_minD else np.nan, np.std(rand_minD) if len(rand_minD) > 1 else np.nan),
            'ANND_improvement_pct': calc_improvement(basic.ANND, np.mean(rand_annd) if rand_annd else np.nan, np.std(rand_annd) if len(rand_annd) > 1 else np.nan),
            'Diversity_improvement_pct': calc_improvement(
                sampled_diversity['diversity_score'], np.mean(rand_diversity) if rand_diversity else np.nan, np.std(rand_diversity) if len(rand_diversity) > 1 else np.nan
            ),
            'RMSD_improvement_pct': calc_improvement(
                sampled_rmsd['rmsd_mean'], np.mean(rand_rmsd) if rand_rmsd else np.nan, np.std(rand_rmsd) if len(rand_rmsd) > 1 else np.nan
            ),

            # 统计显著性（p值）
            'MinD_p_value': calc_significance(basic.MinD, rand_minD),
            'ANND_p_value': calc_significance(basic.ANND, rand_annd),
            'Diversity_p_value': calc_significance(
                sampled_diversity['diversity_score'], rand_diversity
            ),
            'RMSD_p_value': calc_significance(
                sampled_rmsd['rmsd_mean'], rand_rmsd
            ),

            # 相对于均匀采样的改进
            'MinD_vs_uniform_pct': calc_improvement(basic.MinD, basic_uniform.MinD, 0),
            'ANND_vs_uniform_pct': calc_improvement(basic.ANND, basic_uniform.ANND, 0),
            'RMSD_vs_uniform_pct': calc_improvement(sampled_rmsd['rmsd_mean'], uniform_rmsd['rmsd_mean'], 0),
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
    metrics_to_summarize = [
        # 基础距离指标
        ('MinD', 'MinD_sampled', 'MinD_random_mean', 'MinD_uniform'),
        ('ANND', 'ANND_sampled', 'ANND_random_mean', 'ANND_uniform'),
        ('MPD', 'MPD_sampled', 'MPD_random_mean', 'MPD_uniform'),
        
        # 多样性指标
        ('Diversity_Score', 'Diversity_Score_sampled', 'Diversity_random_mean', 'Diversity_Score_uniform'),
        ('Coverage_Ratio', 'Coverage_Ratio_sampled', 'Coverage_random_mean', 'Coverage_Ratio_uniform'),
        
        # 分布相似性指标
        ('JS_Divergence', 'JS_Divergence_sampled', 'JS_random_mean', 'JS_Divergence_uniform'),
        ('EMD_Distance', 'EMD_Distance_sampled', 'EMD_random_mean', 'EMD_Distance_uniform'),
        
        # RMSD指标
        ('RMSD_Mean', 'RMSD_Mean_sampled', 'RMSD_random_mean', 'RMSD_Mean_uniform'),
    ]
    
    logger.info(f"开始处理 {len(rows)} 个系统的数据...")
    
    # 检查是否有数据
    if len(rows) == 0:
        logger.error("没有系统数据")
        return
    
    # 检查第一行数据的结构
    if rows:
        sample_keys = list(rows[0].keys())
        logger.debug(f"数据行包含的列：{sample_keys}")
        # 检查关键列是否存在
        key_columns = ['MinD_sampled', 'MinD_random_mean', 'MinD_uniform', 'RMSD_Mean_sampled', 'MPD_sampled', 'MPD_random_mean']
        for col in key_columns:
            if col in sample_keys:
                value = rows[0][col]
                logger.debug(f"示例值 {col}: {value} (类型: {type(value)})")
            else:
                logger.warning(f"缺少列 {col}")
    
    # 计算每个指标的均值和标准差
    summary_data = {}
    
    for metric_name, sampled_col, random_col, uniform_col in metrics_to_summarize:
        # 收集所有系统的值
        sampled_values = [row[sampled_col] for row in rows if not np.isnan(row.get(sampled_col, np.nan))]
        random_values = [row[random_col] for row in rows if not np.isnan(row.get(random_col, np.nan))]
        uniform_values = [row[uniform_col] for row in rows if not np.isnan(row.get(uniform_col, np.nan))]

        logger.info(f"{metric_name}: Sampled={len(sampled_values)}, Random={len(random_values)}, Uniform={len(uniform_values)}")

        # 调试信息
        if len(sampled_values) > 0:
            logger.debug(f"Sampled values range: {np.min(sampled_values):.6f} - {np.max(sampled_values):.6f}")
        if len(random_values) > 0:
            logger.debug(f"Random values range: {np.min(random_values):.6f} - {np.max(random_values):.6f}")
        if len(uniform_values) > 0:
            logger.debug(f"Uniform values range: {np.min(uniform_values):.6f} - {np.max(uniform_values):.6f}")

        # 计算均值和标准差
        summary_data[f'{metric_name}_Sampled_Mean'] = np.mean(sampled_values) if sampled_values else np.nan
        summary_data[f'{metric_name}_Sampled_Std'] = np.std(sampled_values, ddof=1) if len(sampled_values) >= 2 else (np.std(sampled_values, ddof=0) if len(sampled_values) == 1 else np.nan)
        summary_data[f'{metric_name}_Random_Mean'] = np.mean(random_values) if random_values else np.nan
        summary_data[f'{metric_name}_Random_Std'] = np.std(random_values, ddof=1) if len(random_values) >= 2 else (np.std(random_values, ddof=0) if len(random_values) == 1 else np.nan)
        summary_data[f'{metric_name}_Uniform_Mean'] = np.mean(uniform_values) if uniform_values else np.nan
        summary_data[f'{metric_name}_Uniform_Std'] = np.std(uniform_values, ddof=1) if len(uniform_values) >= 2 else (np.std(uniform_values, ddof=0) if len(uniform_values) == 1 else np.nan)
    
    # 创建汇总DataFrame
    # 行是指标，列是采样方法的均值和标准差
    summary_rows = []
    
    for metric_name, _, _, _ in metrics_to_summarize:
        row = {'Metric': metric_name}
        row.update({
            'Sampled_Mean': summary_data.get(f'{metric_name}_Sampled_Mean', np.nan),
            'Sampled_Std': summary_data.get(f'{metric_name}_Sampled_Std', np.nan),
            'Random_Mean': summary_data.get(f'{metric_name}_Random_Mean', np.nan),
            'Random_Std': summary_data.get(f'{metric_name}_Random_Std', np.nan),
            'Uniform_Mean': summary_data.get(f'{metric_name}_Uniform_Mean', np.nan),
            'Uniform_Std': summary_data.get(f'{metric_name}_Uniform_Std', np.nan),
        })
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # 输出到combined_analysis_results文件夹
    combined_dir = os.path.join(result_dir, 'combined_analysis_results')
    os.makedirs(combined_dir, exist_ok=True)
    summary_path = os.path.join(combined_dir, 'sampling_methods_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"均值对比汇总表格已保存到 {summary_path}")
    logger.info(f"汇总了 {len(rows)} 个系统的数据")

if __name__ == '__main__':
    analyse_sampling_compare()
