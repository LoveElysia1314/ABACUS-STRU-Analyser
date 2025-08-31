import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def calc_metrics(vectors):
    """计算基础距离指标"""
    from scipy.spatial.distance import pdist, squareform
    if len(vectors) < 2:
        return np.nan, np.nan, np.nan
    
    try:
        dists = squareform(pdist(vectors, metric='euclidean'))
        np.fill_diagonal(dists, np.inf)
        
        # 检查是否有有效的距离值
        valid_dists = dists[dists != np.inf]
        if len(valid_dists) == 0:
            return np.nan, np.nan, np.nan
        
        min_d = np.min(valid_dists) if len(valid_dists) > 0 else np.nan
        annd = np.mean(np.min(dists, axis=1)) if np.any(np.isfinite(np.min(dists, axis=1))) else np.nan
        mpd = np.mean(valid_dists) if len(valid_dists) > 0 else np.nan
        
        return min_d, annd, mpd
    except Exception as e:
        print(f"计算基础指标时出错: {e}")
        return np.nan, np.nan, np.nan

def calc_diversity_metrics(vectors):
    """计算多样性相关指标"""
    if len(vectors) < 2:
        return {
            'diversity_score': np.nan,
            'coverage_ratio': np.nan,
            'pca_variance_ratio': np.nan,
            'energy_range': np.nan
        }

    # 检查并处理NaN值
    if np.any(np.isnan(vectors)):
        # 移除包含NaN的行
        valid_mask = ~np.any(np.isnan(vectors), axis=1)
        vectors = vectors[valid_mask]
        if len(vectors) < 2:
            return {
                'diversity_score': np.nan,
                'coverage_ratio': np.nan,
                'pca_variance_ratio': np.nan,
                'energy_range': np.nan
            }

    try:
        # 多样性得分：基于距离矩阵的熵
        dists = squareform(pdist(vectors, metric='euclidean'))
        np.fill_diagonal(dists, np.inf)

        # 检查距离矩阵是否有效
        if np.all(np.isinf(dists)) or np.any(np.isnan(dists)):
            return {
                'diversity_score': np.nan,
                'coverage_ratio': np.nan,
                'pca_variance_ratio': np.nan,
                'energy_range': np.nan
            }

        # 归一化距离矩阵
        max_dist = np.max(dists[dists != np.inf])
        if np.isnan(max_dist) or max_dist == 0:
            dists_norm = dists
        else:
            dists_norm = dists / max_dist

        # 提取有效距离值
        dist_flat = dists_norm[dists_norm != np.inf].flatten()
        dist_flat = dist_flat[~np.isnan(dist_flat)]  # 移除NaN值

        if len(dist_flat) == 0:
            diversity_score = np.nan
        else:
            # 计算多样性得分（距离分布的熵）
            hist, _ = np.histogram(dist_flat, bins=20, density=True)
            diversity_score = entropy(hist + 1e-10)  # 添加小值避免log(0)

        # 覆盖率：凸包体积比（简化版）
        try:
            pca = PCA(n_components=min(3, vectors.shape[1]))
            pca_result = pca.fit_transform(vectors)
            explained_variance = np.sum(pca.explained_variance_ratio_)
            coverage_ratio = explained_variance
        except:
            coverage_ratio = np.nan

        # PCA方差贡献率
        pca_variance_ratio = explained_variance if 'explained_variance' in locals() else np.nan

        # 能量范围（如果有能量列）
        energy_range = np.nan
        if vectors.shape[1] > 0:
            # 假设第一列是能量
            energy_col = vectors[:, 0]
            if not np.all(np.isnan(energy_col)):
                energy_range = np.ptp(energy_col[~np.isnan(energy_col)])

    except Exception as e:
        print(f"计算多样性指标时出错: {e}")
        return {
            'diversity_score': np.nan,
            'coverage_ratio': np.nan,
            'pca_variance_ratio': np.nan,
            'energy_range': np.nan
        }

    return {
        'diversity_score': diversity_score,
        'coverage_ratio': coverage_ratio,
        'pca_variance_ratio': pca_variance_ratio,
        'energy_range': energy_range
    }

def calc_rmsd_metrics(rmsd_values):
    """计算RMSD相关指标"""
    rmsd_values = np.array(rmsd_values)
    
    if len(rmsd_values) == 0 or (len(rmsd_values) > 0 and np.all(np.isnan(rmsd_values))):
        return {
            'rmsd_mean': np.nan,
            'rmsd_std': np.nan,
            'rmsd_min': np.nan,
            'rmsd_max': np.nan
        }
    
    # 过滤NaN值
    valid_values = rmsd_values[~np.isnan(rmsd_values)]
    
    if len(valid_values) == 0:
        return {
            'rmsd_mean': np.nan,
            'rmsd_std': np.nan,
            'rmsd_min': np.nan,
            'rmsd_max': np.nan
        }
    
    return {
        'rmsd_mean': np.mean(valid_values),
        'rmsd_std': np.std(valid_values) if len(valid_values) > 1 else np.nan,
        'rmsd_min': np.min(valid_values),
        'rmsd_max': np.max(valid_values)
    }

def calc_distribution_similarity(sample_vectors, full_vectors):
    """计算采样分布与全集分布的相似性"""
    if len(sample_vectors) < 2 or len(full_vectors) < 2:
        return {
            'js_divergence': np.nan,
            'emd_distance': np.nan,
            'mean_distance': np.nan
        }

    # 检查并处理NaN值
    if np.any(np.isnan(sample_vectors)):
        valid_mask = ~np.any(np.isnan(sample_vectors), axis=1)
        sample_vectors = sample_vectors[valid_mask]

    if np.any(np.isnan(full_vectors)):
        valid_mask = ~np.any(np.isnan(full_vectors), axis=1)
        full_vectors = full_vectors[valid_mask]

    if len(sample_vectors) < 2 or len(full_vectors) < 2:
        return {
            'js_divergence': np.nan,
            'emd_distance': np.nan,
            'mean_distance': np.nan
        }

    try:
        # Jensen-Shannon散度（简化版，使用PCA投影）
        pca = PCA(n_components=min(3, sample_vectors.shape[1]))
        sample_pca = pca.fit_transform(sample_vectors)
        full_pca = pca.transform(full_vectors)

        # 计算一维分布的JS散度
        js_divs = []
        for i in range(sample_pca.shape[1]):
            sample_hist, _ = np.histogram(sample_pca[:, i], bins=20, density=True)
            full_hist, _ = np.histogram(full_pca[:, i], bins=20, density=True)
            # 归一化
            sample_hist = sample_hist / (np.sum(sample_hist) + 1e-10)
            full_hist = full_hist / (np.sum(full_hist) + 1e-10)
            m = 0.5 * (sample_hist + full_hist)
            js_div = 0.5 * (entropy(sample_hist + 1e-10, m + 1e-10) + entropy(full_hist + 1e-10, m + 1e-10))
            js_divs.append(js_div)
        js_divergence = np.mean(js_divs)

        # Earth Mover's Distance
        emd_distances = []
        for i in range(min(3, sample_pca.shape[1])):
            try:
                emd = wasserstein_distance(sample_pca[:, i], full_pca[:, i])
                emd_distances.append(emd)
            except:
                emd_distances.append(np.nan)
        emd_distance = np.nanmean(emd_distances)

        # 平均距离（采样点到全集质心的距离）
        sample_centroid = np.mean(sample_vectors, axis=0)
        full_centroid = np.mean(full_vectors, axis=0)
        mean_distance = np.linalg.norm(sample_centroid - full_centroid)

    except Exception as e:
        print(f"计算分布相似性时出错: {e}")
        js_divergence = np.nan
        emd_distance = np.nan
        mean_distance = np.nan

    return {
        'js_divergence': js_divergence,
        'emd_distance': emd_distance,
        'mean_distance': mean_distance
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
            print("未找到分析结果，请先运行主程序。")
            return
        result_dir = dirs[0]
    single_dir = os.path.join(result_dir, 'single_analysis_results')
    files = glob(os.path.join(single_dir, 'frame_metrics_*.csv'))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        system = os.path.basename(f).replace('frame_metrics_', '').replace('.csv', '')
        # 只保留Energy_Standardized和PC分量作为采样向量
        vector_cols = [col for col in df.columns if (col == 'Energy_Standardized' or col.startswith('PC'))]
        vectors = df[vector_cols].values
        selected = df['Selected'] == 1
        k = selected.sum()
        n = len(df)
        sample_ratio = k / n if n > 0 else 0

        # 读取RMSD数据
        rmsd_data = []
        if 'RMSD' in df.columns:
            rmsd_data = pd.to_numeric(df['RMSD'], errors='coerce').values
        
        # 你的采样
        minD, annd, mpd = calc_metrics(vectors[selected])
        sampled_diversity = calc_diversity_metrics(vectors[selected])
        sampled_similarity = calc_distribution_similarity(vectors[selected], vectors)
        sampled_rmsd = calc_rmsd_metrics(rmsd_data[selected] if len(rmsd_data) > 0 else [])

        # 随机采样（多次运行）
        rand_results = []
        for _ in range(10):
            idx = np.random.choice(n, k, replace=False)
            m, a, p = calc_metrics(vectors[idx])
            div = calc_diversity_metrics(vectors[idx])
            sim = calc_distribution_similarity(vectors[idx], vectors)
            rmsd_metrics = calc_rmsd_metrics(rmsd_data[idx] if len(rmsd_data) > 0 else [])
            rand_results.append({
                'minD': m, 'annd': a, 'mpd': p,
                'diversity_score': div['diversity_score'],
                'coverage_ratio': div['coverage_ratio'],
                'js_divergence': sim['js_divergence'],
                'emd_distance': sim['emd_distance'],
                'rmsd_mean': rmsd_metrics['rmsd_mean']
            })

        # 均匀采样
        idx_uniform = uniform_sample_indices(n, k)
        minD_uni, annd_uni, mpd_uni = calc_metrics(vectors[idx_uniform])
        uniform_diversity = calc_diversity_metrics(vectors[idx_uniform])
        uniform_similarity = calc_distribution_similarity(vectors[idx_uniform], vectors)
        uniform_rmsd = calc_rmsd_metrics(rmsd_data[idx_uniform] if len(rmsd_data) > 0 else [])

        # 计算随机采样的统计量
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
            'MinD_sampled': minD,
            'ANND_sampled': annd,
            'MPD_sampled': mpd,
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
            'MinD_uniform': minD_uni,
            'ANND_uniform': annd_uni,
            'MPD_uniform': mpd_uni,
            'Diversity_Score_uniform': uniform_diversity['diversity_score'],
            'Coverage_Ratio_uniform': uniform_diversity['coverage_ratio'],
            'JS_Divergence_uniform': uniform_similarity['js_divergence'],
            'EMD_Distance_uniform': uniform_similarity['emd_distance'],
            'RMSD_Mean_uniform': uniform_rmsd['rmsd_mean'],

            # 改进百分比（相对于随机采样均值）
            'MinD_improvement_pct': calc_improvement(minD, np.mean(rand_minD) if rand_minD else np.nan, np.std(rand_minD) if len(rand_minD) > 1 else np.nan),
            'ANND_improvement_pct': calc_improvement(annd, np.mean(rand_annd) if rand_annd else np.nan, np.std(rand_annd) if len(rand_annd) > 1 else np.nan),
            'Diversity_improvement_pct': calc_improvement(
                sampled_diversity['diversity_score'], np.mean(rand_diversity) if rand_diversity else np.nan, np.std(rand_diversity) if len(rand_diversity) > 1 else np.nan
            ),
            'RMSD_improvement_pct': calc_improvement(
                sampled_rmsd['rmsd_mean'], np.mean(rand_rmsd) if rand_rmsd else np.nan, np.std(rand_rmsd) if len(rand_rmsd) > 1 else np.nan
            ),

            # 统计显著性（p值）
            'MinD_p_value': calc_significance(minD, rand_minD),
            'ANND_p_value': calc_significance(annd, rand_annd),
            'Diversity_p_value': calc_significance(
                sampled_diversity['diversity_score'], rand_diversity
            ),
            'RMSD_p_value': calc_significance(
                sampled_rmsd['rmsd_mean'], rand_rmsd
            ),

            # 相对于均匀采样的改进
            'MinD_vs_uniform_pct': calc_improvement(minD, minD_uni, 0),
            'ANND_vs_uniform_pct': calc_improvement(annd, annd_uni, 0),
            'RMSD_vs_uniform_pct': calc_improvement(sampled_rmsd['rmsd_mean'], uniform_rmsd['rmsd_mean'], 0),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    
    # 修改输出路径到single_analysis_results文件夹
    single_dir = os.path.join(result_dir, 'single_analysis_results')
    os.makedirs(single_dir, exist_ok=True)
    out_path = os.path.join(single_dir, 'sampling_compare_enhanced.csv')
    out_df.to_csv(out_path, index=False)
    print(f"增强版采样对比结果已保存到 {out_path}")
    
    # 创建均值对比汇总表格
    create_summary_table(rows, result_dir)
    
    print(f"新增统计量包括：")
    print(f"- 变异性指标：标准差、最小值、最大值")
    print(f"- 多样性指标：多样性得分、覆盖率")
    print(f"- 分布相似性：JS散度、EMD距离")
    print(f"- RMSD指标：RMSD均值、标准差、最小值、最大值")
    print(f"- 相对性能：改进百分比、统计显著性")
    print(f"- 效率指标：采样率、帧数统计")

def create_summary_table(rows, result_dir):
    """创建均值对比汇总表格"""
    if not rows:
        print("警告：没有数据行，无法创建汇总表格")
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
    
    print(f"开始处理 {len(rows)} 个系统的数据...")
    
    # 检查是否有数据
    if len(rows) == 0:
        print("错误：没有系统数据")
        return
    
    # 检查第一行数据的结构
    if rows:
        sample_keys = list(rows[0].keys())
        print(f"数据行包含的列：{sample_keys}")
        
        # 检查关键列是否存在
        key_columns = ['MinD_sampled', 'MinD_random_mean', 'MinD_uniform', 'RMSD_Mean_sampled', 'MPD_sampled', 'MPD_random_mean']
        for col in key_columns:
            if col in sample_keys:
                value = rows[0][col]
                print(f"示例值 {col}: {value} (类型: {type(value)})")
            else:
                print(f"警告：缺少列 {col}")
    
    # 计算每个指标的均值和标准差
    summary_data = {}
    
    for metric_name, sampled_col, random_col, uniform_col in metrics_to_summarize:
        # 收集所有系统的值
        sampled_values = [row[sampled_col] for row in rows if not np.isnan(row.get(sampled_col, np.nan))]
        random_values = [row[random_col] for row in rows if not np.isnan(row.get(random_col, np.nan))]
        uniform_values = [row[uniform_col] for row in rows if not np.isnan(row.get(uniform_col, np.nan))]
        
        print(f"{metric_name}: Sampled={len(sampled_values)}, Random={len(random_values)}, Uniform={len(uniform_values)}")
        
        # 调试信息
        if len(sampled_values) > 0:
            print(f"  Sampled values range: {np.min(sampled_values):.6f} - {np.max(sampled_values):.6f}")
        if len(random_values) > 0:
            print(f"  Random values range: {np.min(random_values):.6f} - {np.max(random_values):.6f}")
        if len(uniform_values) > 0:
            print(f"  Uniform values range: {np.min(uniform_values):.6f} - {np.max(uniform_values):.6f}")
        
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
    print(f"均值对比汇总表格已保存到 {summary_path}")
    print(f"汇总了 {len(rows)} 个系统的数据")

if __name__ == '__main__':
    analyse_sampling_compare()
