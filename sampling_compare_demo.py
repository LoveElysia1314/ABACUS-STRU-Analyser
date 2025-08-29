import os
import numpy as np
import pandas as pd
from glob import glob

def calc_metrics(vectors):
    from scipy.spatial.distance import pdist, squareform
    if len(vectors) < 2:
        return np.nan, np.nan, np.nan
    dists = squareform(pdist(vectors, metric='euclidean'))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    annd = np.mean(np.min(dists, axis=1))
    mpd = np.mean(dists[dists != np.inf])
    return min_d, annd, mpd

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
        # 你的采样
        minD, annd, mpd = calc_metrics(vectors[selected])
        # 随机采样
        minD_rand, annd_rand, mpd_rand = [], [], []
        for _ in range(10):
            idx = np.random.choice(n, k, replace=False)
            m, a, p = calc_metrics(vectors[idx])
            minD_rand.append(m)
            annd_rand.append(a)
            mpd_rand.append(p)
        # 均匀采样
        idx_uniform = uniform_sample_indices(n, k)
        minD_uni, annd_uni, mpd_uni = calc_metrics(vectors[idx_uniform])
        rows.append({
            'System': system,
            'MinD_sampled': minD,
            'ANND_sampled': annd,
            'MPD_sampled': mpd,
            'MinD_random': np.mean(minD_rand),
            'ANND_random': np.mean(annd_rand),
            'MPD_random': np.mean(mpd_rand),
            'MinD_uniform': minD_uni,
            'ANND_uniform': annd_uni,
            'MPD_uniform': mpd_uni,
        })
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(result_dir, 'sampling_compare.csv')
    out_df.to_csv(out_path, index=False)
    print(f"采样对比结果已保存到 {out_path}")

if __name__ == '__main__':
    analyse_sampling_compare()
