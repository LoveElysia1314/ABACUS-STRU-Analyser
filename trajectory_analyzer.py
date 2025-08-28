#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本名: trajectory_analyzer.py
功能: ABACUS 轨迹分析器
==================================================

功能：
-----
批量分析 ABACUS 分子动力学模拟生成的 STRU 文件序列，
基于原子间距离向量的标准化分布特性及推荐指标，
评估构象多样性。

核心指标（精简版）：
------------------
1. ConfVol: 构象空间体积（核心多样性指标）
2. nRMSF: 标准化距离均方根波动（整体波动强度）
3. MCV: 平均变异系数（标准化相对波动）
4. nLdRMS: 每帧到平均结构的距离RMS（单帧指标）

所有输出均为无量纲标准化指标，适用于跨体系比较。

输入结构：
---------
当前目录下包含多个体系文件夹：
    struct_mol_<ID>_conf_<N>_T<T>K/
    └── OUT.ABACUS/
        └── STRU/
            ├── STRU_MD_0
            ├── STRU_MD_1
            └── ...

输出结构：
---------
analysis_results/
├── struct_mol_<ID>/
│   ├── standardized_metrics_per_frame_<system>.csv    ← 单帧指标 + 采样结果
│   ├── standardized_metrics_<system>.json             ← 聚合指标（无每帧）
│   └── ...（合并分析同理）
└── standardized_distribution_summary.csv               ← 所有体系汇总对比

使用方式：
----------
python trajectory_analyzer.py [--include_h] [--max_workers N] [--sample_ratio R] [--sample_count N]

依赖：
------
numpy, scipy, stru_parser.py
"""

import os
import glob
import numpy as np
import re
import argparse
from scipy.spatial.distance import pdist, cdist
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import csv
import json
import logging

# 导入解析器模块
import stru_parser

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 核心数据结构 ---
class FrameMetrics:
    """单帧的标准化分析指标（仅保留必要指标）"""
    def __init__(self, frame_id, norm_local_dRMS):
        self.frame_id = frame_id
        self.norm_local_dRMS = norm_local_dRMS  # 标准化局部距离RMS

class DistributionMetrics:
    """整体分布的标准化分析指标（精简版）"""
    def __init__(self, ConfVol, nAPD, nRMSF, MCV, num_frames, dimension):
        self.ConfVol = ConfVol    # 标准化构象空间体积
        self.nAPD = nAPD          # 标准化平均成对距离
        self.nRMSF = nRMSF        # 标准化距离均方根波动
        self.MCV = MCV            # 平均变异系数
        self.num_frames = num_frames
        self.dimension = dimension  # 距离向量维度

class AnalysisResult:
    """单次分析的结果封装"""
    def __init__(self, frame_metrics_list, distribution_metrics, frame_numbers, molecular_formula):
        self.frame_metrics_list = frame_metrics_list
        self.distribution_metrics = distribution_metrics
        self.frame_numbers = frame_numbers
        self.molecular_formula = molecular_formula

class SummaryInfo:
    """用于生成汇总表的信息"""
    def __init__(self, type_, dir_name, mol_id, conf, T,
                 ConfVol, nAPD, nRMSF, MCV,
                 avg_nLdRMS,
                 num_frames, dimension):
        self.type = type_
        self.dir_name = dir_name
        self.mol_id = mol_id
        self.conf = conf
        self.T = T
        self.ConfVol = ConfVol
        self.nAPD = nAPD
        self.nRMSF = nRMSF
        self.MCV = MCV
        self.avg_nLdRMS = avg_nLdRMS
        self.num_frames = num_frames
        self.dimension = dimension

# --- 采样函数 ---
def greedy_max_avg_distance(points, k, frame_nLdRMS_values, num_runs=10, seed=42):
    """
    使用贪心算法选取 k 个点，使其成对距离之和最大。
    
    Parameters:
        points (np.ndarray): 形状为 (n, d) 的点集
        k (int): 采样点数
        frame_nLdRMS_values (list): 每帧的nLdRMS值，用于选择初始点
        num_runs (int): 多次随机初始化取最优 (增加到10次)
        seed (int): 随机种子

    Returns:
        selected_points (np.ndarray): 选中的点 (k, d)
        selected_indices (np.ndarray): 选中点的索引 (k,)
    """
    n, d = points.shape
    if k >= n:
        return points, np.arange(n)

    best_sum = -np.inf
    best_indices = None

    for run in range(num_runs):
        np.random.seed(seed + run)
        # 使用 nLdRMS 最大、最小的两个点作为初始点
        nLdRMS_array = np.array(frame_nLdRMS_values)
        min_idx = np.argmin(nLdRMS_array)
        max_idx = np.argmax(nLdRMS_array)
        if min_idx == max_idx:
            # 如果所有nLdRMS值相同，随机选择两个点
            idxs = np.random.choice(n, 2, replace=False).tolist()
        else:
            idxs = [min_idx, max_idx]
        
        selected = set(idxs)

        # 初始化每个点到已选集合的距离和
        dists_to_S = np.zeros(n)
        for i in range(n):
            if i not in selected:
                dists_to_S[i] = np.sum(cdist(points[list(selected)], points[i:i+1]))

        # 贪心选择剩余点
        for _ in range(2, k):
            # 找到使 dists_to_S 最大的点
            candidates = [i for i in range(n) if i not in selected]
            if not candidates:
                break
            next_idx = candidates[np.argmax(dists_to_S[candidates])]
            selected.add(next_idx)

            # 更新 dists_to_S
            new_dists = cdist(points[next_idx:next_idx+1], points).flatten()
            for i in range(n):
                if i not in selected:
                    dists_to_S[i] += new_dists[i]
                else:
                    dists_to_S[i] = -np.inf  # 已选点不再考虑

        # 计算当前选点的距离和
        selected_list = list(selected)
        if len(selected_list) >= 2:
            subset_points = points[selected_list]
            pairwise_dists = cdist(subset_points, subset_points)
            dist_sum = np.sum(pairwise_dists) / 2  # 每对只算一次

            if dist_sum > best_sum:
                best_sum = dist_sum
                best_indices = np.array(selected_list)

    if best_indices is None:
        # 如果所有运行都失败，返回前k个点
        best_indices = np.arange(min(k, n))

    return points[best_indices], best_indices

# --- 核心分析逻辑 ---
def estimate_mean_distance(vectors):
    """精确计算平均成对距离"""
    N = len(vectors)
    if N <= 1:
        return 0.0
    try:
        pairwise_distances = pdist(vectors, metric='euclidean')
        return float(np.mean(pairwise_distances))
    except Exception as e:
        logger.error(f"pdist 计算失败: {e}")
        return 0.0

def calculate_dRMSF(distance_vectors):
    """
    计算距离均方根波动 (dRMSF)
    distance_vectors: (N_frames, M) 的距离向量矩阵
    """
    if len(distance_vectors) <= 1:
        return 0.0
    # 计算每个原子对距离的方差
    variances = np.var(distance_vectors, axis=0)  # shape: (M,)
    mean_variance = np.mean(variances)
    return float(np.sqrt(mean_variance))

def calculate_MeanCV(distance_vectors):
    """
    计算平均变异系数 (MeanCV)
    distance_vectors: (N_frames, M) 的距离向量矩阵
    """
    if len(distance_vectors) <= 1:
        return 0.0
    means = np.mean(distance_vectors, axis=0)
    stds = np.std(distance_vectors, axis=0)
    # 避免除零错误
    cvs = np.divide(stds, means, out=np.zeros_like(stds), where=means > 1e-12)
    return float(np.mean(cvs))

def calculate_distribution_metrics(distance_vectors, global_mean_distance):
    """计算精简版分布指标"""
    if len(distance_vectors) == 0:
        return DistributionMetrics(0.0, 0.0, 0.0, 0.0, 0, 0)
    if len(distance_vectors) == 1:
        D = distance_vectors.shape[1] if distance_vectors.ndim > 1 else 0
        return DistributionMetrics(0.0, 0.0, 0.0, 0.0, 1, D)

    D = distance_vectors.shape[1]
    
    # 1. 计算 norm_volume (构象空间体积)
    normalized_vectors = distance_vectors / (global_mean_distance + 1e-12)
    try:
        cov = np.cov(normalized_vectors.T)
        det = np.linalg.det(cov)
        norm_volume = det ** (1.0 / D) if det > 0 and D > 0 else 0.0
    except:
        norm_volume = 0.0

    # 2. 计算 dRMSF_norm
    dRMSF = calculate_dRMSF(distance_vectors)
    dRMSF_norm = dRMSF / (global_mean_distance + 1e-12)

    # 3. 计算 MeanCV
    MeanCV = calculate_MeanCV(distance_vectors)

    return DistributionMetrics(
        ConfVol=norm_volume,
        nAPD=1.0,  # 归一化后理论值
        nRMSF=dRMSF_norm,
        MCV=MeanCV,
        num_frames=len(distance_vectors),
        dimension=D
    )

def calculate_all_frame_metrics(distance_vectors, frame_numbers, global_mean_distance, mean_distances):
    """
    计算所有帧的精简版指标
    - Norm_Local_dRMS: 到平均结构的距离RMS
    """
    N = len(distance_vectors)
    if N == 0:
        return []
    if N == 1:
        return [FrameMetrics(frame_numbers[0], 0.0)]

    metrics_list = []
    for i in range(N):
        # 计算 Norm_Local_dRMS: 到平均结构的距离RMS
        diff = distance_vectors[i] - mean_distances
        local_dRMS = np.sqrt(np.mean(diff ** 2))
        norm_local_dRMS = local_dRMS / (global_mean_distance + 1e-12)
        
        metrics_list.append(FrameMetrics(
            frame_id=frame_numbers[i],
            norm_local_dRMS=norm_local_dRMS
        ))
    return metrics_list

def perform_analysis(data_vectors, frame_numbers, sample_ratio=0.1, sample_count=None):
    """执行精简版分析"""
    if len(data_vectors) == 0:
        return None
    
    # 1. 计算全局统计量
    global_mean_distance = estimate_mean_distance(data_vectors)
    mean_distances = np.mean(data_vectors, axis=0)  # 每个原子对的平均距离
    
    # 2. 计算每帧指标
    frame_metrics = calculate_all_frame_metrics(
        data_vectors, 
        frame_numbers,
        global_mean_distance,
        mean_distances
    )
    
    # 3. 计算分布指标
    distribution_metrics = calculate_distribution_metrics(data_vectors, global_mean_distance)
    
    # 4. 执行采样
    selected_indices = None
    n_frames = len(data_vectors)
    if sample_count is not None:
        k = min(sample_count, n_frames)
    else:
        k = max(1, min(int(sample_ratio * n_frames), n_frames))

    if k < n_frames and k > 1:
        # 提取每帧的nLdRMS值用于初始点选择
        frame_nLdRMS_values = [fm.norm_local_dRMS for fm in frame_metrics]
        logger.info(f"  -> 执行采样: 选择 {k}/{n_frames} 帧")
        _, selected_indices = greedy_max_avg_distance(data_vectors, k=k, frame_nLdRMS_values=frame_nLdRMS_values)
    else:
        logger.info(f"  -> 帧数不足或无需采样: {n_frames} 帧")
        selected_indices = np.arange(n_frames)
    
    result = AnalysisResult(
        frame_metrics_list=frame_metrics,
        distribution_metrics=distribution_metrics,
        frame_numbers=frame_numbers,
        molecular_formula=None
    )
    
    return result, selected_indices

# --- 输出函数 ---
def save_per_frame_csv(results_file, analysis_result, selected_indices=None):
    """保存单帧指标 + 采样结果（精简版）"""
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        if selected_indices is not None:
            writer.writerow(['Frame_ID', 'nLdRMS', 'Selected'])
        else:
            writer.writerow(['Frame_ID', 'nLdRMS'])
        
        selected_set = set(selected_indices) if selected_indices is not None else set()
        
        for i, fm in enumerate(analysis_result.frame_metrics_list):
            if selected_indices is not None:
                is_selected = 1 if i in selected_set else 0
                writer.writerow([
                    fm.frame_id,
                    fm.norm_local_dRMS,
                    is_selected
                ])
            else:
                writer.writerow([
                    fm.frame_id,
                    fm.norm_local_dRMS
                ])
    logger.info(f"  -> 逐帧CSV已保存: {results_file}")

def save_results_to_json(results_file, analysis_result):
    """保存精简JSON：仅聚合信息，不含每帧数据"""
    if analysis_result is None or not analysis_result.frame_metrics_list:
        logger.warning(f"分析结果为空，跳过JSON保存: {results_file}")
        return

    dm = analysis_result.distribution_metrics
    fm_list = analysis_result.frame_metrics_list

    # 计算平均单帧指标
    avg_norm_local_dRMS = float(np.mean([fm.norm_local_dRMS for fm in fm_list]))

    result_dict = {
        "molecular_formula": analysis_result.molecular_formula,
        "distribution_metrics": {
            "ConfVol": float(dm.ConfVol),
            "nAPD": float(dm.nAPD),
            "nRMSF": float(dm.nRMSF),
            "MCV": float(dm.MCV),
            "num_frames": dm.num_frames,
            "dimension": dm.dimension
        },
        "average_frame_metrics": {
            "avg_nLdRMS": avg_norm_local_dRMS
        }
    }

    with open(results_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"  -> 精简JSON已保存（无每帧数据）: {results_file}")

def generate_summary_file(summary_info_list, output_base_dir):
    """生成精简版汇总表"""
    summary_file = os.path.join(output_base_dir, 'standardized_distribution_summary.csv')
    logger.info(f"\n-> 生成汇总文件: {summary_file}")
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Analysis_Type', 'Directory', 'Molecule_ID', 'Configuration', 'Temperature',
            'Num_Frames', 'Dimension', 'ConfVol', 'nAPD',
            'nRMSF', 'MCV', 'avg_nLdRMS'
        ])
        for info in summary_info_list:
            writer.writerow([
                info.type, info.dir_name, info.mol_id, info.conf, info.T,
                info.num_frames, info.dimension,
                f"{info.ConfVol:.6e}", f"{info.nAPD:.6f}",
                f"{info.nRMSF:.6f}", f"{info.MCV:.6f}",
                f"{info.avg_nLdRMS:.6f}"
            ])
    logger.info(f"-> 汇总文件已生成: {summary_file}")

# --- 主要分析函数 ---
def analyze_single_system(system_dir, args, max_workers, output_base_dir):
    logger.info(f"--- 开始分析体系: {os.path.basename(system_dir)} ---")
    start_time = time.time()

    stru_dir = os.path.join(system_dir, 'OUT.ABACUS', 'STRU')
    system_name = os.path.basename(system_dir)
    stru_files = stru_parser.sort_stru_files(glob.glob(os.path.join(stru_dir, 'STRU_MD_*')))
    if not stru_files:
        logger.warning(f"未找到 STRU 文件: {stru_dir}")
        return None, None, None, system_dir

    logger.info(f"  -> 找到 {len(stru_files)} 个 STRU 文件")

    all_vecs = [None] * len(stru_files)
    frame_nums = [None] * len(stru_files)
    mol_formula = None
    proc_args = [(f, not args.include_h, i, len(stru_files)) for i, f in enumerate(stru_files)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(stru_parser.process_stru_file, arg): i for i, arg in enumerate(proc_args)}
        for future in as_completed(futures):
            idx, frame_num, vec, elems = future.result()
            if vec is not None:
                all_vecs[idx] = vec
                frame_nums[idx] = frame_num
                if mol_formula is None and elems is not None:
                    mol_formula = stru_parser.get_molecular_formula(elems, not args.include_h)

    valid_idx = [i for i, v in enumerate(all_vecs) if v is not None]
    all_vecs = [all_vecs[i] for i in valid_idx]
    frame_nums = [frame_nums[i] for i in valid_idx]

    if not all_vecs:
        logger.error(f"无有效结构: {system_dir}")
        return None, None, None, system_dir

    data_vectors = np.array(all_vecs)
    
    # 获取采样参数
    sample_ratio = args.sample_ratio if args.sample_ratio is not None else 0.1
    sample_count = args.sample_count
    
    analysis_result, selected_indices = perform_analysis(data_vectors, frame_nums, sample_ratio, sample_count)
    if analysis_result:
        analysis_result.molecular_formula = mol_formula

    # 保存输出
    match = re.search(r'struct_mol_(\d+)', system_name)
    mol_id = match.group(1) if match else "unknown"
    mol_dir = os.path.join(output_base_dir, f"struct_mol_{mol_id}")
    os.makedirs(mol_dir, exist_ok=True)

    csv_file = os.path.join(mol_dir, f"standardized_metrics_per_frame_{system_name}.csv")
    json_file = os.path.join(mol_dir, f"standardized_metrics_{system_name}.json")
    save_per_frame_csv(csv_file, analysis_result, selected_indices)
    save_results_to_json(json_file, analysis_result)

    # 解析元信息
    conf_match = re.match(r'struct_mol_\d+_conf_(\d+)_T(\d+)K', system_name)
    conf, temp = conf_match.groups()[:2] if conf_match else ('unknown', 'unknown')

    # 创建汇总信息
    fm_list = analysis_result.frame_metrics_list
    avg_norm_local_dRMS = np.mean([fm.norm_local_dRMS for fm in fm_list])
    dm = analysis_result.distribution_metrics

    summary_info = SummaryInfo(
        type_='single', dir_name=system_name, mol_id=mol_id, conf=conf, T=temp,
        ConfVol=dm.ConfVol, nAPD=dm.nAPD,
        nRMSF=dm.nRMSF, MCV=dm.MCV,
        avg_nLdRMS=avg_norm_local_dRMS,
        num_frames=dm.num_frames, dimension=dm.dimension
    )

    logger.info(f"  -> 体系分析完成: {time.time() - start_time:.2f}s")
    return summary_info, data_vectors, np.array(frame_nums), mol_formula

def analyze_combined_system(mol_id, combined_data, combined_frame_info, args, max_workers, output_base_dir):
    logger.info(f"--- 合并分析分子 mol_{mol_id} ({len(combined_data)} 帧) ---")
    start_time = time.time()

    data_vectors = np.array(combined_data)
    frame_ids = [f"{info[0]}_frame_{info[1]}" for info in combined_frame_info]
    
    # 获取合并采样参数
    if args.sample_combined_count is not None:
        sample_count = args.sample_combined_count
        sample_ratio = None
    elif args.sample_combined_ratio is not None:
        sample_ratio = args.sample_combined_ratio
        sample_count = None
    else:
        sample_ratio = 0.1  # 默认10%
        sample_count = None
    
    analysis_result, selected_indices = perform_analysis(data_vectors, frame_ids, sample_ratio, sample_count)

    mol_dir = os.path.join(output_base_dir, f"struct_mol_{mol_id}")
    os.makedirs(mol_dir, exist_ok=True)

    csv_file = os.path.join(mol_dir, f"standardized_metrics_per_frame_mol_{mol_id}_combined.csv")
    json_file = os.path.join(mol_dir, f"standardized_metrics_mol_{mol_id}_combined.json")
    save_per_frame_csv(csv_file, analysis_result, selected_indices)
    save_results_to_json(json_file, analysis_result)

    # 创建汇总信息
    fm_list = analysis_result.frame_metrics_list
    avg_norm_local_dRMS = np.mean([fm.norm_local_dRMS for fm in fm_list])
    dm = analysis_result.distribution_metrics

    summary_info = SummaryInfo(
        type_='combined', dir_name=f"mol_{mol_id}_combined", mol_id=mol_id, conf='combined', T='combined',
        ConfVol=dm.ConfVol, nAPD=dm.nAPD,
        nRMSF=dm.nRMSF, MCV=dm.MCV,
        avg_nLdRMS=avg_norm_local_dRMS,
        num_frames=dm.num_frames, dimension=dm.dimension
    )

    logger.info(f"  -> 合并分析完成: {time.time() - start_time:.2f}s")
    return summary_info

def discover_and_group_systems():
    pattern = r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K'
    dirs = glob.glob('struct_mol_*_conf_*_T*K')
    grouped = defaultdict(list)
    for d in dirs:
        match = re.match(pattern, d)
        if match:
            mol_id, conf, temp = match.groups()
            grouped[mol_id].append({'dir_name': d, 'conf': conf, 'temp': temp})
        else:
            logger.warning(f"跳过不匹配目录: {d}")
    return grouped

# --- 主函数 ---
def main():
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description='ABACUS 轨迹分析器（精简版）')
    parser.add_argument('--include_h', action='store_true', help='包含氢原子')
    parser.add_argument('--max_workers', type=int, default=None, help='最大进程数')
    
    # 采样参数
    parser.add_argument('--sample_ratio', type=float, default=None, help='采样比例 (0~1)')
    parser.add_argument('--sample_count', type=int, default=None, help='采样数量')
    parser.add_argument('--sample_combined_ratio', type=float, default=None, help='合并体系采样比例')
    parser.add_argument('--sample_combined_count', type=int, default=None, help='合并体系采样数量')
    
    args = parser.parse_args()

    max_workers = args.max_workers or max(1, min(mp.cpu_count() - 1, 8))
    logger.info(f"使用 {max_workers} 个进程")

    output_base_dir = os.path.join(os.getcwd(), 'analysis_results')
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"结果将保存至: {output_base_dir}")

    grouped = discover_and_group_systems()
    if not grouped:
        logger.error("未发现匹配的体系目录")
        return

    logger.info(f"发现 {len(grouped)} 个分子，共 {sum(len(v) for v in grouped.values())} 个体系")

    summary_info_list = []
    combined_data_dict = defaultdict(list)
    combined_frame_info_dict = defaultdict(list)

    for mol_id, systems in grouped.items():
        logger.info(f"\n========== 分析分子 mol_{mol_id} ==========")
        for sys in systems:
            summary, vecs, frames, formula = analyze_single_system(
                sys['dir_name'], args, max_workers, output_base_dir)
            if summary:
                summary_info_list.append(summary)
                if vecs is not None:
                    combined_data_dict[mol_id].extend(vecs)
                    combined_frame_info_dict[mol_id].extend([(sys['dir_name'], f) for f in frames])

    logger.info("\n" + "="*50)
    logger.info("开始合并分析...")
    for mol_id in combined_data_dict:
        if combined_data_dict[mol_id]:
            combined_summary = analyze_combined_system(
                mol_id, combined_data_dict[mol_id], combined_frame_info_dict[mol_id],
                args, max_workers, output_base_dir)
            if combined_summary:
                summary_info_list.append(combined_summary)

    if summary_info_list:
        generate_summary_file(summary_info_list, output_base_dir)
    else:
        logger.warning("无有效分析结果生成")

    total_time = time.time() - total_start_time
    logger.info(f"\n分析完成，总耗时: {total_time:.2f} 秒")
    logger.info(f"分析分子数: {len(grouped)}")
    logger.info(f"生成汇总条目数: {len(summary_info_list)}")
    logger.info(f"结果路径: {output_base_dir}")

if __name__ == "__main__":
    main()