#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABACUS STRU 轨迹批量分析器 (基于距离矩阵归一化RMSD)
该脚本用于批量分析多个 ABACUS 分子动力学模拟生成的 STRU 文件序列。
它能够处理按特定格式命名的目录结构，并对每个体系进行独立分析，
同时还能对属于同一分子的所有构象进行合并分析。
目录结构假设:
- 当前工作目录
  - struct_mol_1028_conf_0_T400K
    - OUT.ABACUS
      - STRU
        - STRU_MD_*
  - struct_mol_1028_conf_1_T500K
    - OUT.ABACUS
      - STRU
        - STRU_MD_*
  - struct_mol_1029_conf_0_T300K
    - OUT.ABACUS
      - STRU
        - STRU_MD_*
  ...
工作流程：
1.  遍历当前目录，查找所有匹配 'struct_mol_*_conf_*_T*K' 模式的文件夹。
2.  按分子编号 (mol_id) 对这些文件夹进行分组。
3.  对每个找到的体系文件夹 (struct_mol_X_conf_Y_T*):
    a. 读取其 OUT.ABACUS/STRU/STRU_MD_* 文件。
    b. 解析每个 STRU 文件，提取原子坐标（默认排除氢原子）。
    c. 计算每个构象的距离向量。
    d. 使用 KMeans 聚类（默认簇数 = 10）。
    e. 计算每个构象到其归属聚类中心的归一化 RMSD (nRMSD)。
    f. 将结果保存在 analysis_results/struct_mol_X/ 文件夹中，文件名为 nRMSD_<dir_name>.csv。
4.  对每个分子编号组 (mol_id):
    a. 合并该组下所有体系的距离向量。
    b. 对合并后的数据集进行一次全局 KMeans 聚类（默认簇数 = 10）。
    c. 计算每个原始构象到其在全局聚类中归属中心的 nRMSD。
    d. 将合并分析结果保存在 analysis_results/struct_mol_X/ 目录中，文件名为 nRMSD_mol_<mol_id>_combined.csv。
5.  生成一个 analysis_results/comparison_summary.csv 文件，汇总所有单体系和合并分析的关键统计信息，
    包括 Configuration (conf) 和 Temperature (T) 信息，便于分析影响。
特点：
- 多进程并行处理，加速大量构象的读取和计算。
- 自动处理 STRU 文件名排序问题。
- 简洁的输出和详细的执行摘要。
- 聚类标签按簇大小（从大到小）自动排序。
- 默认聚类数为10。
"""
import os
import glob
import numpy as np
import re
import argparse
from scipy.spatial.distance import pdist
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
import csv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入聚类库
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("未安装 scikit-learn。将跳过聚类分析功能。")
    logger.warning("请运行 'pip install scikit-learn' 安装。")

# --- 核心数据结构 ---
class AnalysisResult:
    """封装单次分析的结果数据"""
    def __init__(self, frame_numbers, cluster_labels, nRMSD_to_assigned_center, 
                 avg_nRMSD_to_assigned_center, nRMSD_results_matrix, cluster_centers):
        self.frame_numbers = frame_numbers
        self.cluster_labels = cluster_labels
        self.nRMSD_to_assigned_center = nRMSD_to_assigned_center
        self.avg_nRMSD_to_assigned_center = avg_nRMSD_to_assigned_center
        self.nRMSD_results_matrix = nRMSD_results_matrix
        self.cluster_centers = cluster_centers

class SummaryInfo:
    """封装用于汇总的信息"""
    def __init__(self, type_, dir_name, mol_id, conf, T, min_nRMSD, max_nRMSD, 
                 avg_nRMSD, num_frames, num_clusters):
        self.type = type_
        self.dir_name = dir_name
        self.mol_id = mol_id
        self.conf = conf
        self.T = T
        self.min_nRMSD = min_nRMSD
        self.max_nRMSD = max_nRMSD
        self.avg_nRMSD = avg_nRMSD
        self.num_frames = num_frames
        self.num_clusters = num_clusters

# --- 工具函数 ---
def parse_abacus_stru(stru_file, exclude_hydrogen=True):
    """
    解析 ABACUS STRU 文件，提取原子坐标。
    参数:
        stru_file (str): STRU 文件路径。
        exclude_hydrogen (bool): 是否排除氢原子 (默认: True)。
    返回:
        tuple: (numpy.ndarray: 原子坐标数组 (N_atoms, 3), list: 元素符号列表)。
               如果解析失败，返回 (None, None)。
    """
    try:
        with open(stru_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"无法读取文件 {stru_file}: {e}")
        return None, None

    lattice_constant = 1.0
    atomic_positions = []
    atomic_elements = []
    current_element = None
    element_atoms_count = 0
    element_atoms_collected = 0
    section = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if "LATTICE_CONSTANT" in line:
            section = "LATTICE_CONSTANT"
            continue
        elif "LATTICE_VECTORS" in line:
            section = "LATTICE_VECTORS"
            continue
        elif "ATOMIC_SPECIES" in line:
            section = "ATOMIC_SPECIES"
            continue
        elif "ATOMIC_POSITIONS" in line:
            section = "ATOMIC_POSITIONS"
            continue

        if section == "LATTICE_CONSTANT":
            try:
                lattice_constant = float(re.split(r'\s+', line)[0])
            except (ValueError, IndexError):
                continue
        elif section == "ATOMIC_POSITIONS":
            if re.match(r'^[A-Za-z]{1,2}\s*#', line):
                parts = re.split(r'\s+', line)
                current_element = parts[0]
                element_atoms_count = 0
                element_atoms_collected = 0
                continue
            
            if current_element and "number of atoms" in line:
                try:
                    element_atoms_count = int(re.split(r'\s+', line)[0])
                except (ValueError, IndexError):
                    element_atoms_count = 0
                continue
            
            if current_element and element_atoms_count > 0 and element_atoms_collected < element_atoms_count:
                if exclude_hydrogen and current_element == "H":
                    element_atoms_collected += 1
                    continue
                parts = re.split(r'\s+', line)
                coords = [float(p) for p in parts[:3] if p.replace('.', '', 1).replace('-', '', 1).isdigit()]
                if len(coords) == 3:
                    atomic_elements.append(current_element)
                    atomic_positions.append(coords)
                    element_atoms_collected += 1

    if not atomic_positions:
        return None, None
    
    atomic_positions = np.array(atomic_positions) * lattice_constant
    return atomic_positions, atomic_elements

def calculate_distance_vector(coords):
    """计算原子间距离矩阵的上三角部分（不包含对角线），作为距离向量。"""
    return pdist(coords)

def calculate_normalized_rmsd(dist_vec1, dist_vec2):
    """
    计算两个距离向量之间的归一化 RMSD (nRMSD)。
    nRMSD = sqrt(mean((d1 - d2)^2)) / mean(d1)
    """
    if len(dist_vec1) != len(dist_vec2):
        raise ValueError("距离向量长度不一致")
    try:
        diff = dist_vec1 - dist_vec2
        squared_diff = diff * diff
        mean_squared_diff = np.mean(squared_diff)
        rmsd = np.sqrt(mean_squared_diff)
        mean_dist = np.mean(dist_vec1)
        if mean_dist > 1e-12:
            return rmsd / mean_dist
        else:
            return np.nan
    except Exception:
        return np.nan

def calculate_nRMSD_matrix(vectors_a, vectors_b):
    """计算两组距离向量之间的所有成对归一化 RMSD。"""
    n_a, n_b = vectors_a.shape[0], vectors_b.shape[0]
    nRMSD_matrix = np.full((n_a, n_b), np.nan)
    for i in range(n_a):
        for j in range(n_b):
            nRMSD_matrix[i, j] = calculate_normalized_rmsd(vectors_a[i], vectors_b[j])
    return nRMSD_matrix

def get_molecular_formula(elements, exclude_hydrogen=True):
    """根据元素列表生成分子式字符串。"""
    if exclude_hydrogen:
        elements = [e for e in elements if e != "H"]
    element_counts = defaultdict(int)
    for element in elements:
        element_counts[element] += 1
    formula = ""
    for element in sorted(element_counts.keys()):
        count = element_counts[element]
        formula += element
        if count > 1:
            formula += str(count)
    return formula

def extract_frame_number(filename):
    """从 STRU 文件名中提取帧号。"""
    basename = os.path.basename(filename)
    match = re.search(r'STRU_MD_(\d+)', basename)
    return int(match.group(1)) if match else float('inf')

def sort_stru_files(stru_files):
    """根据帧号对 STRU 文件列表进行排序。"""
    return sorted(stru_files, key=extract_frame_number)

def process_stru_file(args):
    """
    处理单个 STRU 文件的 worker 函数，供多进程调用。
    """
    stru_file, exclude_hydrogen, frame_index, total_frames = args
    try:
        positions, elements = parse_abacus_stru(stru_file, exclude_hydrogen=exclude_hydrogen)
        if positions is not None and elements is not None:
            dist_vector = calculate_distance_vector(positions)
            frame_num = extract_frame_number(stru_file)
            if frame_num == float('inf'):
                frame_num = frame_index
            return frame_index, frame_num, dist_vector, elements
        else:
            return frame_index, frame_index, None, None
    except Exception as e:
        return frame_index, frame_index, None, None

# --- 核心分析逻辑 ---
def perform_kmeans_clustering(data, n_clusters):
    """
    使用 KMeans 对数据进行聚类，并按簇大小对结果进行排序（从大到小）。
    """
    if not SKLEARN_AVAILABLE or len(data) < 2 or n_clusters < 1:
        return None, None, None
    
    if n_clusters >= len(data):
        n_clusters = max(1, len(data) - 1) 
        if n_clusters == 0:
             cluster_labels = np.array([0])
             cluster_centers = data.reshape(1, -1)
             return 1, cluster_labels, cluster_centers

    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        initial_cluster_labels = kmeans.fit_predict(data)
        initial_cluster_centers = kmeans.cluster_centers_
        
        label_counts = Counter(initial_cluster_labels)
        sorted_labels_by_size = [label for label, count in label_counts.most_common()]
        old_to_new_label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_labels_by_size)}
        remapped_cluster_labels = np.array([old_to_new_label_map[label] for label in initial_cluster_labels])
        remapped_cluster_centers = np.array([initial_cluster_centers[old_label] for old_label in sorted_labels_by_size])
        
        return n_clusters, remapped_cluster_labels, remapped_cluster_centers
    except Exception as e:
        logger.error(f"KMeans 聚类过程中出错: {e}")
        return None, None, None

def perform_analysis(data_vectors, n_clusters_to_use):
    """
    执行聚类和 nRMSD 计算的通用逻辑。
    返回 AnalysisResult 对象。
    """
    cluster_labels = None
    cluster_centers = None
    num_clusters = 0
    
    if SKLEARN_AVAILABLE and len(data_vectors) > 1:
        num_clusters, cluster_labels, cluster_centers = perform_kmeans_clustering(
            data_vectors, n_clusters_to_use
        )
    else:
        if not SKLEARN_AVAILABLE:
            logger.info("  -> 跳过聚类分析 (scikit-learn 未安装)。")
        elif len(data_vectors) == 1:
            logger.info("  -> 只有一个结构，跳过聚类分析。")
            num_clusters = 1
            cluster_labels = np.array([0])
            cluster_centers = data_vectors
        else:
            logger.info("  -> 没有有效数据进行聚类分析。")

    nRMSD_results_matrix = None
    if cluster_centers is not None and len(cluster_centers) > 0:
        try:
            nRMSD_results_matrix = calculate_nRMSD_matrix(data_vectors, cluster_centers)
        except Exception as e:
            logger.error(f"  -> 计算 nRMSD 时出错: {e}")
            nRMSD_results_matrix = None

    nRMSD_to_assigned_center = np.full(len(data_vectors), np.nan)
    avg_nRMSD_to_assigned_center = np.nan
    if nRMSD_results_matrix is not None and cluster_labels is not None:
        valid_nRMSD_list = []
        for i in range(len(data_vectors)):
            label = cluster_labels[i]
            if 0 <= label < nRMSD_results_matrix.shape[1]:
                nRMSD_val = nRMSD_results_matrix[i, label]
                nRMSD_to_assigned_center[i] = nRMSD_val
                if not np.isnan(nRMSD_val):
                    valid_nRMSD_list.append(nRMSD_val)
        if valid_nRMSD_list:
            avg_nRMSD_to_assigned_center = np.mean(valid_nRMSD_list)
            
    return AnalysisResult(
        frame_numbers=None, # 需要在调用处设置
        cluster_labels=cluster_labels,
        nRMSD_to_assigned_center=nRMSD_to_assigned_center,
        avg_nRMSD_to_assigned_center=avg_nRMSD_to_assigned_center,
        nRMSD_results_matrix=nRMSD_results_matrix,
        cluster_centers=cluster_centers
    )

def save_results_to_csv(results_file, frame_ids, analysis_result):
    """将分析结果保存到 CSV 文件。"""
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Frame_ID', 'Cluster_Label', 'nRMSD_to_Assigned_Center', 'Avg_nRMSD_to_Assigned_Center']
        if analysis_result.nRMSD_results_matrix is not None:
            for i in range(analysis_result.nRMSD_results_matrix.shape[1]):
                header.append(f'nRMSD_to_Center_{i}')
        writer.writerow(header)
        
        for i in range(len(frame_ids)):
            row = [frame_ids[i]]
            row.append(analysis_result.cluster_labels[i] if analysis_result.cluster_labels is not None else -1)
            row.append(analysis_result.nRMSD_to_assigned_center[i])
            row.append(analysis_result.avg_nRMSD_to_assigned_center)
            if analysis_result.nRMSD_results_matrix is not None:
                for j in range(analysis_result.nRMSD_results_matrix.shape[1]):
                    row.append(analysis_result.nRMSD_results_matrix[i, j])
            writer.writerow(row)

def create_summary_info(type_, dir_name, mol_id, conf, T, analysis_result, num_frames):
    """创建用于汇总的 SummaryInfo 对象。"""
    valid_summary_nRMSD = [d for d in analysis_result.nRMSD_to_assigned_center if not np.isnan(d)]
    min_nRMSD_summary = min(valid_summary_nRMSD) if valid_summary_nRMSD else np.nan
    max_nRMSD_summary = max(valid_summary_nRMSD) if valid_summary_nRMSD else np.nan
    
    return SummaryInfo(
        type_=type_,
        dir_name=dir_name,
        mol_id=mol_id,
        conf=conf,
        T=T,
        min_nRMSD=min_nRMSD_summary,
        max_nRMSD=max_nRMSD_summary,
        avg_nRMSD=analysis_result.avg_nRMSD_to_assigned_center,
        num_frames=num_frames,
        num_clusters=len(analysis_result.cluster_centers) if analysis_result.cluster_centers is not None else 0
    )

# --- 主要分析函数 ---
def analyze_single_system(system_dir, args, max_workers, output_base_dir):
    """
    分析单个体系目录 (struct_mol_*_conf_*_T*K)。
    """
    logger.info(f"--- 开始分析体系: {os.path.basename(system_dir)} ---")
    start_time_sys = time.time()
    
    stru_dir = os.path.join(system_dir, 'OUT.ABACUS', 'STRU')
    system_name = os.path.basename(system_dir)
    stru_files = glob.glob(os.path.join(stru_dir, 'STRU_MD_*'))
    
    if not stru_files:
        logger.warning(f"  警告: 在 {stru_dir} 中没有找到 STRU_MD_* 文件")
        return None, None, None, system_dir

    stru_files = sort_stru_files(stru_files)
    logger.info(f"  -> 找到 {len(stru_files)} 个 STRU 文件")

    all_dist_vectors = [None] * len(stru_files)
    frame_numbers = [None] * len(stru_files)
    molecular_formula = None
    process_args = [(stru_file, not args.include_h, i, len(stru_files)) for i, stru_file in enumerate(stru_files)]
    
    completed = 0
    start_processing = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_stru_file, arg): i for i, arg in enumerate(process_args)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                frame_index, frame_num, dist_vector, elements = future.result()
                if dist_vector is not None and elements is not None:
                    all_dist_vectors[frame_index] = dist_vector
                    frame_numbers[frame_index] = frame_num
                    if molecular_formula is None:
                        molecular_formula = get_molecular_formula(elements, not args.include_h)
                completed += 1
            except Exception:
                completed += 1

    valid_indices = [i for i, vec in enumerate(all_dist_vectors) if vec is not None]
    all_dist_vectors = [all_dist_vectors[i] for i in valid_indices]
    frame_numbers = [frame_numbers[i] for i in valid_indices]
    
    if not all_dist_vectors:
        logger.error(f"  错误: 体系 {system_dir} 没有成功处理任何 STRU 文件")
        return None, None, None, system_dir
        
    logger.info(f"  -> 成功处理 {len(all_dist_vectors)} 个结构")
    logger.info(f"  -> 距离向量计算耗时: {time.time() - start_processing:.2f}秒")

    all_dist_vectors_np = np.array(all_dist_vectors)
    
    # 确定聚类数
    n_clusters_to_use = max(1, args.n_clusters if args.n_clusters is not None else 10)
    analysis_result = perform_analysis(all_dist_vectors_np, n_clusters_to_use)
    analysis_result.frame_numbers = frame_numbers # 设置 frame_numbers

    # 保存结果
    mol_id = re.search(r'struct_mol_(\d+)_', system_name).group(1)
    mol_output_dir = os.path.join(output_base_dir, f'struct_mol_{mol_id}')
    os.makedirs(mol_output_dir, exist_ok=True)
    results_file = os.path.join(mol_output_dir, f'nRMSD_{system_name}.csv')
    
    save_results_to_csv(results_file, frame_numbers, analysis_result)
    logger.info(f"  -> 结果已保存到: {results_file}")

    # 创建摘要信息
    match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', system_name)
    if match:
        mol_id, conf, temp = match.groups()
    else:
        mol_id, conf, temp = 'unknown', 'unknown', 'unknown'
        
    logger.info(f"  -> 体系分析完成，耗时: {time.time() - start_time_sys:.2f}秒")
    
    summary_info = create_summary_info(
        type_='single',
        dir_name=system_name,
        mol_id=mol_id,
        conf=conf,
        T=temp,
        analysis_result=analysis_result,
        num_frames=len(frame_numbers)
    )
    
    return summary_info, all_dist_vectors_np, np.array(frame_numbers), molecular_formula

def analyze_combined_system(mol_id, combined_data, combined_frame_info, args, max_workers, output_base_dir):
    """
    对合并的数据进行分析。
    """
    logger.info(f"--- 开始合并分析分子: mol_{mol_id} (共 {len(combined_data)} 帧) ---")
    start_time_combined = time.time()
    
    all_dist_vectors_np = np.array(combined_data)
    global_frame_ids = [f"{info[0]}_frame_{info[1]}" for info in combined_frame_info]
    
    # 确定聚类数
    n_clusters_to_use = max(1, args.n_clusters if args.n_clusters is not None else 10)
    analysis_result = perform_analysis(all_dist_vectors_np, n_clusters_to_use)
    analysis_result.frame_numbers = global_frame_ids # 设置 frame_ids

    # 保存结果
    mol_output_dir = os.path.join(output_base_dir, f'struct_mol_{mol_id}')
    os.makedirs(mol_output_dir, exist_ok=True)
    results_file = os.path.join(mol_output_dir, f'nRMSD_mol_{mol_id}_combined.csv')
    
    save_results_to_csv(results_file, global_frame_ids, analysis_result)
    logger.info(f"  -> 合并结果已保存到: {results_file}")

    logger.info(f"  -> 合并分析完成，耗时: {time.time() - start_time_combined:.2f}秒")
    
    summary_info = create_summary_info(
        type_='combined',
        dir_name=f"mol_{mol_id}_combined",
        mol_id=mol_id,
        conf='combined',
        T='combined',
        analysis_result=analysis_result,
        num_frames=len(global_frame_ids)
    )
    
    return summary_info

def discover_and_group_systems():
    """在当前目录下发现并按分子ID分组体系文件夹。"""
    pattern = r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K'
    system_dirs = glob.glob('struct_mol_*_conf_*_T*K')
    grouped_systems = defaultdict(list)
    
    for dir_name in system_dirs:
        match = re.match(pattern, dir_name)
        if match:
            mol_id, conf_id, temperature = match.groups()
            grouped_systems[mol_id].append({
                'dir_name': dir_name,
                'mol_id': mol_id,
                'conf_id': conf_id,
                'temperature': temperature
            })
        else:
            logger.warning(f"警告: 目录 '{dir_name}' 不匹配命名规则，已跳过。")

    for mol_id in grouped_systems:
        grouped_systems[mol_id].sort(key=lambda x: (int(x['conf_id']), int(x['temperature'])))
        
    return grouped_systems

def main():
    """主函数，执行整个批量分析流程。"""
    total_start_time = time.time()
    logger.info("=== ABACUS STRU 批量轨迹分析器 (基于归一化RMSD) ===")
    
    parser = argparse.ArgumentParser(description='批量分析ABACUS STRU文件，计算归一化RMSD并聚类。')
    parser.add_argument('--include_h', action='store_true',
                        help='包含氢原子 (默认: 排除以减少计算量)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='最大工作进程数 (默认: CPU数 - 1, 最多1024个)')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='KMeans聚类数 (默认: 10)')
    args = parser.parse_args()

    if args.max_workers is None:
        max_workers = max(1, min(mp.cpu_count() - 1, 1024))
    else:
        max_workers = args.max_workers
    logger.info(f"-> 使用最多 {max_workers} 个工作进程")

    output_base_dir = os.path.join(os.getcwd(), 'analysis_results')
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"-> 所有结果将保存到: {output_base_dir}")

    grouped_systems = discover_and_group_systems()
    if not grouped_systems:
        logger.error("错误: 在当前目录下没有找到匹配 'struct_mol_*_conf_*_T*K' 的体系文件夹。")
        return
        
    logger.info(f"-> 发现 {len(grouped_systems)} 个不同的分子，共 {sum(len(systems) for systems in grouped_systems.values())} 个体系。")
    for mol_id, systems in grouped_systems.items():
        logger.info(f"   - 分子 mol_{mol_id}: {len(systems)} 个构象/体系")
        for s in systems:
            logger.info(f"     * {s['dir_name']}")

    summary_info_list = []
    combined_data_dict = defaultdict(list)
    combined_frame_info_dict = defaultdict(list)
    
    for mol_id, systems in grouped_systems.items():
        logger.info(f"\n========== 分析分子 mol_{mol_id} ==========")
        for system in systems:
            system_dir = system['dir_name']
            summary_info, dist_vectors, frame_nums, formula = analyze_single_system(system_dir, args, max_workers, output_base_dir)
            if summary_info is not None:
                summary_info_list.append(summary_info)
                if dist_vectors is not None and frame_nums is not None:
                    combined_data_dict[mol_id].extend(dist_vectors)
                    combined_frame_info_dict[mol_id].extend([(system_dir, fn) for fn in frame_nums])
            else:
                logger.warning(f"  警告: 体系 {system_dir} 分析失败。")

    logger.info("\n" + "="*50)
    logger.info("开始执行合并分析...")
    for mol_id in combined_data_dict:
        if combined_data_dict[mol_id]:
            combined_summary = analyze_combined_system(
                mol_id, 
                combined_data_dict[mol_id], 
                combined_frame_info_dict[mol_id],
                args, 
                max_workers,
                output_base_dir
            )
            if combined_summary:
                summary_info_list.append(combined_summary)
        else:
            logger.warning(f"  警告: 分子 mol_{mol_id} 没有可用于合并分析的数据。")

    if summary_info_list:
        comparison_file = os.path.join(output_base_dir, 'comparison_summary.csv')
        logger.info(f"\n-> 生成比较摘要文件: {comparison_file}")
        with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Analysis_Type', 'Directory/Group', 'Molecule_ID', 'Configuration', 'Temperature',
                'Number_of_Frames', 'Number_of_Clusters',
                'Min_nRMSD_to_Assigned_Center', 
                'Max_nRMSD_to_Assigned_Center', 
                'Avg_nRMSD_to_Assigned_Center'
            ])
            for info in summary_info_list:
                writer.writerow([
                    info.type,
                    info.dir_name,
                    info.mol_id,
                    info.conf,
                    info.T,
                    info.num_frames,
                    info.num_clusters,
                    f"{info.min_nRMSD:.6f}" if not np.isnan(info.min_nRMSD) else 'NaN',
                    f"{info.max_nRMSD:.6f}" if not np.isnan(info.max_nRMSD) else 'NaN',
                    f"{info.avg_nRMSD:.6f}" if not np.isnan(info.avg_nRMSD) else 'NaN'
                ])
        logger.info(f"-> 比较文件已生成: {comparison_file}")
    else:
        logger.warning("\n警告: 没有生成任何分析摘要，比较文件未创建。")

    total_time = time.time() - total_start_time
    logger.info("\n" + "="*50)
    logger.info("=== 批量分析总览 ===")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"分析的分子数: {len(grouped_systems)}")
    logger.info(f"分析的体系数: {sum(len(systems) for systems in grouped_systems.values())}")
    logger.info(f"生成的比较条目数: {len(summary_info_list)}")
    logger.info(f"所有结果已保存至: {output_base_dir}")
    logger.info("======================")
    logger.info("分析完成。")

if __name__ == "__main__":
    main()