#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import re
import argparse
from scipy.spatial.distance import pdist, cdist
import csv
import logging
import time
from collections import defaultdict
import multiprocessing as mp
import sys

# ====================== 数据结构类 ======================
class FrameData:
    __slots__ = ('frame_id', 'positions', 'distance_vector')
    
    def __init__(self, frame_id, positions):
        self.frame_id = frame_id
        self.positions = positions
        self.distance_vector = None

class SystemMetrics:
    __slots__ = ('system_name', 'mol_id', 'conf', 'temperature', 
                 'num_frames', 'dimension', 'ConfVol', 'nRMSF', 
                 'MCV', 'avg_nLdRMS', 'sampled_frames')
    
    def __init__(self, system_name, mol_id, conf, temperature):
        self.system_name = system_name
        self.mol_id = mol_id
        self.conf = conf
        self.temperature = temperature
        self.num_frames = 0
        self.dimension = 0
        self.ConfVol = 0.0
        self.nRMSF = 0.0
        self.MCV = 0.0
        self.avg_nLdRMS = 0.0
        self.sampled_frames = []

# ====================== 文件解析器 ======================
class StruParser:
    def __init__(self, exclude_hydrogen=True):
        self.exclude_hydrogen = exclude_hydrogen
    
    def parse(self, stru_file):
        """基于原始实现的健壮STRU解析器"""
        try:
            with open(stru_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            return None

        lattice_constant = 1.0
        atomic_positions = []
        current_element = None
        element_atoms_count = 0
        element_atoms_collected = 0
        section = None

        for line in lines:
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
                except:
                    continue
            elif section == "ATOMIC_POSITIONS":
                # 元素行格式: "Element # 原子数"
                if re.match(r'^[A-Za-z]{1,2}\s*#', line):
                    parts = re.split(r'\s+', line)
                    current_element = parts[0]
                    element_atoms_count = 0
                    element_atoms_collected = 0
                    continue

                # 原子数行格式: "原子数 # number of atoms"
                if current_element and "number of atoms" in line:
                    try:
                        element_atoms_count = int(re.split(r'\s+', line)[0])
                    except:
                        element_atoms_count = 0
                    continue

                # 原子坐标行
                if current_element and element_atoms_count > 0 and element_atoms_collected < element_atoms_count:
                    # 排除氢原子（如果设置）
                    if self.exclude_hydrogen and current_element.upper() in ("H", "HYDROGEN"):
                        element_atoms_collected += 1
                        continue
                    
                    parts = re.split(r'\s+', line)
                    if len(parts) < 3:
                        continue
                    
                    try:
                        coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                        atomic_positions.append(np.array(coords) * lattice_constant)
                        element_atoms_collected += 1
                    except ValueError:
                        continue

        if not atomic_positions:
            return None

        return np.array(atomic_positions)

# ====================== 指标计算器 ======================
class MetricCalculator:
    @staticmethod
    def calculate_distance_vectors(positions):
        """计算原子间距离向量并进行归一化处理"""
        if len(positions) < 2:
            return np.array([])
        
        # 计算原始距离向量
        raw_vectors = pdist(positions)
        
        # 计算向量范数（长度）
        norm = np.linalg.norm(raw_vectors)
        
        # 归一化处理：如果范数大于0则除以范数，否则保持原样
        if norm > 1e-12:
            normalized_vectors = raw_vectors / norm
        else:
            normalized_vectors = raw_vectors
        
        return normalized_vectors
    
    @staticmethod
    def compute_global_mean_distance(vectors):
        """计算归一化后的全局平均距离"""
        if len(vectors) == 0:
            return 0.0
        return np.mean(vectors)
    
    @staticmethod
    def compute_conf_volume(vectors, global_mean):
        """计算构象空间体积（基于归一化距离向量）"""
        if len(vectors) == 0:
            return 0.0
        
        # 标准化向量（相对于全局平均）
        normalized = vectors / (global_mean + 1e-12)
        
        # 计算协方差矩阵的行列式
        cov_matrix = np.cov(normalized.T)
        try:
            det = np.linalg.det(cov_matrix)
            D = cov_matrix.shape[0]
            return det ** (1.0 / D) if det > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def compute_normalized_rmsf(vectors, global_mean):
        """计算标准化RMSF（基于归一化距离向量）"""
        if len(vectors) < 2:
            return 0.0
        
        variances = np.var(vectors, axis=0)
        rmsf = np.sqrt(np.mean(variances))
        return rmsf / (global_mean + 1e-12)
    
    @staticmethod
    def compute_mean_cv(vectors):
        """计算平均变异系数（基于归一化距离向量）"""
        if len(vectors) < 2:
            return 0.0
        
        means = np.mean(vectors, axis=0)
        stds = np.std(vectors, axis=0)
        
        # 避免除以零
        with np.errstate(divide='ignore', invalid='ignore'):
            cvs = np.where(means > 1e-12, stds / means, 0.0)
        
        return np.mean(cvs)
    
    @staticmethod
    def compute_frame_metrics(vectors, global_mean):
        """计算每帧的nLdRMS指标（基于归一化距离向量）"""
        if len(vectors) == 0:
            return []
        
        mean_vector = np.mean(vectors, axis=0)
        # 向量化计算
        diff = vectors - mean_vector
        dRMS = np.sqrt(np.mean(diff ** 2, axis=1))
        return dRMS / (global_mean + 1e-12)

# ====================== 采样器 ======================
class PowerMeanSampler:
    @staticmethod
    def power_mean(arr, p):
        arr = np.asarray(arr)
        arr = np.maximum(arr, 1e-12)
        if p == 0:
            return np.exp(np.mean(np.log(arr)))
        elif p == 1:
            return np.mean(arr)
        elif p == -1:
            return len(arr) / np.sum(1.0 / arr)
        else:
            return (np.mean(arr ** p)) ** (1.0 / p)

    @staticmethod
    def select(points, k, nLdRMS_values, p=0.5):
        """
        基于幂平均距离的高效采样策略
        p=1: 算数平均距离
        p=0: 几何平均距离
        p=-1: 调和平均距离
        其它p: 幂平均距离
        """
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0

        # --- 智能双种子初始化 ---
        if nLdRMS_values is not None and len(nLdRMS_values) == n:
            first_seed = np.argmax(nLdRMS_values)
        else:
            first_seed = 0
        selected = [first_seed]

        # 第二个种子：选择距离第一个种子最远的点
        if k > 1 and n > 1:
            current_selected_points = points[selected]
            candidate_points = points
            dist_to_selected = cdist(candidate_points, current_selected_points)
            distances_from_first = dist_to_selected[:, 0]
            second_seed = np.argmax(distances_from_first)
            if second_seed != first_seed:
                selected.append(second_seed)

        remaining_indices = set(range(n)) - set(selected)

        # --- 增量贪心选择 ---
        while len(selected) < k and remaining_indices:
            current_selected_points = points[selected]
            candidate_list = list(remaining_indices)
            candidate_points = points[candidate_list]
            dist_to_selected = cdist(candidate_points, current_selected_points)
            dist_to_selected = np.maximum(dist_to_selected, 1e-12)
            # 计算每个候选点到已选点的幂平均距离
            if p == 0:
                agg = np.exp(np.mean(np.log(dist_to_selected), axis=1))
            elif p == 1:
                agg = np.mean(dist_to_selected, axis=1)
            elif p == -1:
                agg = dist_to_selected.shape[1] / np.sum(1.0 / dist_to_selected, axis=1)
            else:
                agg = (np.mean(dist_to_selected ** p, axis=1)) ** (1.0 / p)
            best_idx = np.argmax(agg)
            best_candidate = candidate_list[best_idx]
            selected.append(best_candidate)
            remaining_indices.remove(best_candidate)

        # --- 局部交换优化 ---
        if len(selected) < 2:
            return selected, 0, 0.0
        selected_points = points[selected]
        sel_dist_pairs = cdist(selected_points, selected_points)
        np.fill_diagonal(sel_dist_pairs, np.inf)
        sel_dist_pairs = np.maximum(sel_dist_pairs, 1e-11)
        # 目标函数：所有采样点对的幂平均距离的均值
        triu_idx = np.triu_indices_from(sel_dist_pairs, k=1)
        pair_dists = sel_dist_pairs[triu_idx]
        initial_obj = PowerMeanSampler.power_mean(pair_dists, p)
        current_obj = initial_obj
        swap_count = 0
        not_selected = list(set(range(n)) - set(selected))
        improved = True
        max_iterations = len(selected) * len(not_selected)
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            best_improvement = 0.0
            best_swap = None
            for i_idx, i in enumerate(selected):
                if not not_selected:
                    break
                current_point = points[i:i+1]
                other_selected_indices = [selected[j] for j in range(len(selected)) if j != i_idx]
                if not other_selected_indices:
                    continue
                other_selected_points = points[other_selected_indices]
                dist_i_to_others = cdist(current_point, other_selected_points).flatten()
                dist_i_to_others = np.maximum(dist_i_to_others, 1e-11)
                old_contrib = PowerMeanSampler.power_mean(dist_i_to_others, p)
                candidate_points = points[not_selected]
                dist_candidates_to_others = cdist(candidate_points, other_selected_points)
                dist_candidates_to_others = np.maximum(dist_candidates_to_others, 1e-11)
                new_contribs = np.array([PowerMeanSampler.power_mean(dist_candidates_to_others[j], p) for j in range(len(not_selected))])
                improvements = new_contribs - old_contrib  # 幂平均距离越大越好
                if len(improvements) > 0:
                    best_candidate_idx = np.argmax(improvements)
                    best_candidate_improvement = improvements[best_candidate_idx]
                    if best_candidate_improvement > best_improvement:
                        best_improvement = best_candidate_improvement
                        best_swap = (i_idx, not_selected[best_candidate_idx])
            if best_swap is not None and best_improvement > 1e-12:
                i_idx, j = best_swap
                selected[i_idx] = j
                not_selected = list(set(range(n)) - set(selected))
                current_obj += best_improvement
                swap_count += 1
                improved = True
        improve_ratio = (current_obj - initial_obj) / initial_obj if initial_obj > 0 else 0.0
        return selected, swap_count, improve_ratio

# ====================== 系统分析器 ======================
class SystemAnalyzer:
    def __init__(self, include_hydrogen, sample_ratio, power_p=0.5):
        self.include_hydrogen = include_hydrogen
        self.sample_ratio = sample_ratio
        self.power_p = power_p
        self.parser = StruParser(exclude_hydrogen=not include_hydrogen)
    
    def analyze(self, system_dir):
        """分析单个分子动力学系统"""
        system_name = os.path.basename(system_dir)
        
        # 解析系统信息
        match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', system_name)
        if not match:
            return None
        
        mol_id, conf, temperature = match.groups()
        
        # 创建结果对象
        metrics = SystemMetrics(system_name, mol_id, conf, temperature)
        
        # 查找STRU文件
        stru_dir = os.path.join(system_dir, 'OUT.ABACUS', 'STRU')
        stru_files = glob.glob(os.path.join(stru_dir, 'STRU_MD_*'))
        if not stru_files:
            return None
        
        # 按帧号排序
        def extract_frame_number(filename):
            match = re.search(r'STRU_MD_(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else float('inf')
        
        stru_files.sort(key=extract_frame_number)
        
        # 解析所有帧（保留原始帧ID）
        frames = []
        for stru_file in stru_files:
            frame_id = extract_frame_number(stru_file)
            if frame_id == float('inf'):
                continue
                
            positions = self.parser.parse(stru_file)
            if positions is None:
                continue
            
            frame_data = FrameData(frame_id, positions)
            frame_data.distance_vector = MetricCalculator.calculate_distance_vectors(positions)
            frames.append(frame_data)
        
        if not frames:
            return None
        
        metrics.num_frames = len(frames)
        
        # 提取距离向量
        vectors = [f.distance_vector for f in frames]
        
        # 检查向量维度一致性
        dims = [len(v) for v in vectors if len(v) > 0]
        if not dims:
            return None
            
        if min(dims) != max(dims):
            # 使用最小维度
            min_dim = min(dims)
            for i in range(len(vectors)):
                if len(vectors[i]) > min_dim:
                    vectors[i] = vectors[i][:min_dim]
        
        metrics.dimension = min(dims)
        vector_matrix = np.array(vectors)
        
        # 计算全局统计量
        global_mean = MetricCalculator.compute_global_mean_distance(vector_matrix)
        metrics.ConfVol = MetricCalculator.compute_conf_volume(vector_matrix, global_mean)
        metrics.nRMSF = MetricCalculator.compute_normalized_rmsf(vector_matrix, global_mean)
        metrics.MCV = MetricCalculator.compute_mean_cv(vector_matrix)
        
        # 计算每帧指标
        frame_nLdRMS_values = MetricCalculator.compute_frame_metrics(vector_matrix, global_mean)
        if frame_nLdRMS_values is not None and len(frame_nLdRMS_values) > 0:
            metrics.avg_nLdRMS = np.mean(frame_nLdRMS_values)
        else:
            metrics.avg_nLdRMS = 0.0
        
        # 执行采样（基于帧总数计算采样数量）
        k = max(2, int(round(self.sample_ratio * metrics.num_frames)))
        p = getattr(self, 'power_p', 0.5)
        if k < metrics.num_frames:
            sampled_indices, swap_count, improve_ratio = PowerMeanSampler.select(vector_matrix, k, frame_nLdRMS_values, p=p)
            metrics.sampled_frames = [frames[i].frame_id for i in sampled_indices]
            return metrics, frames, frame_nLdRMS_values, swap_count, improve_ratio
        else:
            metrics.sampled_frames = [f.frame_id for f in frames]
            return metrics, frames, frame_nLdRMS_values, 0, 0.0

# ====================== 结果保存器 ======================
class ResultSaver:
    @staticmethod
    def save_frame_metrics(output_dir, system_name, frames, frame_nLdRMS_values, sampled_frames):
        """保存每帧指标和采样结果"""
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"metrics_per_frame_{system_name}.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame_ID', 'nLdRMS', 'Selected'])
            
            # 将采样帧ID转换为集合以便快速查找
            sampled_frames_set = set(sampled_frames)
            
            # 输出每帧的原始ID、nLdRMS值和是否被采样
            for i, frame in enumerate(frames):
                frame_id = frame.frame_id
                nLdRMS = frame_nLdRMS_values[i] if i < len(frame_nLdRMS_values) else 0.0
                selected = 1 if frame_id in sampled_frames_set else 0
                writer.writerow([frame_id, f"{nLdRMS:.6f}", selected])
    
    @staticmethod
    def save_system_summary(output_dir, all_metrics):
        """保存系统汇总结果"""
        csv_path = os.path.join(output_dir, "system_summary.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'System', 'Molecule_ID', 'Configuration', 'Temperature(K)',
                'Num_Frames', 'Dimension', 'ConfVol', 'nRMSF', 'MCV', 
                'avg_nLdRMS', 'Sampled_Frames'
            ])
            
            for metrics in all_metrics:
                # 对采样帧进行升序排序
                sorted_sampled_frames = sorted(metrics.sampled_frames)
                writer.writerow([
                    metrics.system_name,
                    metrics.mol_id,
                    metrics.conf,
                    metrics.temperature,
                    metrics.num_frames,
                    metrics.dimension,
                    f"{metrics.ConfVol:.6e}",
                    f"{metrics.nRMSF:.6f}",
                    f"{metrics.MCV:.6f}",
                    f"{metrics.avg_nLdRMS:.6f}",
                    ';'.join(map(str, sorted_sampled_frames))
                ])

# ====================== 目录分析器 ======================
class DirectoryAnalyzer:
    @staticmethod
    def analyze():
        """分析当前目录结构，统计分子数和每个分子的体系数"""
        system_dirs = glob.glob('struct_mol_*_conf_*_T*K')
        if not system_dirs:
            return {}, 0, 0
        
        mol_systems = defaultdict(list)
        
        for system_dir in system_dirs:
            system_name = os.path.basename(system_dir)
            match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', system_name)
            if match:
                mol_id = match.group(1)
                mol_systems[mol_id].append(system_name)
        
        total_molecules = len(mol_systems)
        total_systems = sum(len(systems) for systems in mol_systems.values())
        
        return mol_systems, total_molecules, total_systems

# ====================== 主应用程序 ======================
class MainApp:
    def __init__(self):
        self.logger = None
        self.output_dir = None
    
    def configure_logger(self, output_dir):
        """配置日志记录器"""
        self.output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(self.output_dir, "analysis.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        self.logger.handlers = [file_handler, console_handler]
    
    def run(self):
        start_time = time.time()
        parser = argparse.ArgumentParser(description='ABACUS STRU轨迹分析器')
        parser.add_argument('--include_h', action='store_true', help='包含氢原子')
        parser.add_argument('--sample_ratio', type=float, default=0.05, 
                            help='采样比例 (默认: 0.05)')
        parser.add_argument('--power_p', type=float, default=0.5,
                            help='幂平均距离的p值 (p=1为算数平均, p=0为几何平均, p=-1为调和平均, 其它为幂平均, 默认0.5)')
        parser.add_argument('--workers', type=int, default=-1, 
                            help='并行工作进程数 (默认: 自动使用所有可用核心)')
        parser.add_argument('--output_dir', type=str, default="analysis_results",
                            help='输出目录 (默认: analysis_results)')

        args = parser.parse_args()

        # 自动检测并设置工作进程数
        if args.workers == -1:
            try:
                if 'SLURM_CPUS_PER_TASK' in os.environ:
                    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
                elif 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
                    workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
                else:
                    workers = mp.cpu_count()
            except:
                workers = max(1, mp.cpu_count())
        else:
            workers = max(1, args.workers)

        self.configure_logger(args.output_dir)

        self.logger.info("=" * 60)
        self.logger.info("ABACUS STRU 轨迹分析器启动")
        self.logger.info(f"采样比例: {args.sample_ratio}")
        self.logger.info(f"包含氢原子: {'是' if args.include_h else '否'}")
        self.logger.info(f"工作进程: {workers} (自动设置)")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info("=" * 60)

        mol_systems, total_molecules, total_systems = DirectoryAnalyzer.analyze()
        if not mol_systems:
            self.logger.error("未找到符合格式的系统目录")
            return
        
        self.logger.info(f"发现 {total_molecules} 个分子，共 {total_systems} 个体系")
        for mol_id, systems in sorted(mol_systems.items(), key=lambda x: int(x[0])):
            self.logger.info(f"  分子 {mol_id}: {len(systems)} 个体系")

        all_system_dirs = []
        for mol_id, system_names in mol_systems.items():
            for system_name in system_names:
                match = re.match(r'struct_mol_(\d+)_conf_(\d+)_T(\d+)K', system_name)
                if match:
                    _, conf, temp = match.groups()
                    system_dir = f"struct_mol_{mol_id}_conf_{conf}_T{temp}K"
                    if os.path.exists(system_dir):
                        all_system_dirs.append(system_dir)

        all_metrics = []
        all_swap_counts = []
        all_improve_ratios = []
        mol_count = 0
        processed_systems = 0

        analyzer = SystemAnalyzer(args.include_h, args.sample_ratio, args.power_p)

        for mol_id, system_names in sorted(mol_systems.items(), key=lambda x: int(x[0])):
            mol_count += 1
            mol_start_time = time.time()

            mol_system_dirs = [d for d in all_system_dirs if f"struct_mol_{mol_id}_" in d]
            num_systems = len(mol_system_dirs)

            self.logger.info(f"[分子 {mol_id} ({mol_count}/{total_molecules})] 开始处理 {num_systems} 个体系...")

            mol_metrics = []
            successful_systems = 0

            if workers > 1:
                with mp.Pool(processes=workers) as pool:
                    tasks = []
                    for system_dir in mol_system_dirs:
                        tasks.append(pool.apply_async(
                            analyzer.analyze, 
                            (system_dir,)
                        ))
                    for i, task in enumerate(tasks):
                        try:
                            result = task.get()
                            if result:
                                metrics, frames, frame_nLdRMS_values, swap_count, improve_ratio = result
                                mol_metrics.append((metrics, frames, frame_nLdRMS_values))
                                all_swap_counts.append(swap_count)
                                all_improve_ratios.append(improve_ratio)
                                mol_output_dir = os.path.join(self.output_dir, f"struct_mol_{mol_id}")
                                ResultSaver.save_frame_metrics(
                                    mol_output_dir, metrics.system_name, frames, 
                                    frame_nLdRMS_values, metrics.sampled_frames
                                )
                                successful_systems += 1
                                processed_systems += 1
                                elapsed = time.time() - mol_start_time
                                self.logger.info(f"  [{i+1}/{num_systems}] {metrics.system_name} 完成 ({elapsed:.1f}s)")
                        except Exception as e:
                            self.logger.error(f"处理体系时出错: {str(e)}")
            else:
                for i, system_dir in enumerate(mol_system_dirs):
                    try:
                        result = analyzer.analyze(system_dir)
                        if result:
                            metrics, frames, frame_nLdRMS_values, swap_count, improve_ratio = result
                            mol_metrics.append((metrics, frames, frame_nLdRMS_values))
                            all_swap_counts.append(swap_count)
                            all_improve_ratios.append(improve_ratio)
                            mol_output_dir = os.path.join(self.output_dir, f"struct_mol_{mol_id}")
                            ResultSaver.save_frame_metrics(
                                mol_output_dir, metrics.system_name, frames, 
                                frame_nLdRMS_values, metrics.sampled_frames
                            )
                            successful_systems += 1
                            processed_systems += 1
                            elapsed = time.time() - mol_start_time
                            self.logger.info(f"  [{i+1}/{num_systems}] {metrics.system_name} 完成 ({elapsed:.1f}s)")
                    except Exception as e:
                        self.logger.error(f"处理体系 {system_dir} 时出错: {str(e)}")

            for metrics, frames, frame_nLdRMS_values in mol_metrics:
                all_metrics.append(metrics)

            mol_elapsed = time.time() - mol_start_time
            self.logger.info(f"[分子 {mol_id}] 完成: {successful_systems}/{num_systems} 体系 - 耗时: {mol_elapsed:.1f}s")
        
        # 保存汇总结果
        if all_metrics:
            ResultSaver.save_system_summary(self.output_dir, all_metrics)
        
        # 总体统计
        elapsed = time.time() - start_time
        successful_molecules = len(set(m.mol_id for m in all_metrics))
        successful_systems = len(all_metrics)

        # 输出采样优化统计
        if all_swap_counts:
            mean_swap = np.mean(all_swap_counts)
            mean_improve = np.mean(all_improve_ratios)
            self.logger.info("-" * 60)
            self.logger.info(f"采样优化全局统计：")
            self.logger.info(f"  平均交换次数: {mean_swap:.2f}")
            self.logger.info(f"  平均目标函数改善比例: {mean_improve:.4%}")
            self.logger.info(f"  总交换次数: {sum(all_swap_counts)}")
            self.logger.info("-" * 60)

        self.logger.info("=" * 60)
        self.logger.info("分析完成!")
        self.logger.info(f"总分子数: {total_molecules}, 成功分析: {successful_molecules}")
        self.logger.info(f"总体系数: {total_systems}, 成功分析: {successful_systems}")
        self.logger.info(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        self.logger.info(f"平均每体系: {elapsed/successful_systems:.1f} 秒" if successful_systems > 0 else "")
        self.logger.info(f"结果目录: {self.output_dir}")
        self.logger.info("=" * 60)

if __name__ == "__main__":
    app = MainApp()
    app.run()