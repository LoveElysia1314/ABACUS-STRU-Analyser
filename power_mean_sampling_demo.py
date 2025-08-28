#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
幂平均采样可视化演示脚本

本脚本用于演示不同p值下的幂平均采样效果，支持多维度向量采样，
通过距离分布图展示不同幂平均策略的采样质量。

主要功能：
1. 生成多维度单位球面上的混合分布数据
2. 使用不同p值的幂平均方法进行采样优化
3. 通过距离分布直方图和统计信息可视化采样效果
4. 支持任意维度扩展，高维数据统一使用距离分布展示

作者：AI Assistant
版本：2.0
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from sampler import PowerMeanSampler, SamplingStrategy
from utils import MathUtils

# 忽略可能的警告信息
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 设置中文字体，自动适配常见环境
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STSong']
matplotlib.rcParams['axes.unicode_minus'] = False


# ================================ 配置管理 ================================

class HyperParameters:
    """
    超参数配置类
    
    集中管理所有超参数，便于调优和实验
    """
    
    # ========== 核心算法参数 ==========
    
    # 幂平均p值列表：控制采样策略的核心参数
    # -1: 调和平均（偏向最小值），0: 几何平均，1: 算术平均，2: 二次幂平均，>2: 偏向最大值
    P_VALUES: List[float] = [-1, 0.5, 1, 2]
    
    # ========== 数据生成参数 ==========
    
    # 维度配置：支持的所有维度及其对应的数据规模
    # 格式：{维度: {'total': 总点数, 'sample': 采样数, 'centers': 高斯中心数}}
    DIMENSION_CONFIG: Dict[int, Dict[str, int]] = {
        3:  {'total': 256,  'sample': 32,  'centers': 4},   # 3D：中等规模
        5:  {'total': 512, 'sample': 64, 'centers': 8},   # 5D：高维入门
        9: {'total': 1024, 'sample': 128, 'centers': 16},  # 9D：中高维
        14: {'total': 2048, 'sample': 256, 'centers': 32}  # 14D：高维测试
    }
    
    # 混合分布参数
    MIX_RATIO: float = 0.6      # 均匀分布与高斯分布的混合比例 [0,1]，0=纯高斯，1=纯均匀
    GAUSSIAN_STD: float = 0.15  # 高斯分布标准差，控制聚类紧密程度，越小越紧密
    
    # ========== 可视化参数 ==========
    
    # 图形尺寸参数
    FIGURE_SIZE_PER_COL: float = 4.5    # 每列图形宽度（英寸）
    FIGURE_DPI: int = 300               # 图形分辨率，影响保存质量
    TITLE_PAD: float = 18               # 标题与图形间距，避免重叠
    LAYOUT_PAD: float = 2.5             # 子图间距，整体布局松紧度
    
    # 高级分布曲线可视化参数
    KDE_ALPHA: float = 0.85             # 主密度曲线透明度 [0,1]
    KDE_LINEWIDTH: float = 2.0          # 主密度曲线线宽
    FILL_ALPHA: float = 0.3             # 分布填充透明度
    
    # 自适应核密度估计参数
    MIN_PLOT_POINTS: int = 100          # 最少绘制采样点数
    MAX_PLOT_POINTS: int = 500          # 最多绘制采样点数
    BOUNDARY_BUFFER_RATIO: float = 0.1  # 边界缓冲区比例
    LOCAL_ENHANCEMENT_THRESHOLD: int = 20  # 启用局部增强的最小数据点数
    
    # 统计信息显示参数
    STATS_PRECISION: int = 4            # 统计量显示小数位数
    STATS_FONTSIZE: int = 8             # 统计信息字体大小
    STATS_BOX_ALPHA: float = 0.8        # 统计信息背景透明度
    
    # ========== 输出文件参数 ==========
    
    OUTPUT_MAIN: str = "power_mean_sampling_results.png"     # 主结果图文件名
    
    @classmethod
    def get_dimensions(cls) -> List[int]:
        """获取所有配置的维度列表"""
        return sorted(cls.DIMENSION_CONFIG.keys())
    
    @classmethod
    def get_dimension_config(cls, dim: int) -> Dict[str, int]:
        """获取指定维度的配置信息"""
        if dim not in cls.DIMENSION_CONFIG:
            raise ValueError(f"维度 {dim} 未在配置中定义，支持的维度：{cls.get_dimensions()}")
        return cls.DIMENSION_CONFIG[dim].copy()
    
    @classmethod
    def validate_config(cls) -> None:
        """验证配置参数的合理性"""
        # 检查p值列表
        if not cls.P_VALUES:
            raise ValueError("P_VALUES 不能为空")
        
        # 检查维度配置
        for dim, config in cls.DIMENSION_CONFIG.items():
            if dim < 2:
                raise ValueError(f"维度必须 >= 2，当前维度：{dim}")
            if config['sample'] >= config['total']:
                raise ValueError(f"维度 {dim} 的采样数 ({config['sample']}) 不能大于等于总点数 ({config['total']})")
            if config['centers'] < 1:
                raise ValueError(f"维度 {dim} 的中心数必须 >= 1")
        
        # 检查数值范围
        if not (0 <= cls.MIX_RATIO <= 1):
            raise ValueError(f"MIX_RATIO 必须在 [0,1] 范围内，当前值：{cls.MIX_RATIO}")
        if cls.GAUSSIAN_STD <= 0:
            raise ValueError(f"GAUSSIAN_STD 必须 > 0，当前值：{cls.GAUSSIAN_STD}")


# ================================ 数据生成器 ================================

class VectorGenerator:
    """
    多维向量数据生成器
    
    支持在任意维度的单位球面上生成各种分布的数据点
    """
    
    @staticmethod
    def generate_uniform_sphere(n: int, dim: int) -> np.ndarray:
        """
        在dim维单位球面上均匀生成n个点
        
        Args:
            n: 生成点的数量
            dim: 空间维度
            
        Returns:
            形状为 (n, dim) 的数组，每行为一个单位向量
        """
        # 使用高斯分布生成，然后归一化到单位球面（Muller方法）
        points = np.random.normal(size=(n, dim))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # 避免除零错误
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    @staticmethod
    def generate_multicenter_gaussian(n: int, dim: int, num_centers: int, std_dev: float) -> np.ndarray:
        """
        在dim维单位球面上生成多中心高斯分布的点
        
        Args:
            n: 生成点的总数量
            dim: 空间维度
            num_centers: 高斯中心的数量
            std_dev: 高斯分布的标准差
            
        Returns:
            形状为 (n, dim) 的数组，包含围绕多个中心的高斯分布点
        """
        # 随机生成中心点
        centers = VectorGenerator.generate_uniform_sphere(num_centers, dim)
        
        # 为每个中心分配数据点
        points_per_center = n // num_centers
        remaining_points = n % num_centers
        
        all_points = []
        for i, center in enumerate(centers):
            # 为最后一个中心分配剩余的点
            current_n = points_per_center + (1 if i < remaining_points else 0)
            if current_n == 0:
                continue
                
            # 在中心周围生成高斯分布的点
            cluster_points = np.random.normal(loc=center, scale=std_dev, size=(current_n, dim))
            
            # 投影到单位球面
            norms = np.linalg.norm(cluster_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            cluster_points = cluster_points / norms
            
            all_points.append(cluster_points)
        
        # 合并所有点并随机打乱
        result = np.vstack(all_points)
        np.random.shuffle(result)
        return result[:n]  # 确保返回精确的n个点
    
    @staticmethod
    def generate_mixed_distribution(n: int, dim: int, num_centers: int, 
                                  std_dev: float, mix_ratio: float) -> np.ndarray:
        """
        生成均匀分布与多中心高斯分布的混合
        
        Args:
            n: 生成点的总数量
            dim: 空间维度
            num_centers: 高斯中心的数量
            std_dev: 高斯分布的标准差
            mix_ratio: 均匀分布的比例 [0,1]，1表示纯均匀，0表示纯高斯
            
        Returns:
            形状为 (n, dim) 的混合分布数据
        """
        n_uniform = int(n * mix_ratio)
        n_gaussian = n - n_uniform
        
        points_list = []
        
        # 生成均匀分布部分
        if n_uniform > 0:
            uniform_points = VectorGenerator.generate_uniform_sphere(n_uniform, dim)
            points_list.append(uniform_points)
        
        # 生成高斯分布部分
        if n_gaussian > 0:
            gaussian_points = VectorGenerator.generate_multicenter_gaussian(
                n_gaussian, dim, num_centers, std_dev
            )
            points_list.append(gaussian_points)
        
        # 合并并随机打乱
        if points_list:
            result = np.vstack(points_list)
            np.random.shuffle(result)
            return result
        else:
            # 边界情况：生成最少的均匀分布点
            return VectorGenerator.generate_uniform_sphere(n, dim)


# ================================ 分析工具 ================================

class PowerMeanNameConverter:
    """幂平均名称转换工具"""
    
    _NAME_MAP = {
        -1: "调和平均",
        0: "几何平均", 
        1: "算术平均",
        2: "二次幂平均(RMS)",
        float('inf'): "最大值"
    }
    
    @classmethod
    def get_name(cls, p: float) -> str:
        """根据p值获取对应的数学名称"""
        if p in cls._NAME_MAP:
            return cls._NAME_MAP[p]
        elif p > 0:
            return f"{p:.1f}次幂平均" if p != int(p) else f"{int(p)}次幂平均"
        else:
            return f"{p:.1f}次幂平均" if p != int(p) else f"{int(p)}次幂平均"


class DistanceAnalyzer:
    """距离分布分析工具"""
    
    @staticmethod
    def calculate_pairwise_distances(coords: np.ndarray, selected_indices: List[int]) -> np.ndarray:
        """
        计算选中点之间的成对距离
        
        Args:
            coords: 所有坐标点
            selected_indices: 选中点的索引列表
            
        Returns:
            一维数组，包含所有成对距离（去除重复和自距离）
        """
        if len(selected_indices) < 2:
            return np.array([])
            
        selected_coords = coords[selected_indices]
        distance_matrix = cdist(selected_coords, selected_coords)
        
        # 提取上三角矩阵（排除对角线），避免重复计算
        upper_triangle = np.triu(distance_matrix, k=1)
        return upper_triangle[upper_triangle > 0]
    
    @staticmethod
    def compute_distance_statistics(distances: np.ndarray) -> Dict[str, float]:
        """计算距离的统计信息"""
        if len(distances) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'median': 0.0, 'count': 0
            }
        
        return {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'median': float(np.median(distances)),
            'count': len(distances)
        }


# ================================ 可视化工具 ================================

class DistributionPlotter:
    """距离分布可视化工具"""
    
    def __init__(self, config: HyperParameters):
        self.config = config
    
    def plot_distance_distribution(self, distances: np.ndarray, title: str, ax) -> None:
        """
        绘制距离分布图（自适应核密度估计曲线 + 统计信息）
        
        使用更先进的方法：
        1. 自适应带宽的核密度估计
        2. 边界修正处理
        3. 局部适应性优化
        
        Args:
            distances: 距离数据
            title: 图表标题
            ax: matplotlib轴对象
        """
        # 处理空数据情况
        if len(distances) == 0:
            ax.text(0.5, 0.5, '无距离数据', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=11, pad=self.config.TITLE_PAD)
            return
        
        # 处理单点或双点情况
        if len(distances) <= 2:
            ax.axvline(x=distances[0], color='red', linewidth=2, label=f'单值: {distances[0]:.4f}')
            ax.set_ylim(0, 1)
        else:
            # 使用多种核密度估计方法进行融合绘制
            self._plot_advanced_kde(ax, distances)
        
        # 添加统计信息文本框
        self._add_statistics_box(ax, distances)
        
        # 设置图表属性
        ax.set_xlabel('距离', fontsize=10)
        ax.set_ylabel('密度', fontsize=10)
        ax.set_title(title, fontsize=11, pad=self.config.TITLE_PAD)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_advanced_kde(self, ax, distances: np.ndarray) -> None:
        """
        绘制高级核密度估计曲线
        
        采用多种方法组合：
        1. 自适应核密度估计
        2. 边界修正核密度
        3. 局部线性插值增强
        """
        # 数据范围分析
        data_min, data_max = distances.min(), distances.max()
        data_range = data_max - data_min
        
        # 处理数据范围很小的情况（几乎单值）
        if data_range < 1e-10:
            ax.axvline(x=data_min, color='red', linewidth=2, 
                      label=f'近似单值: {data_min:.4f}')
            ax.set_ylim(0, 1)
            return
        
        # 固定绘制范围到[0,2]，这是距离的理论范围
        x_min = 0.0  # 距离最小值
        x_max = 2.0  # 单位球面上最大距离
        
        # 自适应采样点数
        n_points = min(500, max(100, len(distances) * 10))
        x_range = np.linspace(x_min, x_max, n_points)
        
        try:
            # 方法1: 标准高斯核密度估计
            kde_standard = gaussian_kde(distances)
            
            # 方法2: 使用Silverman规则优化带宽
            n = len(distances)
            std_dev = np.std(distances)
            iqr = np.percentile(distances, 75) - np.percentile(distances, 25)
            
            # Scott规则和Silverman规则的组合
            scott_bw = n**(-1./(4+1)) * std_dev
            silverman_bw = 0.9 * min(std_dev, iqr/1.34) * n**(-1./5)
            optimal_bw = (scott_bw + silverman_bw) / 2
            
            # 应用优化带宽
            kde_optimized = gaussian_kde(distances, bw_method=optimal_bw/std_dev)
            
            # 计算密度值
            density_standard = kde_standard(x_range)
            density_optimized = kde_optimized(x_range)
            
            # 边界修正：对于接近0和接近2的区域进行特殊处理
            boundary_mask_low = x_range < 0.1  # 接近0的区域
            boundary_mask_high = x_range > 1.9  # 接近2的区域
            
            if np.any(boundary_mask_low) or np.any(boundary_mask_high):
                # 在边界区域使用反射方法
                reflected_data = np.concatenate([distances, -distances, 4.0 - distances])
                kde_reflected = gaussian_kde(reflected_data, bw_method=optimal_bw/np.std(reflected_data))
                density_reflected = kde_reflected(x_range) * 3  # 乘以3补偿反射
                
                # 在低边界区域混合使用反射密度
                if np.any(boundary_mask_low):
                    alpha_boundary_low = np.exp(-x_range[boundary_mask_low] / 0.05)
                    density_optimized[boundary_mask_low] = (
                        alpha_boundary_low * density_reflected[boundary_mask_low] + 
                        (1 - alpha_boundary_low) * density_optimized[boundary_mask_low]
                    )
                
                # 在高边界区域混合使用反射密度
                if np.any(boundary_mask_high):
                    alpha_boundary_high = np.exp(-(2.0 - x_range[boundary_mask_high]) / 0.05)
                    density_optimized[boundary_mask_high] = (
                        alpha_boundary_high * density_reflected[boundary_mask_high] + 
                        (1 - alpha_boundary_high) * density_optimized[boundary_mask_high]
                    )
            
            # 绘制主密度曲线（优化版本）
            ax.plot(x_range, density_optimized, 'b-', 
                   alpha=self.config.KDE_ALPHA, 
                   linewidth=self.config.KDE_LINEWIDTH,
                   label='优化密度曲线')
            
            # 如果数据足够多，添加局部适应性曲线
            if len(distances) >= 20:
                # 使用局部加权回归增强细节
                density_enhanced = self._local_enhancement(x_range, density_optimized, distances)
                ax.plot(x_range, density_enhanced, 'r--', 
                       alpha=0.7, linewidth=1.5, label='局部增强曲线')
            
            # 添加数据点位置指示
            y_max = np.max(density_optimized)
            self._add_data_points_indication(ax, distances, y_max)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # 备用方案：简单直方图风格的绘制
            self._fallback_plot(ax, distances, x_range)
    
    def _local_enhancement(self, x_range: np.ndarray, base_density: np.ndarray, 
                          distances: np.ndarray) -> np.ndarray:
        """
        局部线性加权增强密度估计
        
        在数据点密集的区域提供更好的局部适应性
        """
        enhanced_density = base_density.copy()
        
        # 计算每个点的局部密度权重
        for i, x in enumerate(x_range):
            # 找到附近的数据点
            local_distances = np.abs(distances - x)
            
            # 使用指数衰减权重
            weights = np.exp(-local_distances / (np.std(distances) * 0.5))
            local_weight = np.sum(weights) / len(distances)
            
            # 应用局部调整
            enhanced_density[i] = base_density[i] * (1 + local_weight * 0.2)
        
        # 平滑处理避免过度波动
        enhanced_density = gaussian_filter1d(enhanced_density, sigma=1.0)
        
        return enhanced_density
    
    def _add_data_points_indication(self, ax, distances: np.ndarray, y_max: float) -> None:
        """添加数据点位置的可视化指示"""
        # 确保坐标轴范围固定在[0,2]
        ax.set_xlim(0, 2)
        
        # 对于较少的数据点，显示每个点的位置
        if len(distances) <= 50:
            # 在x轴上方显示数据点
            y_tick = y_max * 0.05
            ax.scatter(distances, [y_tick] * len(distances), 
                      alpha=0.6, s=20, color='orange', marker='|', 
                      label=f'数据点 (n={len(distances)})')
        else:
            # 对于大量数据点，显示分位数
            percentiles = [10, 25, 50, 75, 90]
            quantile_values = np.percentile(distances, percentiles)
            y_tick = y_max * 0.05
            
            colors = ['lightcoral', 'orange', 'red', 'orange', 'lightcoral']
            for i, (p, val) in enumerate(zip(percentiles, quantile_values)):
                # 只显示在[0,2]范围内的分位数线
                if 0 <= val <= 2:
                    ax.axvline(x=val, color=colors[i], alpha=0.5, 
                              linestyle=':', linewidth=1)
                    ax.text(val, y_max * 0.95, f'P{p}', rotation=90, 
                           fontsize=7, ha='center', va='top')
    
    def _fallback_plot(self, ax, distances: np.ndarray, x_range: np.ndarray) -> None:
        """备用绘制方案：核密度估计失败时使用"""
        try:
            # 简单的高斯核密度
            kde_simple = gaussian_kde(distances)
            density_simple = kde_simple(x_range)
            ax.plot(x_range, density_simple, 'g-', linewidth=2, 
                   label='简化密度曲线')
        except:
            # 最后备用：直接显示数据分布
            ax.hist(distances, bins=min(20, len(distances)//2 + 1), 
                   alpha=0.5, density=True, color='lightblue', 
                   label='分布近似')
    
    def _add_statistics_box(self, ax, distances: np.ndarray) -> None:
        """在图表上添加统计信息文本框"""
        stats = DistanceAnalyzer.compute_distance_statistics(distances)
        
        # 格式化统计信息文本
        precision = self.config.STATS_PRECISION
        stats_text = (
            f"均值: {stats['mean']:.{precision}f}\n"
            f"标准差: {stats['std']:.{precision}f}\n"
            f"最小值: {stats['min']:.{precision}f}\n"
            f"最大值: {stats['max']:.{precision}f}\n"
            f"中位数: {stats['median']:.{precision}f}"
        )
        
        # 添加文本框
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes,
               fontsize=self.config.STATS_FONTSIZE,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='wheat', 
                        alpha=self.config.STATS_BOX_ALPHA,
                        edgecolor='orange'))


# ================================ 采样处理器 ================================

class SamplingProcessor:
    """采样处理核心类"""
    
    def __init__(self):
        self.sampler = PowerMeanSampler()
    
    def process_sampling_case(self, coords: np.ndarray, sample_count: int, 
                            p_value: float) -> Tuple[List[int], int, float]:
        """
        执行单个采样案例
        
        Args:
            coords: 输入坐标数据
            sample_count: 需要采样的点数
            p_value: 幂平均的p值
            
        Returns:
            (选中的索引列表, 交换次数, 改善百分比)
        """
        try:
            selected_indices, swap_count, improvement = self.sampler.select_frames(
                coords, sample_count, p=p_value
            )
            return selected_indices, swap_count, improvement
        except Exception as e:
            # 采样失败时返回前k个点作为备选
            fallback_indices = list(range(min(sample_count, len(coords))))
            return fallback_indices, 0, 0.0


# ================================ 主演示类 ================================

class PowerMeanSamplingDemo:
    """
    幂平均采样演示主控制器
    
    协调数据生成、采样处理、结果可视化的整个流程
    """
    
    def __init__(self, config: Optional[HyperParameters] = None):
        self.config = config or HyperParameters()
        self.config.validate_config()  # 验证配置
        
        # 初始化组件
        self.vector_generator = VectorGenerator()
        self.sampling_processor = SamplingProcessor()
        self.distance_analyzer = DistanceAnalyzer()
        self.plotter = DistributionPlotter(self.config)
        self.name_converter = PowerMeanNameConverter()
    
    def run_complete_demo(self) -> None:
        """运行完整的演示流程"""
        print("=" * 60)
        print("幂平均采样可视化演示")
        print("=" * 60)
        
        # 1. 显示配置信息
        self._print_configuration()
        
        # 2. 生成多维度数据
        all_data = self._generate_all_dimension_data()
        
        # 3. 执行采样并生成主要结果图
        main_results = self._process_all_sampling_cases(all_data)
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
    
    def _print_configuration(self) -> None:
        """打印当前配置信息"""
        print(f"\n📊 配置信息:")
        print(f"   支持维度: {self.config.get_dimensions()}")
        print(f"   测试p值: {self.config.P_VALUES}")
        print(f"   混合比例: {self.config.MIX_RATIO:.2f} (均匀/高斯)")
        print(f"   高斯标准差: {self.config.GAUSSIAN_STD:.3f}")
        
        print(f"\n📋 数据规模:")
        for dim in self.config.get_dimensions():
            config = self.config.get_dimension_config(dim)
            ratio = config['sample'] / config['total']
            print(f"   {dim:2d}D: {config['total']:4d}点 → {config['sample']:3d}点 "
                  f"({ratio:.1%}, {config['centers']}中心)")
    
    def _generate_all_dimension_data(self) -> Dict[int, np.ndarray]:
        """为所有维度生成测试数据"""
        print(f"\n🔄 生成数据...")
        all_data = {}
        
        for dim in self.config.get_dimensions():
            config = self.config.get_dimension_config(dim)
            
            # 生成混合分布数据
            coords = self.vector_generator.generate_mixed_distribution(
                n=config['total'],
                dim=dim,
                num_centers=config['centers'],
                std_dev=self.config.GAUSSIAN_STD,
                mix_ratio=self.config.MIX_RATIO
            )
            
            all_data[dim] = coords
            print(f"   {dim:2d}D: 生成 {config['total']} 个点 "
                  f"(均匀{int(config['total']*self.config.MIX_RATIO)} + "
                  f"高斯{config['total']-int(config['total']*self.config.MIX_RATIO)})")
        
        return all_data
    
    def _process_single_case(self, p_value: float, dim: int, coords: np.ndarray, config: Dict[str, int]) -> Tuple[float, int, Dict]:
        """处理单个采样案例"""
        sample_count = config['sample']
        selected_indices, swap_count, improvement = self.sampling_processor.process_sampling_case(coords, sample_count, p_value)

        # 计算距离分布
        distances = self.distance_analyzer.calculate_pairwise_distances(coords, selected_indices)

        return p_value, dim, {
            'distances': distances,
            'selected_indices': selected_indices,
            'swap_count': swap_count,
            'improvement': improvement
        }

    def _process_all_sampling_cases(self, all_data: Dict[int, np.ndarray]) -> Dict:
        """处理所有采样案例并生成主结果图"""
        print(f"\n⚡ 执行采样优化...")

        dimensions = self.config.get_dimensions()
        sampling_results = {
            'dimensions': dimensions,
            'p_values': self.config.P_VALUES,
            'results': {}
        }

        # 设置图形布局
        n_rows, n_cols = len(dimensions), len(self.config.P_VALUES)
        fig_width = self.config.FIGURE_SIZE_PER_COL * n_cols
        fig_height = self.config.FIGURE_SIZE_PER_COL * n_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

        # 确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # 并行处理每个p值和维度
        tasks = []
        with ProcessPoolExecutor() as executor:
            for p_value in self.config.P_VALUES:
                for dim in dimensions:
                    coords = all_data[dim]
                    config = self.config.get_dimension_config(dim)
                    tasks.append(executor.submit(self._process_single_case, p_value, dim, coords, config))

            for future in as_completed(tasks):
                try:
                    p_value, dim, result = future.result()
                    sampling_results['results'][(p_value, dim)] = result

                    # 绘制结果
                    p_idx = self.config.P_VALUES.index(p_value)
                    dim_idx = dimensions.index(dim)
                    ax = axes[dim_idx, p_idx]
                    title = f"{dim}D - {self.name_converter.get_name(p_value)}"
                    self.plotter.plot_distance_distribution(result['distances'], title, ax)

                    print(f"     {dim:2d}D (p={p_value}): 交换{result['swap_count']:3d}次, 改善{result['improvement']:6.2%}")
                except Exception as e:
                    print(f"❌ 任务失败: {e}")

        # 保存主结果图
        plt.tight_layout(pad=self.config.LAYOUT_PAD)
        plt.savefig(self.config.OUTPUT_MAIN, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        print(f"\n💾 主结果图已保存: {self.config.OUTPUT_MAIN}")
        plt.show()

        return sampling_results


# ================================ 主程序入口 ================================

def main():
    """主程序入口"""
    try:
        # 创建演示实例并运行
        demo = PowerMeanSamplingDemo()
        demo.run_complete_demo()
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()
