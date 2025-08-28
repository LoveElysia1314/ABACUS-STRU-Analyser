#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据分布策略演示脚本
展示不同分布策略生成的数据样本
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from power_mean_sampling_demo import DataGenerator

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

def demo_distributions():
    """演示不同分布策略"""
    strategies = {
        "uniform": "均匀分布",
        "clustered": "聚类分布", 
        "ring": "环形分布",
        "mixed": "混合分布",
        "noisy": "噪声分布",
        "sparse": "稀疏分布"
    }
    
    n_2d, n_3d = 128, 256
    
    # 策略参数
    params = {
        'cluster_centers': 3,
        'cluster_std': 0.15,
        'inner_ratio': 0.6,
        'noise_level': 0.08,
        'sparse_prob': 0.4
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, len(strategies), figsize=(20, 8))
    
    for i, (strategy, name) in enumerate(strategies.items()):
        print(f"生成 {name} 数据...")
        
        # 生成2D数据
        coords_2d = DataGenerator.generate_2d_circle(n_2d, strategy, **params)
        
        # 生成3D数据
        coords_3d = DataGenerator.generate_3d_sphere(n_3d, strategy, **params)
        
        # 绘制2D分布
        ax_2d = axes[0, i]
        ax_2d.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                     c='blue', s=20, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax_2d.set_aspect('equal')
        ax_2d.set_title(f'2D-{name}', fontsize=12)
        ax_2d.grid(True, alpha=0.3)
        ax_2d.set_xlim(-1.2, 1.2)
        ax_2d.set_ylim(-1.2, 1.2)
        
        # 绘制3D分布的投影
        ax_3d = axes[1, i]
        # 使用x-y投影来展示3D分布
        ax_3d.scatter(coords_3d[:, 0], coords_3d[:, 1], 
                     c=coords_3d[:, 2], s=15, alpha=0.6, 
                     cmap='viridis', edgecolor='black', linewidth=0.3)
        ax_3d.set_aspect('equal')
        ax_3d.set_title(f'3D-{name} (XY投影)', fontsize=12)
        ax_3d.grid(True, alpha=0.3)
        ax_3d.set_xlim(-1.2, 1.2)
        ax_3d.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig('distribution_strategies_demo.png', dpi=300, bbox_inches='tight')
    print(f"\n分布策略演示图已保存到: distribution_strategies_demo.png")
    plt.show()

def compare_sampling_on_distributions():
    """在不同分布上比较采样效果"""
    print("\n=== 不同分布下的采样效果对比 ===")
    
    strategies = ["uniform", "clustered", "mixed", "sparse"]
    strategy_names = ["均匀分布", "聚类分布", "混合分布", "稀疏分布"]
    
    for strategy, name in zip(strategies, strategy_names):
        print(f"\n测试 {name}:")
        print(f"  修改配置文件中的 DATA_STRATEGY = '{strategy}'")
        print(f"  然后运行 python power_mean_sampling_demo.py")

if __name__ == "__main__":
    demo_distributions()
    compare_sampling_on_distributions()
