# ABACUS-STRU-Analyser

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

高效的 ABACUS 分子动力学轨迹分析工具 / Efficient ABACUS MD Trajectory Analysis Suite

---

## 📋 项目简介

ABACUS-STRU-Analyser 是专为 ABACUS 分子动力学轨迹设计的高效批量分析工具，支持从轨迹发现、解析、指标计算到智能采样与统计分析的全流程自动化。

**核心功能**：
- 🎯 **智能采样算法**：Power Mean 贪心采样，显著提升采样质量
- 📊 **多样性指标体系**：ANND、MPD、RMSD、Coverage Ratio、JS Divergence 等
- 🔗 **相关性分析**：温度、构象与多样性指标的统计相关性分析
- ⚖️ **采样方法对比**：智能采样、随机采样、均匀采样性能对比
- 🚀 **批量并行处理**：自动发现多个系统，多进程并行分析
- 🔥 **断点续算**：程序中断后自动检测进度并续算，避免重复计算
- 🔄 **自动 DeepMD 导出**：每个体系分析完成后自动导出 DeepMD 训练数据

---

## ✨ 主要特性

- ⚡ **构象向量化**：基于非氢原子对距离，自动忽略旋转/平移/缩放
- 🧮 **PCA 降维**：主成分分析空间，所有分析在降维空间进行
- 🔬 **多物理量融合**：能量与 PCA 分量综合向量，信息全面
- 🎯 **智能采样**：幂平均距离最大化贪心采样算法
- 📊 **多样性指标**：ANND、MPD、RMSD、Coverage Ratio、JS Divergence 等
- 🔗 **相关性分析**：温度、构象与多样性指标的统计相关性
- ⚖️ **采样对比**：智能采样、随机采样、均匀采样多方法性能对比
- 🚀 **批量并行**：自动发现多个系统，多进程并行分析
- 📁 **参数隔离输出**：输出目录自动按参数组合命名
- 🔥 **热更新续算**：程序中断后自动检测进度并续算
- 💾 **增量保存**：边计算边写入，支持实时数据持久化
- 📝 **命令行友好**：所有参数支持长短选项

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- NumPy ≥ 1.25
- Pandas ≥ 2.2
- SciPy ≥ 1.10
- Scikit-learn ≥ 1.3
- DPData ≥ 0.2.18

### 安装依赖
```bash
pip install numpy>=1.25 pandas>=2.2 scipy>=1.10 scikit-learn>=1.3 dpdata>=0.2.18
```

### 基本用法
```bash
# 分析当前目录
python main_abacus_analyser.py

# 指定参数分析
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90 -w 4

# 相关性分析
python main_correlation_analyser.py

# 采样方法对比
python sampling_compare_demo.py
```

---

## ⚙️ 参数说明

### 主分析器参数
| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| --sample_ratio | -r | 0.05 | 采样比例 |
| --power_p | -p | -0.5 | 幂平均距离 p 值 |
| --pca_variance_ratio | -v | 0.90 | PCA 累计方差贡献率 |
| --workers | -w | -1 | 并行进程数 (-1=自动) |
| --output_dir | -o | analysis_results | 输出目录 |
| --search_path | -s | 当前目录父目录 | 搜索路径 |
| --include_project | -i | False | 包含项目自身目录 |
| --force_recompute | -f | False | 强制重算（忽略进度） |
| --correlation_analysis | -c | True | 启用相关性分析 |
| --sampling_comparison | -sc | True | 启用采样方法对比 |

### 相关性分析器参数
| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| --input | -i | auto | 输入 CSV 文件路径 |
| --output | -o | combined_analysis_results | 输出目录 |
| --no-log-file | - | False | 禁用日志文件输出 |

---

## 📁 输出结构

```
analysis_results/
└── run_r0.1_p-0.5_v0.9/           # 参数组合命名目录
    ├── analysis_targets.json      # 分析目标系统列表
    ├── combined_analysis_results/ # 汇总结果
    │   ├── system_metrics_summary.csv
    │   ├── parameter_analysis_results.csv
    │   ├── sampling_methods_comparison.csv
    │   └── progress.json
    ├── single_analysis_results/   # 单体系结果
    │   ├── frame_metrics_*.csv
    │   └── sampling_compare_enhanced.csv
    ├── mean_structures/           # 平均结构数据
    │   ├── index.json
    │   └── mean_structure_*.json
    └── deepmd_npy_per_system/     # DeepMD 数据集
        ├── system_name_1/
        │   ├── type.raw
        │   ├── set.000/
        │   └── split_*/
        └── system_name_2/
            ├── type.raw
            ├── set.000/
            └── split_*/
```

### 主要输出文件

- **system_metrics_summary.csv**：系统级指标汇总
- **parameter_analysis_results.csv**：相关性分析结果
- **frame_metrics_*.csv**：单体系帧级指标
- **sampling_methods_comparison.csv**：采样方法对比汇总
- **progress.json**：分析进度跟踪（断点续算）
- **mean_structure_*.json**：平均结构数据
- **deepmd_npy_per_system/**：按体系 DeepMD 数据集目录

---

## 🧮 核心算法

### 多样性与距离指标
- **ANND**：平均最近邻距离
- **MPD**：平均成对距离
- **RMSD**：经典均方根偏差（Kabsch 对齐）
- **PCA**：主成分分析（相关系数矩阵，累计方差阈值）
- **综合向量**：[Z-score 能量, PCA 分量]，组间缩放

### 智能采样算法
- **Power Mean 采样**：幂平均距离最大化贪心采样
- **采样复用**：基于哈希判定，无状态管理

### 热更新与断点续算
- **进度跟踪**：自动记录已完成系统，避免重复计算
- **增量保存**：边计算边写入，支持实时数据持久化
- **智能检测**：程序重启时自动检测并跳过已处理系统
- **数据完整性**：确保程序中断时数据不丢失

### 采样方法对比
- **智能采样**：Power Mean 贪心算法
- **随机采样**：多次随机试验
- **均匀采样**：等间隔采样
- **评估指标**：ANND、MPD、Coverage Ratio、JS Divergence、RMSD 等

---

## 💡 使用示例

### 基本工作流程
```bash
# 1. 主分析
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90 -w 4

# 2. 相关性分析
python main_correlation_analyser.py

# 3. 查看结果
ls analysis_results/run_r0.1_p-0.5_v0.9/
```

### 高级用法
```bash
# 指定多个搜索路径
python main_abacus_analyser.py -s /data/exp1 /data/exp2

# 高精度分析
python main_abacus_analyser.py -r 0.02 -v 0.95

# 快速预览
python main_abacus_analyser.py -r 0.10 -v 0.85

# 强制重新计算
python main_abacus_analyser.py -f

# 仅进行相关性分析
python main_abacus_analyser.py -sc false

# 仅进行采样对比
python main_abacus_analyser.py -c false

# 自定义相关性分析输入
python main_correlation_analyser.py -i custom.csv -o custom_results
```

### 断点续算
```bash
# 首次运行
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90

# 程序中断后继续（自动检测进度）
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90

# 强制重新计算所有系统
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90 -f
```

### 采样方法对比
```bash
# 对比采样方法
python sampling_compare_demo.py

# 指定结果目录
python sampling_compare_demo.py --result_dir /path/to/analysis_results/run_r0.1_p-0.5_v0.9
```

---

## 🔧 故障排除

### 常见问题
- **采样比较结果显示 NaN**：数据不足或数值异常
- **随机采样标准差大**：属正常，说明智能采样更稳健
- **采样比较慢**：因需多轮随机采样，可减少轮数或分析部分系统
- **F 值/p 值显示 NaN**：样本量不足，ANOVA 无法计算
- **分析速度慢**：可增加采样比例 (-r)、降低 PCA 阈值 (-v)、增加进程数 (-w)
- **内存不足**：降低采样比例或使用单进程 (-w 1)
- **找不到系统目录**：检查目录命名格式

### 断点续算相关
- **程序中断后如何续算**：直接重新运行相同命令，系统会自动检测进度
- **强制重新计算**：使用 `-f` 参数忽略已有进度
- **进度文件损坏**：删除 `progress.json` 文件重新开始

---

## 📈 更新日志

### 2025-09
- ✨ **DeepMD 导出默认化**：每个体系分析完成后自动导出 DeepMD 数据
- 🐛 **采样对比修复**：修复标准差计算错误，优化指标计算
- ⚡ **并行性能优化**：提升多核利用率和负载均衡
- 🔧 **命令行参数完善**：重构参数，支持灵活配置
- 📦 **包结构更新**：完善模块化设计

### 2025-08
- 🔥 **热更新与断点续算**：实现边计算边写入，支持程序中断后自动续算
- 📊 **增量保存机制**：CSV/JSON 文件支持追加写入，实时数据持久化
- 📁 **进度跟踪系统**：自动记录已完成系统，支持智能跳过
- 💾 **数据完整性保证**：确保程序中断时数据不丢失
- 🔄 **DeepMD 导出优化**：支持检测已有数据集，避免重复导出

### 2025-08 早期
- 🎯 **智能采样算法**：Power Mean 贪心采样，显著提升采样质量
- 📊 **多样性指标体系**：ANND、MPD、Coverage Ratio、JS Divergence 等
- 🔗 **相关性分析**：温度、构象与多样性指标的统计相关性分析
- ⚖️ **采样方法对比**：智能采样、随机采样、均匀采样性能对比
- 🚀 **批量并行处理**：自动发现多个系统，多进程并行分析
- 📁 **参数隔离输出**：输出目录自动按参数组合命名
- 💾 **增量保存**：边计算边写入，支持实时数据持久化

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

---

**Made with ❤️ for the ABACUS community**
