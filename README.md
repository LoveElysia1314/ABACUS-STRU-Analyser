
# ABACUS-STRU-Analyser

- ⚡ **构象向量化 / Conformation Vectorization**：基于非氢原子对距### 主分析器参数 / Main Analyser
| 参数 | 短选项 | 默认值 | 说明 | Description |
|------|--------|--------|------|-------------|
| --sample_ratio | -r | 0.05 | 采样比例 | Sampling ratio |
| --power_p | -p | -0.5 | 幂平均距离p值 | Power mean p value |
| --pca_variance_ratio | -v | 0.90 | PCA累计方差贡献率 | PCA explained variance ratio |
| --workers | -w | -1 | 并行进程数 | Number of workers (-1=auto) |
| --output_dir | -o | analysis_results | 输出目录 | Output directory |
| --search_path | -s | 当前目录父目录 | 搜索路径 | Search path(s) |
| --include_project | -i | False | 包含项目自身 | Include project dir |
| --force_recompute | -f | False | 强制重算（忽略进度） | Force recompute (ignore progress) |
| --correlation_analysis | -c | True | 启用相关性分析 | Enable correlation analysis |
| --sampling_comparison | -sc | True | 启用采样方法对比 | Enable sampling comparison |

### 相关性分析器参数 / Correlation Analyser
| 参数 | 短选项 | 默认值 | 说明 | Description |
|------|--------|--------|------|-------------|
| --input | -i | auto | 输入CSV文件路径 | Input CSV file path |
| --output | -o | combined_analysis_results | 输出目录 | Output directory |
| --no-log-file |  | False | 禁用日志文件输出 | Disable log file output |转/平移/缩放。
- 🧮 **PCA降维 / PCA Dimensionality Reduction**：主成分分析空间，所有分析在降维空间进行。
- 🔬 **多物理量融合 / Multi-Physics Integration**：能量与PCA分量综合向量，信息全面。
- 🎯 **智能采样 / Intelligent Sampling**：幂平均距离最大化贪心采样算法。
- 📊 **多样性与分布指标 / Diversity & Distribution Metrics**：ANND、MPD、RMSD、Coverage Ratio、JS Divergence等。
- 🔗 **相关性分析 / Correlation Analysis**：温度、构象与多样性指标的统计相关性。
- ⚖️ **采样方法对比 / Sampling Method Comparison**：智能采样、随机采样、均匀采样多方法性能对比。
- 🚀 **批量并行 / Batch Parallelism**：自动发现多个系统，多进程并行分析。
- � **参数隔离 / Parameter-Isolated Output**：输出目录自动按参数命名。
- 🔥 **热更新与断点续算 / Hot Update & Resume**：程序中断后自动检测进度并续算，避免重复计算。
- 💾 **增量保存 / Incremental Saving**：边计算边写入文件，支持实时数据持久化。
- 📝 **命令行友好 / CLI Friendly**：所有参数支持长短选项。 **高效的 ABACUS 分子动力学轨迹分析工具 / Efficient ABACUS MD Trajectory Analysis Suite**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 项目简介 | Project Overview

ABACUS-STRU-Analyser 是专为 ABACUS 分子动力学轨迹设计的高效批量分析工具，支持从轨迹发现、解析、指标计算到智能采样与统计分析的全流程自动化。

ABACUS-STRU-Analyser is a high-throughput analysis pipeline for ABACUS MD trajectories, providing automated workflows from system discovery, parsing, metrics calculation, intelligent sampling, to statistical analysis.

---

## 📊 重构进度 | Refactoring Progress

**当前状态**: 66.7% 完成 (4/6阶段)
- ✅ **已完成**: 阶段0-3 (日志管理、RMSD合并、metrics合并、采样相关合并)
- 🔄 **进行中**: 无
- ⏳ **待完成**: 阶段4-5 (IO合并、通用工具合并)

**最新更新**: 阶段3已完成 - 将math_utils.py合并到sampler.py，文件数量从19个减少到18个

详细进度请查看: [`REFACTORING_PROGRESS.md`](REFACTORING_PROGRESS.md)

---

## 主要特性 | Key Features

- ⚡ **构象向量化 / Conformation Vectorization**：基于非氢原子对距离，PCA降维，自动忽略旋转/平移/缩放。
- 🧮 **PCA降维 / PCA Dimensionality Reduction**：主成分分析空间，所有分析在降维空间进行。
- 🔬 **多物理量融合 / Multi-Physics Integration**：能量与PCA分量综合向量，信息全面。
- 🎯 **智能采样 / Intelligent Sampling**：幂平均距离最大化贪心采样算法。
- 📊 **多样性与分布指标 / Diversity & Distribution Metrics**：ANND、MPD、RMSD、Coverage Ratio、JS Divergence等。
- 🔗 **相关性分析 / Correlation Analysis**：温度、构象与多样性指标的统计相关性。
- ⚖️ **采样方法对比 / Sampling Method Comparison**：智能采样、随机采样、均匀采样多方法性能对比。
- 🚀 **批量并行 / Batch Parallelism**：自动发现多个系统，多进程并行分析。
- 📁 **参数隔离 / Parameter-Isolated Output**：输出目录自动按参数命名。
- 🔥 **热更新与断点续算 / Hot Update & Resume**：程序中断后自动检测进度并续算，避免重复计算。
- 💾 **增量保存 / Incremental Saving**：边计算边写入文件，支持实时数据持久化。
- 🔄 **自动DeepMD导出 / Auto DeepMD Export**：每个体系分析完成后自动导出DeepMD训练数据。
- 📝 **命令行友好 / CLI Friendly**：所有参数支持长短选项。

---

## 快速开始 | Quick Start

### 环境要求 | Requirements
- Python 3.8+
- NumPy, Pandas, SciPy, Scikit-learn

### 安装依赖 | Install Dependencies
```bash
pip install numpy>=1.25 pandas>=2.2 scipy>=1.10 scikit-learn>=1.3
```

### 基本用法 | Basic Usage
```bash
# 分析当前目录 / Analyse current directory
python main_abacus_analyser.py

# 指定参数 / Specify parameters
python main_abacus_analyser.py -s "/path/to/data" -r 0.05 -p -0.5 -v 0.90 -w 4

# 相关性分析 / Correlation analysis
python main_correlation_analyser.py

# 采样方法对比分析 / Sampling method comparison
python sampling_compare_demo.py
```

**注意**：主分析完成后，每个体系会自动导出DeepMD训练数据到 `deepmd_npy_per_system/` 目录，并自动进行相关性分析，无需额外配置。

---

## 参数说明 | Parameters

### 主分析器参数 / Main Analyser
| 参数 | 短选项 | 默认值 | 说明 | Description |
|------|--------|--------|------|-------------|
| --sample_ratio | -r | 0.05 | 采样比例 | Sampling ratio |
| --power_p | -p | -0.5 | 幂平均距离p值 | Power mean p value |
| --pca_variance_ratio | -v | 0.90 | PCA累计方差贡献率 | PCA explained variance ratio |
| --workers | -w | -1 | 并行进程数 | Number of workers (-1=auto) |
| --output_dir | -o | analysis_results | 输出目录 | Output directory |
| --search_path | -s | 当前目录父目录 | 搜索路径 | Search path(s) |
| --include_project | -i | False | 包含项目自身 | Include project dir |
| --force_recompute | -f | False | 强制重算（忽略进度） | Force recompute (ignore progress) |
| --correlation_analysis | -c | True | 启用相关性分析 | Enable correlation analysis |
| --sampling_comparison | -sc | True | 启用采样方法对比 | Enable sampling comparison |

---

## 输出结构 | Output Structure

```
analysis_results/
└── run_r0.1_p-0.5_v0.9/
    ├── analysis_targets.json
    ├── combined_analysis_results/
    │   ├── system_metrics_summary.csv
    │   ├── parameter_analysis_results.csv
    │   ├── sampling_methods_comparison.csv
    │   └── progress.json                    # 进度跟踪文件 / Progress tracking
    ├── single_analysis_results/
    │   ├── frame_metrics_*.csv
    │   └── sampling_compare_enhanced.csv
    ├── mean_structures/
    │   ├── index.json
    │   └── mean_structure_*.json
    └── deepmd_npy_per_system/               # DeepMD数据集（按体系） / DeepMD dataset (per-system)
        ├── system_name_1/
        │   ├── type.raw
        │   ├── set.000/
        │   └── split_*/                      # 数据集拆分 / Dataset splits
        └── system_name_2/
            ├── type.raw
            ├── set.000/
            └── split_*/                      # 数据集拆分 / Dataset splits
```

### 主要输出文件 | Main Output Files

- **system_metrics_summary.csv**：系统级指标汇总 / System-level summary
- **parameter_analysis_results.csv**：相关性分析结果 / Correlation analysis
- **frame_metrics_*.csv**：单体系帧级指标 / Per-system frame metrics
- **sampling_compare_enhanced.csv**：采样方法对比明细 / Sampling comparison (detailed)
- **sampling_methods_comparison.csv**：采样方法对比汇总 / Sampling comparison (summary)
- **progress.json**：分析进度跟踪（断点续算） / Analysis progress tracking (resume capability)
- **mean_structure_*.json**：平均结构数据 / Mean structure data
- **deepmd_npy_per_system/**：按体系DeepMD数据集目录 / Per-system DeepMD dataset directory
  - 每个子目录包含单个体系的完整DeepMD数据集 / Each subdirectory contains complete DeepMD dataset for one system

---

## 核心算法 | Core Algorithms

### 多样性与距离指标 / Diversity & Distance Metrics
- **ANND**: 平均最近邻距离 / Average nearest neighbor distance
- **MPD**: 平均成对距离 / Mean pairwise distance
- **RMSD**: 经典均方根偏差，Kabsch对齐 / Root Mean Square Deviation (Kabsch alignment)
- **PCA**: 主成分分析，相关系数矩阵，累计方差阈值 / PCA on correlation matrix, explained variance threshold
- **综合向量 / Comprehensive Vector**: [Z-score能量, PCA分量]，组间缩放 / [Z-score energy, PCA components], group scaling

### 智能采样 / Intelligent Sampling
- **Power Mean采样**: 幂平均距离最大化贪心采样 / Power mean maximization greedy sampling
- **采样复用**: 基于哈希判定，无状态管理 / Hash-based sampling reuse, stateless

### 热更新与断点续算 / Hot Update & Resume
- **进度跟踪**: 自动记录已完成系统，避免重复计算 / Automatic progress tracking to avoid recomputation
- **增量保存**: 边计算边写入，支持实时数据持久化 / Incremental saving with real-time data persistence
- **智能检测**: 程序重启时自动检测并跳过已处理系统 / Smart detection to skip processed systems on restart
- **数据完整性**: 确保程序中断时数据不丢失 / Data integrity guarantee during interruptions

### 采样方法对比 / Sampling Method Comparison
- **智能采样**: Power Mean贪心 / Power Mean greedy
- **随机采样**: 多次随机 / Multiple random trials
- **均匀采样**: 等间隔 / Uniform interval
- **评估指标**: ANND, MPD, Coverage Ratio, JS Divergence, RMSD等

---

## 使用示例 | Usage Examples

### 基本流程 / Basic Workflow
```bash
# 1. 主分析 / Main analysis
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90 -w 4
# 2. 相关性分析 / Correlation analysis
python main_correlation_analyser.py
# 3. 查看结果 / Check results
ls analysis_results/run_r0.1_p-0.5_v0.9/
```

### 高级用法 / Advanced Usage
```bash
# 指定多个路径 / Multiple search paths
python main_abacus_analyser.py -s /data/exp1 /data/exp2
# 高精度分析 / High-precision
python main_abacus_analyser.py -r 0.02 -v 0.95
# 快速预览 / Fast preview
python main_abacus_analyser.py -r 0.10 -v 0.85
# 强制重算 / Force recompute
python main_abacus_analyser.py -f
# 仅进行相关性分析（跳过采样对比） / Correlation analysis only (skip sampling comparison)
python main_abacus_analyser.py -sc false
# 仅进行采样对比（跳过相关性分析） / Sampling comparison only (skip correlation analysis)
python main_abacus_analyser.py -c false
# 相关性分析指定输入 / Custom input for correlation
python main_correlation_analyser.py -i custom.csv -o custom_results
```

### 断点续算 / Resume from Checkpoint
```bash
# 首次运行 / First run
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90
# 程序中断后继续 / Resume after interruption
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90
# 系统会自动检测进度并跳过已完成的系统
# 强制重新计算所有系统 / Force recompute all systems
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90 -f
```

### 采样方法对比 / Sampling Comparison
```bash
# 采样方法对比分析 / Compare sampling methods
python sampling_compare_demo.py
# 指定结果目录 / Specify result dir
python sampling_compare_demo.py --result_dir /path/to/analysis_results/run_r0.1_p-0.5_v0.9
```

---

## 故障排除 | Troubleshooting

- 采样比较结果中某些指标显示NaN：数据不足或数值异常。
- 随机采样标准差大：属正常，说明智能采样更稳健。
- 采样比较慢：因需多轮随机采样，可减少轮数或分析部分系统。
- F值/p值为nan：样本量不足，ANOVA无法计算。
- 分析慢：可增采样比例(-r)、降PCA阈值(-v)、增进程数(-w)。
- 内存不足：降采样比例或用单进程(-w 1)。
- 找不到系统目录：检查目录命名格式。
- **程序中断后如何续算**：直接重新运行相同命令，系统会自动检测进度并续算。
- **强制重新计算**：使用 `-f` 参数忽略已有进度，重新计算所有系统。
- **进度文件损坏**：删除 `progress.json` 文件，系统会重新开始计算。

---

## 开发与测试 | Development & Testing

### 单元测试 / Unit Test
```bash
pip install pytest pytest-cov
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest --cov=src tests/
```

### 代码质量 / Code Quality
```bash
pip install ruff black mypy
black src/
ruff check src/
mypy src/
```

---


## 更新日志 | Changelog

> 主要变更按时间归纳，详见 [Git 提交历史](https://github.com/LoveElysia1314/ABACUS-STRU-Analyser/commits/main)

### 2025-09 (最新)
- ✨ **DeepMD导出默认化**：将`--export_deepmd_per_system`设为默认行为，每个体系分析完成后自动导出DeepMD数据
- 🗑️ **参数简化**：移除`--export_deepmd_per_system`、`--deepmd_output_subdir`、`--force_deepmd_overwrite`、`--enable_legacy_global_deepmd_export`参数
- 🐛 **采样对比修复**：修复采样效果对比脚本标准差计算错误，移除MinD、Diversity_Score、EMD_Distance、Mean_Centroid_Distance指标
- ⚡ **并行性能优化**：并行分析时将chunksize固定为1，提升多核利用率和负载均衡
- 🔧 **命令行参数完善**：重构命令行参数，支持体系粒度DeepMD导出与legacy全局导出可选
- 📦 **包结构更新**：更新`__init__.py`文件，完善包结构

### 2025-09
- 🔥 **热更新与断点续算功能**：实现边计算边写入，支持程序中断后自动续算，避免重复计算。
- 📊 **增量保存机制**：CSV文件支持追加写入，JSON文件支持智能更新，实时数据持久化。
- 📁 **进度跟踪系统**：新增progress.json文件，自动记录已完成系统，支持智能跳过。
- 💾 **数据完整性保证**：所有文件写入后强制同步，确保程序中断时数据不丢失。
- 🔄 **DeepMD导出优化**：支持检测已有数据集，避免重复导出。
- 📝 **文档完善**：更新README，详细说明热更新和断点续算功能。

- 采样复用逻辑彻底重构，移除status字段，采样复用仅依赖哈希判定，所有输出强制重写，简化增量/全量切换逻辑。
- 采样方法对比、相关性分析、系统指标输出三者统一，采样效果验证集成主流程，指标注册表驱动。
- 多样性与分布相似性指标（Diversity Score, Coverage Ratio, JS Divergence, EMD Distance）正式纳入主流程。
- 结构指标与RMSD计算委托至structural_metrics模块，采样算法与指标计算进一步模块化。
- 日志系统重构，统一多进程安全日志输出。
- 文档与代码质量优化，完善开发与测试说明。

### 2025-08
- 项目架构模块化，主流程、采样、相关性分析、采样对比等功能独立。
- 支持参数化输出目录、增量与全量分析切换。
- 采样方法对比脚本（sampling_compare_demo.py）上线，支持智能采样、随机采样、均匀采样多方法性能对比。
- 采样多样性与分布指标完善，采样对比结果自动输出详细与汇总表。
- 支持能量/力解析，能量标准化，采样帧导出为DeepMD数据集。
- 相关性分析器完善，支持温度/构象效应统计检验，输出显著性与效应量。
- 采样与分析流程支持多进程并行，提升大规模批量分析效率。
- 代码风格、类型注解、测试与CI/CD集成。

---

---

## 许可证 | License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
MIT License, see [LICENSE](LICENSE).

## 贡献指南 | Contributing

1. Fork 项目 / Fork the repo
2. 创建分支 / Create a branch: `git checkout -b feature/your-feature`
3. 提交更改 / Commit: `git commit -am 'Add new feature'`
4. 推送分支 / Push: `git push origin feature/your-feature`
5. 创建 Pull Request

---

**Made with ❤️ for the ABACUS community**
