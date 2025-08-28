# ABACUS-STRU-Analyser

🔬 **高效的 ABACUS 分子动力学轨迹分析工具 v2.3**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 项目概述

ABACUS-STRU-Analyser 是专为 ABACUS 分子动力学轨迹设计的高效批量分析工具。提供从轨迹发现、解析、指标计算到智能采样和统计分析的完整流水线。

### 🌟 主要特性

- **⚡ 构象向量化**: 基于非氢原子对间距离归一化构建高维方向向量，从原理上忽略旋转、平移、缩放，便于表征分子构象变化
- **🧮 PCA降维**: 基于主成分分析的空间降维，所有分析在降维空间进行，增加可视化程度
- **🎯 智能采样**: 基于 点对间幂平均距离最大化 贪心算法 采样策略
- **📊 多样性指标**: 计算 MinD、ANND、MPD、PCA方差贡献率 等构象多样性指标，且包含采样前后指标对比，直观展示采样优势
- **🔗 相关性分析**: 温度和构象与分子构象丰富程度的相关性统计分析
- **🚀 批量并行**: 自动发现和处理多个分子动力学系统，多进程并行提升分析效率
- **📁 参数隔离**: 输出目录按分析参数自动命名
- **🔄 增量计算**: 智能跳过已完成的分析任务
- **📝 命令行支持**: 支持传递命令行参数调整（长、短选项均可）

## 快速开始

### 环境要求
- Python 3.8+
- NumPy, Pandas, SciPy, Scikit-learn

### 安装依赖

```bash
pip install numpy>=1.25 pandas>=2.2 scipy>=1.10 scikit-learn>=1.3
```

### 基本使用

```bash
# 分析当前目录
python main_abacus_analyser.py

# 指定搜索路径和参数（使用短选项）
python main_abacus_analyser.py \
    -s "/path/to/data" \
    -r 0.05 \
    -p -0.5 \
    -v 0.90 \
    -w 4

# 相关性分析
python main_correlation_analyser.py
```

## 项目结构

```
ABACUS-STRU-Analyser/
├── src/                    # 源代码
│   ├── utils/             # 工具模块
│   │   ├── data_utils.py  # 数据验证和处理
│   │   └── file_utils.py  # 文件和目录操作
│   ├── logging/           # 日志管理
│   │   └── manager.py     # 集中化日志管理
│   ├── io/                # 输入输出
│   │   ├── path_manager.py # 路径管理
│   │   ├── result_saver.py # 结果保存
│   │   └── stru_parser.py  # 结构文件解析
│   ├── core/              # 核心算法
│   │   ├── metrics.py     # 统计指标计算
│   │   ├── sampler.py     # 采样算法
│   │   └── system_analyser.py # 系统分析器
│   └── analysis/          # 分析模块
│       └── correlation_analyser.py # 相关性分析
├── analysis_results/      # 分析结果输出目录
├── main_abacus_analyser.py    # 主分析器入口
├── main_correlation_analyser.py # 相关性分析器入口
├── requirements.txt       # 运行依赖
└── README.md             # 项目文档
```

## 配置参数

### 主分析器参数

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--sample_ratio` | `-r` | 0.05 | 采样比例 (0.01-1.0) |
| `--power_p` | `-p` | -0.5 | 幂平均距离的p值 |
| `--pca_variance_ratio` | `-v` | 0.90 | PCA降维累计方差贡献率 (0-1) |
| `--workers` | `-w` | auto | 并行工作进程数 |
| `--output_dir` | `-o` | analysis_results | 输出根目录 |
| `--search_path` | `-s` | 当前目录父目录 | 递归搜索路径，支持多个路径 |
| `--include_project` | `-i` | False | 允许搜索项目自身目录 |
| `--force_recompute` | `-f` | False | 强制重新计算所有系统 |

### 相关性分析器参数

| 参数 | 短选项 | 默认值 | 说明 |
|------|--------|--------|------|
| `--input` | `-i` | auto | 输入的system_metrics_summary.csv文件路径 |
| `--output` | `-o` | combined_analysis_results | 输出目录路径 |
| `--no-log-file` | - | False | 禁用日志文件输出，仅输出到控制台 |

## 输出说明

### 输出目录结构

```
analysis_results/
└── run_r0.05_p-0.5_v0.9/           # 按参数命名的输出目录
    ├── analysis_targets.json       # 分析目标状态
    ├── combined_analysis_results/  # 汇总分析结果
    │   ├── system_metrics_summary.csv     # 系统汇总指标
    │   └── parameter_analysis_results.csv # 相关性分析结果
    └── single_analysis_results/    # 单体系详细结果
        └── frame_metrics_*.csv     # 各体系的帧级指标和PCA分量
```

### 主要输出文件

#### 1. system_metrics_summary.csv
包含所有系统的汇总指标：

| 字段 | 说明 |
|------|------|
| System | 系统名称（如：struct_mol_100_conf_0_T300K） |
| Molecule_ID | 分子ID |
| Configuration | 构象编号 |
| Temperature(K) | 温度 |
| Num_Frames | 总帧数 |
| Dimension | 降维后的维度 |
| MinD | 最小间距 |
| ANND | 平均最近邻距离 |
| MPD | 平均成对距离 |
| MinD_sampled | 采样后最小间距 |
| ANND_sampled | 采样后平均最近邻距离 |
| MPD_sampled | 采样后平均成对距离 |
| MinD_ratio | 最小间距比值 |
| ANND_ratio | 平均最近邻距离比值 |
| MPD_ratio | 平均成对距离比值 |
| PCA_Variance_Ratio | PCA方差贡献率 |
| PCA_Cumulative_Variance_Ratio | PCA累计方差贡献率 |
| PCA_Num_Components_Retained | 保留主成分数量 |
| PCA_Variance_Ratios | 各主成分方差贡献率（JSON格式） |

#### 2. parameter_analysis_results.csv
相关性分析结果：

| 字段 | 说明 |
|------|------|
| Analysis_Type | 分析类型（Temp_Corr/Config_Effect） |
| Indicator | 指标名称 |
| Statistic_Value | 统计值（r值或F值） |
| P_Value | p值 |
| Effect_Size | 效应量 |
| Significance | 是否显著 |
| Interpretation | 结果解释 |
| Valid_Samples | 有效样本数 |
| Statistic_Info | 统计信息摘要 |
| Group_Info | 分组信息 |
| Additional_Details | 附加详情 |

#### 3. frame_metrics_*.csv
单体系详细结果：

| 字段 | 说明 |
|------|------|
| Frame_ID | 帧ID |
| Selected | 是否被选中（1=选中，0=未选中） |
| PC1, PC2, ... | 各主成分坐标 |

## 核心算法

### 多样性指标计算

#### 1. 最小间距 (MinD)
计算所有点对之间的最小欧氏距离：
```
MinD = min(distance(p_i, p_j) for all i < j)
```

#### 2. 平均最近邻距离 (ANND)
计算每个点到其最近邻点的平均距离：
```
ANND = mean(min(distance(p_i, p_j) for j ≠ i) for all i)
```

#### 3. 平均成对距离 (MPD)
计算所有点对之间距离的平均值：
```
MPD = mean(distance(p_i, p_j) for all i < j)
```

#### 4. PCA降维分析
使用主成分分析将高维数据投影到低维空间：
- 保留主要方差方向
- 减少计算复杂度
- 便于数据可视化
- 默认保留90%的累计方差贡献率

### 智能采样策略

#### Power Mean 采样
基于点对间幂平均距离的采样策略：
```
PowerMean(x, p) = (Σxᵢᵖ / n)^(1/p)
```

#### 贪婪最大距离采样
迭代选择与已选点集距离最大的点，并在达到设定选点数后尝试交换采样与未采样的点对以优化结果

## 相关性分析

### 温度相关性分析 (Temp_Corr)
- **分析类型**: 单变量控制（固定分子和构象，分析温度效应）
- **统计方法**: Pearson相关系数 + Spearman秩相关
- **输出指标**: 相关系数r、p值、效应量、显著性

### 构象效应分析 (Config_Effect)
- **分析类型**: 单变量控制（固定分子和温度，分析构象效应）
- **统计方法**: 单因素方差分析(ANOVA)
- **输出指标**: F统计量、p值、效应量η²、显著性

### 分析条件
- **温度分析**: 每个(分子,构象)组至少需要2个温度点
- **构象分析**: 每个(分子,温度)组至少需要2个构象
- **样本量不足**: 当组内样本数<2时，F值和p值显示为nan

## 开发指南

### 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行单元测试
python -m pytest tests/unit/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 生成覆盖率报告
python -m pytest --cov=src tests/
```

### 代码质量检查

```bash
# 安装开发依赖
pip install ruff black mypy

# 代码格式化
black src/

# 代码检查
ruff check src/

# 类型检查
mypy src/
```

## 使用示例

### 基本分析流程

```bash
# 1. 执行主分析（自动发现并分析所有系统）
python main_abacus_analyser.py -r 0.05 -p -0.5 -v 0.90 -w 4

# 2. 执行相关性分析
python main_correlation_analyser.py

# 3. 查看结果
ls analysis_results/run_r0.05_p-0.5_v0.9/
```

### 高级用法

```bash
# 指定搜索路径
python main_abacus_analyser.py -s /data/experiment1 /data/experiment2

# 高精度分析（降低采样比例）
python main_abacus_analyser.py -r 0.02 -v 0.95

# 快速预览（增加采样比例）
python main_abacus_analyser.py -r 0.10 -v 0.85

# 强制重新计算
python main_abacus_analyser.py -f

# 相关性分析指定输入文件
python main_correlation_analyser.py -i custom_data.csv -o custom_results
```

### 并行处理优化

```bash
# 自动检测CPU核心数
python main_abacus_analyser.py -w -1

# 指定进程数
python main_abacus_analyser.py -w 8

# 单进程调试模式
python main_abacus_analyser.py -w 1
```

## 故障排除

### 常见问题

#### Q: F值和p值显示为nan
A: 构象分析中每个(分子,温度)组至少需要2个构象。如果样本量不足，ANOVA无法计算，会显示nan值。

#### Q: 分析速度很慢
A: 尝试增加采样比例(-r)或减少PCA方差贡献率阈值(-v)，或者增加并行进程数(-w)。

#### Q: 内存不足
A: 降低采样比例，或使用单进程模式(-w 1)处理大系统。

#### Q: 找不到系统目录
A: 检查目录命名格式是否符合 `struct_mol_{id}_conf_{conf}_T{temp}K` 模式。

### 日志调试

```bash
# 查看主分析日志
cat analysis_results/main_analysis_*.log

# 查看相关性分析日志
cat analysis_results/correlation_analysis.log
```

## 更新日志

### v2.0.1 (2025-08-28)

#### ✨ 新特性
- **短选项支持**: 所有命令行参数支持短选项，提升使用体验
- **输出目录优化**: 输出目录按参数值命名，便于管理
- **CSV编码修复**: 所有CSV文件使用UTF-8编码，确保兼容性
- **MPD指标**: 新增平均成对距离（Mean Pairwise Distance）指标

#### 🔧 改进
- **指标计算**: 优化距离计算算法，提升性能
- **结果输出**: 合并CSV输出为单个summary文件
- **代码结构**: 改进模块化设计，增强可维护性
- **相关性分析**: 简化Analysis_Type值为Temp_Corr和Config_Effect

#### 🐛 修复
- **编码问题**: 修复CSV文件编码问题，确保跨平台兼容
- **参数验证**: 增强参数验证和错误处理
- **内存管理**: 优化大文件处理时的内存使用

### v2.0.0 (2025-08-21)

#### 🎉 重大改进
- **现代化架构**: 模块化 `src/` 目录结构
- **开发工具链**: 完整的测试、CI/CD、代码质量工具
- **类型安全**: 全面类型注解支持
- **PCA降维分析**: 所有分析流程基于主成分分析空间进行

#### ✨ 新特性
- 统一日志管理系统
- 智能路径管理和增量分析
- 并行处理支持
- 自动化迁移工具
- 完整的测试覆盖

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 贡献指南

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建 Pull Request

---

**Made with ❤️ for the ABACUS community**
