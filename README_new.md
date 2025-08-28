# ABACUS-STRU-Analyser

🔬 **高效的 ABACUS 分子动力学轨迹分析工具 v2.0**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/LoveElysia1314/ABACUS-STRU-Analyser/workflows/CI/badge.svg)](https://github.com/LoveElysia1314/ABACUS-STRU-Analyser/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 v2.0 核心优势

### 📦 现代化架构
- **模块化设计**: 清晰的 `src/` 目录结构，代码可维护性大幅提升
- **类型安全**: 全面的类型注解，提升代码可靠性
- **标准化配置**: 使用 `pyproject.toml` 现代 Python 项目配置

### 🔧 开发体验
- **完整测试**: 单元测试 + 集成测试，确保代码质量
- **自动化CI/CD**: GitHub Actions 自动测试、代码质量检查
- **代码规范**: Black + Ruff + MyPy 确保代码风格一致性

### 🛡️ 可靠性与维护性
- **统一日志系统**: 集中化日志管理，便于调试和监控
- **异常处理**: 完善的错误处理机制，优雅降级
- **向后兼容**: 保持 v1.x API 兼容，无缝升级

### 🔄 可扩展性
- **插件化结构**: 易于添加新的分析方法和指标
- **路径管理**: 智能路径处理，支持增量分析
- **配置灵活**: 丰富的命令行参数和配置选项

## 项目概述

ABACUS-STRU-Analyser 是专为 ABACUS 分子动力学轨迹设计的高效批量分析工具。提供从轨迹发现、解析、指标计算到智能采样和统计分析的完整流水线。

### 🌟 主要特性

- **🚀 批量分析**: 自动发现和处理多个分子动力学系统
- **🎯 智能采样**: 支持 Power Mean、贪婪最大距离等采样策略
- **📊 多样性指标**: 计算 MinD、ANND、MPD、PCA方差贡献率 等构象多样性指标
- **🔗 相关性分析**: 温度和构象相关性统计分析
- **⚡ 并行处理**: 多进程并行提升分析效率
- **📁 结果管理**: 参数隔离的输出目录
- **🔄 增量计算**: 智能跳过已完成的分析任务
- **🧮 PCA降维**: 基于主成分分析的空间降维，所有分析在降维空间进行

## 快速开始

### 环境要求
- Python 3.8+
- NumPy, Pandas, SciPy, Scikit-learn

### 安装

```bash
# 克隆项目
git clone https://github.com/LoveElysia1314/ABACUS-STRU-Analyser.git
cd ABACUS-STRU-Analyser

# 安装依赖
pip install -r requirements.txt

# 开发环境（可选）
pip install -r requirements-dev.txt
```

### 基本使用

```bash
# 分析当前目录
python main_abacus_analyzer.py

# 指定搜索路径和参数
python main_abacus_analyzer.py \
    --search_path "/path/to/data" \
    --sample_ratio 0.05 \
    --workers 4

# 相关性分析
python main_correlation_analyzer.py
```

## 项目结构

```
ABACUS-STRU-Analyser/
├── src/                    # 源代码
│   ├── utils/             # 工具模块
│   │   ├── __init__.py
│   │   ├── data_utils.py  # 数据验证和处理
│   │   └── file_utils.py  # 文件和目录操作
│   ├── logging/           # 日志管理
│   │   ├── __init__.py
│   │   └── manager.py     # 集中化日志管理
│   ├── io/                # 输入输出
│   │   ├── __init__.py
│   │   ├── path_manager.py # 路径管理
│   │   ├── result_saver.py # 结果保存
│   │   └── stru_parser.py  # 结构文件解析
│   ├── core/              # 核心算法
│   │   ├── __init__.py
│   │   ├── metrics.py     # 统计指标计算
│   │   ├── pca_analyzer.py # PCA分析
│   │   ├── sampler.py     # 采样算法
│   │   └── system_analyzer.py # 系统分析器
│   └── analysis/          # 分析模块
│       ├── __init__.py
│       ├── correlation_analyzer.py # 相关性分析
│       └── trajectory_analyzer.py # 轨迹分析
├── tests/                 # 测试文件
├── tools/                 # 迁移工具
├── .github/               # CI/CD配置
├── pyproject.toml         # 项目配置
├── requirements.txt       # 运行依赖
└── requirements-dev.txt   # 开发依赖
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample_ratio` | 0.05 | 采样比例 (0.01-1.0) |
| `--power_p` | -0.5 | 幂平均距离的p值 |
| `--pca_components` | 5 | PCA降维后的主成分数量 |
| `--workers` | auto | 并行工作进程数 |
| `--search_path` | 当前目录 | 递归搜索路径 |
| `--output_dir` | analysis_results | 输出目录 |
| `--force_recompute` | False | 强制重新计算 |

## 输出说明

### 主要输出文件

- `system_metrics_summary.csv`: 所有系统的汇总指标
- `pca_components.csv`: 每帧的主成分分量数据
- `analysis_statistics.csv`: 分析统计信息
- `parameter_analysis_results.csv`: 相关性分析结果
- `single_analysis_results/`: 单系统详细结果

### 指标说明

- **MinD**: 最小间距（Minimum Distance）
- **ANND**: 平均最近邻距离（Average Nearest Neighbor Distance）
- **MPD**: 平均成对距离（Mean Pairwise Distance）
- **PCA方差贡献率**: 各主成分的方差贡献率
- **PCA分量**: 每帧在降维空间中的主成分坐标

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
- 提供数据可视化

### 智能采样策略

#### Power Mean 采样
基于幂平均函数的采样策略：
```
PowerMean(x, p) = (Σxᵢᵖ / n)^(1/p)
```

#### 贪婪最大距离采样
迭代选择与已选点集距离最大的点。

## 开发指南

### 运行测试

```bash
# 单元测试
python -m pytest tests/unit/ -v

# 烟雾测试
python tests/smoke_test.py

# 代码质量检查
ruff check src/
black src/ --check
mypy src/
```

### 从 v1.x 升级

v2.0 保持向后兼容，现有代码无需修改。如需使用新特性：

```python
# v1.x 方式（仍然支持）
from src.utils import create_logger

# v2.0 推荐方式
from src.logging import LoggerManager
logger = LoggerManager.create_logger("my_logger")
```

详细升级指南请参考 `UPGRADE_TO_V2.md`。

## 贡献指南

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 更新日志

### v2.0.1 (2025-08-28)

#### ✨ 新特性
- **MPD指标**: 新增平均成对距离（Mean Pairwise Distance）指标
- **增强统计**: 完善多样性评估指标体系

#### 🔧 改进
- **指标计算**: 优化距离计算算法
- **结果输出**: 扩展CSV输出字段
- **代码结构**: 改进模块化设计

### v2.0.0 (2025-08-21)

#### 🎉 重大改进
- **现代化架构**: 模块化 `src/` 目录结构
- **开发工具链**: 完整的测试、CI/CD、代码质量工具
- **类型安全**: 全面类型注解支持
- **配置标准化**: 迁移到 `pyproject.toml`

#### ✨ 新特性
- 统一日志管理系统
- 智能路径管理和增量分析
- 自动化迁移工具
- 完整的测试覆盖
- **PCA降维分析**: 所有分析流程基于主成分分析空间进行

#### 🔧 改进
- 优化异常处理机制
- 提升代码可维护性
- 增强向后兼容性
- 简化开发流程

#### 📚 文档
- 重构项目文档
- 添加升级指南
- 完善API文档

---

**Made with ❤️ for the ABACUS community**
