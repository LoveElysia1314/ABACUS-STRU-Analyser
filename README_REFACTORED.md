# ABACUS分析器重构版本使用指南

## 概述

重构版本引入了两种分析模式：
1. **仅采样模式** (`--sampling_only`)：只执行采样算法，不计算统计指标，快速获得采样结果并导出为DeepMD格式
2. **完整分析模式**（默认）：执行完整的分析流程，包括统计指标计算、相关性分析和采样效果评估

## 主要改进

### 1. 模块化设计
- **AnalysisConfig**：统一的配置管理类
- **AnalysisOrchestrator**：核心逻辑编排器，减少代码重复
- **AnalysisMode**：明确的模式枚举

### 2. 流程优化
- 提取公共函数，减少代码冗余
- 更清晰的流程控制
- 支持即时DeepMD数据导出

### 3. 新增仅采样模式
- 跳过耗时的统计指标计算
- 保留采样算法的完整功能
- 直接输出DeepMD训练数据

## 使用方法

### 仅采样模式（推荐用于快速数据准备）

```bash
# 基本用法：仅执行采样，获得DeepMD数据
python main_abacus_analyser_refactored.py --sampling_only

# 指定采样参数
python main_abacus_analyser_refactored.py --sampling_only \
    --sample_ratio 0.1 \
    --power_p -0.5 \
    --pca_variance_ratio 0.90

# 指定搜索路径和并行进程
python main_abacus_analyser_refactored.py --sampling_only \
    --search_path /path/to/abacus/data \
    --workers 4

# 强制重新计算（忽略已有采样结果）
python main_abacus_analyser_refactored.py --sampling_only \
    --force_recompute
```

### 完整分析模式（用于详细研究）

```bash
# 基本用法：执行完整分析
python main_abacus_analyser_refactored.py

# 禁用采样效果评估
python main_abacus_analyser_refactored.py \
    --disable_sampling_eval

# 多路径搜索
python main_abacus_analyser_refactored.py \
    --search_path /path1 /path2 /path3

# 包含项目目录搜索
python main_abacus_analyser_refactored.py \
    --include_project
```

### 采样复用计划评估

```bash
# 评估采样复用情况（不实际执行分析）
python main_abacus_analyser_refactored.py --dry_run_reuse
```

## 输出差异

### 仅采样模式输出
```
analysis_results/
├── run_r0.1_p-0.5_v0.9/
│   ├── sampling_summary.json          # 采样汇总信息
│   ├── deepmd_npy_per_system/         # DeepMD格式数据（按系统分组）
│   │   ├── system1/
│   │   │   ├── coord.npy
│   │   │   ├── energy.npy
│   │   │   └── force.npy
│   │   └── system2/
│   └── analysis_targets.json          # 分析目标信息
```

### 完整分析模式输出
```
analysis_results/
├── run_r0.1_p-0.5_v0.9/
│   ├── combined_analysis_results/      # 汇总分析结果
│   ├── single_analysis_results/       # 单系统详细结果
│   ├── mean_structures/               # 平均结构
│   ├── deepmd_npy_per_system/         # DeepMD格式数据
│   ├── sampling_compare_enhanced.csv   # 采样效果比较
│   └── correlation_analysis_results/   # 相关性分析结果
```

## 性能对比

| 模式 | 计算时间 | 内存使用 | 输出文件 | 适用场景 |
|------|----------|----------|----------|----------|
| 仅采样模式 | ~30% | ~40% | 轻量 | 快速数据准备、DeepMD训练 |
| 完整分析模式 | 100% | 100% | 完整 | 详细分析、研究探索 |

## 配置选项

### 核心参数
- `--sample_ratio` (float): 采样比例 [默认: 0.1]
- `--power_p` (float): 幂平均距离的p值 [默认: -0.5]
- `--pca_variance_ratio` (float): PCA累计方差贡献率 [默认: 0.90]

### 运行配置
- `--workers` (int): 并行工作进程数 [默认: 自动检测]
- `--output_dir` (str): 输出根目录 [默认: analysis_results]
- `--search_path` (list): 搜索路径列表，支持通配符
- `--include_project`: 允许搜索项目自身目录
- `--force_recompute`: 强制重新计算，忽略已有结果

### 模式控制
- `--sampling_only`: 仅采样模式
- `--dry_run_reuse`: 仅评估采样复用计划
- `--disable_sampling_eval`: 禁用采样效果评估

## 兼容性

重构版本与原版本完全兼容：
- 支持相同的命令行参数
- 产生相同格式的输出文件
- 可以复用原版本的分析结果

## 迁移建议

1. **快速数据准备**：使用仅采样模式
2. **详细研究分析**：使用完整分析模式
3. **混合工作流**：先运行仅采样模式获得DeepMD数据，后续需要时运行完整分析模式

## 示例工作流

### 场景1：快速准备DeepMD训练数据

```bash
# 步骤1：仅采样，快速获得训练数据
python main_abacus_analyser_refactored.py --sampling_only \
    --search_path /data/md_trajectories \
    --sample_ratio 0.05 \
    --workers 8

# 步骤2：使用DeepMD数据训练模型
# 数据位置: analysis_results/run_r0.05_p-0.5_v0.9/deepmd_npy_per_system/
```

### 场景2：完整分析流程

```bash
# 一步完成：采样 + 统计分析 + 相关性分析 + 效果评估
python main_abacus_analyser_refactored.py \
    --search_path /data/md_trajectories \
    --sample_ratio 0.1 \
    --workers 8
```

### 场景3：渐进式分析

```bash
# 步骤1：快速采样
python main_abacus_analyser_refactored.py --sampling_only

# 步骤2：后续需要详细分析时，运行完整模式（会复用采样结果）
python main_abacus_analyser_refactored.py
```
