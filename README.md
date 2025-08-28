
# ABACUS-STRU-Analyser

## 简介

ABACUS-STRU-Analyser 是针对 ABACUS 分子动力学轨迹的高效分析工具，支持批量处理、智能采样与统计相关性分析。代码已模块化、无冗余，主流程清晰，输出结构规范。

## 快速上手

### 1. 环境准备

- Python >= 3.7
- 推荐：Anaconda/Miniconda
- 依赖安装：
  ```bash
  pip install numpy scipy pandas
  # 或 conda install numpy scipy pandas
  ```

### 2. 数据结构规范

```
project/
├── struct_mol_xxx_conf_xxx_TxxxK/
│   └── OUT.ABACUS/STRU/STRU_MD_*
└── ...
```
目录与文件命名需严格遵循上述格式。

### 3. 分析命令

```bash
# 默认分析（在当前目录的父目录递归搜索）
python abacus_dist_analyzer.py

# 指定搜索路径
python abacus_dist_analyzer.py --search_path /path/to/project_root

# 自定义参数
python abacus_dist_analyzer.py --include_h --sample_ratio 0.1 --power_p 1 --workers 8 --search_path /data/abacus_projects

# 仅相关性分析
python correlation_analyzer.py -i analysis_results/combined_system_summary.csv -o correlation_results --log-file
```

## 代码结构

项目采用模块化设计，主要文件：

| 文件 | 功能 | 说明 |
|---|---|---|
| `abacus_dist_analyzer.py` | 主程序 | 分析流程控制，约200行 |
| `utils.py` | 通用工具 | 文件操作、数学工具、目录发现 |
| `stru_parser.py` | STRU解析器 | 解析ABACUS STRU文件 |
| `metrics.py` | 指标计算器 | 计算nRMSF、MCV、nLdRMS等指标 |
| `sampler.py` | 采样器 | 基于幂平均距离的智能采样 |
| `system_analyzer.py` | 系统分析器 | 集成解析、计算、采样功能 |
| `result_saver.py` | 结果保存器 | 保存各种分析结果到CSV |
| `path_manager.py` | 路径管理器 | 管理分析目标和进度状态 |
| `correlation_analyzer.py` | 相关性分析器 | 统计相关性分析 |

### 4. 输出说明

分析结果位于 `analysis_results/`：

| 文件/目录 | 说明 |
|---|---|
| combined_system_summary.csv | 系统级指标总表 |
| sampled_frames.csv | 采样帧列表 |
| parameter_analysis_results.csv | 详细相关性分析 |
| parameter_analysis_summary.csv | 相关性分析汇总 |
| analysis.log | 主程序日志 |
| analysis_targets.json | 分析目标详细信息 |
| path_summary.json | 路径发现摘要 |
| target_paths.txt | 人类可读的路径列表 |
| struct_mol_<ID>/metrics_per_frame_*.csv | 每帧详细指标 |

## 参数说明

### 主程序 (`abacus_dist_analyzer.py`)

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| --include_h | flag | False | 是否包含氢原子 |
| --sample_ratio | float | 0.05 | 采样比例 |
| --power_p | float | 0.5 | 幂平均参数 (0:几何, 1:算术, -1:调和) |
| --workers | int | -1 | 并行进程数 |
| --output_dir | str | analysis_results | 输出目录 |
| --search_path | str | 当前目录的父目录 | 递归搜索STRU文件的根路径 |

### 相关性分析 (`correlation_analyzer.py`)

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| -i, --input | str | 自动查找 | 输入CSV路径 |
| -o, --output | str | correlation_analysis_results | 输出目录 |
| --log-file | flag | False | 启用文件日志 |

## 输出结果结构

- **combined_system_summary.csv**：系统/分子/温度等主指标，含采样前后对比
- **parameter_analysis_results.csv**：全局/分子层次的相关性与显著性统计
- **parameter_analysis_summary.csv**：全局趋势与分子级概览
- **analysis.log / correlation_analysis.log**：详细日志，含进度、警告、统计解释

## 结果解读

- 相关系数 |r| < 0.3：弱相关；0.3-0.7：中等；≥0.7：强相关
- Eta² < 0.01：无效应；0.01-0.06：小效应；0.06-0.14：中等；≥0.14：大效应
- p < 0.05：统计学显著

## 最佳实践

- 严格遵循目录/文件命名规范
- 使用 `--search_path` 指定包含多个项目的根目录
- 先小规模测试采样参数
- 优先解读全局结果，分子级仅作参考
- 检查 `analysis_targets.json` 和 `path_summary.json` 了解发现的体系
- 检查日志获取分析建议和状态信息

## 技术支持

如遇问题请：
1. 检查日志和本说明
2. 确认数据/依赖
3. 提交 GitHub Issue

## 许可证

详见 LICENSE 文件。
