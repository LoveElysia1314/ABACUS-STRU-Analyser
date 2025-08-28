
# ABACUS-STRU-Analyser

## 简介

ABACUS-STRU-Analyser 是针对 ABACUS 分子动力学轨迹的高效分析工具，支持批量处理、智能采样与统计相关性分析。代码已模块化、无冗余，主流程清晰，输出结构规范。

## 快速上手
# ABACUS-STRU-Analyser

项目概览
---------

ABACUS-STRU-Analyser 是面向 ABACUS STRU 分子动力学轨迹的高效批量分析工具。功能包括：批量发现/管理系统目录、轨迹解析、距离/结构指标计算、智能采样（基于幂平均/层次策略）、汇总与可选的相关性统计分析。目标是为分子/材料研究提供可重复、可扩展的轨迹指标分析流水线。

目录与关键模块
----------------

- `main_abacus_analyzer.py`：主入口，负责目标发现、并行/顺序分析调用、结果保存与调用相关性分析模块。
- `src/utils.py`：通用工具（日志、文件操作、数学/统计工具、目录发现等）。
- `src/io/path_manager.py`：管理分析目标（发现/加载/保存/导出/验证/状态跟踪）。
- `src/io/stru_parser.py`：解析 ABACUS `OUT.ABACUS/STRU` 轨迹文件（用于生成帧数据）。
- `src/core/system_analyzer.py`：系统级分析流程（计算指标、执行采样），包含 `SystemAnalyzer` 与 `BatchAnalyzer`。
- `src/core/metrics.py`：指标计算（nRMSF、MCV、frame_nLdRMS、avg_nLdRMS 等）。
- `src/core/sampler.py`：采样策略（PowerMeanSampler / RandomSampler / UniformSampler）。
- `src/io/result_saver.py`：把每体系与汇总结果写入 `analysis_results` 下的 CSV 文件。
- `src/analysis/correlation_analyzer.py`：可选的独立相关性分析模块（利用 pandas / scipy 做 Pearson/Spearman/ANOVA，并输出 `parameter_analysis_results.csv` 等）。

快速上手（最短路径）
----------------------

1. 建议创建并激活虚拟环境（推荐 conda）：

  ```powershell
  # Windows PowerShell 示例
  conda create -n abacus python=3.11 -y; conda activate abacus
  pip install -r requirements.txt
  ```

2. 运行帮助查看参数：

  ```powershell
  python .\main_abacus_analyzer.py --help
  ```

3. **运行主分析（推荐模式）**

在含有多个 `struct_mol_*_conf_*_T*K` 父目录的上层目录执行，或使用 `--search_path` 指定：

```powershell
python .\main_abacus_analyzer.py --include_h --sample_ratio 0.05 --power_p 0.5 --workers 4 --output_dir analysis_results --search_path D:\path\to\projects
```

**增量计算**：程序会基于参数自动创建或查找对应目录，如果发现已有完整结果则直接跳过计算并执行相关性分析。

4. **仅运行相关性分析模块**

如果已有分析结果（`analysis_results/run_*/combined_analysis_results/system_metrics_summary.csv`），可直接调用相关性分析：

  ```powershell
  python .\src\analysis\correlation_analyzer.py  # 或以模块方式在代码中引用 CorrelationAnalyzer
  ```

依赖（最小）
-------------

- numpy
- pandas
- scipy

（已在项目根添加 `requirements.txt`，建议在虚拟环境中安装。）

输入数据与目录约定
-------------------

项目期望的目录结构：

```
project_root/
  struct_mol_<ID>_conf_<C>_T<Temp>K/
   OUT.ABACUS/STRU/STRU_MD_*
```

主脚本默认会以当前工作目录的父目录为搜索根（可用 `--search_path` 覆盖）。路径名称需遵循 `struct_mol_{mol_id}_conf_{conf}_T{temperature}K` 格式以便自动解析。

输出文件（基于参数的目录结构）
----------------------------------------------------------------

**新特性：参数隔离的输出目录**

程序会根据分析参数自动创建专用目录，避免不同参数的结果相互覆盖：

```
analysis_results/
├── run_include_h_False_power_p_-0.5_sample_ratio_0.05/
│   ├── analysis_targets.json
│   ├── single_analysis_results/
│   └── combined_analysis_results/
├── run_include_h_True_power_p_-0.5_sample_ratio_0.05/
│   ├── analysis_targets.json
│   ├── single_analysis_results/
│   └── combined_analysis_results/
└── main_analysis_20250820_094824.log  # 带时间戳的日志文件
```

**输出文件说明：**
- `analysis_targets.json`：发现/加载的目标列表和状态，包含元数据、汇总信息和详细体系信息
- `single_analysis_results/frame_metrics_<system>.csv`：每体系按帧的 nLdRMS 与采样标记  
- `combined_analysis_results/system_metrics_summary.csv`：体系级汇总指标（用于相关性分析）
- `combined_analysis_results/parameter_analysis_results.csv`：相关性分析详细结果
- `main_analysis_YYYYMMDD_HHMMSS.log`：带时间戳的日志文件，避免覆盖

**增量计算机制：**
- **智能检测**：程序会检查对应参数目录是否已有完整结果，如有则直接跳过计算
- **参数隔离**：不同参数的结果完全独立，支持并行运行不同配置
- **源文件监控**：通过 SHA-256 哈希检测源文件变更，自动触发重新计算
- **状态恢复**：支持中断恢复，只计算未完成的系统

主要参数说明（`main_abacus_analyzer.py`）
------------------------------------------------

- `--include_h`：包含氢原子（默认不包含）。
- `--sample_ratio`：采样比例（默认 0.05）。
 - `--power_p`：幂平均采样的 p 值（默认 -0.5；0 表示几何平均，1 表示算术平均，-1 表示调和平均）。
- `--workers`：并行进程数（默认 -1，表示使用环境或 cpu_count）。
- `--output_dir`：输出目录（默认 `analysis_results`）。
- `--search_path`：用于递归查找目标体系的根目录（默认为当前目录的父目录）。
- `--skip_single_results`：跳过单个系统详细结果输出，仅保留汇总结果（节省磁盘空间）。
- `--force_recompute`：强制重新计算所有系统，禁用增量计算功能。


**增量计算机制（高效跳过已完成分析）**
-------------------------------------------------
程序支持智能的增量计算，基于参数分目录和双哈希校验机制：

**参数分目录机制**：
- **自动目录生成**：根据关键分析参数（`include_h`、`sample_ratio`、`power_p`、`skip_single_results`）自动生成唯一的输出目录
- **参数隔离**：不同参数组合的结果完全独立，支持并行运行不同配置
- **目录命名规则**：`run_include_h_{True/False}_power_p_{value}_sample_ratio_{value}/`

**增量计算逻辑**：
- **目录级检测**：程序首先检查对应参数目录是否已有完整结果，如有则直接跳过计算并执行相关性分析
- **状态管理**：程序会在参数目录中自动加载/保存 `analysis_targets.json`，记录每个体系的分析状态
- **源数据哈希校验**：自动计算每个体系的源文件哈希（基于 `OUT.ABACUS/STRU/` 中帧数最大的 STRU 文件）
- **参数兼容性检查**：只有当前参数与已保存的参数完全一致时，才启用增量计算
- **智能增量判断**：若体系在 `analysis_targets.json` 标记为 "completed"，且源数据哈希未变化，且结果文件存在，则自动跳过分析
- **强制重算支持**：`--force_recompute` 可禁用增量计算并强制重新分析所有系统

**源数据哈希校验优化**
----------------------
为确保增量计算的安全性和高效性，项目采用了**最大帧数STRU文件哈希法**：

- 只对 `OUT.ABACUS/STRU/` 文件夹中帧数最大的 `STRU_MD_*` 文件做哈希（基于文件名、帧号、大小、修改时间）
- 极大减少了文件系统调用（从O(n)降至O(1)），同时能有效检测轨迹是否完整
- 若最大帧文件缺失或被修改，系统会自动重新分析该体系

**参数管理机制**
--------------------
程序自动管理分析参数，确保结果一致性：

- **关键参数**：`include_h`（是否包含氢原子）、`sample_ratio`（采样比例）、`power_p`（幂平均参数）、`skip_single_results`（是否跳过详细结果）
- **非关键参数**：`workers`（并行数）、`output_dir`（输出根目录）、`search_path`（搜索路径）等不影响计算结果的参数
- **参数哈希**：使用 SHA-256 算法计算关键参数的哈希值，确保参数唯一性
- **兼容性检查**：只有当前参数与已保存的参数完全一致时，才启用增量计算；否则重置所有体系状态

**结果排序与采样增强**
----------------------
- 所有输出的汇总结果（如 `system_metrics_summary.csv`、`sampling_records.csv`）均按分子编号、构象、温度排序，便于对比与下游分析
- 采样结果文件中新增 `OUT_ABACUS_Path` 列，便于追溯每个体系的原始数据目录

**自动去重机制**：程序会自动检测并去除重复的体系（基于 `struct_mol_<ID>_conf_<C>_T<Temp>K` 名称），保留文件夹修改时间最晚的版本。这确保了即使存在备份或多版本数据也不会重复分析。

相关性分析（`src/analysis/correlation_analyzer.py`）
-------------------------------------------------

功能：对 `system_metrics_summary.csv` 中的指标（例如 nRMSF、MCV、avg_nLdRMS 及其采样后值）进行全局温度相关性（Pearson/Spearman）和构型影响（ANOVA）分析，输出 `parameter_analysis_results.csv`、`parameter_analysis_summary.csv` 与 `correlation_analysis.log`。

输入要求：CSV 需包含列 `Molecule_ID`, `Configuration`, `Temperature(K)` 与指标列 (`nRMSF`, `MCV`, `avg_nLdRMS`, `nRMSF_sampled`, `MCV_sampled`, `avg_nLdRMS_sampled`)。

设计要点与已作的改进
----------------------

- 并行稳定性：已把多进程 worker 改为顶层可序列化函数（在子进程内构造 `SystemAnalyzer`），以避免在 Windows 下的 pickling 问题。
- 模块化：相关性分析为可选独立模块，导入失败会被主程序安全忽略并记录警告。
- 结果格式标准化：输出目录下的 CSV 命名与列约定已统一，便于下游脚本处理。

运行建议与调试
-----------------

- 首次运行建议在小数据集上测试（`--sample_ratio` 设大一点、`--workers` 设 1），确认解析与指标计算逻辑。若通过，再放大规模并发。
- 并行问题排查：查看 `analysis_results/main_analysis.log` 与 `correlation_analysis.log`。
- 若缺失 `system_metrics_summary.csv`，相关性分析会被跳过。

贡献与许可证
----------------

欢迎通过 GitHub 提交 Issue 与 Pull Request（仓库：https://github.com/LoveElysia1314/ABACUS-STRU-Analyser）。请在贡献前阅读仓库中的 LICENSE 与贡献说明。

更多文档
---------

项目中还包含若干补充文档：

- `CORRELATION_ANALYZER_README.md`：相关性分析模块的详细说明与统计解释。
- `不同距离指标的的特性.md`：关于不同距离/聚合指标（最小距离、算术/几何/幂平均等）的性质与权衡讨论，适合指标选择参考。
- `PROJECT_OVERVIEW.md`, `PROJECT_FEATURES.md`：项目演进、特性与设计目标说明。

附录：不同距离指标的特性
-------------------------

下面的表格总结了常用距离/聚合指标的性质、适用性与计算复杂度，供指标选择参考：

| 指标                       | 性质说明                         | 聚焦最小距离 ↑ | 聚集敏感度 | 计算复杂度 |
|--------------------------|--------------------------------|---------------:|----------:|-----------:|
| 最小化最小夹角（最小距离） | 强制最坏点对远离                 | ✅ **最优**     | ★★★★      | ★★★★★     |
| 调和/算术平均距离         | 全局平均/稳健                   | ★★★★          | ✅ **最强**| ★★        |
| 幂平均距离（几何/算术等）  | 对极端值有可调的敏感度           | ★★★           | ★★★       | ★★★       |
| 层平平均距离 (p=0.5)      | 修正静电/极值影响               | ★★★★          | ★★★★      | ★★★★      |
| 算术平均距离             | 距离总和/最大化                 | ★             | ★         | ★★★★★     |

说明：
- 最小距离敏感于最近邻的密集集群，适用于检测局部紧密区域；但对全局统计可能不稳健。
- 聚集敏感度表示指标对小尺度密集簇的响应强度；选择时需平衡全局与局部信息。
- 计算复杂度列给出相对复杂度指示，实际时间依赖于帧数 N 与特征维度 k（例如某些算法有 O(N k^3) 成本）。

注意：该表来自项目补充文档，已在本 README 中保留关键结论。如需更详细的数值或讨论，请告诉我将把更深层内容保留或另存为长文档。

最后更新
---------

最后修改：2025-08-20 （v4.0）——实现参数分目录机制、日志文件唯一化、优化增量计算逻辑。

如果需要，我可以：

- 把 README 翻译为英文版本并生成 `README_EN.md`。
- 添加示例数据并运行端到端 smoke test（需要短小的 STRU 文件样例）。
- 为关键模块添加单元测试并配置简单的 CI。

文档修正与代码一致性说明
---------------------------
在审查项目实现后，发现 README 中若干表述与当前代码实现不完全一致或可能导致误解。为便于维护与使用，建议如下：

1) 增量计算与 `analysis_targets.json` 的持久化
  - 发现：程序会尝试加载 `analysis_targets.json`（由 `PathManager.load_targets()`），并利用其中的 `source_hash` 来做增量判断。但当前主程序并未在运行结束时自动写回/更新 `analysis_targets.json`（相关的 save_targets/save_summary 功能被移除），因此只有当用户或上游工具事先准备并保存了该文件时，才会启用持久化的增量缓存。
  - 建议：如果希望可靠的增量语义，需在程序结束时由 `PathManager` 将当前 `targets` 写回 `analysis_targets.json`（注意写入必须在结果文件完全写入后执行，以避免未完成任务被标记为 completed）。

2) 标记 `completed` 的时机
  - 发现：主流程中在分析成功后会调用 `path_manager.update_target_status(system_path, "completed")`（在并行/顺序处理里）。但在并行模式中，`_parallel_analysis` 使用 `imap_unordered`，且在某些分支中对失败或异常的映射对状态更新并不严格——并且结果写出与状态更新的顺序可能导致短时间内 `analysis_targets.json`（若由外部保存）与磁盘上的结果不一致。
  - 建议：保证只有在所有结果文件（单体系的 `frame_metrics_*.csv` 与汇总 `system_metrics_summary.csv`）成功写入磁盘后，再将对应 target 标记为 `completed` 并在最后一次性持久化 `analysis_targets.json`。这可以通过在 ResultSaver 保存成功后再更新状态并在主流程结束时保存 targets 实现。

3) 参数变更检测与 `analysis_params.json`
  - 发现：参数检测逻辑依赖 `analysis_params.json`，如果该文件丢失，程序会把运行视为首次（并返回 False 导致开启增量检查）或在读取失败时选择完整重算。当前实现会在每次运行末尾写入 `analysis_params.json`，但若程序在中途被中断，文件可能会缺失或包含旧值。
  - 建议：写入 `analysis_params.json` 时先写入到临时文件再原子重命名，或在写入前再做一次完整性检查；并把参数变更的范围记录在日志中以便审计。

4) 增量完整性校验增强
  - 发现：`PathManager.check_existing_results()` 依赖 summary CSV 中的 `System` 列以及单体系 `frame_metrics_*.csv` 的存在性与旧的 `source_hash` 值来判定是否已完成。该逻辑对“被中断但文件存在且不完整”的场景无法检测（例如 CSV 存在但行数不完整），存在误判风险。
  - 建议：增加对文件完整性的简单校验，例如检查 `frame_metrics_*.csv` 的行数是否与 `num_frames` 匹配，或对 summary 行的必需字段做非空校验；在发现不一致时把状态设置为 `pending` 并记录到日志。

5) README 文档更新（已应用部分）
  - 已修正 `--power_p` 的默认值为 `-0.5`，并补充了哪些文件是由程序实际写入/导出的说明。

如需，我可以继续：
 - 实现并提交上述第1和第2点的最小修复补丁（确保持久化 `analysis_targets.json`、写回前保证结果文件已写入）。
 - 为 `PathManager.check_existing_results()` 添加文件完整性检查并编写对应的单元测试。


---


---

# 性能优化与使用示例（合并自 PERFORMANCE_GUIDE.md 与 USAGE_EXAMPLES.md）

## ABACUS主分析器性能优化指南

### 问题诊断与常见卡顿点分析

1. **目录发现阶段**：大规模系统时，文件系统遍历和STRU文件匹配可能导致长时间无输出。
2. **路径管理器初始化阶段**：处理大量系统路径和元数据提取时可能卡顿，内置去重机制会自动移除重复体系。
3. **重复体系去重阶段**：检测并移除同名体系目录，保留修改时间最晚版本。
4. **目标验证阶段**：已移除，依赖SystemAnalyzer内部错误处理。
5. **结果保存阶段**：生成大量单体系详细结果文件时可能卡顿。

### 性能优化策略
- 自动优化机制：自动检测并去除重复体系，减少冗余分析。
- 去重策略：保留文件夹修改时间最晚的版本。
- 性能提升：减少冗余分析，适合有备份或多版本数据环境。
- 验证移除：完全移除耗时的预验证步骤，依赖SystemAnalyzer内部错误处理。
- 日志记录：去重过程详细记录在日志中，便于审查。

#### 快速诊断模式
```powershell
python main_abacus_analyzer.py --search_path "你的数据路径" --skip_validation --workers 1 --sample_ratio 0.001
```

#### 大批量处理模式
```powershell
python main_abacus_analyzer.py --search_path "数据路径1" "数据路径2" --skip_validation --skip_single_results --workers 16 --sample_ratio 0.05
```

#### 分批处理策略
```powershell
python main_abacus_analyzer.py --search_path "path/batch1/*" --skip_validation --skip_single_results
python main_abacus_analyzer.py --search_path "path/batch2/*" --skip_validation --skip_single_results
python main_abacus_analyzer.py --search_path "path/batch3/*" --skip_validation --skip_single_results
```

#### 参数说明与性能影响

| 参数 | 性能影响 | 适用场景 |
|------|----------|----------|
| `--skip_validation` | 跳过文件验证，节省大量I/O时间 | 确认数据完整的大批量处理 |
| `--skip_single_results` | 不生成frame_metrics_*.csv，节省磁盘I/O | 只需汇总统计的场景 |
| `--workers N` | 并行处理，理论上N倍加速 | CPU和I/O密集的分析阶段 |
| `--sample_ratio 0.01` | 更小的采样比例减少计算量 | 快速评估或预览结果 |

#### 存储空间考虑
- 单个系统详细结果：每个系统约1-10MB
- 1000个系统：可能产生1-10GB详细结果文件
- 跳过详细结果：仅保留汇总文件，通常<100MB

#### 集群环境优化
SLURM作业脚本示例：
```bash
#!/bin/bash
#SBATCH --job-name=abacus_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=64G

module load python/3.8
source activate deepmd

python main_abacus_analyzer.py \
  --search_path "/path/to/data/*" \
  --skip_validation \
  --skip_single_results \
  --workers ${SLURM_CPUS_PER_TASK} \
  --output_dir "results_$(date +%Y%m%d_%H%M%S)"
```

#### 内存使用优化
```powershell
python main_abacus_analyzer.py --workers 4 --skip_single_results
```

#### 故障排除
1. 程序卡在发现阶段：缩小搜索路径范围，检查无关子目录。
2. 程序卡在验证阶段：加 `--skip_validation`，检查网络存储访问速度。
3. 内存不足：减少 `--workers` 数量，使用 `--skip_single_results`，分批处理数据。
4. 磁盘空间不足：用 `--skip_single_results`，定期清理旧结果，使用更大存储分区。

#### 性能基准参考
- 小规模 (10-50系统): 1-5分钟
- 中规模 (100-500系统): 10-60分钟
- 大规模 (1000+系统): 2-10小时

---

## ABACUS主分析器使用示例

### 基本用法
1. 使用默认路径（当前目录的父目录）
```powershell
python main_abacus_analyzer.py
```
2. 指定单个搜索路径
```powershell
python main_abacus_analyzer.py --search_path "D:\abacus_data"
```
3. 指定多个搜索路径
```powershell
python main_abacus_analyzer.py --search_path "D:\data1" "D:\data2" "E:\projects"
```

### 通配符支持
4. 匹配所有以project开头的目录
```powershell
python main_abacus_analyzer.py --search_path "D:\datasets\project*"
```
5. 匹配特定模式目录
```powershell
python main_abacus_analyzer.py --search_path "D:\data\2024\*\simulation"
```
6. 递归匹配（**模式）
```powershell
python main_abacus_analyzer.py --search_path "D:\projects\*\*\abacus"
```
7. 组合多个通配符模式
```powershell
python main_abacus_analyzer.py --search_path "D:\data1\*" "E:\backup\project*" "F:\archive\2024*"
```

### 性能优化选项
8. 大批量处理优化（默认高性能模式）
```powershell
python main_abacus_analyzer.py --search_path "D:\large_dataset\*" --skip_single_results --workers 16
```
9. 仅跳过单个系统详细结果
```powershell
python main_abacus_analyzer.py --search_path "D:\data\*" --skip_single_results
```
10. 增量计算模式（新功能）
```powershell
python main_abacus_analyzer.py --search_path "D:\data\*"
# 程序会自动检测已有结果，仅计算新的或失败的系统

# 强制重新计算所有系统（禁用增量计算）
python main_abacus_analyzer.py --search_path "D:\data\*" --force_recompute
```

### 高级选项
11. 允许搜索项目自身目录（用于测试）
```powershell
python main_abacus_analyzer.py --search_path "." --include_project
```
12. 完整参数示例
```powershell
python main_abacus_analyzer.py --search_path "D:\data\*\simulation" "E:\projects\abacus*" --workers 8 --sample_ratio 0.1 --include_h --output_dir "results_2024"
```

### 注意事项
- 自动去重机制：自动检测并去除重复体系，保留修改时间最晚版本，详细记录在日志。
- 性能优化建议：已完全优化，默认获得最佳性能。增量计算机制避免重复计算。
- 只需汇总结果时，使用 `--skip_single_results` 可节省大量磁盘空间。
- SystemAnalyzer内置完善的错误处理，无需预先验证。
- 增量计算机制：检测 `analysis_targets.json` 状态和已有结果文件，已完成的系统自动跳过。用 `--force_recompute` 可强制全部重算。
- 通配符需加引号，避免PowerShell展开。
- 路径分隔符建议用 `\`。
- 性能考虑：通配符展开可能匹配大量目录，建议精确指定模式。
- 项目目录屏蔽：默认屏蔽项目自身目录，`--include_project` 可关闭。

### 输出示例
```
ABACUS主分析器启动 | 采样比例: 0.05 | 工作进程: 8
搜索路径: ['D:\data\project1\simulation', 'D:\data\project2\simulation', 'E:\projects\abacus_test']
项目目录屏蔽: 开启
通配符 'D:\data\*\simulation' 展开为 2 个目录
搜索路径: D:\data\project1\simulation
在 D:\data\project1\simulation 中发现 3 个分子类别
搜索路径: D:\data\project2\simulation
在 D:\data\project2\simulation 中发现 5 个分子类别
```
