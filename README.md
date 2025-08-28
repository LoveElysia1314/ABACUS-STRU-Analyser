
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

3. 运行主分析（在含有多个 `struct_mol_*_conf_*_T*K` 父目录的上层目录执行，或使用 `--search_path` 指定）：

  ```powershell
  python .\main_abacus_analyzer.py --include_h --sample_ratio 0.05 --power_p 0.5 --workers 4 --output_dir analysis_results --search_path D:\path\to\projects
  ```

说明：若只想运行相关性分析模块（已有 `analysis_results/combined_analysis_results/system_metrics_summary.csv`），可直接调用 `src/analysis/correlation_analyzer.py`（项目提供为模块形式）：

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

输出文件（位于 `analysis_results`，或通过 `--output_dir` 指定）
----------------------------------------------------------------

- `analysis_targets.json`：发现/加载的目标列表和状态。
- `path_summary.json`：路径摘要与统计。
- `target_paths.txt`：可读的目标路径列表与状态。
- `single_analysis_results/frame_metrics_<system>.csv`：每体系按帧的 nLdRMS 与采样标记。
- `combined_analysis_results/system_metrics_summary.csv`：体系级汇总指标（用于相关性分析）。
- `combined_analysis_results/parameter_analysis_results.csv`：相关性分析详细结果（若启用相关性分析）。

主要参数说明（`main_abacus_analyzer.py`）
------------------------------------------------

- `--include_h`：包含氢原子（默认不包含）。
- `--sample_ratio`：采样比例（默认 0.05）。
- `--power_p`：幂平均采样的 p 值（默认 0.5；0 表示几何平均，1 表示算术平均，-1 表示调和平均）。
- `--workers`：并行进程数（默认 -1，表示使用环境或 cpu_count）。
- `--output_dir`：输出目录（默认 `analysis_results`）。
- `--search_path`：用于递归查找目标体系的根目录（默认为当前目录的父目录）。

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

最后修改：2025-08-16 （v3.0）——模块化相关性分析并增强并行稳定性。

如果需要，我可以：

- 把 README 翻译为英文版本并生成 `README_EN.md`。
- 添加示例数据并运行端到端 smoke test（需要短小的 STRU 文件样例）。
- 为关键模块添加单元测试并配置简单的 CI。

---

（已自动将主项目文档整合到本 README；若需调整某部分的表述或增删章节，请告诉我想保留/删除的文档片段。）
