# ABACUS-STRU-Analyser 代码质量与后续重构路线图（精简版）

**版本：** v1.2  
**日期：** 2025年9月1日  
**维护者：** GitHub Copilot  
**当前代码状态：** Level 4 所有既定目标已完成（指标适配、采样后指标剥离、列分组排序、分布相似性接入）。

本版本开始：移除已交付/完成的历史任务条目，只保留“尚未完成 / 新增的演进需求”。

---

## 🥇 当前最高优先级 (P0) — 续算（增量复用）重构

### 背景
现有执行模式对长轨迹（帧数大、采样耗时高）每次需全量重新计算：
- 采样策略执行（Greedy / PowerMean 等）→ 计算量最大
- 平均构象（Mean Structure）迭代求解 → 次高耗时
其余统计（MinD / ANND / MPD / PCA / 多样性 / 分布相似性）相对廉价，可在复用基础数据后快速重算。

### 目标
若检测到“可续算条件”满足：
1. 直接从 `single_analysis_results/*.csv` 与 `analysis_targets.json`、以及缓存 JSON 中加载：
   - 每帧原子坐标向量（或其投影/派生特征）
   - 已选采样帧列表（sampled_frames）
   - 平均构象坐标（从 system_metrics_summary.csv 中移出，单独 JSON 缓存）
2. 跳过重新采样与均值构象迭代；重建 `TrajectoryMetrics` 与后续指标。
3. 重新执行：PCA / 多样性 / 分布相似性 / 统计聚合 / 报表写出 / 相关性分析 / 采样方法对比。

### 成功判定
| 维度 | 判据 |
|------|------|
| 速度 | 同一数据集第二次运行 ≥ 70% 时间缩短 |
| 正确性 | 与全量重新计算差异 < 1e-8 (核心标量指标) |
| 鲁棒性 | 缺失或部分损坏缓存时自动回退全量计算 |

### 续算判定条件 (ResumeEligibility)
触发条件全部满足时进入续算模式：
1. 存在 `single_analysis_results/frame_metrics_*.csv` 且行数 ≥ 目标帧数
2. 存在 `analysis_targets.json` 且体系集合一致
3. 存在 对应 `cache/mean_structure_<system>.json`（新建目录）
4. 采样帧列表可成功解析（Selected=1）且非空
5. 无“版本不兼容”标记（通过写入缓存 `metadata.version` 匹配当前代码声明）

### 拟新增/调整输出
1. `system_metrics_summary.csv` 移除列：`Mean_Structure_Coordinates`
2. 新增：`mean_structures/mean_structure_<system>.json`
   ```json
   {
     "system": "struct_mol_1028_conf_0_T400K",
     "num_frames": 5000,
     "dimension": 3,
     "mean_structure": [[x,y,z], ...],
     "version": "ms-v1",
     "generated_at": "2025-09-01T12:00:00Z"
   }
   ```
3. 新增：`cache/index.json` 维护 { system: { checksum/frame_count/version } }

### 拟新增公共工具方法 (抽象层)
| 方法 | 位置建议 | 功能 |
|------|----------|------|
| `detect_resume_candidates(output_dir) -> Dict[str, ResumeState]` | `src/utils/resume_utils.py` | 扫描输出目录判定可续算体系 |
| `load_cached_mean_structure(system)` | 同上 | 读取 JSON & 校验维度/帧数 |
| `parse_sampled_frames(frame_metrics_csv)` | 同上 | 返回有序采样帧列表 |
| `reconstruct_metrics_from_cache(system, sampled_frames, mean_structure)` | `core/system_analyser.py` 或独立工厂 | 构建基础 `TrajectoryMetrics` 骨架 |
| `compute_remaining_metrics(metrics, frames_data)` | `utils/metrics_utils.py` | 在已有均值 & 采样集上补算其余指标 |
| `export_mean_structure(system, mean_structure, meta)` | `io/result_saver.py` | 首次或回退全量时输出 JSON |
| `validate_cache_compat(meta) -> bool` | `resume_utils` | 版本/维度/数量一致性检查 |

### 关键数据流（续算模式）
1. 读取 frame_metrics CSV → 采样帧集合 + 每帧能量（可选）
2. 读取 mean_structure JSON → ndarray(mean)
3. 加载原始帧源数据（必要时只加载用于补算的最小数据：坐标矩阵/能量数组）
4. 重建 metrics → 执行补算 (PCA、多样性、JS/EMD、MinD/ANND/MPD 若未缓存)
5. 写回 system_metrics_summary（无均值坐标列）+ 采样对比 & 相关性分析

### 回退策略
任一关键步骤（JSON损坏 / 维度不符 / 采样帧缺失）→ 记录 WARN → 走全量路径，并重新生成缓存。

### 迭代拆分（建议执行次序）
1. (PR1) 结构调整：移除 CSV 中 `Mean_Structure_Coordinates` 列 & 新增 JSON 导出
2. (PR2) 新增 `resume_utils` + 判定逻辑（检测 + 日志输出试运行，不启用短路）
3. (PR3) 启用真正短路：跳过采样与均值计算（加运行参数 `--enable-resume` 防止误触）
4. (PR4) 增加校验/版本元数据与回退测试
5. (PR5) 采样方法对比 & 相关性分析扩展到使用“所有 system_metrics_summary 指标”

### 风险 & 缓解
| 风险 | 说明 | 缓解 |
|------|------|------|
| 缓存腐坏 | JSON/CSV 部分丢失 | 校验 + 回退全量 + 记录warning |
| 指标回归 | 缓存模式与全量差异 | 添加 A/B 校验脚本：运行双模式 diff |
| 版本漂移 | 字段变化导致误续算 | `metadata.schema_version` 强校验 |

---

## � 中优先级 (P1) — 仍待处理事项

1. 数据验证与清洗统一：扩展 `data_utils` 提供 `DataValidator`（数组空/NaN/统计安全包装）。
2. 常量与枚举集中：建立 `utils/constants.py`（列名、阈值、版本号、特征开关）。
3. 指标开关机制：允许通过配置禁用高成本指标（EMD / PCA），减少首次全量时间。
4. 报表列自动校验：运行时对 `system_metrics_summary.csv` 头部进行 schema 校验（防漂移）。

---

## 🟢 低优先级 (P2) — 优化 / 增强

1. 类型注解补全 & mypy 校验基线。
2. 文档字符串统一（NumPy 风格）。
3. 计算性能微优化（向量化 / 临时数组重用 / EMD 可选快速近似）。
4. 结果再现性：在根目录生成 `run_metadata.json`（参数、种子、git commit）。

---

## 🔄 采样方法对比与相关性分析扩展

### 现状
`sampling_methods_comparison.csv` 当前不含全部结构/分布指标；相关性分析部分字段缺失。

### 目标
1. 采样方法对比文件包含：与 `system_metrics_summary.csv` 完全一致的指标集合（除系统基础标识外可另设第一列描述采样策略）。
2. 相关性分析：对全指标矩阵运行（可选参数过滤高维 JSON / 列表字段）。

### 拟公共函数
| 函数 | 说明 |
|------|------|
| `collect_system_metrics_for_sampling(strategies_results)` | 汇总不同策略返回的 `TrajectoryMetrics` 为统一表结构 |
| `export_sampling_comparison(metrics_table)` | 输出 comparison CSV |
| `correlation_on_metrics(df, exclude_patterns=None)` | 按需排除列表/JSON列后计算皮尔逊/斯皮尔曼 |

---

## � 精简路线图（更新后）

| 阶段 | 名称 | 状态 | 说明 |
|------|------|------|------|
| 已完成 | Level 1-4 | ✅ | 不再在文档中展开 |
| P0-PR1 | 均值结构迁移 & 列精简 | 待开始 | JSON 输出 / CSV 列更新 |
| P0-PR2 | 续算检测 | 待开始 | 仅检测 + 日志 |
| P0-PR3 | 跳过采样/均值 | 待开始 | 增参 `--enable-resume` |
| P0-PR4 | 校验与回退 | 待开始 | A/B diff 工具 |
| P0-PR5 | 采样对比 & 全指标相关性 | 待开始 | 指标收集公共层 |
| P1 | 数据验证 & 配置化开关 | 排队 | 与 P0 可并行晚期插入 |
| P2 | 风格 / 类型 / 再现性元数据 | 排队 | 低风险穿插 |

---

## ✅ 已完成（仅列举，不再维护详细条目）
- 重复结构/采样逻辑抽离与统一
- 日志标准化 & 输出分组排序
- 分布相似性（JS / EMD）与多样性指标并入主流程
- 采样后指标剥离，主 CSV 仅保留全集指标

---

## � 下一步立即行动建议
1. 执行 PR1：修改 `result_saver` 移除均值坐标列；新增 JSON 导出函数框架（空实现返回占位）。
2. 为 `TrajectoryMetrics` 添加可选字段 `mean_structure_cached: bool` 以标记来源。
3. 创建 `utils/resume_utils.py` 骨架（含 TODO 注释），先行提交，便于后续填充。

---

## 文档版本控制
- v1.0 (2025-08-31): 初始版本
- v1.1 (2025-09-01): 原分阶段重构明细（至 Level 4 前）
- v1.2 (2025-09-01): 精简已完成内容，新增续算重构为最高优先级 & 统一指标对比/相关性扩展计划
<parameter name="filePath">d:\drzqr\Documents\GitHub\ABACUS-STRU-Analyser\CODE_QUALITY_OPTIMIZATION_REPORT.md
