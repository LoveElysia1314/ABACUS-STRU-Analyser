# ABACUS-STRU-Analyser 代码质量与后续重构路线图（精简版）

**版本：** v1.3  
**日期：** 2025年9月1日  
**维护者：** GitHub Copilot  
**当前代码状态：** Level 4 目标已完成 + “采样结果复用 v1” 已落地（仅跳过采样阶段，其余输出强制重算 & 自动补全缺失文件）。

v1.3 更新要点：
- 实现：采样复用（基于 `analysis_targets.json` + 源数据哈希），路径归一化 + system_name 回退匹配解决跨目录重用失败。
- 调整：增量续算路线图，新增 PR0（已完成），其余步骤顺延。
- 文档：统一成功判定与数据流描述，清理冗余。 

---

## 🥇 当前最高优先级 (P0) — 增量续算重构（进行中）

### 已完成（PR0：采样复用 v1）
条件：历史 `analysis_targets.json` 中目标体系存在且哈希匹配 → 跳过采样；仍重新生成全部指标/汇总/对比/相关性输出。若衍生输出缺失或损坏 → 状态回退 pending 自动补写。

### 待实现范围（v2+）
在采样短路基础上，加入均值构象短路、指标差异补算与多层缓存。

### 目标（v2 完成定义）
1. 可独立短路：采样 / 均值构象 / 高成本分布指标（EMD）。
2. 指标新增/移除自动检测并最小补算。
3. 分层缓存：frame_metrics → mean_structure → derived_metrics → correlation。
4. schema_version + metric_versions 双层版本校验。

### 成功判定 (KPIs)
| 维度 | 判据 |
|------|------|
| 性能 | 第二次运行耗时缩短 ≥ 70%（采样+均值均被短路） |
| 准确 | 与全量模式核心标量差异 < 1e-8 |
| 鲁棒 | 损坏/不兼容缓存自动回退全量，不中断 |

### 续算判定条件（目标状态）
| 级别 | 条件 | 描述 |
|------|------|------|
| 采样短路 (已实现) | 哈希匹配 + sampled_frames 可解析 | 跳过采样策略执行 |
| 均值短路 | mean_structure JSON 维度/帧数/哈希/版本匹配 | 跳过均值迭代 |
| 指标差异补算 | index.json 存在指标版本映射 | 仅补缺失或过期指标 |

### 拟新增 / 调整输出
1. `system_metrics_summary.csv` 移除 `Mean_Structure_Coordinates` 列。
2. 新增 `mean_structures/mean_structure_<system>.json`：
     ```json
     {
         "system": "struct_xxx",
         "num_frames": 5000,
         "natoms": 2048,
         "dimension": 3,
         "mean_structure": [[x,y,z], ...],
         "version": "ms-v1",
         "generated_at": "2025-09-01T12:00:00Z"
     }
     ```
3. 新增 `cache/index.json`：{ system: { source_hash, frame_count, natoms, metric_versions:{...}, schema_version } }。

### 计划公共工具 (抽象层)
| 函数 | 作用 |
|------|------|
| detect_resume_state(output_dir) -> Dict[str, ResumeState] | 聚合可短路组件判定 |
| load_mean_structure(system) | 读取 + 校验 mean JSON |
| parse_sampled_frames(csv) | 提取采样帧索引序列 |
| reconstruct_metrics_from_sampling(...) | 构建最小 metrics 骨架 |
| compute_remaining_metrics(...) | 差异检测后补算 |
| export_mean_structure_json(...) | 输出/覆盖均值缓存 |
| validate_cache_meta(meta) | 统一版本/哈希/维度校验 |

### 数据流（目标 v2）
1. 解析 analysis_targets.json → sampled_frames, source_hash。
2. 尝试加载 mean_structure 缓存（可选）。
3. 重建基础 metrics（惰性帧数据访问）。
4. 差异检测 → 生成需补算指标列表。
5. 补算并写出所有产物（刷新时间戳）。

### 回退策略
任一层缓存缺失/不兼容 → WARN & 回退全量；全量成功后刷新缓存层。

### 迭代拆分（更新后）
| 顺序 | 名称 | 内容 | 产出 |
|------|------|------|------|
| PR0 (Done) | 采样复用 v1 | hash + name fallback | 采样短路 |
| PR1 | 均值结构迁移 | CSV 列移除 + JSON 输出 | mean JSON |
| PR2 | resume_utils & index | 检测+元数据骨架 | index.json |
| PR3 | 均值短路实现 | 参数 `--enable-resume` 控制 | 均值短路 |
| PR4 | 差异补算 & 校验 | metric_versions + A/B diff | 局部补算框架 |
| PR5 | 全指标比较/相关性扩展 | 指标全集覆盖 | 扩展报表 |

### 风险 & 缓解
| 风险 | 说明 | 缓解 |
|------|------|------|
| 缓存腐坏 | JSON/CSV 损坏 | 严格校验 + 自动回退 |
| 逻辑漂移 | 指标新增未捕获 | 版本映射 + 差异检测 |
| IO 过大 | 反复加载大帧数据 | 惰性/分块加载 |
| 误命中 | 非同源数据被短路 | 哈希+版本双校验 |

---

## � 中优先级 (P1) — 仍待处理事项
1. DataValidator：统一空/NaN/维度不符检查。
2. constants.py：列名、schema_version、metric_versions 常量集中。
3. 指标开关：配置禁用 EMD / PCA 等高成本指标。
4. CSV schema 校验：启动校验列集合与顺序。

---

## 🟢 低优先级 (P2) — 优化 / 增强
1. mypy 类型基线。
2. NumPy 风格 docstrings 统一。
3. 性能微调（向量化 / 缓冲复用 / 可选近似 EMD）。
4. run_metadata.json：参数、随机种子、git commit 追踪。

---

## 🔄 采样方法对比与相关性分析扩展
目标：comparison CSV 与 system_metrics_summary 指标全集对齐；相关性分析覆盖全部数值型指标（排除高维数组）。
新增函数：collect_system_metrics_for_sampling / export_sampling_comparison / correlation_on_metrics。

---

## � 精简路线图（概览）
| 阶段 | 名称 | 状态 | 说明 |
|------|------|------|------|
| 已完成 | Level 1-4 | ✅ | 基础重构阶段 |
| P0-PR0 | 采样复用 v1 | ✅ | 跳过采样，其他重算 |
| P0-PR1 | 均值结构迁移 | 待开始 | JSON 输出 / CSV 精简 |
| P0-PR2 | resume_utils & index | 待开始 | 检测 & 元数据骨架 |
| P0-PR3 | 均值短路 | 待开始 | 需参数控制 |
| P0-PR4 | 差异补算 & 校验 | 待开始 | A/B diff 工具 |
| P0-PR5 | 全指标对比/相关性 | 待开始 | 指标统一收集层 |
| P1 | 数据验证 & 开关 | 排队 | 与 P0 后期并行 |
| P2 | 风格 / 类型 / 再现性 | 排队 | 低风险穿插 |

---

## ✅ 已完成（摘要）
- 采样结果复用 v1（哈希 + system_name 回退）
- 自动补写缺失输出（复用模式）
- 采样后指标剥离 & 主 CSV 指标归一
- 分布相似性 (JS / EMD) 与多样性指标整合
- 日志标准化与输出排序
- 采样/结构逻辑抽离

---

## � 下一步立即行动（聚焦 PR1）
1. result_saver: 添加 export_mean_structure_json() 并移除 CSV 中均值坐标列。
2. TrajectoryMetrics: 增加 mean_structure_cached 字段。
3. utils/resume_utils.py 骨架：detect_resume_state / load_mean_structure 占位。
4. constants.py: 定义 SCHEMA_VERSION / METRIC_VERSION 占位常量。
5. 日志增强：复用时输出匹配模式 + 哈希前后 8 位。

---

## 文档版本
- v1.0 (2025-08-31): 初始版本
- v1.1 (2025-09-01): 阶段性重构明细（至 Level 4）
- v1.2 (2025-09-01): 精简路线 + 增量续算规划
- v1.3 (2025-09-01): 采样复用 v1 完成，路线图重排序

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
