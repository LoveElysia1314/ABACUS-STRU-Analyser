# ABACUS-STRU-Analyser 项目重构计划 (带进度)

## 🎯 总体目标
- **减少脚本数量**：从19个Python文件减少到约10个
- **均衡代码量**：每个脚本控制在300-600行
- **提升聚合度**：减少冗余和高耦合，规范项目架构

## 📊 当前进度
- ✅ **已完成阶段**：阶段0、阶段1、阶段2、阶段3、阶段4 (共完成5个阶段)
- 🔄 **进行中阶段**：无
- ⏳ **待完成阶段**：阶段5 (共1个阶段)
- 📈 **总体进度**：83.3% (5/6阶段完成)

---

## ✅ 已完成阶段详情

### 阶段0：日志管理模块合并 (✅ 已完成)
- **合并内容**：manager.py → logmanager.py
- **涉及文件**：删除`src/logmanager/`目录，更新11个文件的import
- **结果**：文件数量从20个减少到19个

### 阶段1：RMSD/结构比对相关合并 (✅ 已完成)
- **合并内容**：structural_metrics.py → system_analyser.py
- **涉及文件**：更新trajectory_analyser.py的import
- **结果**：system_analyser.py从421行增加到463行，文件数量从19个减少到18个

### 阶段2：metrics相关合并 (✅ 已完成)
- **合并内容**：metrics_utils.py + metrics_registry.py → metrics.py
- **涉及文件**：更新5个文件的import路径
- **结果**：metrics.py从148行增加到471行，文件数量从18个减少到16个

### 阶段3：采样相关合并 (✅ 已完成)
- **合并内容**：math_utils.py → sampler.py
- **涉及文件**：更新sampling_compare_demo.py的import，删除math_utils.py
- **结果**：sampler.py从252行增加到308行，文件数量从19个减少到18个

---

## 🔄 待完成阶段详情

### 阶段4：IO相关合并 (✅ 已完成)
- **合并内容**：deepmd_exporter.py + sampled_frames_to_deepmd.py → result_saver.py
- **涉及文件**：更新main_abacus_analyser.py的import，删除deepmd_exporter.py和sampled_frames_to_deepmd.py
- **结果**：result_saver.py从498行增加到808行，文件数量从18个减少到16个

### 阶段5：通用工具合并 (⏳ 待开始)
**目标**：将工具类合并到common.py
- **合并内容**：
  - data_utils.py → common.py
  - file_utils.py → common.py
- **涉及文件**：更新所有引用工具类的import
- **预期结果**：common.py成为统一的工具模块，文件数量减少到14个

---

## 📈 预期最终架构

```
src/
├── __init__.py
├── utils/
│   ├── __init__.py
│   ├── common.py           # 合并所有通用工具 (预期500+行)
│   └── logmanager.py       # 日志管理工具 (309行)
├── core/
│   ├── __init__.py
│   ├── analysis_orchestrator.py (116行)
│   ├── metrics.py          # 合并metrics相关 (471行)
│   ├── sampler.py          # 合并采样相关 (308行)
│   └── system_analyser.py  # 合并结构分析 (463行)
├── analysis/
│   ├── __init__.py
│   ├── correlation_analyser.py (887行 - 可能需要拆分)
│   ├── force_energy_parser.py (61行)
│   └── trajectory_analyser.py (149行)
└── io/
    ├── __init__.py
    ├── path_manager.py    (731行 - 可能需要拆分)
    ├── result_saver.py    # 合并IO导出 (808行)
    └── stru_parser.py     (280行)
```

## 🎯 下一步建议

**立即开始阶段5**：通用工具合并
- 理由：这是最后一个合并阶段，完成后将达成重构目标
- 预期用时：60-90分钟
- 风险等级：中

**阶段5完成后建议**：
- 评估correlation_analyser.py(887行)和path_manager.py(731行)是否需要拆分
- 进行最终的功能测试和代码审查
- 更新项目文档和README

## 📝 继续工作的检查点

当在新话题中继续时，请检查：
1. 当前git状态是否干净
2. utils目录中是否还有data_utils.py、file_utils.py
3. io目录中是否还有deepmd_exporter.py、sampled_frames_to_deepmd.py
4. 所有现有功能是否正常工作

---

**🎉 当前状态**：已完成83.3%的重构工作，项目架构更加清晰，冗余代码显著减少。只剩最后一个合并阶段即可达成最终目标！
