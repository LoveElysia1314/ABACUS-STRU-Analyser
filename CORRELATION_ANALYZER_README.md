# 相关性分析器使用说明

## 概述

`correlation_analyzer.py` 是一个专业的统计分析模块，专门用于分析 ABACUS STRU 轨迹数据中参数与指标间的关系。该模块经过优化，确保统计分析的可靠性和结果的科学意义。

### 设计理念

- **统计可靠性优先**：只输出样本量充足、统计意义明确的结果
- **模块化设计**：可独立运行或作为主程序的组件
- **科学严谨性**：自动过滤小样本分析，避免误导性结论
- **用户友好**：提供清晰的结果解释和研究建议

## 使用方式

### 1. 作为独立脚本运行

适用于已有 `combined_system_summary.csv` 文件，需要单独进行相关性分析的情况。

### 2. 作为模块被主程序调用

无缝集成到 `abacus_dist_analyzer.py` 的分析流程中。

## 核心功能

### 统计分析类型

1. **全局温度相关性分析**（重点推荐）
   - 跨所有分子和构象的整体趋势分析
   - 大样本量，统计可靠性高
   - Pearson 和 Spearman 相关系数
   
2. **分子级分析统计概览**
   - 样本量统计和显著性比例
   - 仅作为趋势参考，不输出具体结果
   - 避免小样本量的过度解读

### 质量控制特性

- **自动样本量检查**：跳过样本数不足的分析
- **统计警告过滤**：确保分析过程无警告
- **效应量评估**：提供 Eta 平方效应量指标
- **显著性检验**：完整的 p 值和置信区间

## 独立使用指南

### 基本命令

```bash
# 自动查找并分析（推荐）
python correlation_analyzer.py

# 指定输入文件
python correlation_analyzer.py -i analysis_results/combined_system_summary.csv

# 完整配置
python correlation_analyzer.py -i data.csv -o correlation_results --log-file
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-i, --input` | str | 自动查找 | 输入CSV文件路径 |
| `-o, --output` | str | correlation_analysis_results | 输出目录 |
| `--log-file` | flag | False | 启用详细日志文件 |

### 输入文件要求

**必需列：**
- `Molecule_ID`：分子标识符
- `Configuration`：构象编号
- `Temperature(K)`：温度值
- 指标列：`nRMSF`, `MCV`, `avg_nLdRMS`, `nRMSF_sampled`, `MCV_sampled`, `avg_nLdRMS_sampled`

**数据格式示例：**
```csv
System,Molecule_ID,Configuration,Temperature(K),nRMSF,MCV,avg_nLdRMS,...
struct_mol_1028_conf_0_T400K,1028,0,400,0.145,0.089,0.234,...
struct_mol_1028_conf_1_T450K,1028,1,450,0.156,0.092,0.241,...
```

## 输出结果说明

### 1. 详细分析结果 (`parameter_analysis_results.csv`)

**内容：** 仅包含全局分析结果，确保统计可靠性

**关键列：**
- `Analysis_Level`：分析层次（Global）
- `Parameter`：分析参数（Temperature）
- `Indicator`：目标指标
- `Statistic_Value`：相关系数值
- `P_value`：显著性检验 p 值
- `Significance`：是否统计显著
- `Interpretation`：相关性强度解释

**示例记录：**
```csv
Analysis_Level,Parameter,Indicator,Statistic_Value,P_value,Significance,Interpretation
Global,Temperature,nRMSF,0.228,0.000,Yes,弱相关
Global,Temperature,MCV_sampled,0.251,0.000,Yes,弱相关
```

### 2. 汇总分析结果 (`parameter_analysis_summary.csv`)

**内容：** 分子级统计概览和全局结果汇总

**分子级概览：**
- 仅提供统计数字（显著性比例、效应量均值等）
- 不包含具体分子的详细结果
- 用于了解整体分析情况

**全局结果汇总：**
- 每个指标的完整统计信息
- 最可靠的研究结论来源

### 3. 详细日志 (`correlation_analysis.log`)

**包含信息：**
- 数据加载和验证过程
- 样本量统计和过滤记录
- 完整的分析结果输出
- 研究建议和解读指导

## 结果解读指南

### 相关性强度判断

| 相关系数绝对值 | 解释 | 研究价值 |
|----------------|------|----------|
| < 0.3 | 弱相关 | 存在趋势，需结合其他证据 |
| 0.3 - 0.7 | 中等相关 | 有实际意义，值得深入研究 |
| ≥ 0.7 | 强相关 | 重要发现，优先研究对象 |

### 统计显著性

- **p < 0.05**：统计学显著 ⭐
- **p < 0.01**：高度显著 ⭐⭐
- **p < 0.001**：极高度显著 ⭐⭐⭐

### 样本量考虑

- **全局分析**：通常 N > 500，结果最可靠
- **分子级分析**：通常 N < 10，仅供参考
- **建议**：主要关注全局分析结果

## 高级特性

### 质量保证机制

1. **自动过滤**：
   - 跳过样本量 < 2 的组合
   - 过滤缺失值和异常数据
   - 避免统计警告和错误

2. **稳健性检查**：
   - 同时提供 Pearson 和 Spearman 相关系数
   - 交叉验证线性和单调关系
   - 效应量评估

3. **透明度**：
   - 详细记录分析过程
   - 说明数据过滤原因
   - 提供方法学解释

### 与主程序集成

当作为模块调用时：


```python
from correlation_analyzer import CorrelationAnalyzer

# 创建分析器
analyzer = CorrelationAnalyzer(logger=your_logger)

# 执行分析
success = analyzer.analyze_correlations(csv_file_path, output_dir)
```

## 最佳实践建议

### 1. 数据准备

- 确保数据完整性和一致性
- 检查温度和构象的分布
- 验证指标值的合理性

### 2. 结果分析

- **优先关注全局结果**：统计可靠性最高
- **谨慎解读分子级统计**：仅作为趋势参考
- **结合多个指标**：寻找一致的模式

### 3. 科学报告

- 报告样本量和统计方法
- 说明效应量的实际意义
- 讨论结果的局限性

### 4. 进一步分析

基于相关性分析结果，可以：
- 设计针对性实验验证
- 深入分析特定温度范围
- 研究关键指标的物理机制

## 故障排除

### 常见问题

**问题：** 没有找到输入文件
**解决：** 确保当前目录或指定路径存在 `combined_system_summary.csv`

**问题：** 分析结果为空
**解决：** 检查数据中是否有足够的温度变化和样本量

**问题：** 统计警告
**解决：** 当前版本已自动处理，如仍出现请检查数据质量

### 日志分析

查看日志文件中的关键信息：
- `样本量统计`：了解数据分布
- `数据不足`：理解过滤原因
- `研究建议`：获取分析指导

## 技术细节

### 统计方法

- **Pearson 相关**：检验线性关系
- **Spearman 相关**：检验单调关系
- **ANOVA**：检验组间差异（已禁用小样本）
- **效应量**：Eta 平方评估实际意义

### 数据处理

- 自动缺失值处理
- 样本量充足性检查
- 异常值识别和处理
- 分组有效性验证

## 版本更新

**v3.0** (2025年8月16日)：
- ✨ 重构为独立模块
- 🛡️ 增强统计稳健性
- 📊 优化输出策略
- 📖 完善使用文档

**v2.1** (2025年8月15日)：
- 🔧 修复小样本警告
- 📈 改进结果解释

---

**使用建议：** 相关性分析是探索性数据分析的重要工具，结果应结合领域知识和进一步实验验证来解释。
