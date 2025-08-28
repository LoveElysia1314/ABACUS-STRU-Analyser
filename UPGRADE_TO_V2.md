# ABACUS-STRU-Analyser v2.0 升级指南

## 概述

ABACUS-STRU-Analyser v2.0 是一个重大版本更新，包含了架构重构、代码质量改进和新功能。本文档将指导您从 v1.x 升级到 v2.0。

## ⚠️ 重要提醒

**这是一个破坏性更新**。虽然核心分析功能保持不变，但内部架构和一些 API 接口发生了变化。升级前请备份您的数据和配置文件。

## 🆕 v2.0 新特性

### 架构改进
- **模块化设计**：将大型 `utils.py` 拆分为专门的模块
- **统一日志管理**：集中的日志配置和管理
- **改进的异常处理**：更详细的错误信息和更好的错误恢复
- **类型注解**：添加了类型提示以提高代码质量

### 开发工具
- **静态分析工具**：集成了 ruff、black、mypy
- **测试框架**：完整的单元测试和集成测试
- **CI/CD 流程**：自动化的代码质量检查

### 功能增强
- **增强的路径管理**：更好的文件组织和路径处理
- **改进的结果保存**：更灵活的结果保存选项
- **更好的错误处理**：详细的错误日志和堆栈跟踪

## 📋 升级前准备

### 1. 备份现有数据
```bash
# 备份整个项目目录
cp -r /path/to/old/project /path/to/backup/project_v1_backup

# 或者只备份重要的结果文件
cp -r analysis_results analysis_results_backup
```

### 2. 检查 Python 版本
v2.0 要求 Python 3.8 或更高版本：
```bash
python --version  # 应该显示 3.8+ 
```

### 3. 记录当前配置
记录您当前使用的命令行参数和配置文件，以便在升级后重新配置。

## 🔄 升级步骤

### 方法一：自动迁移（推荐）

1. **获取 v2.0 代码**
```bash
git checkout refactor/v2
# 或下载 v2.0 release
```

2. **安装依赖**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 可选，用于开发
```

3. **运行迁移脚本**
```bash
python tools/migration/migrate_v1_to_v2.py /path/to/old/data /path/to/new/data

# 先进行干运行测试
python tools/migration/migrate_v1_to_v2.py /path/to/old/data /path/to/new/data --dry-run
```

4. **验证迁移结果**
```bash
# 运行烟雾测试
python tests/smoke_test.py

# 检查迁移报告
cat /path/to/new/data/migration_report.json
```

### 方法二：手动升级

如果自动迁移不适用，可以手动升级：

1. **重新组织文件结构**
```
新的 v2.0 结构：
├── src/
│   ├── utils/           # 拆分后的工具模块
│   ├── logging/         # 专门的日志管理
│   ├── core/           # 核心分析功能
│   ├── io/             # 输入输出处理
│   └── analysis/       # 分析算法
├── tests/              # 测试文件
├── tools/              # 工具脚本
└── analysis_results/   # 分析结果
```

2. **更新导入语句**
```python
# v1.x
from src.utils import create_standard_logger, ValidationUtils

# v2.0
from src.logging import create_standard_logger
from src.utils import ValidationUtils
```

3. **更新配置文件**
检查并更新配置文件以使用新的格式。

## 🔧 API 变更

### 日志管理
```python
# v1.x
from src.utils import create_standard_logger

# v2.0
from src.logging import create_standard_logger, LoggerManager

# 新的高级用法
logger = LoggerManager.create_analysis_logger("analysis", output_dir)
```

### 文件操作
```python
# v1.x
from src.utils import FileUtils

# v2.0 - 基本用法不变，但有新功能
from src.utils import FileUtils

# 新增功能
FileUtils.safe_read_csv(filepath)
FileUtils.safe_remove(filepath)
```

### 数据验证
```python
# v1.x
from src.utils import ValidationUtils

# v2.0 - 基本用法不变
from src.utils import ValidationUtils

# 现在有更多验证方法
ValidationUtils.is_empty(data)  # 改进的空检查
```

## 🚨 已知的破坏性变更

### 1. 模块重组
- `src.utils` 中的一些类移动到了专门的子模块
- 日志相关功能移动到 `src.logging`

### 2. 异常处理
- 不再静默忽略异常，会提供详细的错误信息
- 某些错误现在会记录到日志文件

### 3. 文件路径
- 日志文件位置可能发生变化
- 某些临时文件的命名约定可能不同

## 🧪 验证升级

### 1. 运行测试套件
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/unit/
pytest tests/integration/
```

### 2. 烟雾测试
```bash
python tests/smoke_test.py
```

### 3. 功能验证
使用一个小的测试数据集运行完整的分析流程：
```bash
python main_abacus_analyzer.py --sample-ratio 0.1 --power-p -0.5 test_data/
```

## 🐛 常见问题

### Q: 升级后无法找到某些模块
A: 检查导入语句是否需要更新。参考上面的 API 变更部分。

### Q: 日志格式发生变化
A: v2.0 使用了统一的日志格式。如果需要自定义，使用 `LoggerManager` 的参数。

### Q: 性能是否受到影响？
A: v2.0 的核心算法没有变化，性能应该保持相同或略有改善。

### Q: 可以同时保留 v1.x 和 v2.0 吗？
A: 可以，建议将它们安装在不同的目录或虚拟环境中。

## 📞 获取帮助

如果在升级过程中遇到问题：

1. **检查迁移日志**：`migration.log` 和 `migration_report.json`
2. **运行诊断**：`python tests/smoke_test.py --verbose`
3. **查看文档**：README.md 和代码注释
4. **提交 Issue**：在项目仓库中报告问题

## 🔄 回滚计划

如果升级后遇到严重问题，可以回滚到 v1.x：

1. **恢复备份**
```bash
rm -rf /path/to/project
cp -r /path/to/backup/project_v1_backup /path/to/project
```

2. **切换到 v1.x 分支**
```bash
git checkout main  # 或之前的稳定分支
```

3. **恢复依赖**
```bash
pip install -r requirements.txt
```

## ✅ 升级检查清单

- [ ] 备份现有数据和配置
- [ ] 检查 Python 版本 (≥3.8)
- [ ] 获取 v2.0 代码
- [ ] 安装新依赖
- [ ] 运行迁移脚本
- [ ] 验证迁移结果
- [ ] 运行测试套件
- [ ] 执行功能验证
- [ ] 更新使用文档
- [ ] 培训团队成员（如适用）

---

**升级愉快！如有疑问，请查阅文档或联系维护团队。**
