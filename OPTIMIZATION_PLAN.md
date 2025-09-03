# 代码优化计划 - 提高效率，降低冗余逻辑和过严格验证

## 性能瓶颈分析（基于当前运行日志）

### 核心问题识别
从运行日志分析，"进程模式任务: 3989" 后半小时无输出的根本原因是：

1. **海量文件IO**: 3989个系统 × ~1000帧/系统 = 400万个STRU文件读取
2. **O(n²)距离计算**: 每个系统计算原子间距离，复杂度极高
3. **缺乏进度反馈**: ProcessScheduler仅每10秒输出一次进度
4. **内存压力**: 同时处理多个大系统的数据

### 优化优先级

## 🚀 Phase 1: 快速优化（立即见效）

### 1. 减少文件验证开销
**当前问题**: 每个STRU文件都进行完整性验证
```python
# 当前代码在stru_parser.py
def parse_file(self, stru_file: str):
    try:
        with open(stru_file, encoding="utf-8") as f:
            lines = f.readlines()  # 读取整个文件
        # 然后进行详细解析和验证...
```

**优化方案**:
```python
# 优化后：轻量级预检 + 延迟加载
def parse_file_lightweight(self, stru_file: str):
    # 只检查文件基本信息，不读取内容
    if not os.path.exists(stru_file):
        return None
    stat = os.stat(stru_file)
    if stat.st_size < 100:  # 太小的文件可能是损坏的
        return None
    return stru_file  # 返回文件路径，延迟解析
```

### 2. 优化距离计算算法
**当前问题**: 完整O(n²)距离计算
```python
# 当前代码在system_analyser.py
for frame in frames:
    dist_vec = MetricCalculator.calculate_distance_vectors(frame.positions)
    # 对所有原子对计算距离
```

**优化方案**:
```python
# 优化1: 采样计算（减少计算量90%）
def calculate_distance_vectors_sampled(positions, sample_ratio=0.1):
    n_atoms = len(positions)
    sample_size = max(100, int(n_atoms * sample_ratio))
    indices = np.random.choice(n_atoms, sample_size, replace=False)
    sampled_positions = positions[indices]
    # 只计算采样原子的距离
    return calculate_distance_vectors(sampled_positions)

# 优化2: 近似计算（使用kd-tree等空间数据结构）
def calculate_distance_vectors_approximate(positions, max_distance=10.0):
    # 只计算距离小于阈值的原子对
    # 使用空间索引加速
    pass
```

### 3. 增加细粒度进度反馈
**当前问题**: 仅每10秒输出一次进度
```python
# 当前ProcessScheduler
if now - last_log >= 10 or completed == len(future_map):
    logger.info(f"进度 {completed}/{len(future_map)}")
```

**优化方案**:
```python
# 优化后：每处理完一个系统就输出
def _worker(task, analyser_params):
    start = time.time()
    try:
        # ... 处理逻辑 ...
        result = analyser.analyse_system(task.system_path)
        # 新增：立即输出完成信息
        logger.info(f"✓ 完成系统 {task.system_name} ({time.time()-start:.1f}s)")
        return task.system_name, result, time.time() - start
    except Exception as e:
        logger.error(f"✗ 失败系统 {task.system_name}: {e}")
        return task.system_name, (None, str(e)), time.time() - start
```

### 4. 内存优化
**当前问题**: 同时加载所有帧数据
```python
# 当前代码
frames = self.parser.parse_trajectory(stru_dir)  # 加载所有帧
for frame in frames:  # 处理所有帧
    # 计算距离向量等
```

**优化方案**:
```python
# 优化后：分批处理
def parse_trajectory_batched(self, stru_dir, batch_size=100):
    stru_files = glob.glob(os.path.join(stru_dir, "STRU_MD_*"))
    for i in range(0, len(stru_files), batch_size):
        batch_files = stru_files[i:i+batch_size]
        batch_frames = []
        for file in batch_files:
            frame = self.parse_file(file)
            if frame:
                batch_frames.append(frame)
        yield batch_frames  # 返回批次，处理完后释放内存
```

## 📊 Phase 2: 核心算法优化

### 1. 实现计算结果缓存
```python
class ComputationCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, system_path, operation):
        # 生成基于文件修改时间的缓存键
        stat = os.stat(system_path)
        return f"{operation}_{stat.st_mtime}_{hash(system_path)}"

    def load_cached_result(self, key):
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def save_cached_result(self, key, result):
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
```

### 2. 并行计算优化
```python
# 在系统内部也使用并行
def analyse_system_parallel(self, system_dir):
    frames = self.parser.parse_trajectory(system_dir)

    # 并行计算距离向量
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        distance_futures = [
            executor.submit(MetricCalculator.calculate_distance_vectors, frame.positions)
            for frame in frames
        ]
        distance_vectors = [f.result() for f in distance_futures]

    # 并行计算RMSD
    with ThreadPoolExecutor(max_workers=4) as executor:
        rmsd_futures = [
            executor.submit(self._calculate_rmsd_single, frame, mean_structure)
            for frame in frames
        ]
        rmsd_values = [f.result() for f in rmsd_futures]

    return self._build_metrics(distance_vectors, rmsd_values)
```

## 🔧 Phase 3: 架构级优化

### 1. 分布式处理框架
- 使用Dask或Ray实现分布式计算
- 支持跨机器扩展

### 2. 增量计算机制
- 只重新计算变更的数据
- 实现智能的依赖跟踪

### 3. 自适应算法选择
- 根据系统规模自动选择最优算法
- 实现算法A/B测试框架

## 📈 预期性能提升

| 优化阶段 | 预期提速 | 主要改进 |
|---------|---------|---------|
| Phase 1 | 2-3倍 | 减少IO验证，优化算法，增加进度反馈 |
| Phase 2 | 5-10倍 | 缓存机制，并行计算，内存优化 |
| Phase 3 | 10-50倍 | 分布式处理，增量计算，智能调度 |

## 🎯 实施建议

### 立即开始
1. 实现轻量级文件预检
2. 增加细粒度进度日志
3. 添加内存使用监控

### 短期目标（1周内）
1. 实现计算结果缓存
2. 优化距离计算算法
3. 分批处理大系统

### 长期规划（1月内）
1. 分布式处理框架
2. 增量计算机制
3. 性能监控仪表板

## 📋 验证方法

### 性能基准测试
```python
def benchmark_system(system_path):
    start_time = time.time()

    # 测试不同配置的性能
    configs = [
        {"validation_level": "full", "algorithm": "exact"},
        {"validation_level": "light", "algorithm": "approximate"},
        {"validation_level": "minimal", "algorithm": "sampled"}
    ]

    results = {}
    for config in configs:
        analyser = SystemAnalyser(**config)
        result = analyser.analyse_system(system_path)
        results[str(config)] = time.time() - start_time

    return results
```

### 准确性验证
- 对比优化前后结果的数值差异
- 确保优化不影响科学计算的准确性
- 建立回归测试套件
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

# 使用示例：
@handle_exceptions(logger=self.logger, default_return=[])
def load_data(self, filepath: str) -> List[dict]:
    return FileUtils.safe_read_json(filepath)
```

## 3. 数据结构优化（提高内存和计算效率）

### 当前问题：
- 频繁使用 list/dict，内存效率低
- 大量循环操作
- 数据转换开销大

### 优化方案：
```python
# 使用更高效的数据结构
from collections import defaultdict, deque
from typing import Deque

# 示例：使用 deque 替代 list 进行频繁的 append/popleft 操作
class EfficientQueue:
    def __init__(self):
        self._queue: Deque = deque()

    def add_task(self, task):
        self._queue.append(task)

    def get_next_task(self):
        return self._queue.popleft() if self._queue else None

# 使用 numpy 数组替代 Python list 进行数值计算
import numpy as np

# 示例：批量处理数值数据
def process_batch_data(data_list: List[List[float]]) -> np.ndarray:
    """将数据转换为 numpy 数组进行高效处理"""
    return np.array(data_list, dtype=np.float32)
```

## 4. 缓存机制（减少重复计算）

### 当前问题：
- 重复计算相同的结果
- 文件读取没有缓存
- 系统信息重复查询

### 优化方案：
```python
# 在 src/utils/common.py 中添加缓存装饰器
from functools import lru_cache
import time

class CachedFileReader:
    _cache = {}
    _cache_timeout = 300  # 5分钟缓存

    @classmethod
    @lru_cache(maxsize=128)
    def read_json_cached(cls, filepath: str) -> Optional[dict]:
        """带缓存的 JSON 文件读取"""
        cache_key = filepath
        if cache_key in cls._cache:
            cached_time, data = cls._cache[cache_key]
            if time.time() - cached_time < cls._cache_timeout:
                return data

        data = FileUtils.safe_read_json(filepath)
        if data is not None:
            cls._cache[cache_key] = (time.time(), data)
        return data

# 系统信息缓存
@lru_cache(maxsize=100)
def get_system_info_cached(system_path: str) -> dict:
    """缓存系统信息，避免重复解析"""
    return parse_system_info(system_path)
```

## 5. 简化验证逻辑（减少过严格检查）

### 当前问题：
- 过多不必要的文件存在性检查
- 重复的数据验证
- 性能影响的调试验证

### 优化方案：
```python
# 在 src/utils/common.py 中添加条件验证
class ValidationLevel:
    STRICT = "strict"  # 严格验证（开发/调试）
    NORMAL = "normal"  # 正常验证（生产）
    MINIMAL = "minimal"  # 最小验证（高性能）

_validation_level = ValidationLevel.NORMAL

def set_validation_level(level: str):
    global _validation_level
    _validation_level = level

def should_validate(level: str) -> bool:
    """根据当前验证级别决定是否执行验证"""
    levels = {
        ValidationLevel.STRICT: ["strict", "normal", "minimal"],
        ValidationLevel.NORMAL: ["normal", "minimal"],
        ValidationLevel.MINIMAL: ["minimal"]
    }
    return level in levels.get(_validation_level, [])

# 使用示例：
def load_system_data(filepath: str) -> dict:
    if should_validate("file_exists"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

    # 核心加载逻辑
    return FileUtils.safe_read_json(filepath)
```

## 6. 批量操作优化（减少IO次数）

### 当前问题：
- 频繁的小文件写入
- 单条记录处理
- 网络/磁盘IO开销大

### 优化方案：
```python
# 在 src/io/result_saver.py 中添加批量写入
class BatchWriter:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.buffer = []
        self.output_dir = None

    def add_record(self, record: dict):
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        # 批量写入逻辑
        try:
            # 将 buffer 中的记录批量写入文件
            FileUtils.safe_write_csv(
                os.path.join(self.output_dir, "batch_data.csv"),
                self.buffer,
                headers=["col1", "col2", "col3"]
            )
            self.buffer.clear()
        except Exception as e:
            logger.error(f"批量写入失败: {e}")

# 使用示例：
batch_writer = BatchWriter(batch_size=50)
for record in records:
    batch_writer.add_record(record)
batch_writer.flush()  # 确保最后一批数据被写入
```

## 7. 内存管理优化

### 当前问题：
- 大数据集处理时内存使用过高
- 临时对象没有及时释放
- 内存泄漏风险

### 优化方案：
```python
# 使用生成器替代列表
def process_large_dataset(filepaths: List[str]):
    """使用生成器处理大文件，避免一次性加载所有数据"""
    for filepath in filepaths:
        data = FileUtils.safe_read_json(filepath)
        if data:
            yield data
        # 及时释放内存
        del data

# 内存池管理
import gc
from contextlib import contextmanager

@contextmanager
def memory_context():
    """内存管理上下文"""
    try:
        yield
    finally:
        gc.collect()  # 强制垃圾回收

# 使用示例：
with memory_context():
    for data in process_large_dataset(filepaths):
        process_data(data)
```

## 8. 性能监控和分析

### 当前问题：
- 缺乏性能监控
- 难以识别瓶颈
- 优化效果无法量化

### 优化方案：
```python
# 在 src/utils/common.py 中添加性能监控
import time
from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger = logging.getLogger(func.__module__)
        logger.info(f"函数 {func.__name__} 执行时间: {duration:.4f}秒")
        return result
    return wrapper

# 使用示例：
@timed
def heavy_computation(self):
    # 耗时操作
    pass

# 性能分析工具
def profile_function(func):
    """使用 cProfile 分析函数性能"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        from io import StringIO

        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()

        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        logger.info(f"性能分析结果 for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper
```

## 实施计划

### Phase 1: 基础优化（1-2周）
1. 实现统一日志装饰器
2. 添加异常处理装饰器
3. 优化主要数据结构

### Phase 2: 性能优化（2-3周）
1. 实现缓存机制
2. 添加批量操作
3. 优化内存管理

### Phase 3: 验证简化（1周）
1. 实现条件验证系统
2. 简化不必要的检查
3. 添加性能监控

### Phase 4: 测试和调优（1-2周）
1. 性能测试
2. 内存分析
3. 最终优化

## 预期效果

- **效率提升**: 20-50% 的执行时间减少
- **内存优化**: 30-60% 的内存使用减少
- **代码简化**: 减少 40% 的重复代码
- **维护性**: 提高代码可读性和可维护性
