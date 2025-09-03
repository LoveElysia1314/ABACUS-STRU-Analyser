"""
并行执行通用工具
支持进程池/线程池任务分发、日志队列、异常处理、进度显示等
"""
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Tuple


def setup_multiprocess_logging(output_dir: str, log_filename: str = "main.log", when: str = "D", backup_count: int = 14):
    """创建多进程日志队列和监听器，返回(log_queue, log_listener)"""
    from src.utils.logmanager import LoggerManager
    log_queue, log_listener = LoggerManager.create_multiprocess_logging_setup(
        output_dir=output_dir,
        log_filename=log_filename,
        when=when,
        backup_count=backup_count
    )
    log_listener.start()
    return log_queue, log_listener


def stop_multiprocess_logging(log_listener):
    """关闭多进程日志监听器"""
    from src.utils.logmanager import LoggerManager
    LoggerManager.stop_listener(log_listener)


def run_parallel_tasks(
    tasks: List[Any],
    worker_fn: Callable,
    workers: int = 1,
    mode: str = "process",
    initializer: Optional[Callable] = None,
    initargs: Optional[Tuple] = None,
    log_queue: Any = None,
    logger: Optional[logging.Logger] = None,
    desc: str = "任务",
    chunksize: int = 1
) -> List[Any]:
    """
    通用并行任务分发与收集
    支持mode=process/thread，自动处理异常与进度日志
    """
    results = []
    total = len(tasks)
    completed = 0
    failures = 0
    pool_cls = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
    # 自动并行：workers=-1 时用 CPU 核心数
    if workers == -1:
        try:
            workers = mp.cpu_count()
        except NotImplementedError:
            workers = 1
    if workers < 1:
        workers = 1
    pool_kwargs = {"max_workers": workers}
    if mode == "process" and initializer:
        pool_kwargs["initializer"] = initializer
        if initargs:
            pool_kwargs["initargs"] = initargs
    
    with pool_cls(**pool_kwargs) as pool:
        future_to_task = {}
        for task in tasks:
            if mode == "process":
                future = pool.submit(worker_fn, task)
            else:
                future = pool.submit(worker_fn, task)
            future_to_task[future] = task
        for future in as_completed(future_to_task):
            completed += 1
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                if logger:
                    logger.info(f"({completed}/{total}) {desc} 完成")
            except Exception as e:
                failures += 1
                if logger:
                    logger.error(f"({completed}/{total}) {desc} 失败: {e}")
    if logger:
        logger.info(f"{desc} 并行调度完成: 成功 {len(results)} 失败 {failures}")
    return results
