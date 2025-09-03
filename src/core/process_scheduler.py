#!/usr/bin/env python
"""Process based scheduler for per-system parallel analysis.

设计目标:
 - 体系为粒度, 动态工作窃取 (进程池自动调度)
 - 单体系内部 numpy / BLAS / OpenMP 限制为 1 线程, 避免超订阅
 - 复用轻量发现的 selected_files, 避免重复 I/O

Windows 下使用 spawn, 需在调用端放在 `if __name__ == "__main__":` 后。
"""

from __future__ import annotations
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 全局上下文用于进程间通信
class ProcessSchedulerContext:
    """ProcessScheduler的全局上下文"""
    _log_queue = None

    @classmethod
    def set_log_queue(cls, log_queue: Optional[mp.Queue]):
        """设置日志队列"""
        cls._log_queue = log_queue

    @classmethod
    def get_log_queue(cls) -> Optional[mp.Queue]:
        """获取日志队列"""
        return cls._log_queue

logger = logging.getLogger(__name__)


@dataclass
class ProcessAnalysisTask:
    system_path: str
    system_name: str
    pre_sampled_frames: Optional[List[int]]
    pre_stru_files: Optional[List[str]]


def _set_single_thread_env():
    # 通用环境变量限制线程数
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")


def _worker(task: ProcessAnalysisTask, analyser_params: Dict[str, Any]) -> Tuple[str, Any, float]:
    start = time.time()
    _set_single_thread_env()
    
    # Setup multiprocess logging for worker
    log_queue = ProcessSchedulerContext.get_log_queue()
    if log_queue is not None:
        from ..utils.logmanager import LoggerManager  # type: ignore
        LoggerManager.setup_worker_logger(
            name="WorkerProcess",
            queue=log_queue,
            level=logging.INFO,
            add_console=False
        )
    
    try:
        # 延迟导入（确保环境变量已生效）
        from .system_analyser import SystemAnalyser  # type: ignore
        analyser = SystemAnalyser(
            include_hydrogen=False,
            sample_ratio=analyser_params.get('sample_ratio', 0.1),
            power_p=analyser_params.get('power_p', 0.5),
            pca_variance_ratio=analyser_params.get('pca_variance_ratio', 0.90),
        )
        result = analyser.analyse_system(
            task.system_path,
            pre_sampled_frames=task.pre_sampled_frames,
            pre_stru_files=task.pre_stru_files,
        )
        return task.system_name, result, time.time() - start
    except Exception as e:  # noqa
        return task.system_name, (None, str(e)), time.time() - start


class ProcessScheduler:
    def __init__(self, max_workers: int, analyser_params: Dict[str, Any], log_queue: Optional[mp.Queue] = None):
        self.max_workers = max_workers
        self.analyser_params = analyser_params
        # 设置全局日志队列
        ProcessSchedulerContext.set_log_queue(log_queue)
        self.tasks: List[ProcessAnalysisTask] = []

    def add_task(self, task: ProcessAnalysisTask):
        self.tasks.append(task)

    def run(self) -> List[Any]:
        if not self.tasks:
            return []
        t0 = time.time()
        logger.info(f"ProcessScheduler: 提交 {len(self.tasks)} 个体系, 进程数={self.max_workers}")
        results: List[Any] = []
        failures = 0
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {pool.submit(_worker, task, self.analyser_params): task for task in self.tasks}
            completed = 0
            last_log = t0
            for fut in as_completed(future_map):
                completed += 1
                sys_name, result, dur = fut.result()
                if result is None or (isinstance(result, tuple) and result[0] is None):
                    failures += 1
                    logger.warning(f"失败 {sys_name} 用时 {dur:.2f}s")
                else:
                    results.append(result)
                now = time.time()
                if now - last_log >= 10 or completed == len(future_map):
                    logger.info(
                        f"进度 {completed}/{len(future_map)} 总耗时 {now-t0:.1f}s 失败 {failures}"
                    )
                    last_log = now
        logger.info(
            f"ProcessScheduler 完成: 成功 {len(results)} 失败 {failures} 总耗时 {time.time()-t0:.1f}s"
        )
        return results
