#!/usr/bin/env python
"""Simple threaded task scheduler for deferred system analysis."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class AnalysisTask:
    system_path: str
    system_name: str
    pre_sampled_frames: Optional[List[int]] = None
    pre_stru_files: Optional[List[str]] = None
    reuse_sampling: bool = False
    enqueue_time: float = field(default_factory=time.time)
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = 'pending'  # pending/running/done/failed
    error: Optional[str] = None

    def mark_start(self):
        self.start_time = time.time()
        self.status = 'running'

    def mark_done(self):
        self.end_time = time.time()
        self.status = 'done'

    def mark_failed(self, err: str):
        self.end_time = time.time()
        self.status = 'failed'
        self.error = err


class TaskScheduler:
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.tasks: List[AnalysisTask] = []
        self.results: List[Any] = []
        self.failed: List[AnalysisTask] = []

    def add_task(self, task: AnalysisTask):
        self.tasks.append(task)

    def run(self, analyse_fn: Callable[[AnalysisTask], Any]) -> List[Any]:
        if not self.tasks:
            return []
        logger.info(f"TaskScheduler: 启动 {len(self.tasks)} 个任务，线程数={self.max_workers}")
        start_overall = time.time()
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for task in self.tasks:
                def _wrap(t: AnalysisTask):
                    def _call():
                        t.mark_start()
                        try:
                            result = analyse_fn(t)
                            t.mark_done()
                            return (t, result)
                        except Exception as e:  # noqa
                            t.mark_failed(str(e))
                            logger.warning(f"任务失败 {t.system_name}: {e}")
                            return (t, None)
                    return _call
                futures.append(executor.submit(_wrap(task)))

            completed = 0
            last_log = time.time()
            for fut in as_completed(futures):
                task, result = fut.result()
                completed += 1
                if result is not None:
                    self.results.append(result)
                else:
                    self.failed.append(task)
                now = time.time()
                if now - last_log >= 10 or completed == len(futures):
                    logger.info(
                        f"进度: {completed}/{len(futures)} 完成, 耗时 {now-start_overall:.1f}s, 失败 {len(self.failed)}"
                    )
                    last_log = now
        total = time.time() - start_overall
        logger.info(
            f"TaskScheduler: 全部完成 成功 {len(self.results)} 失败 {len(self.failed)} 总耗时 {total:.1f}s"
        )
        return self.results
