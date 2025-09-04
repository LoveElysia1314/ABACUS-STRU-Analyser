import os
import json
import csv
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from ...core.metrics import MetricsToolkit
from ...core.sampler import SamplingStrategy
from ...core.system_analyser import RMSDCalculator
from ...utils import FileUtils


ORDERED_METRICS = ["RMSD_Mean", "ANND", "MPD", "Coverage_Ratio", "Energy_Range", "JS_Divergence"]


class _Welford:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta * 1.0 / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            return self.mean if self.n == 1 else np.nan, np.nan
        return self.mean, (self.M2 / (self.n - 1)) ** 0.5


class StreamingSamplingComparisonManager:
    """流式维护采样方法统计（不再输出 sampling_compare_enhanced.csv）。"""

    def __init__(self, result_dir: str, logger: Optional[logging.Logger] = None):
        self.result_dir = result_dir
        self.logger = logger or logging.getLogger(__name__)
        # 新旧目录兼容与迁移
        legacy_single = os.path.join(result_dir, 'single_analysis_results')
        new_single = os.path.join(result_dir, 'single_analysis')
        if os.path.isdir(legacy_single) and not os.path.isdir(new_single):
            # 不做移动，仅创建新目录用于后续写入
            pass
        self.single_dir = new_single
        FileUtils.ensure_dir(self.single_dir)
        # 采样比较缓存目录
        self.sampling_dir = os.path.join(result_dir, 'sampling_comparison')
        FileUtils.ensure_dir(self.sampling_dir)
    # 状态文件
        self.state_path = os.path.join(self.sampling_dir, 'sampling_methods_state.json')
        # 最终汇总文件直接放在 run_* 根目录
        self.final_summary_path = os.path.join(result_dir, 'sampling_methods_comparison.csv')
        self._init_state()

    def _init_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
                return
            except Exception:
                pass
        # state 结构：{ metric: { sampled: welford_data, random: welford_data, uniform: welford_data } }
        self.state = {m: {"sampled": {"n":0,"mean":0.0,"M2":0.0},
                          "random": {"n":0,"mean":0.0,"M2":0.0},
                          "uniform": {"n":0,"mean":0.0,"M2":0.0}} for m in ORDERED_METRICS}
        self._persist_state()

    def _persist_state(self):
        tmp = self.state_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    def _welford_update_dict(self, slot: Dict[str, Any], value: float):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        n = slot['n'] + 1
        delta = value - slot['mean']
        mean = slot['mean'] + delta / n
        delta2 = value - mean
        M2 = slot['M2'] + delta * delta2
        slot['n'] = n
        slot['mean'] = mean
        slot['M2'] = M2

    def update_per_system(self, system_name: str, system_path: str, vectors: np.ndarray,
                          sampled_mask: np.ndarray, frame_ids: np.ndarray) -> None:
        try:
            k = sampled_mask.sum()
            n = len(vectors)
            if k == 0 or n == 0:
                return
            sampled_vectors = vectors[sampled_mask]
            sampled_indices = frame_ids[sampled_mask]

            # 采样组 RMSD
            sampled_rmsd = self._calculate_group_rmsd(system_path, sampled_indices.tolist())
            sampled_metrics = MetricsToolkit.adapt_sampling_metrics(
                sampled_vectors, vectors, sampled_rmsd if len(sampled_rmsd)>0 else []
            )

            # 随机 10 次
            rand_results = []
            for _ in range(10):
                idx = np.random.choice(n, k, replace=False)
                sel_vectors = vectors[idx]
                rand_rmsd = self._calculate_group_rmsd(system_path, frame_ids[idx].tolist())
                sel_metrics = MetricsToolkit.adapt_sampling_metrics(
                    sel_vectors, vectors, rand_rmsd if len(rand_rmsd)>0 else []
                )
                rand_results.append(sel_metrics)

            # 均匀
            idx_uniform = SamplingStrategy.uniform_sample_indices(n, k)
            uni_vectors = vectors[idx_uniform]
            uni_rmsd = self._calculate_group_rmsd(system_path, frame_ids[idx_uniform].tolist())
            uniform_metrics = MetricsToolkit.adapt_sampling_metrics(
                uni_vectors, vectors, uni_rmsd if len(uni_rmsd)>0 else []
            )

            # 汇总随机均值
            def rand_collect(name):
                vals = [r.get(name) for r in rand_results if r.get(name) is not None]
                return float(np.mean(vals)) if vals else float('nan')

            row = {
                'System': system_name,
                'Sample_Ratio': k / n,
                'Total_Frames': n,
                'Sampled_Frames': k,
            }
            for metric in ORDERED_METRICS:
                sampled_key = metric.replace('RMSD_Mean','RMSD').replace('Coverage_Ratio','Coverage').replace('JS_Divergence','JS')  # for random/uniform naming alignment later if needed
                # 保持与原增强 CSV 格式接近（使用 *_sampled / _random_mean / _uniform 后缀）
                sampled_val = sampled_metrics.get(metric)
                random_mean_val = rand_collect(metric)
                uniform_val = uniform_metrics.get(metric)
                row[f'{metric}_sampled'] = sampled_val
                row[f'{metric.split("_Mean")[0]}_random_mean' if metric=='RMSD_Mean' else f'{metric.split("_Ratio")[0] if metric.endswith("_Ratio") else metric.split("_Divergence")[0]}_random_mean'] = random_mean_val
                row[f'{metric}_uniform'] = uniform_val

                # 更新状态
                self._welford_update_dict(self.state[metric]['sampled'], sampled_val)
                self._welford_update_dict(self.state[metric]['random'], random_mean_val)
                self._welford_update_dict(self.state[metric]['uniform'], uniform_val)

            # 持久化 state（不再写增强CSV）
            self._persist_state()
        except Exception as e:
            self.logger.warning(f"流式采样比较更新失败 {system_name}: {e}")

    def finalize(self):
        try:
            rows = []
            for metric in ORDERED_METRICS:
                def finalize_slot(name):
                    slot = self.state[metric][name]
                    n = slot['n']
                    if n < 1:
                        return np.nan, np.nan
                    mean = slot['mean']
                    std = (slot['M2'] / (n - 1)) ** 0.5 if n > 1 else np.nan
                    return mean, std
                s_mean, s_std = finalize_slot('sampled')
                r_mean, r_std = finalize_slot('random')
                u_mean, u_std = finalize_slot('uniform')
                rows.append({
                    'Metric': metric,
                    'Sampled_Mean': s_mean,
                    'Sampled_Std': s_std,
                    'Random_Mean': r_mean,
                    'Random_Std': r_std,
                    'Uniform_Mean': u_mean,
                    'Uniform_Std': u_std,
                })
            with open(self.final_summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            self.logger.info(f"流式采样方法对比汇总已生成: {self.final_summary_path}")
        except Exception as e:
            self.logger.warning(f"流式采样方法对比汇总生成失败: {e}")

    # ---- Helper ----
    def _calculate_group_rmsd(self, system_path: str, frame_indices: List[int]):
        return RMSDCalculator.calculate_group_rmsd(system_path, frame_indices, self.logger)