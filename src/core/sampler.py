#!/usr/bin/env python

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from ..utils import MathUtils


class PowerMeanSampler:
    @staticmethod
    def select_frames(
        points: np.ndarray,
        k: int,
        nLdRMS_values: Optional[np.ndarray] = None,
        p: float = 0.5,
    ) -> Tuple[List[int], int, float]:
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0
        selected = PowerMeanSampler._initialize_seeds(points, k, nLdRMS_values)
        selected = PowerMeanSampler._incremental_selection(points, selected, k, p)
        selected, swap_count, improve_ratio = PowerMeanSampler._swap_optimization(
            points, selected, p
        )
        return selected, swap_count, improve_ratio

    @staticmethod
    def _initialize_seeds(
        points: np.ndarray, k: int, nLdRMS_values: Optional[np.ndarray] = None
    ) -> List[int]:
        n = len(points)
        if nLdRMS_values is not None and len(nLdRMS_values) == n:
            first_seed = np.argmax(nLdRMS_values)
        else:
            first_seed = 0
        selected = [first_seed]
        if k > 1 and n > 1:
            dists = cdist(points, points[[first_seed]])
            second_seed = np.argmax(dists)
            if second_seed != first_seed:
                selected.append(second_seed)
        return selected

    @staticmethod
    def _incremental_selection(
        points: np.ndarray, selected: List[int], k: int, p: float
    ) -> List[int]:
        n = len(points)
        remaining = set(range(n)) - set(selected)
        while len(selected) < k and remaining:
            current_points = points[selected]
            candidate_indices = list(remaining)
            candidate_points = points[candidate_indices]
            dists = cdist(candidate_points, current_points)
            dists = np.maximum(dists, 1e-12)
            agg_scores = []
            for i in range(len(candidate_indices)):
                candidate_dists = dists[i, :]
                agg_scores.append(MathUtils.power_mean(candidate_dists, p))
            best_idx = np.argmax(agg_scores)
            selected.append(candidate_indices[best_idx])
            remaining.remove(selected[-1])
        return selected

    @staticmethod
    def _swap_optimization(
        points: np.ndarray, selected: List[int], p: float
    ) -> Tuple[List[int], int, float]:
        if len(selected) < 2:
            return selected, 0, 0.0
        n = len(points)
        selected = selected.copy()
        selected_points = points[selected]
        sel_dists = cdist(selected_points, selected_points)
        np.fill_diagonal(sel_dists, np.inf)
        triu_idx = np.triu_indices_from(sel_dists, k=1)
        pair_dists = sel_dists[triu_idx]
        initial_obj = MathUtils.power_mean(pair_dists, p)
        current_obj = initial_obj
        swap_count = 0
        not_selected = list(set(range(n)) - set(selected))
        improved = True
        while improved:
            improved = False
            best_improvement = 0.0
            best_swap = None
            for i_idx, i in enumerate(selected):
                current_point = points[i : i + 1]
                other_indices = [s for j, s in enumerate(selected) if j != i_idx]
                other_points = points[other_indices]
                if not other_points.size:
                    continue
                dist_i = cdist(current_point, other_points).flatten()
                old_contrib = MathUtils.power_mean(dist_i, p)
                for j in not_selected:
                    dist_j = cdist(points[j : j + 1], other_points).flatten()
                    new_contrib = MathUtils.power_mean(dist_j, p)
                    improvement = new_contrib - old_contrib
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i_idx, j)
            if best_swap and best_improvement > 1e-12:
                i_idx, j = best_swap
                old_point = selected[i_idx]
                selected[i_idx] = j
                not_selected.remove(j)
                not_selected.append(old_point)
                current_obj += best_improvement
                swap_count += 1
                improved = True
        improve_ratio = (
            (current_obj - initial_obj) / initial_obj if initial_obj > 0 else 0.0
        )
        return selected, swap_count, improve_ratio


class RandomSampler:
    @staticmethod
    def select_frames(
        points: np.ndarray, k: int, **kwargs
    ) -> Tuple[List[int], int, float]:
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0
        selected = np.random.choice(n, k, replace=False).tolist()
        return selected, 0, 0.0


class UniformSampler:
    @staticmethod
    def select_frames(
        points: np.ndarray, k: int, **kwargs
    ) -> Tuple[List[int], int, float]:
        n = len(points)
        if k >= n:
            return list(range(n)), 0, 0.0
        indices = np.linspace(0, n - 1, k, dtype=int)
        selected = list(indices)
        return selected, 0, 0.0


class SamplingStrategy:
    POWER_MEAN = "power_mean"
    RANDOM = "random"
    UNIFORM = "uniform"
    GREEDY_MAX_DISTANCE = "greedy_max_distance"

    @staticmethod
    def uniform_sample_indices(n: int, k: int) -> np.ndarray:
        """Generate uniformly spaced indices for sampling.

        Args:
            n: Total number of items
            k: Number of samples to select

        Returns:
            Array of selected indices
        """
        if k >= n:
            return np.arange(n)
        return np.round(np.linspace(0, n-1, k)).astype(int)

    @staticmethod
    def create_sampler(strategy: str = POWER_MEAN, **kwargs):
        if strategy == SamplingStrategy.POWER_MEAN:
            return PowerMeanSampler()
        elif strategy == SamplingStrategy.RANDOM:
            return RandomSampler()
        elif strategy == SamplingStrategy.UNIFORM:
            return UniformSampler()
        elif strategy == SamplingStrategy.GREEDY_MAX_DISTANCE:
            return GreedyMaxDistanceSampler()
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")


class GreedyMaxDistanceSampler:
    """贪婪最大距离采样器（从trajectory_analyser迁移）"""

    @staticmethod
    def select_frames(
        points: np.ndarray,
        k: int,
        frame_nLdRMS_values: Optional[np.ndarray] = None,
        num_runs: int = 10,
        seed: int = 42,
        **kwargs,
    ) -> Tuple[List[int], int, float]:
        """
        使用贪婪策略选择最大化平均距离的帧子集

        Args:
                points: 帧向量矩阵
                k: 要选择的帧数
                frame_nLdRMS_values: 每帧的nLdRMS值（用于初始种子选择）
                num_runs: 运行次数
                seed: 随机种子

        Returns:
                选中的帧索引列表、交换次数、改进比例
        """
        n, _ = points.shape
        if k >= n:
            return list(range(n)), 0, 0.0

        best_sum = -np.inf
        best_indices = None

        for run in range(num_runs):
            np.random.seed(seed + run)

            # 初始化种子点
            if frame_nLdRMS_values is not None and len(frame_nLdRMS_values) == n:
                nLdRMS_array = np.array(frame_nLdRMS_values)
                min_idx = np.argmin(nLdRMS_array)
                max_idx = np.argmax(nLdRMS_array)
                if min_idx == max_idx:
                    idxs = np.random.choice(n, 2, replace=False).tolist()
                else:
                    idxs = [min_idx, max_idx]
            else:
                idxs = np.random.choice(n, min(2, k), replace=False).tolist()

            selected = set(idxs)
            dists_to_S = np.zeros(n)

            # 计算初始距离
            for i in range(n):
                if i not in selected:
                    dists_to_S[i] = np.sum(
                        cdist(points[list(selected)], points[i : i + 1])
                    )

            # 贪婪选择剩余点
            for _ in range(len(selected), k):
                candidates = [i for i in range(n) if i not in selected]
                if not candidates:
                    break
                next_idx = candidates[np.argmax(dists_to_S[candidates])]
                selected.add(next_idx)

                # 更新距离
                new_dists = cdist(points[next_idx : next_idx + 1], points).flatten()
                for i in range(n):
                    if i not in selected:
                        dists_to_S[i] += new_dists[i]
                    else:
                        dists_to_S[i] = -np.inf

            # 评估当前选择
            selected_list = list(selected)
            if len(selected_list) >= 2:
                subset_points = points[selected_list]
                pairwise_dists = cdist(subset_points, subset_points)
                dist_sum = np.sum(pairwise_dists) / 2

                if dist_sum > best_sum:
                    best_sum = dist_sum
                    best_indices = np.array(selected_list)

        if best_indices is None:
            best_indices = np.arange(min(k, n))

        return best_indices.tolist(), 0, 0.0


# Statistical analysis utilities (merged from math_utils.py)
def calculate_improvement(sample_val: float, baseline_mean: float, baseline_std: float = None) -> float:
    """Calculate percentage improvement over baseline.

    Args:
        sample_val: Sample value
        baseline_mean: Baseline mean value
        baseline_std: Baseline standard deviation (unused in current implementation)

    Returns:
        Improvement percentage
    """
    if np.isnan(sample_val) or np.isnan(baseline_mean) or baseline_mean == 0:
        return np.nan
    improvement = (sample_val - baseline_mean) / abs(baseline_mean) * 100
    return improvement


def calculate_significance(sample_val: float, baseline_vals: List[float]) -> float:
    """Calculate statistical significance using t-test.

    Args:
        sample_val: Sample value to test
        baseline_vals: List of baseline values

    Returns:
        p-value from t-test
    """
    if len(baseline_vals) < 2 or np.isnan(sample_val) or np.all(np.isnan(baseline_vals)):
        return np.nan
    # Filter out NaN values
    baseline_vals = [v for v in baseline_vals if not np.isnan(v)]
    if len(baseline_vals) < 2:
        return np.nan
    from scipy.stats import ttest_1samp
    try:
        t_stat, p_val = ttest_1samp(baseline_vals, sample_val)
        return p_val
    except:
        return np.nan
