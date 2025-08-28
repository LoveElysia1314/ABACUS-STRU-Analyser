#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist
from ..utils import MathUtils


class PowerMeanSampler:
	@staticmethod
	def select_frames(points: np.ndarray, k: int, 
					 nLdRMS_values: Optional[np.ndarray] = None, 
					 p: float = 0.5) -> Tuple[List[int], int, float]:
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
	def _initialize_seeds(points: np.ndarray, k: int, 
						 nLdRMS_values: Optional[np.ndarray] = None) -> List[int]:
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
	def _incremental_selection(points: np.ndarray, selected: List[int], 
							  k: int, p: float) -> List[int]:
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
	def _swap_optimization(points: np.ndarray, selected: List[int], 
						  p: float) -> Tuple[List[int], int, float]:
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
				current_point = points[i:i+1]
				other_indices = [s for j, s in enumerate(selected) if j != i_idx]
				other_points = points[other_indices]
				if not other_points.size:
					continue
				dist_i = cdist(current_point, other_points).flatten()
				old_contrib = MathUtils.power_mean(dist_i, p)
				for j in not_selected:
					dist_j = cdist(points[j:j+1], other_points).flatten()
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
		improve_ratio = (current_obj - initial_obj) / initial_obj if initial_obj > 0 else 0.0
		return selected, swap_count, improve_ratio


class RandomSampler:
	@staticmethod
	def select_frames(points: np.ndarray, k: int, **kwargs) -> Tuple[List[int], int, float]:
		n = len(points)
		if k >= n:
			return list(range(n)), 0, 0.0
		selected = np.random.choice(n, k, replace=False).tolist()
		return selected, 0, 0.0


class UniformSampler:
	@staticmethod
	def select_frames(points: np.ndarray, k: int, **kwargs) -> Tuple[List[int], int, float]:
		n = len(points)
		if k >= n:
			return list(range(n)), 0, 0.0
		indices = np.linspace(0, n-1, k, dtype=int)
		selected = list(indices)
		return selected, 0, 0.0


class SamplingStrategy:
	POWER_MEAN = "power_mean"
	RANDOM = "random"
	UNIFORM = "uniform"
	@staticmethod
	def create_sampler(strategy: str = POWER_MEAN, **kwargs):
		if strategy == SamplingStrategy.POWER_MEAN:
			return PowerMeanSampler()
		elif strategy == SamplingStrategy.RANDOM:
			return RandomSampler()
		elif strategy == SamplingStrategy.UNIFORM:
			return UniformSampler()
		else:
			raise ValueError(f"Unknown sampling strategy: {strategy}")
