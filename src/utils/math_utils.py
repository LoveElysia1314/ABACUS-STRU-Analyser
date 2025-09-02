#!/usr/bin/env python
"""Mathematical utility functions for sampling and analysis."""

import numpy as np
from typing import List


class SamplingUtils:
    """Utilities for sampling operations."""

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


class StatisticalUtils:
    """Statistical analysis utilities."""

    @staticmethod
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

    @staticmethod
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


# Convenience functions for backward compatibility
uniform_sample_indices = SamplingUtils.uniform_sample_indices
calc_improvement = StatisticalUtils.calculate_improvement
calc_significance = StatisticalUtils.calculate_significance
