#!/usr/bin/env python
"""Unified metrics calculation utilities.

This module consolidates scattered metric computations (ANND, MPD,
RMSD summary, diversity metrics, distribution similarity) into a single
reusable toolkit to reduce duplication across scripts and analysis modules.

Phase 1 scope (2025-09-01):
- Migrated sampling_compare_demo.py local metric functions here
- Reused existing MetricCalculator where possible

Planned Phase 2:
- Refactor core.metrics.MetricCalculator to delegate to this toolkit fully
- Deduplicate any overlapping distance computations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, List, Any
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, wasserstein_distance
from sklearn.decomposition import PCA

from . import ValidationUtils
# 注意：避免循环导入，不再从 core.metrics 导入 MetricCalculator。
# 若需要核心额外辅助函数，请在运行时延迟导入或在此工具内部重写。


@dataclass
class BasicDistanceMetrics:
    ANND: float
    MPD: float


@dataclass
class DiversityMetrics:
    coverage_ratio: float
    pca_variance_ratio: float
    energy_range: float


@dataclass
class DistributionSimilarity:
    js_divergence: float


@dataclass
class RMSDSummary:
    rmsd_mean: float
    rmsd_std: float
    rmsd_min: float
    rmsd_max: float


class MetricsToolkit:
    """Unified static methods for metric calculations."""

    # ---- Basic distance metrics ----
    @staticmethod
    def compute_basic_distance_metrics(vectors: np.ndarray) -> BasicDistanceMetrics:
        if vectors is None or len(vectors) < 2:
            return BasicDistanceMetrics(np.nan, np.nan)
        try:
            # Reuse logic similar to sampling_compare_demo but centralized
            dists = squareform(pdist(vectors, metric="euclidean"))
            np.fill_diagonal(dists, np.inf)
            valid = dists[dists != np.inf]
            if len(valid) == 0:
                return BasicDistanceMetrics(np.nan, np.nan)
            annd = float(np.mean(np.min(dists, axis=1))) if np.any(np.isfinite(np.min(dists, axis=1))) else np.nan
            mpd = float(np.mean(valid)) if len(valid) else np.nan
            return BasicDistanceMetrics(annd, mpd)
        except Exception:
            return BasicDistanceMetrics(np.nan, np.nan)

    # ---- Diversity metrics ----
    @staticmethod
    def compute_diversity_metrics(vectors: np.ndarray) -> DiversityMetrics:
        if vectors is None or len(vectors) < 2:
            return DiversityMetrics(np.nan, np.nan, np.nan)
        # Drop rows with any NaN
        if np.any(np.isnan(vectors)):
            mask = ~np.any(np.isnan(vectors), axis=1)
            vectors = vectors[mask]
            if len(vectors) < 2:
                return DiversityMetrics(np.nan, np.nan, np.nan)
        try:
            # PCA coverage
            try:
                pca = PCA(n_components=min(3, vectors.shape[1]))
                pca_result = pca.fit_transform(vectors)
                explained_variance = float(np.sum(pca.explained_variance_ratio_))
                coverage_ratio = explained_variance
                pca_variance_ratio = explained_variance
            except Exception:
                coverage_ratio = np.nan
                pca_variance_ratio = np.nan
            # Energy range (assume col0 energy if present)
            energy_range = np.nan
            if vectors.shape[1] > 0:
                energy_col = vectors[:, 0]
                if not np.all(np.isnan(energy_col)):
                    energy_range = float(np.ptp(energy_col[~np.isnan(energy_col)]))
            return DiversityMetrics(coverage_ratio, pca_variance_ratio, energy_range)
        except Exception:
            return DiversityMetrics(np.nan, np.nan, np.nan)

    # ---- Distribution similarity ----
    @staticmethod
    def compute_distribution_similarity(sample_vectors: np.ndarray, full_vectors: np.ndarray) -> DistributionSimilarity:
        if sample_vectors is None or full_vectors is None or len(sample_vectors) < 2 or len(full_vectors) < 2:
            return DistributionSimilarity(np.nan)
        # Clean NaNs
        if np.any(np.isnan(sample_vectors)):
            sample_vectors = sample_vectors[~np.any(np.isnan(sample_vectors), axis=1)]
        if np.any(np.isnan(full_vectors)):
            full_vectors = full_vectors[~np.any(np.isnan(full_vectors), axis=1)]
        if len(sample_vectors) < 2 or len(full_vectors) < 2:
            return DistributionSimilarity(np.nan)
        try:
            pca = PCA(n_components=min(3, sample_vectors.shape[1]))
            sample_pca = pca.fit_transform(sample_vectors)
            full_pca = pca.transform(full_vectors)
            js_components = []
            for i in range(sample_pca.shape[1]):
                s_hist, _ = np.histogram(sample_pca[:, i], bins=20, density=True)
                f_hist, _ = np.histogram(full_pca[:, i], bins=20, density=True)
                s_hist = s_hist / (np.sum(s_hist) + 1e-10)
                f_hist = f_hist / (np.sum(f_hist) + 1e-10)
                m = 0.5 * (s_hist + f_hist)
                js = 0.5 * (entropy(s_hist + 1e-10, m + 1e-10) + entropy(f_hist + 1e-10, m + 1e-10))
                js_components.append(js)
            js_divergence = float(np.mean(js_components))
            return DistributionSimilarity(js_divergence)
        except Exception:
            return DistributionSimilarity(np.nan)

    # ---- RMSD summary ----
    @staticmethod
    def summarize_rmsd(values: Sequence[float]) -> RMSDSummary:
        arr = np.array(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return RMSDSummary(np.nan, np.nan, np.nan, np.nan)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return RMSDSummary(np.nan, np.nan, np.nan, np.nan)
        return RMSDSummary(
            float(np.mean(valid)),
            float(np.std(valid)) if valid.size > 1 else np.nan,
            float(np.min(valid)),
            float(np.max(valid)),
        )

    # ---- Adapters (Level 4) ----
    @staticmethod
    def adapt_frame_metrics(vectors_per_frame: List[np.ndarray]) -> List[BasicDistanceMetrics]:
        """Compute basic metrics for each frame's internal coordinate vectors.

        vectors_per_frame: list of (n_points, dim) arrays or flattened vector sets.
        Returns list with NaN metrics where insufficient data.
        """
        results: List[BasicDistanceMetrics] = []
        for vectors in vectors_per_frame:
            try:
                results.append(MetricsToolkit.compute_basic_distance_metrics(vectors))
            except Exception:
                results.append(BasicDistanceMetrics(np.nan, np.nan))
        return results

    @staticmethod
    def adapt_distribution_metrics(vectors: np.ndarray) -> Dict[str, Any]:
        """Aggregate distribution-level metrics (basic + diversity)."""
        basic = MetricsToolkit.compute_basic_distance_metrics(vectors)
        diversity = MetricsToolkit.compute_diversity_metrics(vectors)
        return {
            "ANND": basic.ANND,
            "MPD": basic.MPD,
            "coverage_ratio": diversity.coverage_ratio,
            "pca_variance_ratio": diversity.pca_variance_ratio,
            "energy_range": diversity.energy_range,
            "num_frames": int(vectors.shape[0]) if hasattr(vectors, 'shape') else 0,
            "dimension": int(vectors.shape[1]) if hasattr(vectors, 'shape') and vectors.ndim == 2 else 0,
        }
    @staticmethod
    def adapt_sampling_metrics(selected_vectors: np.ndarray, full_vectors: np.ndarray, rmsd_values: np.ndarray | List[float] | None = None) -> Dict[str, Any]:
        """Convenience wrapper for sampling comparison.

        Returns a unified dict with canonical keys aligned to system summary headers.
        """
        try:
            basic = MetricsToolkit.compute_basic_distance_metrics(selected_vectors)
            diversity = MetricsToolkit.compute_diversity_metrics(selected_vectors)
            similarity = MetricsToolkit.compute_distribution_similarity(selected_vectors, full_vectors)
            rmsd_summary = MetricsToolkit.summarize_rmsd(rmsd_values if rmsd_values is not None else [])
            return {
                "ANND": basic.ANND,
                "MPD": basic.MPD,
                "Coverage_Ratio": diversity.coverage_ratio,
                "Energy_Range": diversity.energy_range,
                "JS_Divergence": similarity.js_divergence,
                "RMSD_Mean": rmsd_summary.rmsd_mean,
            }
        except Exception:
            return {}

    # ---- Wrapper methods for legacy compatibility ----
    @staticmethod
    def wrap_diversity(vectors: np.ndarray) -> Dict[str, float]:
        """Legacy wrapper for diversity metrics."""
        m = MetricsToolkit.compute_diversity_metrics(vectors)
        return {
            'coverage_ratio': m.coverage_ratio,
            'pca_variance_ratio': m.pca_variance_ratio,
            'energy_range': m.energy_range,
        }

    @staticmethod
    def wrap_rmsd(values: Sequence[float]) -> Dict[str, float]:
        """Legacy wrapper for RMSD summary."""
        s = MetricsToolkit.summarize_rmsd(values)
        return {
            'rmsd_mean': s.rmsd_mean,
            'rmsd_std': s.rmsd_std,
            'rmsd_min': s.rmsd_min,
            'rmsd_max': s.rmsd_max,
        }

    @staticmethod
    def wrap_similarity(sample_vectors: np.ndarray, full_vectors: np.ndarray) -> Dict[str, float]:
        """Legacy wrapper for distribution similarity."""
        m = MetricsToolkit.compute_distribution_similarity(sample_vectors, full_vectors)
        return {
            'js_divergence': m.js_divergence,
        }

    @staticmethod
    def collect_metric_values(results: List[Dict[str, Any]], key: str) -> List[float]:
        """Collect non-NaN values for a specific metric from results list."""
        vals = [r.get(key) for r in results]
        return [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]


# Convenience re-exports for legacy style imports if needed
compute_basic_distance_metrics = MetricsToolkit.compute_basic_distance_metrics
compute_diversity_metrics = MetricsToolkit.compute_diversity_metrics
compute_distribution_similarity = MetricsToolkit.compute_distribution_similarity
summarize_rmsd = MetricsToolkit.summarize_rmsd
adapt_sampling_metrics = MetricsToolkit.adapt_sampling_metrics
wrap_diversity = MetricsToolkit.wrap_diversity
wrap_rmsd = MetricsToolkit.wrap_rmsd
wrap_similarity = MetricsToolkit.wrap_similarity
collect_metric_values = MetricsToolkit.collect_metric_values
