#!/usr/bin/env python
"""Unified metrics calculation utilities.

This module consolidates scattered metric computations (MinD, ANND, MPD,
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
from typing import Dict, Sequence
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, wasserstein_distance
from sklearn.decomposition import PCA

from . import ValidationUtils
# 注意：避免循环导入，不再从 core.metrics 导入 MetricCalculator。
# 若需要核心额外辅助函数，请在运行时延迟导入或在此工具内部重写。


@dataclass
class BasicDistanceMetrics:
    MinD: float
    ANND: float
    MPD: float


@dataclass
class DiversityMetrics:
    diversity_score: float
    coverage_ratio: float
    pca_variance_ratio: float
    energy_range: float


@dataclass
class DistributionSimilarity:
    js_divergence: float
    emd_distance: float
    mean_distance: float


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
            return BasicDistanceMetrics(np.nan, np.nan, np.nan)
        try:
            # Reuse logic similar to sampling_compare_demo but centralized
            dists = squareform(pdist(vectors, metric="euclidean"))
            np.fill_diagonal(dists, np.inf)
            valid = dists[dists != np.inf]
            if len(valid) == 0:
                return BasicDistanceMetrics(np.nan, np.nan, np.nan)
            min_d = float(np.min(valid)) if len(valid) else np.nan
            annd = float(np.mean(np.min(dists, axis=1))) if np.any(np.isfinite(np.min(dists, axis=1))) else np.nan
            mpd = float(np.mean(valid)) if len(valid) else np.nan
            return BasicDistanceMetrics(min_d, annd, mpd)
        except Exception:
            return BasicDistanceMetrics(np.nan, np.nan, np.nan)

    # ---- Diversity metrics ----
    @staticmethod
    def compute_diversity_metrics(vectors: np.ndarray) -> DiversityMetrics:
        if vectors is None or len(vectors) < 2:
            return DiversityMetrics(np.nan, np.nan, np.nan, np.nan)
        # Drop rows with any NaN
        if np.any(np.isnan(vectors)):
            mask = ~np.any(np.isnan(vectors), axis=1)
            vectors = vectors[mask]
            if len(vectors) < 2:
                return DiversityMetrics(np.nan, np.nan, np.nan, np.nan)
        try:
            dists = squareform(pdist(vectors, metric="euclidean"))
            np.fill_diagonal(dists, np.inf)
            if np.all(np.isinf(dists)) or np.any(np.isnan(dists)):
                return DiversityMetrics(np.nan, np.nan, np.nan, np.nan)
            max_dist = np.max(dists[dists != np.inf])
            if np.isnan(max_dist) or max_dist == 0:
                d_norm = dists
            else:
                d_norm = dists / max_dist
            dist_flat = d_norm[d_norm != np.inf]
            dist_flat = dist_flat[~np.isnan(dist_flat)]
            if len(dist_flat) == 0:
                diversity_score = np.nan
            else:
                hist, _ = np.histogram(dist_flat, bins=20, density=True)
                diversity_score = float(entropy(hist + 1e-10))
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
            return DiversityMetrics(diversity_score, coverage_ratio, pca_variance_ratio, energy_range)
        except Exception:
            return DiversityMetrics(np.nan, np.nan, np.nan, np.nan)

    # ---- Distribution similarity ----
    @staticmethod
    def compute_distribution_similarity(sample_vectors: np.ndarray, full_vectors: np.ndarray) -> DistributionSimilarity:
        if sample_vectors is None or full_vectors is None or len(sample_vectors) < 2 or len(full_vectors) < 2:
            return DistributionSimilarity(np.nan, np.nan, np.nan)
        # Clean NaNs
        if np.any(np.isnan(sample_vectors)):
            sample_vectors = sample_vectors[~np.any(np.isnan(sample_vectors), axis=1)]
        if np.any(np.isnan(full_vectors)):
            full_vectors = full_vectors[~np.any(np.isnan(full_vectors), axis=1)]
        if len(sample_vectors) < 2 or len(full_vectors) < 2:
            return DistributionSimilarity(np.nan, np.nan, np.nan)
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
            emd_vals = []
            for i in range(min(3, sample_pca.shape[1])):
                try:
                    emd_vals.append(wasserstein_distance(sample_pca[:, i], full_pca[:, i]))
                except Exception:
                    emd_vals.append(np.nan)
            emd_distance = float(np.nanmean(emd_vals))
            sample_centroid = np.mean(sample_vectors, axis=0)
            full_centroid = np.mean(full_vectors, axis=0)
            mean_distance = float(np.linalg.norm(sample_centroid - full_centroid))
            return DistributionSimilarity(js_divergence, emd_distance, mean_distance)
        except Exception:
            return DistributionSimilarity(np.nan, np.nan, np.nan)

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


# Convenience re-exports for legacy style imports if needed
compute_basic_distance_metrics = MetricsToolkit.compute_basic_distance_metrics
compute_diversity_metrics = MetricsToolkit.compute_diversity_metrics
compute_distribution_similarity = MetricsToolkit.compute_distribution_similarity
summarize_rmsd = MetricsToolkit.summarize_rmsd
