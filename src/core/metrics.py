#!/usr/bin/env python

import os
from typing import Dict, List, Any, Callable, Sequence, Optional, Union
from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import entropy, wasserstein_distance
from sklearn.decomposition import PCA

from ..utils import Constants, ValidationUtils


# Data classes for metrics
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


# Metrics registry classes
@dataclass
class MetricSpec:
    key: str
    header: str
    category: str  # identity | scale | core_distance | diversity | distribution | pca
    extractor: Callable[[Any], Any]
    formatter: Optional[Callable[[Any], str]] = None  # 可选格式化

    def get_value(self, obj: Any) -> str:
        raw = None
        try:
            raw = self.extractor(obj)
        except Exception:
            raw = None
        if self.formatter:
            try:
                return self.formatter(raw)
            except Exception:
                return ""
        # 默认格式化规则
        if raw is None:
            return ""
        if isinstance(raw, float):
            return f"{raw:.6f}" if raw == raw else ""  # NaN 处理
        if isinstance(raw, (list, tuple)):
            try:
                return json.dumps(raw, ensure_ascii=False)
            except Exception:
                return ""
        return str(raw)


# Schema version
SCHEMA_VERSION = "summary-v2"

# Group order
GROUP_ORDER = [
    "identity",          # 基础标识
    "scale",             # 规模/维度
    "core_distance",     # 核心结构距离指标
    "diversity",         # 多样性与覆盖 & 能量
    "distribution",      # 分布 / 采样相似性 (已提前)
    "pca",               # PCA 概览
]

# Formatting utilities
_float6 = lambda v: "" if v is None else (f"{float(v):.6f}" if not (isinstance(v, float) and v != v) else "")
_int = lambda v: "" if v is None else str(int(v))
_json_list = lambda v: json.dumps(v, ensure_ascii=False) if v is not None else ""
_passthrough = lambda v: "" if v is None else str(v)

# Metrics registry
REGISTRY: List[MetricSpec] = [
    # identity
    MetricSpec("system", "System", "identity", lambda m: getattr(m, "system_name", ""), _passthrough),
    MetricSpec("mol_id", "Molecule_ID", "identity", lambda m: getattr(m, "mol_id", ""), _passthrough),
    MetricSpec("conf", "Configuration", "identity", lambda m: getattr(m, "conf", ""), _passthrough),
    MetricSpec("temperature", "Temperature(K)", "identity", lambda m: getattr(m, "temperature", ""), _passthrough),
    # scale
    MetricSpec("num_frames", "Num_Frames", "scale", lambda m: getattr(m, "num_frames", None), _int),
    MetricSpec("dimension", "Dimension", "scale", lambda m: getattr(m, "dimension", None), _int),
    # core distance
    MetricSpec("rmsd_mean", "RMSD_Mean", "core_distance", lambda m: getattr(m, "rmsd_mean", None), _float6),
    MetricSpec("ANND", "ANND", "core_distance", lambda m: getattr(m, "ANND", None), _float6),
    MetricSpec("MPD", "MPD", "core_distance", lambda m: getattr(m, "MPD", None), _float6),
    # diversity & energy
    MetricSpec("coverage_ratio", "Coverage_Ratio", "diversity", lambda m: getattr(m, "coverage_ratio", None), _float6),
    MetricSpec("energy_range", "Energy_Range", "diversity", lambda m: getattr(m, "energy_range", None), _float6),
    # distribution similarity (order swapped before PCA)
    MetricSpec("js_divergence", "JS_Divergence", "distribution", lambda m: getattr(m, "js_divergence", None), _float6),
    # PCA
    MetricSpec("pca_components", "PCA_Num_Components_Retained", "pca", lambda m: getattr(m, "pca_components", None), _int),
    MetricSpec("pca_variance_ratio", "PCA_Variance_Ratio", "pca", lambda m: getattr(m, "pca_variance_ratio", None), _float6),
    MetricSpec("pca_cumulative_variance_ratio", "PCA_Cumulative_Variance_Ratio", "pca", lambda m: getattr(m, "pca_cumulative_variance_ratio", None), _float6),
    MetricSpec("pca_explained_variance_ratio", "PCA_Variance_Ratios", "pca", lambda m: getattr(m, "pca_explained_variance_ratio", None), _json_list),
]

# Pre-built: group -> specs list
_GROUP_TO_SPECS: Dict[str, List[MetricSpec]] = {g: [] for g in GROUP_ORDER}
for spec in REGISTRY:
    _GROUP_TO_SPECS.setdefault(spec.category, []).append(spec)

# Final CSV header order
SYSTEM_SUMMARY_HEADERS: List[str] = [spec.header for g in GROUP_ORDER for spec in _GROUP_TO_SPECS[g]]

HEADER_TO_SPEC: Dict[str, MetricSpec] = {spec.header: spec for spec in REGISTRY}
KEY_TO_SPEC: Dict[str, MetricSpec] = {spec.key: spec for spec in REGISTRY}


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
        """Compute basic metrics for each frame's internal coordinate vectors."""
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
    def adapt_sampling_metrics(
        selected_vectors: np.ndarray,
        full_vectors: np.ndarray,
        rmsd_values: Optional[Union[np.ndarray, List[float]]] = None
    ) -> Dict[str, Any]:
        """Convenience wrapper for sampling comparison."""
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


# Registry functions
def build_summary_row(metrics_obj: Any) -> List[str]:
    """按统一顺序生成一行指标字符串列表。"""
    return [spec.get_value(metrics_obj) for g in GROUP_ORDER for spec in _GROUP_TO_SPECS[g]]


def iter_metric_specs(category: Optional[str] = None):
    if category is None:
        for g in GROUP_ORDER:
            for spec in _GROUP_TO_SPECS[g]:
                yield spec
    else:
        for spec in _GROUP_TO_SPECS.get(category, []):
            yield spec


def get_headers_by_categories(categories: List[str]) -> List[str]:
    """返回按给定分类过滤后的 header 列表，保持 registry 原顺序。"""
    cat_set = set(categories)
    return [spec.header for g in GROUP_ORDER for spec in _GROUP_TO_SPECS[g] if spec.category in cat_set]


class MetricCalculator:
    @staticmethod
    def calculate_distance_vectors(positions: np.ndarray) -> np.ndarray:
        # positions is expected to be a 2D numpy array; be robust if it's None or too short
        if (
            ValidationUtils.is_empty(positions)
            or getattr(positions, "shape", [0])[0] < 2
        ):
            return np.array([])
        # 移除L2归一化步骤，直接返回原始距离向量
        return pdist(positions)

    @staticmethod
    def compute_all_metrics(vector_matrix: np.ndarray) -> Dict[str, object]:
        """统一计算距离相关指标。

        已重构为调用 MetricsToolkit.compute_basic_distance_metrics 以消除重复。
        保持原返回结构向后兼容。
        """
        if ValidationUtils.is_empty(vector_matrix) or getattr(vector_matrix, "shape", [0])[0] < 2:
            return {"global_mean": 0.0, "ANND": 0.0, "MPD": 0.0}

        global_mean = float(np.mean(vector_matrix))
        basic = MetricsToolkit.compute_basic_distance_metrics(vector_matrix)
        return {
            "global_mean": global_mean,
            "ANND": basic.ANND,
            "MPD": basic.MPD,
        }



    @staticmethod
    def _calculate_ANND(vector_matrix: np.ndarray) -> float:  # deprecated
        """(Deprecated) 保留旧接口，内部委托统一工具。"""
        basic = MetricsToolkit.compute_basic_distance_metrics(vector_matrix)
        return 0.0 if np.isnan(basic.ANND) else float(basic.ANND)



    @staticmethod
    def estimate_mean_distance(vectors: np.ndarray) -> float:  # deprecated
        """(Deprecated) 平均成对距离，委托统一工具 MPD。"""
        basic = MetricsToolkit.compute_basic_distance_metrics(vectors)
        return 0.0 if np.isnan(basic.MPD) else float(basic.MPD)

    @staticmethod
    def calculate_dRMSF(distance_vectors: np.ndarray) -> float:
        """计算距离均方根波动（从trajectory_analyser迁移）"""
        if ValidationUtils.is_empty(distance_vectors) or len(distance_vectors) <= 1:
            return 0.0
        variances = np.var(distance_vectors, axis=0)
        mean_variance = np.mean(variances)
        return float(np.sqrt(mean_variance))

    @staticmethod
    def calculate_MeanCV(distance_vectors: np.ndarray) -> float:
        """计算平均变异系数（从trajectory_analyser迁移）"""
        if ValidationUtils.is_empty(distance_vectors) or len(distance_vectors) <= 1:
            return 0.0
        means = np.mean(distance_vectors, axis=0)
        stds = np.std(distance_vectors, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cvs = np.where(means > Constants.EPSILON, stds / means, 0.0)
        return float(np.mean(cvs))


class TrajectoryMetrics:
    def __init__(
        self,
        system_name: str,
        mol_id: str,
        conf: str,
        temperature: str,
        system_path: str = "",
    ):
        self.system_name = system_name
        self.mol_id = mol_id
        self.conf = conf
        self.temperature = temperature
        self.system_path = system_path  # 添加系统路径
        self.num_frames = 0
        self.dimension = 0
        self.ANND = 0.0
        self.MPD = 0.0
        self.sampled_frames = []
        # RMSD相关字段
        self.rmsd_mean = 0.0  # RMSD均值（总体指标）
        self.rmsd_per_frame = []  # 每帧RMSD（单帧指标）
        # PCA相关字段
        self.pca_variance_ratio = 0.0  # PCA目标方差贡献率
        self.pca_explained_variance_ratio = []  # 各主成分方差贡献率
        self.pca_cumulative_variance_ratio = 0.0  # 累积方差贡献率
        self.pca_components = 0  # PCA主成分数量
        # 综合向量相关字段
        self.comprehensive_dimension = 0  # 综合向量维度
        self.energy_available = False  # 是否有能量数据
        # 平均构象坐标（用于后续构象统一）
        self.mean_structure = None  # 平均构象坐标，numpy数组形状为(n_atoms, 3)
        # Level 4: 多样性 / 分布相似性扩展字段（可选输出）
        self.coverage_ratio = None
        self.energy_range = None
        self.js_divergence = None

    def set_diversity_metrics(self, diversity_obj):  # diversity_obj: DiversityMetrics
        try:
            self.coverage_ratio = float(diversity_obj.coverage_ratio)
            self.energy_range = float(diversity_obj.energy_range)
        except Exception:
            pass

    def set_distribution_similarity(self, sim_obj):  # sim_obj: DistributionSimilarity
        try:
            self.js_divergence = float(sim_obj.js_divergence)
        except Exception:
            pass

    @property
    def out_abacus_path(self) -> str:
        """返回OUT.ABACUS文件夹的路径"""
        if self.system_path:
            return os.path.join(self.system_path, "OUT.ABACUS")
        return ""

    def set_original_metrics(self, metrics_data: Dict[str, float]):
        self.ANND = metrics_data["ANND"]
        self.MPD = metrics_data["MPD"]

    def set_sampled_metrics(self, metrics_data: Dict[str, float]):
        """Deprecated placeholder for backward compatibility."""
        return None

    def get_ratio_metrics(self) -> Dict[str, float]:
        """Deprecated: ratio metrics removed."""
        return {}
