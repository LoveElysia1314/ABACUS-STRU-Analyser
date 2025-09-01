#!/usr/bin/env python
"""Unified metrics registry.

目的:
1. 消除 `ResultSaver` 中手写列名与取值逻辑的重复/漂移风险。
2. 为相关性分析、采样方法对比等二级分析提供**单一指标来源**。
3. 支持后续: 版本化(schema_version) / 指标子集开关 / 动态扩展。

使用:
from ..utils.metrics_registry import SYSTEM_SUMMARY_HEADERS, build_summary_row

设计要点:
- 通过 MetricSpec 描述: key(内部标识) / header(CSV列名) / category(分组) / extractor(取值函数)
- 顺序由 GROUP_ORDER + 每组内注册顺序共同决定。
- 新增指标: 仅需在 REGISTRY 列表中 append 对应 MetricSpec。
- 分布 / 采样相似性 位置已调至 PCA 之前 (与最新需求一致)。

后续扩展建议:
- 增加 `metric_version_map` 记录各指标的计算版本。
- 增加 `ENABLED_CATEGORIES` 以支持命令行开关高成本指标(例如 EMD)。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, List, Dict
import json

# schema 版本（当列集合或取值语义发生破坏性调整时递增）
SCHEMA_VERSION = "summary-v2"  # v1 -> 原始硬编码; v2 -> 引入 registry + 分组重排

@dataclass
class MetricSpec:
    key: str
    header: str
    category: str  # identity | scale | core_distance | diversity | distribution | pca
    extractor: Callable[[Any], Any]
    formatter: Callable[[Any], str] | None = None  # 可选格式化

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

# ---- 通用格式化工具 ----
_float6 = lambda v: "" if v is None else (f"{float(v):.6f}" if not (isinstance(v, float) and v != v) else "")
_int = lambda v: "" if v is None else str(int(v))
_json_list = lambda v: json.dumps(v, ensure_ascii=False) if v is not None else ""
_passthrough = lambda v: "" if v is None else str(v)

# 组顺序（外部依赖此顺序时只需修改此列表）
GROUP_ORDER = [
    "identity",          # 基础标识
    "scale",             # 规模/维度
    "core_distance",     # 核心结构距离指标
    "diversity",         # 多样性与覆盖 & 能量
    "distribution",      # 分布 / 采样相似性 (已提前)
    "pca",               # PCA 概览
]

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

# 预构建: group -> specs 列表
_GROUP_TO_SPECS: Dict[str, List[MetricSpec]] = {g: [] for g in GROUP_ORDER}
for spec in REGISTRY:
    _GROUP_TO_SPECS.setdefault(spec.category, []).append(spec)

# 最终 CSV 头顺序
SYSTEM_SUMMARY_HEADERS: List[str] = [spec.header for g in GROUP_ORDER for spec in _GROUP_TO_SPECS[g]]

HEADER_TO_SPEC: Dict[str, MetricSpec] = {spec.header: spec for spec in REGISTRY}
KEY_TO_SPEC: Dict[str, MetricSpec] = {spec.key: spec for spec in REGISTRY}


def build_summary_row(metrics_obj: Any) -> List[str]:
    """按统一顺序生成一行指标字符串列表。"""
    return [spec.get_value(metrics_obj) for g in GROUP_ORDER for spec in _GROUP_TO_SPECS[g]]


def iter_metric_specs(category: str | None = None):
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

__all__ = [
    "SCHEMA_VERSION",
    "MetricSpec",
    "REGISTRY",
    "GROUP_ORDER",
    "SYSTEM_SUMMARY_HEADERS",
    "HEADER_TO_SPEC",
    "KEY_TO_SPEC",
    "build_summary_row",
    "iter_metric_specs",
    "get_headers_by_categories",
]
