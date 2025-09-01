#!/usr/bin/env python
"""Structural metrics and alignment utilities (Level 3 extraction).

This module centralizes RMSD/RMSF and structural alignment routines
(previously scattered in `trajectory_analyser`).

Functions are intentionally lightweight and pure for easy unit testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np

__all__ = [
    "kabsch_align",
    "iterative_mean_structure",
    "compute_rmsd_series",
    "compute_rmsf",
    "RMSFSummary",
    "summarize_rmsf",
]


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align coordinates P onto Q using Kabsch algorithm.

    Args:
        P: (n_atoms, 3)
        Q: (n_atoms, 3)
    Returns:
        Aligned copy of P.
    """
    if P.size == 0 or Q.size == 0:
        return P.copy()
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1, 1, d]) @ Wt
    return Pc @ U


def iterative_mean_structure(positions_list: Sequence[np.ndarray], max_iter: int = 20, tol: float = 1e-6) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Iteratively align frames to a reference and recompute mean structure.

    Returns mean structure and list of aligned frames.
    """
    if not positions_list:
        return np.array([]), []
    ref = positions_list[0].copy()
    aligned_positions = list(positions_list)
    for _ in range(max_iter):
        aligned_positions = [kabsch_align(pos, ref) for pos in positions_list]
        mean_structure = np.mean(aligned_positions, axis=0)
        if np.linalg.norm(mean_structure - ref) < tol:
            break
        ref = mean_structure
    return mean_structure, aligned_positions


def compute_rmsd_series(positions_list: Sequence[np.ndarray], reference: np.ndarray | None = None) -> np.ndarray:
    """Compute per-frame RMSD to reference (default: iterative mean)."""
    if not positions_list:
        return np.array([])
    if reference is None:
        reference = np.mean(np.stack(positions_list, axis=0), axis=0)
    rmsds = []
    for pos in positions_list:
        diff = pos - reference
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        rmsds.append(rmsd)
    return np.array(rmsds, dtype=float)


def compute_rmsf(positions_list: Sequence[np.ndarray]) -> np.ndarray:
    """Compute per-atom RMSF over trajectory."""
    if not positions_list:
        return np.array([])
    arr = np.stack(positions_list, axis=0)
    mean_pos = np.mean(arr, axis=0)
    diff = arr - mean_pos
    rmsf = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=0))
    return rmsf


@dataclass
class RMSFSummary:
    mean: float
    std: float
    min: float
    max: float


def summarize_rmsf(values: Sequence[float]) -> RMSFSummary:
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return RMSFSummary(np.nan, np.nan, np.nan, np.nan)
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        return RMSFSummary(np.nan, np.nan, np.nan, np.nan)
    return RMSFSummary(float(np.mean(v)), float(np.std(v)) if v.size > 1 else np.nan, float(np.min(v)), float(np.max(v)))
