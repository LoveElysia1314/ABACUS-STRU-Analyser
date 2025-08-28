#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List
import numpy as np
from scipy.spatial.distance import pdist
from ..utils import Constants, MathUtils


class MetricCalculator:
	@staticmethod
	def calculate_distance_vectors(positions: np.ndarray) -> np.ndarray:
		if len(positions) < 2:
			return np.array([])
		raw_vectors = pdist(positions)
		norm = np.linalg.norm(raw_vectors)
		return raw_vectors / norm if norm > Constants.EPSILON else raw_vectors

	@staticmethod
	def compute_all_metrics(vector_matrix: np.ndarray) -> Dict[str, object]:
		if len(vector_matrix) == 0:
			return {
				'global_mean': 0.0,
				'nRMSF': 0.0,
				'MCV': 0.0,
				'frame_nLdRMS': [],
				'avg_nLdRMS': 0.0
			}

		global_mean = np.mean(vector_matrix) if vector_matrix.size > 0 else 0.0
		nRMSF = MetricCalculator._calculate_nRMSF(vector_matrix, global_mean)
		MCV = MetricCalculator._calculate_MCV(vector_matrix)
		frame_nLdRMS = MetricCalculator._calculate_frame_nLdRMS(vector_matrix, global_mean)
		avg_nLdRMS = np.mean(frame_nLdRMS) if len(frame_nLdRMS) > 0 else 0.0
		return {
			'global_mean': global_mean,
			'nRMSF': nRMSF,
			'MCV': MCV,
			'frame_nLdRMS': frame_nLdRMS,
			'avg_nLdRMS': avg_nLdRMS
		}

	@staticmethod
	def _calculate_nRMSF(vector_matrix: np.ndarray, global_mean: float) -> float:
		variances = np.var(vector_matrix, axis=0)
		nRMSF = MathUtils.safe_sqrt(np.mean(variances)) / (global_mean + Constants.EPSILON)
		return nRMSF

	@staticmethod
	def _calculate_MCV(vector_matrix: np.ndarray) -> float:
		means = np.mean(vector_matrix, axis=0)
		stds = np.std(vector_matrix, axis=0)
		with np.errstate(divide='ignore', invalid='ignore'):
			cvs = np.where(means > Constants.EPSILON, stds / means, 0.0)
		return np.mean(cvs)

	@staticmethod
	def _calculate_frame_nLdRMS(vector_matrix: np.ndarray, global_mean: float) -> np.ndarray:
		mean_vector = np.mean(vector_matrix, axis=0)
		diff = vector_matrix - mean_vector
		frame_nLdRMS = MathUtils.safe_sqrt(np.mean(diff ** 2, axis=1)) / (global_mean + Constants.EPSILON)
		return frame_nLdRMS


class TrajectoryMetrics:
	def __init__(self, system_name: str, mol_id: str, conf: str, temperature: str, system_path: str = ""):
		self.system_name = system_name
		self.mol_id = mol_id
		self.conf = conf
		self.temperature = temperature
		self.system_path = system_path  # 添加系统路径
		self.num_frames = 0
		self.dimension = 0
		self.nRMSF = 0.0
		self.MCV = 0.0
		self.avg_nLdRMS = 0.0
		self.nRMSF_sampled = 0.0
		self.MCV_sampled = 0.0
		self.avg_nLdRMS_sampled = 0.0
		self.sampled_frames = []
	
	@property
	def out_abacus_path(self) -> str:
		"""返回OUT.ABACUS文件夹的路径"""
		if self.system_path:
			return os.path.join(self.system_path, "OUT.ABACUS")
		return ""

	def set_original_metrics(self, metrics_data: Dict[str, float]):
		self.nRMSF = metrics_data['nRMSF']
		self.MCV = metrics_data['MCV']
		self.avg_nLdRMS = metrics_data['avg_nLdRMS']

	def set_sampled_metrics(self, metrics_data: Dict[str, float]):
		self.nRMSF_sampled = metrics_data['nRMSF']
		self.MCV_sampled = metrics_data['MCV']
		self.avg_nLdRMS_sampled = metrics_data['avg_nLdRMS']

	def get_ratio_metrics(self) -> Dict[str, float]:
		return {
			'nRMSF_ratio': self.nRMSF_sampled / (self.nRMSF + Constants.EPSILON),
			'MCV_ratio': self.MCV_sampled / (self.MCV + Constants.EPSILON),
			'avg_nLdRMS_ratio': self.avg_nLdRMS_sampled / (self.avg_nLdRMS + Constants.EPSILON)
		}
