#!/usr/bin/env python

import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FrameData:
    frame_id: int
    positions: np.ndarray
    elements: List[str]
    distance_vector: Optional[np.ndarray] = None
    energy: Optional[float] = None
    energy_standardized: Optional[float] = None
    forces: Optional[List[Tuple[str, float, float, float]]] = None


class StrUParser:
    def __init__(self, exclude_hydrogen: bool = True):
        self.exclude_hydrogen = exclude_hydrogen
        self.logger = logging.getLogger(__name__)

    def parse_md_dumpfreq(self, input_file: str) -> int:
        """Parse md_dumpfreq value from an ABACUS INPUT file.

        Args:
            input_file: Absolute path to INPUT file.
        Returns:
            int: md_dumpfreq (>=1). Default 1 if missing / invalid / file not found.
        """
        freq = 1
        try:
            if not os.path.isfile(input_file):
                return freq
            with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Strip inline comments after '#'
                    raw = line.strip()
                    if not raw or raw.startswith('#'):
                        continue
                    if raw.lower().startswith('md_dumpfreq'):
                        # Split and take the first numeric token after the key
                        parts = raw.split()
                        if len(parts) >= 2:
                            try:
                                val = int(float(parts[1]))  # tolerate "10" or "10.0"
                                if val >= 1:
                                    freq = val
                            except ValueError:
                                pass
                        break
        except Exception:
            # Silently fall back to default (1)
            return 1
        return freq

    def parse_md_parameters(self, input_file: str) -> tuple:
        """Parse md_dumpfreq and md_nstep from an ABACUS INPUT file.

        Args:
            input_file: Absolute path to INPUT file.
        Returns:
            tuple: (md_dumpfreq, md_nstep). Both default to 1 if missing/invalid.
        """
        dumpfreq = 1
        nstep = 1
        try:
            if not os.path.isfile(input_file):
                return dumpfreq, nstep
            with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Strip inline comments after '#'
                    raw = line.strip()
                    if not raw or raw.startswith('#'):
                        continue
                    if raw.lower().startswith('md_dumpfreq'):
                        parts = raw.split()
                        if len(parts) >= 2:
                            try:
                                val = int(float(parts[1]))
                                if val >= 1:
                                    dumpfreq = val
                            except ValueError:
                                pass
                    elif raw.lower().startswith('md_nstep'):
                        parts = raw.split()
                        if len(parts) >= 2:
                            try:
                                val = int(float(parts[1]))
                                if val >= 1:
                                    nstep = val
                            except ValueError:
                                pass
        except Exception:
            # Silently fall back to defaults
            return 1, 1
        return dumpfreq, nstep

    def select_frames_by_md_dumpfreq(self, stru_files, md_dumpfreq: int):
        """Filter STRU_MD_* file list according to md_dumpfreq, excluding frame 0.

        Rule: select frames with indices i = md_dumpfreq * k, k=1,2,3,... (excluding 0)
        That is: all multiples of md_dumpfreq, but not including 0.

        Args:
            stru_files (List[str]): Full list of STRU_MD_* file paths.
            md_dumpfreq (int): Dump frequency (>=1).
        Returns:
            List[str]: Filtered list, sorted by frame index.
        """
        if md_dumpfreq <= 1:
            # No downsampling needed (frequency 1 => every step dumped), but exclude 0
            filtered = [f for f in stru_files if self._frame_id_from_path(f) != 0]
            return sorted(filtered, key=self._frame_id_from_path)
        filtered = []
        for f in stru_files:
            fid = self._frame_id_from_path(f)
            if fid is None:
                continue
            if fid % md_dumpfreq == 0 and fid != 0:
                filtered.append(f)
        return sorted(filtered, key=self._frame_id_from_path)

    def _frame_id_from_path(self, path: str) -> Optional[int]:
        name = os.path.basename(path)
        m = re.search(r"STRU_MD_(\d+)$", name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    def parse_file(self, stru_file: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        try:
            with open(stru_file, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            self.logger.warning(f"Cannot read file {stru_file}: {e}")
            return None
        try:
            return self._parse_lines(lines)
        except Exception as e:
            self.logger.warning(f"Parse error for {stru_file}: {e}")
            return None

    def _parse_lines(self, lines: List[str]) -> Optional[Tuple[np.ndarray, List[str]]]:
        lattice_constant = 1.0
        positions = []
        elements = []
        current_element = None
        element_atoms_count = 0
        element_atoms_collected = 0
        section = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "LATTICE_CONSTANT" in line:
                section = "LATTICE_CONSTANT"
                continue
            elif "LATTICE_VECTORS" in line:
                section = "LATTICE_VECTORS"
                continue
            elif "ATOMIC_SPECIES" in line:
                section = "ATOMIC_SPECIES"
                continue
            elif "ATOMIC_POSITIONS" in line:
                section = "ATOMIC_POSITIONS"
                continue
            if section == "LATTICE_CONSTANT":
                lattice_constant = self._parse_lattice_constant(line)
            elif section == "ATOMIC_POSITIONS":
                result = self._parse_atomic_positions_line(
                    line,
                    current_element,
                    element_atoms_count,
                    element_atoms_collected,
                    positions,
                    elements,
                    lattice_constant,
                )
                if result:
                    current_element, element_atoms_count, element_atoms_collected = (
                        result
                    )
        if not positions:
            return None
        return np.array(positions), elements

    def _parse_lattice_constant(self, line: str) -> float:
        try:
            return float(re.split(r"\s+", line)[0])
        except (ValueError, IndexError):
            return 1.0

    def _parse_atomic_positions_line(
        self,
        line: str,
        current_element: str,
        element_atoms_count: int,
        element_atoms_collected: int,
        positions: List,
        elements: List,
        lattice_constant: float,
    ) -> Optional[Tuple]:
        if re.match(r"^[A-Za-z]{1,2}\s*#", line):
            parts = re.split(r"\s+", line)
            current_element = parts[0]
            element_atoms_count = 0
            element_atoms_collected = 0
            return current_element, element_atoms_count, element_atoms_collected
        if current_element and "number of atoms" in line:
            try:
                element_atoms_count = int(re.split(r"\s+", line)[0])
            except (ValueError, IndexError):
                element_atoms_count = 0
            return current_element, element_atoms_count, element_atoms_collected
        if (
            current_element
            and element_atoms_count > 0
            and element_atoms_collected < element_atoms_count
        ):
            if self.exclude_hydrogen and current_element.upper() in ("H", "HYDROGEN"):
                element_atoms_collected += 1
                return current_element, element_atoms_count, element_atoms_collected
            try:
                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    return current_element, element_atoms_count, element_atoms_collected
                coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                positions.append(np.array(coords) * lattice_constant)
                elements.append(current_element)
                element_atoms_collected += 1
            except (ValueError, IndexError):
                pass
            return current_element, element_atoms_count, element_atoms_collected
        return None

    def parse_trajectory(self, stru_dir: str, pre_files: Optional[List[str]] = None) -> List[FrameData]:
        """解析轨迹。

        Args:
            stru_dir: STRU 目录
            pre_files: 可选，已过滤好的 STRU 文件路径列表（绝对路径）。提供后将跳过 glob 与 md_dumpfreq 过滤。
        """
        if pre_files is not None:
            stru_files = list(pre_files)
        else:
            stru_files = glob.glob(os.path.join(stru_dir, "STRU_MD_*"))
            if not stru_files:
                self.logger.warning(f"No STRU_MD_* files in {stru_dir}")
                return []
            # 基于 INPUT 中的 md_dumpfreq 进行筛选，保持与 PathManager 一致
            try:
                input_file = os.path.abspath(os.path.join(stru_dir, os.pardir, "INPUT"))
                md_dumpfreq = self.parse_md_dumpfreq(input_file)
                original = len(stru_files)
                stru_files = self.select_frames_by_md_dumpfreq(stru_files, md_dumpfreq)
                if not stru_files:
                    self.logger.warning(
                        f"After applying md_dumpfreq={md_dumpfreq} filter, no STRU frames remain in {stru_dir} (original {original})"
                    )
                    return []
                if len(stru_files) != original:
                    self.logger.info(
                        f"Filtered STRU frames by md_dumpfreq={md_dumpfreq}: {len(stru_files)}/{original} retained (including 0)"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to apply md_dumpfreq filtering, using all frames: {e}")
        frames = []
        for stru_file in stru_files:
            match = re.search(r"STRU_MD_(\d+)", os.path.basename(stru_file))
            if not match:
                continue
            frame_id = int(match.group(1))
            if frame_id == 0:
                continue  # 排除第0帧
            result = self.parse_file(stru_file)
            if result is None:
                continue
            positions, elements = result
            if len(positions) < 2:
                continue
            frames.append(FrameData(frame_id, positions, elements))
        frames.sort(key=lambda x: x.frame_id)

        # 解析能量和力信息
        self._parse_energy_and_forces(stru_dir, frames)

        self.logger.info(f"Parsed trajectory: {len(frames)} frames")
        return frames

    def _parse_energy_and_forces(self, stru_dir: str, frames: List[FrameData]) -> None:
        """解析能量和力信息，并将其添加到帧数据中"""
        try:
            # 尝试导入能量力解析器
            from ..analysis.trajectory_analyser import parse_running_md_log

            # 构建日志文件路径
            log_path = os.path.join(stru_dir, "..", "running_md.log")
            log_path = os.path.abspath(log_path)

            if os.path.exists(log_path):
                energies, forces = parse_running_md_log(log_path)

                # 将能量和力信息添加到对应的帧中
                for frame in frames:
                    if frame.frame_id in energies:
                        frame.energy = energies[frame.frame_id]
                    if frame.frame_id in forces:
                        frame.forces = forces[frame.frame_id]

                self.logger.info(f"Parsed energy and force data for {len(frames)} frames")
            else:
                self.logger.warning(f"Energy/force log file not found: {log_path}")

        except ImportError:
            self.logger.warning("force_energy_parser module not available, skipping energy/force parsing")
        except Exception as e:
            self.logger.warning(f"Failed to parse energy and force data: {e}")


def parse_abacus_stru(
    stru_file: str, exclude_hydrogen: bool = True
) -> Optional[Tuple[np.ndarray, List[str]]]:
    parser = StrUParser(exclude_hydrogen=exclude_hydrogen)
    return parser.parse_file(stru_file)
