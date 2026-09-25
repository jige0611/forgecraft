"""
零件朝向优化器 — 自动选择最优打印方向

功能:
  - 24 候选方向生成 (6 主轴 × 4 旋转)
  - 3 种评分策略: min_support / max_strength / assembly_aware
  - 自动悬垂分析 + 支撑体积估计
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from forgecraft.cam.types import OrientationResult

__all__ = ["PartOrienter"]


class PartOrienter:
    """零件朝向优化器

    Usage:
        orienter = PartOrienter(strategy="min_support")
        result = orienter.optimize(mesh)
    """

    # 6 个主轴方向
    _AXES = [
        np.array([0, 0, 1]),    # +Z
        np.array([0, 0, -1]),   # -Z
        np.array([0, 1, 0]),    # +Y
        np.array([0, -1, 0]),   # -Y
        np.array([1, 0, 0]),    # +X
        np.array([-1, 0, 0]),   # -X
    ]

    # 4 个绕轴旋转角度
    _ROTATIONS = [0, math.pi / 2, math.pi, 3 * math.pi / 2]

    def __init__(self, strategy: str = "min_support"):
        self.strategy = strategy

    def optimize(self, mesh: trimesh.Trimesh) -> OrientationResult:
        """优化零件朝向，返回最优结果"""
        candidates = self._generate_candidates()

        best_result = None
        best_score = float('-inf')

        for matrix in candidates:
            result = self._evaluate(mesh, matrix)
            if result.score > best_score:
                best_score = result.score
                best_result = result

        return best_result if best_result else OrientationResult(
            matrix=np.eye(4), score=0.0,
        )

    def _generate_candidates(self) -> List[np.ndarray]:
        """生成 24 个候选朝向的 4x4 变换矩阵"""
        matrices = []
        for axis in self._AXES:
            for angle in self._ROTATIONS:
                M = self._build_orientation_matrix(axis, angle)
                matrices.append(M)
        return matrices

    def _build_orientation_matrix(
        self, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        """构建 4x4 变换矩阵：先轴对齐，再绕 Z 旋转"""
        # 计算使 axis 对齐到 +Z 的旋转
        z_axis = np.array([0, 0, 1.0])
        if np.allclose(axis, z_axis):
            R_align = np.eye(3)
        elif np.allclose(axis, -z_axis):
            R_align = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        else:
            v = np.cross(axis, z_axis)
            v = v / np.linalg.norm(v)
            cos_theta = np.dot(axis, z_axis)
            sin_theta = np.linalg.norm(np.cross(axis, z_axis))
            Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_align = np.eye(3) + sin_theta * Vx + (1 - cos_theta) * np.dot(Vx, Vx)

        # 绕 Z 旋转
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        Rz = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

        R = Rz @ R_align

        M = np.eye(4)
        M[:3, :3] = R
        return M

    def _evaluate(self, mesh: trimesh.Trimesh, matrix: np.ndarray) -> OrientationResult:
        """评估单个朝向"""
        rotated = mesh.copy()
        rotated.apply_transform(matrix)

        if self.strategy == "min_support":
            return self._score_min_support(rotated, matrix)
        elif self.strategy == "max_strength":
            return self._score_max_strength(rotated, matrix)
        elif self.strategy == "assembly_aware":
            return self._score_assembly_aware(rotated, matrix)
        else:
            return self._score_min_support(rotated, matrix)

    def _score_min_support(
        self, mesh: trimesh.Trimesh, matrix: np.ndarray
    ) -> OrientationResult:
        """最少支撑策略：悬垂面积最小"""
        overhang_faces = self._detect_overhang_faces(mesh)
        overhang_area = mesh.area_faces[overhang_faces].sum() if len(overhang_faces) > 0 else 0.0
        total_area = mesh.area

        # 支撑体积估算 (简化)
        z_min = mesh.bounds[0, 2]
        face_centers = mesh.triangles_center[overhang_faces] if len(overhang_faces) > 0 else np.empty((0, 3))
        if len(face_centers) > 0:
            avg_height = np.mean(face_centers[:, 2] - z_min)
            support_volume = overhang_area * avg_height * 0.5  # 锥形支撑近似
        else:
            support_volume = 0.0
            avg_height = 0.0

        # 综合评分：面积越小 + 体积越小 = 越好 (负值 → 最大化)
        score = -(overhang_area + support_volume * 1000)

        return OrientationResult(
            matrix=matrix.copy(),
            support_volume=support_volume,
            support_area=overhang_area,
            build_height=mesh.bounds[1, 2] - mesh.bounds[0, 2],
            strategy="min_support",
            score=score,
        )

    def _score_max_strength(
        self, mesh: trimesh.Trimesh, matrix: np.ndarray
    ) -> OrientationResult:
        """最大强度策略：XY 截面积大 + Z 方向层少"""
        # 计算水平截面积
        z_mid = (mesh.bounds[0, 2] + mesh.bounds[1, 2]) / 2
        section = mesh.section(plane_origin=[0, 0, z_mid], plane_normal=[0, 0, 1])
        section_area = 0.0
        if section is not None and hasattr(section, 'area'):
            section_area = section.area

        build_height = mesh.bounds[1, 2] - mesh.bounds[0, 2]

        # 截面积大 (强度高) + 高度小 (层数少) = 好
        score = section_area * 1000 - build_height * 100

        overhang_faces = self._detect_overhang_faces(mesh)
        overhang_area = mesh.area_faces[overhang_faces].sum() if len(overhang_faces) > 0 else 0.0

        return OrientationResult(
            matrix=matrix.copy(),
            support_area=overhang_area,
            build_height=build_height,
            strategy="max_strength",
            score=score,
        )

    def _score_assembly_aware(
        self, mesh: trimesh.Trimesh, matrix: np.ndarray
    ) -> OrientationResult:
        """装配感知策略 (简化版)"""
        # 孔轴线与 Z 平行 = 圆度最高
        overhang_faces = self._detect_overhang_faces(mesh)
        overhang_area = mesh.area_faces[overhang_faces].sum() if len(overhang_faces) > 0 else 0.0
        build_height = mesh.bounds[1, 2] - mesh.bounds[0, 2]

        score = -overhang_area * 0.7 - build_height * 50

        return OrientationResult(
            matrix=matrix.copy(),
            support_area=overhang_area,
            build_height=build_height,
            strategy="assembly_aware",
            score=score,
        )

    @staticmethod
    def _detect_overhang_faces(mesh: trimesh.Trimesh, threshold_deg: float = 45.0) -> np.ndarray:
        """复用现有悬垂检测"""
        from forgecraft.manufacturing.supports import detect_overhang_faces
        faces, _ = detect_overhang_faces(mesh, threshold_deg)
        return faces
