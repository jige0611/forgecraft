"""
直线填充图案 — 平行线 + 逐层旋转

最基础的填充图案：在轮廓内生成平行线，每层旋转 90° 以增加层间结合力。
"""

import math
from typing import List

import numpy as np

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.types import PathSegment


class RectilinearInfill(InfillPattern):
    """直线填充"""

    def __init__(self, density: float = 0.2, extrusion_width: float = 0.45):
        super().__init__(density, extrusion_width)

    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        angle: float = None,
        **kwargs,
    ) -> List[PathSegment]:
        """生成直线填充路径

        Args:
            angle: 指定填充角度 (度)，None 则按层自动旋转
        """
        if angle is None:
            angle = 45.0 + (layer_index % 2) * 90.0  # 奇数层旋转 90°

        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # 计算旋转后的包围盒
        bbox_min, bbox_max = self._rotated_bbox(boundary, cos_a, sin_a)

        spacing = self.line_spacing
        lines = []

        # 在旋转坐标系中生成水平线，然后转回
        y = bbox_min[1]
        while y <= bbox_max[1]:
            # 水平线（在旋转坐标系中）
            line_rotated = np.array([
                [bbox_min[0], y],
                [bbox_max[0], y],
            ], dtype=np.float64)

            # 逆旋转回原始坐标系
            line_original = self._unrotate(line_rotated, cos_a, sin_a)
            lines.append(line_original)
            y += spacing

        # 裁剪
        segments = self._clip_lines_to_polygon(lines, boundary)

        # 交替方向 (减少回抽)
        for i, seg in enumerate(segments):
            if i % 2 == 0:
                seg.points = seg.points[::-1]

        return segments

    def _rotated_bbox(self, polygon: np.ndarray, cos_a: float, sin_a: float):
        """计算旋转后的包围盒"""
        rotated = np.column_stack([
            polygon[:, 0] * cos_a + polygon[:, 1] * sin_a,
            -polygon[:, 0] * sin_a + polygon[:, 1] * cos_a,
        ])
        return rotated.min(axis=0), rotated.max(axis=0)

    def _unrotate(self, points: np.ndarray, cos_a: float, sin_a: float) -> np.ndarray:
        """逆旋转"""
        return np.column_stack([
            points[:, 0] * cos_a - points[:, 1] * sin_a,
            points[:, 0] * sin_a + points[:, 1] * cos_a,
        ])
