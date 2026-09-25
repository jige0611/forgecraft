"""
立方填充图案 — 三维立方晶格投影

在每层生成正交的方形网格，层间错位形成 3D 立方晶格效果。
"""

import math
from typing import List

import numpy as np

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.types import PathSegment


class CubicInfill(InfillPattern):
    """立方晶格填充

    每隔一层生成方向变化的网格：层 0=x方向, 层1=y方向, 层2=复合45°
    """

    def __init__(self, density: float = 0.2, extrusion_width: float = 0.45):
        super().__init__(density, extrusion_width)

    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        **kwargs,
    ) -> List[PathSegment]:
        """生成立方填充"""
        spacing = self.line_spacing

        # 每 3 层循环：0=x, 1=y, 2=45°复合
        pattern_idx = layer_index % 3

        if pattern_idx == 0:
            angle = 0.0
        elif pattern_idx == 1:
            angle = 90.0
        else:
            angle = 45.0

        return self._generate_parallel(boundary, spacing, angle)

    def _generate_parallel(
        self, boundary: np.ndarray, spacing: float, angle_deg: float
    ) -> List[PathSegment]:
        """生成平行线填充"""
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # 旋转包围盒
        rotated = np.column_stack([
            boundary[:, 0] * cos_a + boundary[:, 1] * sin_a,
            -boundary[:, 0] * sin_a + boundary[:, 1] * cos_a,
        ])
        bbox_min = rotated.min(axis=0)
        bbox_max = rotated.max(axis=0)

        lines = []
        y = bbox_min[1]
        while y <= bbox_max[1]:
            line_rot = np.array([[bbox_min[0], y], [bbox_max[0], y]], dtype=np.float64)
            line_orig = np.column_stack([
                line_rot[:, 0] * cos_a - line_rot[:, 1] * sin_a,
                line_rot[:, 0] * sin_a + line_rot[:, 1] * cos_a,
            ])
            lines.append(line_orig)
            y += spacing

        segments = self._clip_lines_to_polygon(lines, boundary)

        for i, seg in enumerate(segments):
            if i % 2 == 0:
                seg.points = seg.points[::-1]

        return segments
