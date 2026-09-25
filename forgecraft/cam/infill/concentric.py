"""
同心填充图案 — 轮廓反复向内偏置

类似等高线填充，从外轮廓开始逐层向内偏置直到填满。
"""

from typing import List

import numpy as np

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.types import PathSegment


class ConcentricInfill(InfillPattern):
    """同心轮廓偏置填充"""

    def __init__(self, density: float = 0.2, extrusion_width: float = 0.45):
        super().__init__(density, extrusion_width)

    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        **kwargs,
    ) -> List[PathSegment]:
        """生成同心填充

        从外轮廓开始，逐步向内偏置。
        """
        spacing = self.line_spacing
        segments = []

        # 从外边界向内偏置
        current = boundary.copy()
        # 计算中心和最大偏置距离
        center = boundary.mean(axis=0)
        max_dist = max(np.linalg.norm(p - center) for p in boundary)

        iteration = 0
        while len(current) >= 3:
            segments.append(PathSegment(
                points=current.copy(),
                is_extrude=True,
            ))
            # 向内偏置
            current = self._offset_inward(current, spacing)
            if current is None or len(current) < 3:
                break
            iteration += 1
            if iteration > 200:  # 安全上限
                break

        return segments

    def _offset_inward(self, polygon: np.ndarray, distance: float) -> np.ndarray:
        """多边形向内偏置"""
        from forgecraft.cam.toolpath import _offset_polygon
        return _offset_polygon(polygon, -distance)
