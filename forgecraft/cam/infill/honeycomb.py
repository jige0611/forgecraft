"""
蜂窝填充图案 — 正六边形镶嵌

在轮廓内生成正六边形网格路径。
"""

import math
from typing import List

import numpy as np

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.types import PathSegment


class HoneycombInfill(InfillPattern):
    """正六边形蜂窝填充"""

    def __init__(self, density: float = 0.2, extrusion_width: float = 0.45):
        super().__init__(density, extrusion_width)

    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        **kwargs,
    ) -> List[PathSegment]:
        """生成蜂窝填充

        正六边形边长 L = spacing / sqrt(3)
        """
        spacing = self.line_spacing
        # 六边形边长
        hex_side = spacing / math.sqrt(3)

        # 包围盒
        bbox_min = boundary.min(axis=0)
        bbox_max = boundary.max(axis=0)

        # 六边形网格行列间距
        dx = hex_side * 3.0
        dy = hex_side * math.sqrt(3)

        # 生成所有六边形中心
        segments = []
        y = bbox_min[1] - dy
        row = 0
        while y <= bbox_max[1] + dy:
            x_offset = (row % 2) * (hex_side * 1.5)
            x = bbox_min[0] - dx + x_offset
            while x <= bbox_max[0] + dx:
                hex_pts = self._hexagon_vertices(x, y, hex_side)
                # 裁剪：只保留在多边形内的部分
                clipped = self._clip_hex_to_boundary(hex_pts, boundary)
                if clipped is not None:
                    for seg in clipped:
                        segments.append(seg)
                x += dx
            y += dy
            row += 1

        return segments

    def _hexagon_vertices(self, cx: float, cy: float, side: float) -> np.ndarray:
        """计算正六边形 6 个顶点"""
        verts = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            verts.append([
                cx + side * math.cos(angle),
                cy + side * math.sin(angle),
            ])
        verts.append(verts[0])  # 闭合
        return np.array(verts, dtype=np.float64)

    def _clip_hex_to_boundary(
        self, hex_pts: np.ndarray, boundary: np.ndarray
    ) -> List[PathSegment]:
        """裁剪六边形到边界内"""
        from forgecraft.cam.infill.base import _clip_single_line, _points_in_polygon

        inside = _points_in_polygon(hex_pts, boundary)

        # 全在内部 → 直接返回
        if np.all(inside[:-1]):  # 最后一个点是闭合点
            return [PathSegment(points=hex_pts, is_extrude=True)]

        # 全在外面 → 跳过
        if not np.any(inside[:-1]):
            return []

        # 部分在内 → 裁剪
        clipped = _clip_single_line(hex_pts, boundary)
        if clipped is not None and len(clipped) >= 2:
            return [PathSegment(points=clipped, is_extrude=True)]

        return []
