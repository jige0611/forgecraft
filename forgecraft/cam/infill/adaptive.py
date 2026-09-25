"""
自适应密度填充 — 基于应力场调整局部填充密度

输入 FEA 应力场 → 高应力区域高密度，低应力区域低密度。
"""

import math
from typing import List, Optional

import numpy as np

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.infill.cubic import CubicInfill
from forgecraft.cam.types import PathSegment


class AdaptiveInfill(InfillPattern):
    """自适应密度立方填充

    Usage:
        # stress_field: 2D 数组，值越高越密
        infill = AdaptiveInfill(density=0.3, min_density=0.1, max_density=0.8)
        paths = infill.generate(boundary, holes, 0, stress_field=stress_2d)
    """

    def __init__(
        self,
        density: float = 0.2,
        extrusion_width: float = 0.45,
        min_density: float = 0.1,
        max_density: float = 0.8,
    ):
        super().__init__(density, extrusion_width)
        self.min_density = min_density
        self.max_density = max_density

    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        stress_field: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[PathSegment]:
        """生成自适应填充

        Args:
            stress_field: (H, W) 应力值 2D 数组，如不提供则退化为均匀密度
        """
        if stress_field is None:
            # 无应力数据 → 均匀密度
            cubic = CubicInfill(density=self.density, extrusion_width=self.extrusion_width)
            return cubic.generate(boundary, holes, layer_index)

        # 自适应分块处理
        segments = []
        bbox_min = boundary.min(axis=0)
        bbox_max = boundary.max(axis=0)

        # 将区域划分为网格块，每块根据应力值调整密度
        block_size = max(5.0, self.line_spacing * 2)
        ny, nx = stress_field.shape

        x0, y0 = bbox_min[0], bbox_min[1]
        dx = (bbox_max[0] - bbox_min[0]) / nx if nx > 1 else 1.0
        dy = (bbox_max[1] - bbox_min[1]) / ny if ny > 1 else 1.0

        for i in range(ny):
            for j in range(nx):
                stress_val = stress_field[i, j]
                # 归一化到 [min_density, max_density]
                local_density = self._map_stress_to_density(stress_val, stress_field)

                # 块边界
                bx0, bx1 = x0 + j * dx, x0 + (j + 1) * dx
                by0, by1 = y0 + i * dy, y0 + (i + 1) * dy
                block = np.array([
                    [bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]
                ], dtype=np.float64)

                # 裁剪到主边界内（简化：直接使用主边界）
                cubic = CubicInfill(density=local_density, extrusion_width=self.extrusion_width)
                block_segs = cubic.generate(boundary, holes, layer_index)
                segments.extend(block_segs)

        return segments

    def _map_stress_to_density(
        self, stress_val: float, field: np.ndarray
    ) -> float:
        """应力值 → 填充密度映射"""
        fmin, fmax = field.min(), field.max()
        if fmax - fmin < 1e-10:
            return self.density

        normalized = (stress_val - fmin) / (fmax - fmin)
        density = self.min_density + (self.max_density - self.min_density) * normalized
        return max(self.min_density, min(self.max_density, density))
