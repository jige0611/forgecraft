"""
螺旋体填充图案 — Gyroid 三重周期极小曲面等高线

Gyroid: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0

在固定 Z 层上，取 gyroid 曲面的等高线作为填充路径。
"""

import math
from typing import List

import numpy as np

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.types import PathSegment


class GyroidInfill(InfillPattern):
    """Gyroid 螺旋体填充"""

    def __init__(self, density: float = 0.2, extrusion_width: float = 0.45):
        super().__init__(density, extrusion_width)

    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        z: float = None,
        **kwargs,
    ) -> List[PathSegment]:
        """生成 gyroid 填充

        gyroid_func(x,y,z) = sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx)
        等高线 gyroid_func = 0 即为填充路径。
        """
        spacing = self.line_spacing
        # 频率 k = 2π / wavelength, wavelength 与 spacing 成正比
        wavelength = spacing * 2.0
        k = 2.0 * math.pi / wavelength

        z_val = z if z is not None else layer_index * 0.2

        bbox_min = boundary.min(axis=0)
        bbox_max = boundary.max(axis=0)

        # 网格采样
        resolution = spacing * 0.5
        nx = max(3, int((bbox_max[0] - bbox_min[0]) / resolution) + 1)
        ny = max(3, int((bbox_max[1] - bbox_min[1]) / resolution) + 1)
        xs = np.linspace(bbox_min[0], bbox_max[0], nx)
        ys = np.linspace(bbox_min[1], bbox_max[1], ny)
        X, Y = np.meshgrid(xs, ys)

        # 计算 gyroid 函数值
        F = (
            np.sin(k * X) * np.cos(k * Y)
            + np.sin(k * Y) * np.cos(k * z_val)
            + np.sin(k * z_val) * np.cos(k * X)
        )

        # 提取等高线
        contours = self._extract_contours(F, xs, ys, level=0.0)

        # 裁剪到边界
        segments = []
        for contour in contours:
            if len(contour) < 2:
                continue
            clipped = self._clip_lines_to_polygon([contour], boundary)
            segments.extend(clipped)

        return segments

    def _extract_contours(
        self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, level: float = 0.0
    ) -> List[np.ndarray]:
        """从 2D 标量场提取等高线 (marching squares 简化版)"""
        segments = []
        ny, nx = field.shape

        for i in range(ny - 1):
            for j in range(nx - 1):
                # 四个角的值
                v00, v10 = field[i, j], field[i, j + 1]
                v01, v11 = field[i + 1, j], field[i + 1, j + 1]

                # 四个角相对于 level 的值
                case = 0
                if v00 > level: case |= 1
                if v10 > level: case |= 2
                if v11 > level: case |= 4
                if v01 > level: case |= 8

                if case == 0 or case == 15:
                    continue

                x0, x1 = xs[j], xs[j + 1]
                y0, y1 = ys[i], ys[i + 1]

                # 边上的插值交点
                def edge_interp(a, b, va, vb):
                    if abs(vb - va) < 1e-12:
                        return 0.5
                    return (level - va) / (vb - va)

                top = np.array([x0 + edge_interp(v00, v10, v00, v10) * (x1 - x0), y0])
                right = np.array([x1, y0 + edge_interp(v10, v11, v10, v11) * (y1 - y0)])
                bottom = np.array([x0 + edge_interp(v01, v11, v01, v11) * (x1 - x0), y1])
                left = np.array([x0, y0 + edge_interp(v00, v01, v00, v01) * (y1 - y0)])

                # 16 cases
                if case == 1 or case == 14:
                    segments.append(np.array([top, left]))
                elif case == 2 or case == 13:
                    segments.append(np.array([top, right]))
                elif case == 3 or case == 12:
                    segments.append(np.array([left, right]))
                elif case == 4 or case == 11:
                    segments.append(np.array([bottom, right]))
                elif case == 5:
                    segments.append(np.array([top, left]))
                    segments.append(np.array([bottom, right]))
                elif case == 6 or case == 9:
                    segments.append(np.array([top, bottom]))
                elif case == 7 or case == 8:
                    segments.append(np.array([left, bottom]))
                elif case == 10:
                    segments.append(np.array([top, right]))
                    segments.append(np.array([left, bottom]))

        return segments
