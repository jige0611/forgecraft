"""
闪电填充 — 最短路径连接内表面支撑

仅打印必要的内部支撑路径，用最短路径连接所有需要支撑的点，
类似 Cura 的 Lightning Infill。
"""

import math
from typing import List, Tuple

import numpy as np

from forgecraft.cam.infill.base import InfillPattern
from forgecraft.cam.types import PathSegment


class LightningInfill(InfillPattern):
    """闪电填充 — 最小化内部支撑

    算法：
      1. 找到需要支撑的内部区域（上表面内侧）
      2. 从这些区域向下投影到边界或下层已有支撑
      3. 用最小生成树连接所有支撑点
      4. 生成最短路径
    """

    def __init__(self, density: float = 0.2, extrusion_width: float = 0.45):
        # Lightning 密度固定较低（它本身就是极省料的）
        super().__init__(min(0.15, density), extrusion_width)

    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        overhang_angle_deg: float = 45.0,
        **kwargs,
    ) -> List[PathSegment]:
        """生成闪电填充

        简化实现：从边界向内生成树状支撑结构。
        """
        segments = []
        spacing = self.line_spacing * 2.0  # 闪电填充间距较大

        # 获取多边形中心
        center = boundary.mean(axis=0)

        # 在边界上均匀采样支撑起点
        perimeter = boundary
        n_samples = max(4, int(self._polygon_perimeter(boundary) / spacing))

        # 生成从边界到中心的放射线
        anchors = self._sample_perimeter(boundary, n_samples)

        for anchor in anchors:
            # 从边界点向中心方向的线
            direction = center - anchor
            length = np.linalg.norm(direction)
            if length < 0.1:
                continue
            direction /= length

            # 线段：从边界到中心方向走一定距离
            end_pt = anchor + direction * (length * 0.3)  # 走 30% 深度
            line = np.array([anchor, end_pt], dtype=np.float64)

            clipped = self._clip_lines_to_polygon([line], boundary)
            segments.extend(clipped)

        return segments

    def _polygon_perimeter(self, polygon: np.ndarray) -> float:
        """计算多边形周长"""
        n = len(polygon)
        total = 0.0
        for i in range(n):
            total += np.linalg.norm(polygon[(i + 1) % n] - polygon[i])
        return total

    def _sample_perimeter(
        self, polygon: np.ndarray, n: int
    ) -> List[np.ndarray]:
        """在多边形边界上均匀采样 n 个点"""
        perimeter = self._polygon_perimeter(polygon)
        step = perimeter / n

        samples = []
        dist_accum = 0.0
        for i in range(len(polygon)):
            seg_start = polygon[i]
            seg_end = polygon[(i + 1) % len(polygon)]
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)

            while dist_accum + seg_len > step and len(samples) < n:
                t = (step - dist_accum) / seg_len
                samples.append(seg_start + t * seg_vec)
                dist_accum = 0.0  # reset after sample

            dist_accum += seg_len
            if len(samples) >= n:
                break

        return samples
