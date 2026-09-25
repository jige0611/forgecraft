"""
填充图案抽象基类
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from forgecraft.cam.types import PathSegment


class InfillPattern(ABC):
    """填充图案基类

    子类需实现 generate() 方法，接收轮廓多边形，返回 PathSegment 列表。
    """

    def __init__(self, density: float = 0.2, extrusion_width: float = 0.45):
        """
        Args:
            density: 填充密度 0.0 ~ 1.0
            extrusion_width: 挤出线宽 (mm)
        """
        self.density = max(0.05, min(1.0, density))
        self.extrusion_width = extrusion_width

    @property
    def line_spacing(self) -> float:
        """根据密度计算线间距"""
        if self.density <= 0.0:
            return 100.0  # 几乎无线
        return self.extrusion_width / self.density

    @abstractmethod
    def generate(
        self,
        boundary: np.ndarray,
        holes: List[np.ndarray],
        layer_index: int = 0,
        **kwargs,
    ) -> List[PathSegment]:
        """生成填充路径

        Args:
            boundary: 外轮廓多边形 (N, 2)
            holes: 内轮廓/孔洞列表
            layer_index: 层索引 (用于逐层旋转)
            **kwargs: 图案特定参数

        Returns:
            填充路径段列表
        """
        ...

    def _clip_lines_to_polygon(
        self, lines: List[np.ndarray], boundary: np.ndarray
    ) -> List[PathSegment]:
        """将线段裁剪到多边形内部 (Sutherland-Hodgman 变体)

        简化版：检查每条线的端点是否在多边形内，对跨边界线段做交点计算。
        """
        segments = []
        for line in lines:
            if len(line) < 2:
                continue
            clipped = _clip_single_line(line, boundary)
            if clipped is not None and len(clipped) >= 2:
                segments.append(PathSegment(
                    points=clipped,
                    is_extrude=True,
                ))
        return segments


def _clip_single_line(line: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """对单条折线用多边形裁剪 (简单版：取线段在多边形内的部分)"""
    # 如果都在内部，直接返回
    inside_mask = _points_in_polygon(line, boundary)
    if np.all(inside_mask):
        return line

    # 如果全在外面，检查是否穿过 (两端都在外面，但线段穿过多边形)
    if not np.any(inside_mask):
        # 找出所有与边界的交点
        intersections = []
        for i in range(len(line) - 1):
            p0, p1 = line[i], line[i + 1]
            inter = _intersect_with_boundary(p0, p1, boundary)
            if inter is not None:
                intersections.append((_param_on_segment(p0, p1, inter), inter))
        if len(intersections) >= 2:
            # 按参数排序 → 取最小和最大交点为内部段
            intersections.sort(key=lambda x: x[0])
            return np.array([intersections[0][1], intersections[-1][1]], dtype=np.float64)
        return None

    # 找出跨越边界的线段并插入交点
    result_pts = [line[0]]
    for i in range(1, len(line)):
        p0, p1 = line[i - 1], line[i]
        i0, i1 = inside_mask[i - 1], inside_mask[i]

        if i1:
            if not i0:
                # 从外到内 → 先加交点
                inter = _intersect_with_boundary(p0, p1, boundary)
                if inter is not None:
                    result_pts.append(inter)
            result_pts.append(p1)
        else:
            if i0:
                # 从内到外 → 加交点
                inter = _intersect_with_boundary(p0, p1, boundary)
                if inter is not None:
                    result_pts.append(inter)

    if len(result_pts) < 2:
        return None
    return np.array(result_pts, dtype=np.float64)


def _param_on_segment(p0, p1, pt):
    """返回 pt 在 p0→p1 上的参数 t"""
    d = p1 - p0
    denom = np.dot(d, d)
    if denom < 1e-14:
        return 0.0
    return np.dot(pt - p0, d) / denom


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """批量判断点是否在多边形内"""
    result = np.zeros(len(points), dtype=bool)
    n = len(polygon)
    for k, p in enumerate(points):
        x, y = p
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        result[k] = inside
    return result


def _intersect_with_boundary(p0: np.ndarray, p1: np.ndarray, boundary: np.ndarray) -> Optional[np.ndarray]:
    """线段 p0→p1 与多边形边界的交点"""
    best_t = float('inf')
    best_pt = None
    n = len(boundary)
    for i in range(n):
        q0 = boundary[i]
        q1 = boundary[(i + 1) % n]
        t = _segment_intersection_param(p0, p1, q0, q1)
        if t is not None and 0 < t < best_t:
            best_t = t
            best_pt = p0 + t * (p1 - p0)
    return best_pt


def _segment_intersection_param(a0, a1, b0, b1):
    """计算线段 a0a1 和 b0b1 的交点参数 (a0→a1 的参数 t)"""
    d1 = a1 - a0
    d2 = b1 - b0
    cross = np.cross(d1, d2)
    if abs(cross) < 1e-10:
        return None
    t = np.cross(b0 - a0, d2) / cross
    u = np.cross(b0 - a0, d1) / cross
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t
    return None
