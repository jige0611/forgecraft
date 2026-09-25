"""
路径规划器 — 外壳 + 填充 + 顶底皮 + 桥接 + 支撑路径生成

功能:
  - PerimeterGenerator : 轮廓向内偏置生成外壳
  - InfillGenerator    : 填充图案分发
  - SkinGenerator      : 顶/底层实心填充
  - BridgeGenerator    : 桥接区域路径
  - ToolpathOptimizer  : 路径排序优化
"""

import math
import logging
from typing import Dict, List, Optional

import numpy as np

from forgecraft.cam.types import SliceLayer, LayerToolpath, PathSegment
from forgecraft.cam.infill import (
    InfillPattern,
    RectilinearInfill,
    HoneycombInfill,
    GyroidInfill,
    CubicInfill,
    ConcentricInfill,
    AdaptiveInfill,
    LightningInfill,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "ToolpathGenerator",
    "PerimeterGenerator",
    "SkinGenerator",
    "BridgeGenerator",
    "ToolpathOptimizer",
    "generate_toolpath",
]

# 填充图案注册表
_INFILL_REGISTRY = {
    "rectilinear": RectilinearInfill,
    "honeycomb": HoneycombInfill,
    "gyroid": GyroidInfill,
    "cubic": CubicInfill,
    "concentric": ConcentricInfill,
    "adaptive_cubic": AdaptiveInfill,
    "lightning": LightningInfill,
    "solid": RectilinearInfill,  # solid = rectilinear with density=1.0
}


# ══════════════════════════════════════════════════════════
# Perimeter 生成器
# ══════════════════════════════════════════════════════════

class PerimeterGenerator:
    """外壳路径生成 — 轮廓向内偏置"""

    def __init__(self, perimeter_count: int = 3, extrusion_width: float = 0.45):
        self.perimeter_count = perimeter_count
        self.extrusion_width = extrusion_width

    def generate(self, outer_contours: List[np.ndarray]) -> List[PathSegment]:
        """对外轮廓向内偏置生成多层外壳"""
        segments = []
        for contour in outer_contours:
            for i in range(self.perimeter_count):
                offset_dist = self.extrusion_width * (0.5 + i)
                offset_poly = _offset_polygon(contour, -offset_dist)
                if offset_poly is None or len(offset_poly) < 3:
                    break
                segments.append(PathSegment(
                    points=offset_poly,
                    is_extrude=True,
                ))
        return segments


def _offset_polygon(polygon: np.ndarray, distance: float) -> Optional[np.ndarray]:
    """简单多边形偏置

    使用顶点法线偏移近似。对凸多边形有效，凹多边形可能产生自交。
    """
    if len(polygon) < 3:
        return None

    n = len(polygon)
    offset_pts = np.zeros_like(polygon)

    for i in range(n):
        prev = polygon[(i - 1) % n]
        curr = polygon[i]
        next_pt = polygon[(i + 1) % n]

        # 计算顶点法线 (前后边的角平分线方向)
        e1 = curr - prev
        e2 = next_pt - curr
        e1_len = np.linalg.norm(e1)
        e2_len = np.linalg.norm(e2)

        if e1_len < 1e-10 or e2_len < 1e-10:
            offset_pts[i] = curr
            continue

        e1_n = e1 / e1_len
        e2_n = e2 / e2_len

        # 角平分线方向 = normalize(e1_n + e2_n)
        bisector = e1_n + e2_n
        bis_len = np.linalg.norm(bisector)

        if bis_len < 1e-10:
            # 平角 → 用垂直方向
            normal = np.array([-e1_n[1], e1_n[0]])
            offset_pts[i] = curr + normal * distance
        else:
            bisector /= bis_len
            # 判断内/外方向 (CCW → 法线朝内，向内偏置 direction = -bisector 的法线)
            normal = np.array([-bisector[1], bisector[0]])
            offset_pts[i] = curr + normal * distance

    return offset_pts


# ══════════════════════════════════════════════════════════
# Skin 生成器 (顶/底层)
# ══════════════════════════════════════════════════════════

class SkinGenerator:
    """顶/底层实心填充"""

    def __init__(
        self,
        top_layers: int = 4,
        bottom_layers: int = 3,
        skin_angle_deg: float = 45.0,
        extrusion_width: float = 0.45,
    ):
        self.top_layers = top_layers
        self.bottom_layers = bottom_layers
        self.skin_angle_deg = skin_angle_deg
        self.extrusion_width = extrusion_width

    def generate(
        self,
        layer_index: int,
        total_layers: int,
        boundary: np.ndarray,
        holes: List[np.ndarray],
    ) -> Optional[List[PathSegment]]:
        """判断当前层是否需要 skin 填充"""
        is_top = layer_index >= total_layers - self.top_layers
        is_bottom = layer_index < self.bottom_layers

        if not is_top and not is_bottom:
            return None

        # 实心 = rectilinear 密度 1.0
        infill = RectilinearInfill(density=1.0, extrusion_width=self.extrusion_width)
        angle = self.skin_angle_deg + (layer_index % 2) * 90.0
        return infill.generate(boundary, holes, layer_index, angle=angle)


# ══════════════════════════════════════════════════════════
# Bridge 生成器
# ══════════════════════════════════════════════════════════

class BridgeGenerator:
    """桥接区域路径生成"""

    def __init__(self, extrusion_width: float = 0.45):
        self.extrusion_width = extrusion_width

    def generate(self, bridge_regions: List[np.ndarray]) -> List[PathSegment]:
        """为桥接区域生成 100% 填充路径"""
        segments = []
        for region in bridge_regions:
            if len(region) < 3:
                continue
            # 使用直线填充，密度极高 (桥接需要密实)
            infill = RectilinearInfill(density=1.0, extrusion_width=self.extrusion_width)
            segs = infill.generate(region, [], 0, angle=0.0)
            segments.extend(segs)
        return segments


# ══════════════════════════════════════════════════════════
# Toolpath 优化器
# ══════════════════════════════════════════════════════════

class ToolpathOptimizer:
    """路径排序优化 — Greedy NN + 2-opt"""

    @staticmethod
    def optimize(segments: List[PathSegment]) -> List[PathSegment]:
        """对路径段排序，减少空驶距离"""
        if len(segments) <= 1:
            return segments

        # 贪心最近邻排序
        ordered = [segments[0]]
        remaining = segments[1:]

        while remaining:
            last_pt = ordered[-1].points[-1] if len(ordered[-1].points) > 0 else np.zeros(2)
            # 找最近起点
            best_idx, best_dist = 0, float('inf')
            for i, seg in enumerate(remaining):
                d = np.linalg.norm(seg.points[0] - last_pt)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            ordered.append(remaining.pop(best_idx))

        return ordered

    @staticmethod
    def insert_travel_moves(segments: List[PathSegment]) -> List[PathSegment]:
        """在非连续路径间插入空驶移动"""
        if not segments:
            return []

        result = [segments[0]]
        for i in range(1, len(segments)):
            prev_end = segments[i - 1].points[-1]
            next_start = segments[i].points[0]
            dist = np.linalg.norm(next_start - prev_end)

            if dist > 0.5:  # > 0.5mm 插入空驶
                result.append(PathSegment(
                    points=np.array([prev_end, next_start]),
                    is_extrude=False,
                ))
            result.append(segments[i])

        return result


# ══════════════════════════════════════════════════════════
# ToolpathGenerator 主类
# ══════════════════════════════════════════════════════════

class ToolpathGenerator:
    """路径规划总控

    Usage:
        gen = ToolpathGenerator(infill_pattern="rectilinear", infill_density=0.2)
        layers = gen.generate_all(sliced_layers)
    """

    def __init__(
        self,
        infill_pattern: str = "rectilinear",
        infill_density: float = 0.2,
        perimeter_count: int = 3,
        top_layers: int = 4,
        bottom_layers: int = 3,
        extrusion_width: float = 0.45,
        optimize: bool = True,
    ):
        self.infill_pattern = infill_pattern
        self.infill_density = infill_density
        self.extrusion_width = extrusion_width
        self.optimize = optimize

        self.perimeter_gen = PerimeterGenerator(
            perimeter_count=perimeter_count,
            extrusion_width=extrusion_width,
        )
        self.skin_gen = SkinGenerator(
            top_layers=top_layers,
            bottom_layers=bottom_layers,
            extrusion_width=extrusion_width,
        )
        self.bridge_gen = BridgeGenerator(extrusion_width=extrusion_width)

        # 创建填充实例
        infill_cls = _INFILL_REGISTRY.get(infill_pattern, RectilinearInfill)
        if infill_pattern == "solid":
            density = 1.0
        else:
            density = infill_density
        self.infill_engine = infill_cls(density=density, extrusion_width=extrusion_width)

    def generate_all(self, layers: List[SliceLayer]) -> List[LayerToolpath]:
        """为所有切片层生成完整路径"""
        total = len(layers)
        toolpaths = []

        for i, layer in enumerate(layers):
            tp = self._generate_layer(layer, i, total)
            toolpaths.append(tp)

        return toolpaths

    def _generate_layer(
        self, layer: SliceLayer, layer_idx: int, total_layers: int
    ) -> LayerToolpath:
        tp = LayerToolpath(z=layer.z, height=layer.height)

        if not layer.outer_contours:
            return tp

        boundary = layer.outer_contours[0]  # 主外轮廓

        # 1. 外壳
        tp.perimeters = self.perimeter_gen.generate(layer.outer_contours)

        # 2. 填充
        tp.infill = self.infill_engine.generate(
            boundary, layer.inner_contours, layer_idx,
        )

        # 3. 顶/底层
        skin = self.skin_gen.generate(
            layer_idx, total_layers, boundary, layer.inner_contours,
        )
        if skin:
            tp.skin = skin

        # 4. 桥接
        if layer.is_bridge and layer.bridge_regions:
            tp.bridges = self.bridge_gen.generate(layer.bridge_regions)

        # 5. 路径优化
        if self.optimize:
            all_extrude = tp.perimeters + tp.infill + tp.skin + tp.bridges
            optimized = ToolpathOptimizer.optimize(all_extrude)
            with_travel = ToolpathOptimizer.insert_travel_moves(optimized)

            # 分类回各字段
            tp.perimeters = [s for s in with_travel if s in tp.perimeters]
            tp.infill = [s for s in with_travel if s in tp.infill]
            tp.travel = [s for s in with_travel if not s.is_extrude]

        return tp


# ══════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════

def generate_toolpath(
    layers: List[SliceLayer],
    infill_pattern: str = "rectilinear",
    infill_density: float = 0.2,
    perimeter_count: int = 3,
    extrusion_width: float = 0.45,
) -> List[LayerToolpath]:
    """便捷全流程函数"""
    gen = ToolpathGenerator(
        infill_pattern=infill_pattern,
        infill_density=infill_density,
        perimeter_count=perimeter_count,
        extrusion_width=extrusion_width,
    )
    return gen.generate_all(layers)
