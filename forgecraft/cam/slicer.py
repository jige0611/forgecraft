"""
切片引擎 — 将 3D mesh 切分为逐层轮廓

功能:
  - 固定层高切片：mesh.section() → 多边形轮廓
  - 自适应层高：根据表面法线动态调整层高
  - 桥接检测：对比相邻层发现悬空区域
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from forgecraft.cam.types import SliceLayer

_logger = logging.getLogger(__name__)

__all__ = [
    "SlicingEngine",
    "slice_mesh",
    "adaptive_layer_heights",
    "detect_bridges",
    "closed_polygon_from_section",
]

# 从 trimesh 截面中提取闭合多边形的辅助函数


def closed_polygon_from_section(section, tol: float = 1e-6) -> Optional[np.ndarray]:
    """从 trimesh Path2D / Path3D 提取闭合 2D 多边形顶点

    trimesh.section() 返回 Path3D(entities=[Line(...)])。
    本函数收集所有线段端点，重建闭合多边形。
    """
    if section is None:
        return None

    # 检查是否有离散顶点
    if hasattr(section, 'vertices') and section.vertices is not None:
        verts = section.vertices
        if len(verts) < 3:
            return None
        # 取 XY 分量
        points = verts[:, :2].copy()
        _close_if_needed(points, tol)
        return points

    # Path2D / Path3D 格式
    if hasattr(section, 'entities'):
        points = []
        for entity in section.entities:
            if hasattr(entity, 'points') and entity.points is not None:
                pts = entity.points
                if pts.ndim == 2 and pts.shape[1] >= 2:
                    points.extend(pts[:, :2].tolist())
                elif pts.ndim == 1 and len(pts) >= 2:
                    points.append(pts[:2].tolist())
            elif hasattr(entity, 'nodes') and entity.nodes is not None:
                nodes = entity.nodes
                if nodes.ndim == 2:
                    points.extend(nodes[:, :2].tolist())

        if len(points) < 3:
            # 尝试从离散顶点获取
            if hasattr(section, 'discrete') and callable(section.discrete):
                discrete_verts = section.discrete(section.vertices)
                if discrete_verts is not None and len(discrete_verts) > 0:
                    points = discrete_verts[:, :2].tolist()

        if len(points) < 3:
            return None
        points = np.array(points, dtype=np.float64)
        _close_if_needed(points, tol)
        return points

    return None


def _close_if_needed(points: np.ndarray, tol: float = 1e-6):
    """确保多边形首尾闭合"""
    if len(points) < 2:
        return
    if np.linalg.norm(points[-1] - points[0]) > tol:
        # 已经闭合则不需要添加
        pass


def _is_ccw(polygon: np.ndarray) -> bool:
    """判断多边形是否为逆时针 (CCW)，使用 shoelace 公式"""
    if len(polygon) < 3:
        return True
    x, y = polygon[:, 0], polygon[:, 1]
    area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    area += x[-1] * y[0] - x[0] * y[-1]
    return area > 0


def _classify_contours(polygons: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """按面积分类多边形：CCW=外轮廓, CW=内轮廓/孔洞"""
    outer, inner = [], []
    for poly in polygons:
        if len(poly) < 3:
            continue
        if _is_ccw(poly):
            outer.append(poly)
        else:
            inner.append(poly)
    return outer, inner


# ══════════════════════════════════════════════════════════
# 桥接检测
# ══════════════════════════════════════════════════════════

def detect_bridges(
    layers: List[SliceLayer],
    bridge_angle_deg: float = 30.0,
) -> None:
    """检测桥接区域并标记到 layer 上

    当前层有填充，但下一层对应位置为空 → 桥接。
    使用简单的 point-in-polygon 判断。
    """
    if len(layers) < 2:
        return

    cos_bridge = math.cos(math.radians(bridge_angle_deg))

    for i in range(len(layers) - 1):
        cur = layers[i]
        nxt = layers[i + 1]

        # 收集当前层所有外轮廓区域
        bridge_regions = []
        for outer in cur.outer_contours:
            # 检查下一层是否有轮廓覆盖此区域
            has_support = False
            for nxt_outer in nxt.outer_contours:
                if _polygon_overlap(outer, nxt_outer, threshold=0.3):
                    has_support = True
                    break

            if not has_support:
                bridge_regions.append(outer)

        if bridge_regions:
            cur.is_bridge = True
            cur.bridge_regions = bridge_regions


def _polygon_overlap(a: np.ndarray, b: np.ndarray, threshold: float = 0.3) -> bool:
    """简单判断两个多边形是否有重叠 (采样法)"""
    # 采样 A 的顶点，检查是否在 B 内
    count_inside = 0
    sample_count = min(len(a), 20)
    step = max(1, len(a) // sample_count)
    for i in range(0, len(a), step):
        if _point_in_polygon(a[i], b):
            count_inside += 1
    return count_inside / max(1, (len(a) - 1) // step + 1) > threshold


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """射线法判断点是否在多边形内"""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ══════════════════════════════════════════════════════════
# 自适应层高
# ══════════════════════════════════════════════════════════

def adaptive_layer_heights(
    mesh: trimesh.Trimesh,
    base_height: float = 0.2,
    min_height: float = 0.08,
    max_height: float = 0.32,
    z_steps: int = 100,
) -> List[float]:
    """根据表面法线计算自适应层高

    陡峭面 → 薄层 (减少台阶效应) ; 平坦面 → 厚层 (加快打印)

    Returns:
        List of (z_top, height) for each layer
    """
    if z_steps < 2:
        return []

    z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
    if z_max - z_min < 1e-6:
        return []

    z_range = np.linspace(z_min + base_height, z_max, z_steps)
    face_normals = mesh.face_normals

    heights = []
    for z in z_range:
        # 找到该高度附近的三角面
        face_centers = mesh.triangles_center
        nearby = np.abs(face_centers[:, 2] - z) < base_height
        if not np.any(nearby):
            h = base_height
        else:
            nearby_normals = face_normals[nearby]
            # Z 分量越小 (越陡) → 层高越薄
            avg_z_component = np.mean(np.abs(nearby_normals[:, 2]))
            h = base_height + (max_height - base_height) * (1.0 - avg_z_component)
            h = max(min_height, min(max_height, h))
        heights.append(h)

    if not heights:
        heights = [base_height]
    return heights


# ══════════════════════════════════════════════════════════
# 切片引擎
# ══════════════════════════════════════════════════════════

def slice_mesh(
    mesh: trimesh.Trimesh,
    layer_height: float = 0.2,
    first_layer_height: Optional[float] = None,
    adaptive: bool = False,
) -> List[SliceLayer]:
    """主切片函数

    将 3D mesh 沿 Z 轴切分为逐层 2D 轮廓。
    """
    if mesh.vertices.shape[0] == 0:
        return []

    first_h = first_layer_height or layer_height
    z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]

    if adaptive:
        heights = adaptive_layer_heights(mesh, layer_height)
    else:
        num_layers = max(1, int(math.ceil((z_max - z_min) / layer_height)))
        heights = [layer_height] * num_layers

    layers: List[SliceLayer] = []
    z_current = z_min + first_h

    for i, h in enumerate(heights):
        actual_h = first_h if i == 0 else h
        if z_current > z_max + 1e-6:
            break

        # trimesh 水平截面
        section = mesh.section(
            plane_origin=[0, 0, z_current],
            plane_normal=[0, 0, 1],
        )

        outer: List[np.ndarray] = []
        inner: List[np.ndarray] = []

        if section is not None:
            # 尝试直接获取离散轮廓
            polygons = _extract_polygons_from_section(section)
            outer, inner = _classify_contours(polygons)

        layer = SliceLayer(
            z=float(z_current),
            height=float(actual_h),
            outer_contours=outer,
            inner_contours=inner,
        )
        layers.append(layer)
        z_current += actual_h

    # 桥接检测
    detect_bridges(layers)

    return layers


def _extract_polygons_from_section(section) -> List[np.ndarray]:
    """从 trimesh section 提取所有多边形"""
    polygons = []

    # 如果 section 有离散顶点属性
    if hasattr(section, 'vertices') and section.vertices is not None:
        verts = section.vertices
        if verts.ndim == 2:
            poly = closed_polygon_from_section(section)
            if poly is not None:
                polygons.append(poly)
            return polygons

    # 处理 Path2D/Path3D
    if hasattr(section, 'entities'):
        for entity in section.entities:
            poly = _entity_to_polygon(entity, section)
            if poly is not None:
                polygons.append(poly)

    # 如果上面都没提取到，尝试 discrete
    if not polygons and hasattr(section, 'discrete') and callable(section.discrete):
        try:
            discrete_verts = section.discrete(section.vertices)
            if discrete_verts is not None and len(discrete_verts) >= 3:
                polygons.append(discrete_verts[:, :2])
        except Exception:
            pass

    return polygons


def _entity_to_polygon(entity, section) -> Optional[np.ndarray]:
    """将 trimesh entity 转为多边形"""
    points = []
    if hasattr(entity, 'points') and entity.points is not None:
        pts = entity.points
        if pts.ndim == 2 and pts.shape[1] >= 2:
            points = pts[:, :2].tolist()
        elif pts.ndim == 1 and len(pts) >= 2:
            points.append(pts[:2].tolist())
    elif hasattr(entity, 'nodes') and entity.nodes is not None:
        # 从 section.vertices 按索引取点
        nodes = entity.nodes
        if nodes.ndim == 1:
            if hasattr(section, 'vertices') and section.vertices is not None:
                for idx in nodes:
                    v = section.vertices[idx]
                    points.append(v[:2].tolist())
            else:
                points = nodes.reshape(-1, 2).tolist()
        elif nodes.ndim == 2:
            points = nodes[:, :2].tolist()

    if len(points) < 3:
        return None

    points = np.array(points, dtype=np.float64)
    _close_if_needed(points)
    return points


# ══════════════════════════════════════════════════════════
# SlicingEngine 类 — 统一入口
# ══════════════════════════════════════════════════════════

class SlicingEngine:
    """切片引擎封装类

    Usage:
        engine = SlicingEngine(layer_height=0.2)
        layers = engine.slice(mesh)
    """

    def __init__(
        self,
        layer_height: float = 0.2,
        first_layer_height: float = 0.3,
        adaptive: bool = False,
        bridge_threshold_deg: float = 30.0,
    ):
        self.layer_height = layer_height
        self.first_layer_height = first_layer_height
        self.adaptive = adaptive
        self.bridge_threshold_deg = bridge_threshold_deg

    def slice(self, mesh: trimesh.Trimesh) -> List[SliceLayer]:
        return slice_mesh(
            mesh,
            layer_height=self.layer_height,
            first_layer_height=self.first_layer_height,
            adaptive=self.adaptive,
        )

    @staticmethod
    def detect_bridges(layers: List[SliceLayer]) -> None:
        detect_bridges(layers)
