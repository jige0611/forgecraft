"""
凸分解碰撞体 — 从 trimesh → MuJoCo convex mesh

用迭代凸包分解替代简单原语 (cylinder/box/sphere) 碰撞体。
每个零件生成 1-8 个凸包 mesh，大幅提升碰撞精度。

算法:
  1. 计算零件 mesh 的凸包 (convex_hull)
  2. 找到距离凸包最远的点簇
  3. 如果 max_distance > threshold，在凹陷处分片
  4. 递归每片
  5. 导出凸包 STL → MuJoCo <geom type="mesh">

不依赖 V-HACD，纯 trimesh + scipy/numpy。

使用:
  >>> from forgecraft.simulation.convex_collision import convex_decompose
  >>> hulls = convex_decompose(part_mesh, max_hulls=4)
  >>> for i, h in enumerate(hulls):
  >>>     h.export(f"collision_{i}.stl")
"""

from typing import List, Optional, Tuple
import logging

import numpy as np
import trimesh

_logger = logging.getLogger(__name__)

__all__ = [
    "convex_decompose",
    "mesh_to_collision_geoms",
    "export_collision_meshes",
    "ConvexCollisionBuilder",
]


def convex_decompose(
    mesh: trimesh.Trimesh,
    max_hulls: int = 4,
    concavity_threshold: float = 0.02,
) -> List[trimesh.Trimesh]:
    """迭代凸包分解
    
    算法: 体积比法
      1. 计算凸包体积 V_hull
      2. 计算 mesh 体积 V_mesh
      3. 如果 V_mesh / V_hull > 1 - threshold → 近似凸，返回 [hull]
      4. 否则沿 PCA 主轴分两半，递归
    """
    if max_hulls <= 1 or len(mesh.vertices) < 4:
        hull = mesh.convex_hull
        return [hull] if isinstance(hull, trimesh.Trimesh) else [mesh]
    
    hull = mesh.convex_hull
    if not isinstance(hull, trimesh.Trimesh):
        return [mesh]
    
    # 体积比检测凹陷
    try:
        v_mesh = abs(mesh.volume) if mesh.is_watertight else abs(mesh.convex_hull.volume) * 0.8
    except Exception:
        v_mesh = abs(hull.volume) * 0.8
    v_hull = abs(hull.volume)
    
    if v_hull < 1e-12:
        return [hull]
    
    vol_ratio = v_mesh / v_hull
    
    # 体积比 > 0.85 → 接近凸，不分解
    if vol_ratio > 0.85 or max_hulls <= 1:
        return [hull]
    
    # 沿最长 PCA 轴分片
    centered = mesh.vertices - mesh.vertices.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        split_axis_vec = vh[0]  # 第一主成分
    except Exception:
        split_axis_vec = np.array([1.0, 0.0, 0.0])
    
    projections = centered @ split_axis_vec
    split_val = np.median(projections)
    
    left_mask = projections <= split_val
    right_mask = ~left_mask
    
    # 按顶点分面
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)
    face_proj = (face_centers - mesh.vertices.mean(axis=0)) @ split_axis_vec
    
    left_faces = face_proj <= split_val
    right_faces = ~left_faces
    
    pieces = []
    for fmask in [left_faces, right_faces]:
        if fmask.sum() < 3:
            continue
        submesh = mesh.submesh([fmask], append=True)
        if len(submesh.vertices) >= 4:
            pieces.append(submesh)
    
    if len(pieces) < 2:
        return [hull]
    
    # 递归
    all_hulls = []
    remaining = max_hulls
    per_piece = max(1, remaining // len(pieces))
    for piece in pieces:
        piece_hulls = convex_decompose(piece, per_piece, concavity_threshold)
        all_hulls.extend(piece_hulls)
        if len(all_hulls) >= max_hulls:
            break
    
    if len(all_hulls) > max_hulls:
        all_hulls = all_hulls[:max_hulls]
    
    return all_hulls


def mesh_to_collision_geoms(
    mesh: trimesh.Trimesh,
    quality: str = "medium",
    max_hulls: int = 4,
) -> List[trimesh.Trimesh]:
    """将零件 mesh 转为 MuJoCo 碰撞体列表
    
    Returns:
        凸包 mesh 列表 (每个都是 convex hull, 可导出 STL)
    """
    if mesh.is_convex:
        return [mesh]
    
    # Decimate for collision (600 verts max per hull)
    target_verts = min(600, len(mesh.vertices))
    if len(mesh.vertices) > target_verts:
        try:
            mesh = mesh.simplify_quadric_decimation(target_verts)
        except Exception:
            pass
    
    return convex_decompose(mesh, max_hulls=max_hulls)


def export_collision_meshes(
    hulls: List[trimesh.Trimesh],
    output_dir: str,
    prefix: str = "collision",
) -> List[str]:
    """导出碰撞体 STL 到指定目录"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    files = []
    for i, hull in enumerate(hulls):
        fname = os.path.join(output_dir, f"{prefix}_{i}.stl")
        hull.export(fname)
        files.append(fname)
    return files


class ConvexCollisionBuilder:
    """凸分解碰撞体构建器 — 集成到仿真管线
    
    使用:
        builder = ConvexCollisionBuilder(max_hulls=3, quality="high")
        collision_files = builder.build_for_body(mesh, "output/collision/")
    """
    
    def __init__(self, max_hulls: int = 3, quality: str = "medium"):
        self.max_hulls = max_hulls
        self.quality = quality
    
    def build_single(self, mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """单零件碰撞体"""
        return mesh_to_collision_geoms(mesh, self.quality, self.max_hulls)
    
    def export_all(self, hulls: List[trimesh.Trimesh],
                   output_dir: str, prefix: str = "collision") -> List[str]:
        return export_collision_meshes(hulls, output_dir, prefix)
    
    def estimate_quality(self, hulls: List[trimesh.Trimesh],
                          original: trimesh.Trimesh) -> dict:
        """估算碰撞体质量"""
        hull_vol = sum(h.volume for h in hulls)
        orig_vol = original.volume if original.is_watertight else hull_vol
        return {
            "num_hulls": len(hulls),
            "total_vertices": sum(len(h.vertices) for h in hulls),
            "volume_ratio": hull_vol / max(orig_vol, 1e-9),
            "is_watertight": all(h.is_watertight for h in hulls),
        }
