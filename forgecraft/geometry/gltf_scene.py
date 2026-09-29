"""
glTF 2.0 场景构建器

将参数化装配体构建为带 PBR 材质的 glTF 场景, 导出 .glb 文件。
支持:
- 装配体节点层级 (匹配关节树)
- PBR metallic-roughness 材质
- 关节可视化 (铰链/固定的连接指示器)
- 正常视图 + 爆炸视图双模式
- 可选标注线/尺寸标注
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from forgecraft.geometry.materials import get_material, to_trimesh_material, PBRMaterial
from forgecraft.geometry.exploded import compute_exploded_positions, _build_adjacency, _bfs_order, _find_root


def build_joint_indicator(p1: np.ndarray, p2: np.ndarray,
                          joint_type: str = "fixed",
                          radius: float = 0.0015) -> Optional[trimesh.Trimesh]:
    """构建关节连接指示器 (细圆柱/箭头)"""
    vec = np.asarray(p2) - np.asarray(p1)
    dist = np.linalg.norm(vec)
    if dist < 1e-8:
        return None

    sec = 8
    cyl = trimesh.creation.cylinder(radius=radius, height=dist, sections=sec)
    direction = vec / dist
    z_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, z_axis)) > 0.9999:
        if direction[2] < 0:
            rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            rot = np.eye(4)
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction))
        rot = trimesh.transformations.rotation_matrix(angle, axis)
    mid = (np.asarray(p1) + np.asarray(p2)) / 2
    cyl.apply_transform(rot)
    cyl.apply_translation(mid)

    # 关节类型着色
    if joint_type == "hinge":
        color = (80, 180, 80, 200)   # 绿色
    elif joint_type == "prismatic":
        color = (80, 80, 220, 200)   # 蓝色
    else:
        color = (180, 180, 180, 160)  # 灰色
    cyl.visual.face_colors = color
    return cyl


def _build_parts_meshes(
    parts: List[dict],
    gen,
    positions_map: Optional[Dict[str, np.ndarray]] = None,
) -> List[Tuple[str, str, trimesh.Trimesh]]:
    """为所有零件生成参数化网格

    Returns: [(part_id, part_type, mesh), ...]
    """
    meshes = []
    for p in parts:
        pid = p.get("part_id", "")
        pt = p.get("part_type", "unknown")
        params = p.get("params", {})
        pos = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)

        # 使用指定的位置 (爆炸视图等)
        if positions_map is not None and pid in positions_map:
            pos = np.asarray(positions_map[pid], dtype=np.float64)

        mesh = gen.make(pt, params, position=pos)
        if mesh is not None:
            meshes.append((pid, pt, mesh))
    return meshes


def build_gltf_scene(
    body_data: dict,
    gen,
    exploded: bool = False,
    explode_distance: float = 0.15,
    show_joints: bool = True,
    joint_radius: float = 0.0015,
    tight: bool = True,
) -> trimesh.Scene:
    """构建带 PBR 材质的 trimesh Scene

    Args:
        body_data: {parts, joints}
        gen: ParametricGenerator 实例
        exploded: 是否使用爆炸视图位置
        explode_distance: 爆炸距离
        show_joints: 是否显示关节指示器
        joint_radius: 关节指示器半径

    Returns:
        trimesh.Scene 对象, 可直接 export('file.glb')
    """
    scene = trimesh.Scene()

    # 紧装配
    if tight and not exploded:
        try:
            from forgecraft.geometry.exploded import tight_assemble
            body_data = tight_assemble(body_data, gen)
        except Exception:
            pass

    parts = body_data.get("parts", [])
    joints = body_data.get("joints", [])

    # 爆炸视图位置
    positions_map = None
    if exploded:
        result = compute_exploded_positions(body_data, explode_distance)
        if result and "exploded" in result:
            positions_map = {
                pid: mat[:3, 3] for pid, mat in result["exploded"].items()
            }

    # 位置映射
    pos_map = {}
    for p in parts:
        pid = p.get("part_id", "")
        pos = np.array(p.get("position", [0, 0, 0]), dtype=np.float64)
        if positions_map and pid in positions_map:
            pos = positions_map[pid]
        pos_map[pid] = pos

    # 生成零件网格
    part_meshes = _build_parts_meshes(parts, gen, positions_map)

    for pid, pt, mesh in part_meshes:
        mat = get_material(pt)
        tri_mat = to_trimesh_material(mat)
        mesh.visual.material = tri_mat
        # 同时设置 face colors 作为 fallback
        c = mat.base_color
        mesh.visual.face_colors = [int(c[0] * 255), int(c[1] * 255),
                                   int(c[2] * 255), int(c[3] * 255)]
        scene.add_geometry(mesh, node_name=f"{pt}_{pid}",
                          geom_name=f"{pt}_{pid}")

    # 关节指示器
    if show_joints and not exploded:
        from forgecraft.geometry.exploded import get_joint_ends
        for j in joints:
            p1_id, p2_id, jtype = get_joint_ends(j)
            if p1_id in pos_map and p2_id in pos_map:
                indicator = build_joint_indicator(
                    pos_map[p1_id], pos_map[p2_id],
                    joint_type=jtype,
                    radius=joint_radius,
                )
                if indicator is not None:
                    scene.add_geometry(indicator, node_name=f"joint_{p1_id}_{p2_id}")

    return scene


def export_glb(
    body_data: dict,
    gen,
    output_path: str,
    exploded: bool = False,
    explode_distance: float = 0.15,
    show_joints: bool = True,
) -> str:
    """导出 .glb (Binary glTF 2.0)

    Returns:
        输出文件路径
    """
    scene = build_gltf_scene(
        body_data, gen,
        exploded=exploded,
        explode_distance=explode_distance,
        show_joints=show_joints,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scene.export(output_path, file_type="glb")
    return output_path


def export_gltf(
    body_data: dict,
    gen,
    output_dir: str,
    prefix: str = "assembly",
    explode_distance: float = 0.15,
) -> Dict[str, str]:
    """导出完整 glTF 套件 (正常 + 爆炸 + 独立 JSON)

    Returns:
        {mode: filepath, ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    result = {}

    # 正常装配体
    path = os.path.join(output_dir, f"{prefix}.glb")
    export_glb(body_data, gen, path, exploded=False, show_joints=True)
    result["normal"] = path

    # 爆炸视图
    path_exp = os.path.join(output_dir, f"{prefix}_exploded.glb")
    export_glb(body_data, gen, path_exp, exploded=True,
               explode_distance=explode_distance, show_joints=False)
    result["exploded"] = path_exp

    return result
