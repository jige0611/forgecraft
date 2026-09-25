"""
连接结构生成器 — 使零件可物理组装

功能：
1. 螺栓孔 — 关节处生成 M2/M3 通孔，用于螺丝组装
2. 卡扣 — 固定连接处生成 snap-fit 凸凹结构
3. 轴孔配合 — 铰链处生成轴承座，公差可调
4. 燕尾榫 — 免工具组装，3D打印友好

算法：
  - 螺栓孔 = CSG 差集 (零件 - 圆柱体)
  - 卡扣 = 凸 (hook) + 凹 (socket) 对偶 mesh
  - 燕尾榫 = 梯形截面拉伸体
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "add_bolt_holes_to_part",
    "add_snap_fit_to_parts",
    "add_hinge_bearing",
    "add_dovetail_to_parts",
]




def _make_bolt_hole(radius: float, depth: float = 0.05, hole_type: str = "through") -> trimesh.Trimesh:
    height = depth * 2.0 if hole_type == "through" else depth
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    return cyl


def _make_counterbore(bolt_r: float, head_r: float, head_depth: float, total_depth: float) -> trimesh.Trimesh:
    through = trimesh.creation.cylinder(radius=bolt_r, height=total_depth * 2, sections=16)
    cbore = trimesh.creation.cylinder(radius=head_r, height=head_depth, sections=16)
    cbore.apply_translation([0, 0, total_depth - head_depth / 2])
    return trimesh.util.concatenate([through, cbore])


def _make_snap_hook(width: float, depth: float, height: float) -> trimesh.Trimesh:
    hook = trimesh.creation.box(extents=(width, depth, height))
    latch = trimesh.creation.box(extents=(width * 1.3, depth * 0.6, height * 0.3))
    latch.apply_translation([0, 0, height / 2 + height * 0.15])
    return trimesh.util.concatenate([hook, latch])


def _make_snap_socket(width: float, depth: float, height: float, clearance: float) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(width + clearance, depth + clearance, height + clearance * 2))


def _make_dovetail_male(width: float, length: float, angle_deg: float = 15.0) -> trimesh.Trimesh:
    angle = math.radians(angle_deg)
    top_width = width
    bot_width = width - 2 * length * math.tan(angle)
    bot_width = max(bot_width, width * 0.3)

    verts = np.array([
        [-top_width / 2, -length / 2, 0],
        [top_width / 2, -length / 2, 0],
        [top_width / 2, length / 2, 0],
        [-top_width / 2, length / 2, 0],
        [-bot_width / 2, -length / 2, length],
        [bot_width / 2, -length / 2, length],
        [bot_width / 2, length / 2, length],
        [-bot_width / 2, length / 2, length],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 1], [4, 1, 0],
        [5, 6, 2], [5, 2, 1],
        [6, 7, 3], [6, 3, 2],
        [7, 4, 0], [7, 0, 3],
        [4, 7, 6], [4, 6, 5],
    ])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.remove_unreferenced_vertices()
    return mesh


def add_bolt_holes_to_part(
    part_mesh: trimesh.Trimesh,
    connection_points: List[Tuple[np.ndarray, np.ndarray]],
    bolt_radius: float = 0.0015,
    clearance: float = 0.0002,
) -> trimesh.Trimesh:
    """为零件网格添加螺栓孔: CSG 差集 (零件 - 圆柱体)"""
    result = part_mesh.copy()

    for origin, direction in connection_points:
        bolt_r = bolt_radius + clearance
        hole = _make_bolt_hole(bolt_r, depth=0.04, hole_type="through")

        direction = direction / (np.linalg.norm(direction) + 1e-9)
        z_axis = np.array([0, 0, 1.0])
        angle = math.acos(np.clip(np.dot(z_axis, direction), -1, 1))
        if angle > 1e-6 and angle < math.pi - 1e-6:
            rot_axis = np.cross(z_axis, direction)
            rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-9)
            rot = trimesh.transformations.rotation_matrix(angle, rot_axis)
            hole.apply_transform(rot)

        hole.apply_translation(origin)
        try:
            result = result.difference(hole, engine="scad")
        except Exception:
            pass

    return result


def add_snap_fit_to_parts(
    parent_mesh: trimesh.Trimesh,
    child_mesh: trimesh.Trimesh,
    joint_position: np.ndarray,
    joint_axis: np.ndarray,
    snap_width: float = 0.008,
    snap_depth: float = 0.006,
    snap_height: float = 0.004,
    clearance: float = 0.0003,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    parent = parent_mesh.copy()
    child = child_mesh.copy()

    hook = _make_snap_hook(snap_width, snap_depth, snap_height)
    socket = _make_snap_socket(snap_width, snap_depth, snap_height, clearance)

    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-9)
    z_axis = np.array([0, 0, 1.0])
    angle = math.acos(np.clip(np.dot(z_axis, joint_axis), -1, 1))
    if angle > 1e-6 and angle < math.pi - 1e-6:
        rot_axis = np.cross(z_axis, joint_axis)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-9)
        rot = trimesh.transformations.rotation_matrix(angle, rot_axis)
        hook.apply_transform(rot)
        socket.apply_transform(rot)

    hook.apply_translation(joint_position)
    socket.apply_translation(joint_position)

    try:
        parent = parent.union(hook, engine="scad")
        child = child.difference(socket, engine="scad")
    except Exception:
        pass

    return parent, child


def add_hinge_bearing(
    part_mesh: trimesh.Trimesh,
    joint_position: np.ndarray,
    joint_axis: np.ndarray,
    shaft_radius: float = 0.003,
    bearing_thickness: float = 0.004,
    clearance: float = 0.0003,
) -> trimesh.Trimesh:
    result = part_mesh.copy()

    hole_r = shaft_radius + clearance
    hole = _make_bolt_hole(hole_r, depth=bearing_thickness * 2, hole_type="through")

    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-9)
    z_axis = np.array([0, 0, 1.0])
    angle = math.acos(np.clip(np.dot(z_axis, joint_axis), -1, 1))
    if angle > 1e-6 and angle < math.pi - 1e-6:
        rot_axis = np.cross(z_axis, joint_axis)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-9)
        rot = trimesh.transformations.rotation_matrix(angle, rot_axis)
        hole.apply_transform(rot)

    hole.apply_translation(joint_position)

    try:
        result = result.difference(hole, engine="scad")
    except Exception:
        pass

    return result


def add_dovetail_to_parts(
    parent_mesh: trimesh.Trimesh,
    child_mesh: trimesh.Trimesh,
    joint_position: np.ndarray,
    joint_axis: np.ndarray,
    dovetail_width: float = 0.01,
    dovetail_length: float = 0.008,
    clearance: float = 0.0003,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    parent = parent_mesh.copy()
    child = child_mesh.copy()

    male = _make_dovetail_male(dovetail_width, dovetail_length, angle_deg=15.0)
    female = _make_box(dovetail_width + clearance, dovetail_length + clearance, dovetail_length + clearance * 3)

    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-9)
    z_axis = np.array([0, 0, 1.0])
    angle = math.acos(np.clip(np.dot(z_axis, joint_axis), -1, 1))
    if angle > 1e-6 and angle < math.pi - 1e-6:
        rot_axis = np.cross(z_axis, joint_axis)
        rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-9)
        rot = trimesh.transformations.rotation_matrix(angle, rot_axis)
        male.apply_transform(rot)
        female.apply_transform(rot)

    male.apply_translation(joint_position)
    female.apply_translation(joint_position)

    try:
        parent = parent.union(male, engine="scad")
        child = child.difference(female, engine="scad")
    except Exception:
        pass

    return parent, child


def _make_box(w: float, d: float, h: float) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(w, d, h))


BOLT_SPECS = {
    "M2": {"bolt_r": 0.0010, "head_r": 0.0019, "head_d": 0.0016},
    "M3": {"bolt_r": 0.0015, "head_r": 0.0028, "head_d": 0.0024},
}

CONNECTION_PRESETS = {
    "snap": {"width": 0.008, "depth": 0.006, "height": 0.004, "clearance": 0.0003},
    "dovetail": {"width": 0.010, "length": 0.008, "clearance": 0.0003},
    "bolt_M2": {"bolt_r": 0.0010, "clearance": 0.0002},
    "bolt_M3": {"bolt_r": 0.0015, "clearance": 0.0002},
    "hinge_3mm": {"shaft_r": 0.0015, "clearance": 0.0003, "bearing_thickness": 0.004},
    "hinge_5mm": {"shaft_r": 0.0025, "clearance": 0.0004, "bearing_thickness": 0.005},
}
