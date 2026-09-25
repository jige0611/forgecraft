"""
FDM 支撑结构优化器

功能：
1. 悬垂检测 — 找出 >45° 的悬垂面
2. 支撑柱生成 — 自动生成可折断支撑柱
3. 支撑最少化 — 根据零件方向优化减少支撑
4. 筏层生成 — 为小接触面生成打印筏层
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import trimesh
__all__ = [
    "detect_overhang_faces",
    "detect_overhang_regions",
    "generate_support_pillars",
    "generate_raft",
    "generate_breakaway_supports",
    "compute_support_volume_ratio",
    "optimize_orientation",
]




# ── 支撑参数常量 ──
_OVERHANG_THRESHOLD = math.cos(math.radians(45))    # 悬垂角 45° → 法线 cos 阈值
_BRIDGE_THRESHOLD = math.cos(math.radians(30))       # 桥接角 30°
_SUPPORT_PILLAR_RADIUS = 0.0015      # 支撑柱默认半径 (1.5mm)
_SUPPORT_PILLAR_MIN_RADIUS = 0.0010  # 支撑柱最小半径 (1mm)
_SUPPORT_BASE_RADIUS = 0.004         # 支撑底垫半径 (4mm)
_RAFT_THICKNESS = 0.001              # 筏层厚度 (1mm)
_RAFT_EXTRA_MARGIN = 0.003           # 筏层外延边距 (3mm)


def detect_overhang_faces(
    mesh: trimesh.Trimesh,
    threshold_angle_deg: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """检测悬垂面: face_normal · (0,0,-1) < cos(threshold)"""
    if mesh.faces.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    normals = mesh.face_normals
    z_down = np.array([0, 0, -1.0])
    # 面法线朝下的分量越小 → 越需要支撑
    cos_threshold = math.cos(math.radians(threshold_angle_deg))

    dot_products = np.dot(normals, z_down)

    overhang_mask = dot_products > cos_threshold
    overhang_faces = np.where(overhang_mask)[0]

    bridge_mask = overhang_mask & (np.abs(dot_products) < _BRIDGE_THRESHOLD)
    bridge_faces = np.where(bridge_mask)[0]

    return overhang_faces, bridge_faces


def detect_overhang_regions(
    mesh: trimesh.Trimesh,
    threshold_angle_deg: float = 45.0,
) -> List[np.ndarray]:
    overhang_faces, _ = detect_overhang_faces(mesh, threshold_angle_deg)

    if len(overhang_faces) == 0:
        return []

    adjacency = mesh.face_adjacency
    face_to_face = {}
    for a, b in adjacency:
        face_to_face.setdefault(a, set()).add(b)
        face_to_face.setdefault(b, set()).add(a)

    oh_set = set(overhang_faces.tolist())
    visited = set()
    regions = []

    for face_idx in oh_set:
        if face_idx in visited:
            continue
        region = set()
        stack = [face_idx]
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in oh_set:
                continue
            visited.add(cur)
            region.add(cur)
            for nb in face_to_face.get(cur, set()):
                if nb not in visited and nb in oh_set:
                    stack.append(nb)
        if region:
            regions.append(np.array(list(region)))

    return regions


def generate_support_pillars(
    mesh: trimesh.Trimesh,
    regions: List[np.ndarray] = None,
    pillar_spacing: float = 0.015,
    pillar_radius: float = None,
    min_height: float = 0.001,
) -> trimesh.Trimesh:
    if pillar_radius is None:
        pillar_radius = _SUPPORT_PILLAR_RADIUS

    if regions is None:
        regions = detect_overhang_regions(mesh)

    if not regions:
        return trimesh.Trimesh()

    supports = []
    mesh_centroids = mesh.triangles_center

    for region in regions:
        region_points = mesh_centroids[region]
        if len(region_points) == 0:
            continue

        sampled = _sample_region_points(region_points, pillar_spacing)

        for pt in sampled:
            pillar = _make_support_pillar(pt, mesh.vertices, pillar_radius, min_height)
            if pillar is not None:
                supports.append(pillar)

    if not supports:
        return trimesh.Trimesh()

    combined = trimesh.util.concatenate(supports)
    combined.merge_vertices()
    return combined


def _sample_region_points(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) <= 1:
        return points

    sampled = [points[0]]
    for pt in points[1:]:
        if np.min(np.linalg.norm(sampled - pt, axis=1)) > spacing:
            sampled.append(pt)
    return np.array(sampled)


def _make_support_pillar(
    point: np.ndarray,
    all_vertices: np.ndarray,
    radius: float,
    min_height: float = 0.001,
) -> Optional[trimesh.Trimesh]:
    z_min = np.min(all_vertices[:, 2]) - 0.002
    point_z = point[2]

    height = point_z - z_min
    if height < min_height:
        return None

    if height < radius * 2:
        return None

    segments = max(1, int(height / radius))
    pillar_verts = []
    pillar_faces = []
    total_verts = 0

    for i in range(segments + 1):
        t = i / segments
        z = point_z - t * height
        r = radius * (0.6 + 0.4 * t)
        for j in range(6):
            angle = 2 * math.pi * j / 6
            x = point[0] + r * math.cos(angle)
            y = point[1] + r * math.sin(angle)
            pillar_verts.append([x, y, z])

    for i in range(segments):
        for j in range(6):
            j_next = (j + 1) % 6
            v0 = i * 6 + j
            v1 = i * 6 + j_next
            v2 = (i + 1) * 6 + j
            v3 = (i + 1) * 6 + j_next
            pillar_faces.append([v0, v2, v1])
            pillar_faces.append([v1, v2, v3])

    if len(pillar_verts) < 4:
        return None

    vertices = np.array(pillar_verts)
    faces = np.array(pillar_faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def generate_raft(
    mesh: trimesh.Trimesh,
    raft_thickness: float = None,
    extra_margin: float = None,
) -> Optional[trimesh.Trimesh]:
    if raft_thickness is None:
        raft_thickness = _RAFT_THICKNESS
    if extra_margin is None:
        extra_margin = _RAFT_EXTRA_MARGIN

    z_min = np.min(mesh.vertices[:, 2])
    z_candidates = mesh.vertices[mesh.vertices[:, 2] < z_min + raft_thickness * 3]

    if len(z_candidates) < 3:
        return None

    hull_2d = z_candidates[:, :2]
    min_xy = np.min(hull_2d, axis=0) - extra_margin
    max_xy = np.max(hull_2d, axis=0) + extra_margin

    raft_w = max_xy[0] - min_xy[0]
    raft_d = max_xy[1] - min_xy[1]
    raft_cx = (min_xy[0] + max_xy[0]) / 2
    raft_cy = (min_xy[1] + max_xy[1]) / 2
    raft_z = z_min - raft_thickness / 2

    raft = trimesh.creation.box(extents=(raft_w, raft_d, raft_thickness))
    raft.apply_translation([raft_cx, raft_cy, raft_z])
    return raft


def generate_breakaway_supports(
    mesh: trimesh.Trimesh,
    contact_radius: float = 0.0008,
) -> trimesh.Trimesh:
    regions = detect_overhang_regions(mesh)
    if not regions:
        return trimesh.Trimesh()

    pillars = generate_support_pillars(mesh, regions, pillar_spacing=0.012,
                                        pillar_radius=contact_radius)
    return pillars


def compute_support_volume_ratio(
    mesh: trimesh.Trimesh,
    supports: trimesh.Trimesh,
) -> float:
    if mesh.volume <= 0:
        return 0.0
    return supports.volume / mesh.volume


def optimize_orientation(
    mesh: trimesh.Trimesh,
    num_rotations: int = 12,
) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
    best_mesh = mesh.copy()
    best_score = float("inf")
    best_rot = np.eye(4)

    for i in range(num_rotations):
        angle = 2 * math.pi * i / num_rotations
        rot_x = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        rot_y = trimesh.transformations.rotation_matrix(angle * 0.618, [0, 1, 0])

        test = mesh.copy()
        test.apply_transform(rot_x)
        test.apply_transform(rot_y)

        oh_faces, _ = detect_overhang_faces(test)
        score = len(oh_faces)

        if score < best_score:
            best_score = score
            best_mesh = test
            best_rot = rot_y @ rot_x

    return best_mesh, best_rot, best_score
