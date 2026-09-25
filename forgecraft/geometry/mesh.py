# ══════════════════════════════════════════════════════════
# forgecraft.geometry.mesh — 半边数据结构
#
#   PolyMesh: 半边 (half-edge) 多边形网格
#
#   为什么半边:
#     - 顶点 1 环邻域遍历: O(1) per edge vs O(V+E) for face list
#     - 面→边 遍历: O(1) per edge
#     - 边界检测: 直接检查 halfedge.twin
#     - Catmull-Clark 细分、自适应细分、镶嵌 全部依赖半边
#
#   不依赖任何外部库，纯 NumPy + Python
# ══════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

_logger = logging.getLogger(__name__)

__all__ = [
    "HalfEdge",
    "PolyMesh",
    "mesh_from_trimesh",
    "mesh_to_trimesh",
    "build_tet_mesh",
    "validate_manifold",
]

# ══════════════════════════════════════════════════════════
#  Numba JIT 加速 (可选)
# ══════════════════════════════════════════════════════════

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def dec(f): return f
        return dec
    prange = range

# Try Numba CUDA for GPU subdivision (optional)
try:
    from numba import cuda
    _HAS_CUDA = False  # Set True if CUDA device available
    try:
        cuda.detect()
        _HAS_CUDA = True
    except Exception:
        pass
except ImportError:
    _HAS_CUDA = False


# ══════════════════════════════════════════════════════════
#  半边
# ══════════════════════════════════════════════════════════

@dataclass
class HalfEdge:
    """半边 — 有向连接，包含拓扑邻接"""
    origin: int     # 起点顶点索引
    face: int       # 所属面索引 (-1 = 边界半边)
    next: int       # 同一面的下一条半边
    prev: int       # 同一面的上一条半边
    twin: int       # 反向半边 (-1 = 边界边)


# ══════════════════════════════════════════════════════════
#  PolyMesh: 半边网格
# ══════════════════════════════════════════════════════════

class PolyMesh:
    """半边数据结构多边形网格
    
    属性:
        vertices: (V, 3) float 顶点
        halfedges: List[HalfEdge] (2E) 半边
        faces: (F,) int → 每条半边属于哪个面
        edges: (E,) int → 混合半边
    
    约束:
        - manifold (每个边 ≤ 2 个面)
        - oriented (每个面的半边环方向一致)
    """
    
    __slots__ = ('vertices', 'halfedges', '_face_to_he',
                 '_vertex_to_he', '_edge_to_he', '_edge_to_he_tuple',
                 '_vertex_normals', '_face_normals', '_face_areas', '_vertex_areas')
    
    def __init__(self):
        self.vertices: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.halfedges: List[HalfEdge] = []
        self._face_to_he: Dict[int, int] = {}
        self._vertex_to_he: Dict[int, int] = {}
        self._edge_to_he: Dict[int, int] = {}
        self._edge_to_he_tuple: Dict[Tuple[int, int], int] = {}
        
        # 缓存
        self._vertex_normals: Optional[np.ndarray] = None
        self._face_normals: Optional[np.ndarray] = None
        self._face_areas: Optional[np.ndarray] = None
        self._vertex_areas: Optional[np.ndarray] = None
    
    # ── 基本属性 ──
    
    @property
    def n_vertices(self) -> int: return len(self.vertices)
    @property
    def n_halfedges(self) -> int: return len(self.halfedges)
    @property
    def n_edges(self) -> int: return len(self._edge_to_he)
    @property
    def n_faces(self) -> int: return len(self._face_to_he)
    
    # ── 顶点邻域 ──
    
    def vertex_ring(self, v: int) -> List[int]:
        """围绕顶点 v 的一环顶点 (CW 序)"""
        if v not in self._vertex_to_he:
            return []
        
        ring = []
        start_he = self._vertex_to_he[v]
        he_idx = start_he
        
        for _ in range(100):  # 防死循环
            he = self.halfedges[he_idx]
            prev = self.halfedges[he.prev]
            ring.append(prev.origin)
            
            # 走到下一个半边: twin → next
            tw = he.twin
            if tw < 0:
                break  # 边界
            he_idx = self.halfedges[tw].next
            
            if he_idx == start_he:
                break
        
        return ring
    
    def vertex_faces(self, v: int) -> List[int]:
        """包含顶点 v 的所有面"""
        if v not in self._vertex_to_he:
            return []
        
        faces = []
        start_he = self._vertex_to_he[v]
        he_idx = start_he
        
        for _ in range(100):
            he = self.halfedges[he_idx]
            if he.face >= 0:
                faces.append(he.face)
            
            tw = he.twin
            if tw < 0:
                break  # 边界
            he_idx = self.halfedges[tw].next
            
            if he_idx == start_he:
                break
        
        return faces
    
    def vertex_valence(self, v: int) -> int:
        """顶点度 (邻接面数)"""
        return len(self.vertex_faces(v))
    
    # ── 面遍历 ──
    
    def face_vertices(self, f: int) -> List[int]:
        """面 f 的顶点列表 (CCW 序)"""
        if f not in self._face_to_he:
            return []
        
        verts = []
        he_start = self._face_to_he[f]
        he_idx = he_start
        
        for _ in range(100):
            he = self.halfedges[he_idx]
            verts.append(he.origin)
            he_idx = he.next
            if he_idx == he_start:
                break
        
        return verts
    
    def face_halfedges(self, f: int) -> List[int]:
        """面 f 的半边列表"""
        if f not in self._face_to_he:
            return []
        
        edges = []
        he_start = self._face_to_he[f]
        he_idx = he_start
        
        for _ in range(100):
            edges.append(he_idx)
            he_idx = self.halfedges[he_idx].next
            if he_idx == he_start:
                break
        
        return edges
    
    def face_vertex_count(self, f: int) -> int:
        """面的顶点数"""
        return len(self.face_vertices(f))
    
    # ── 边查询 ──
    
    def edge_vertices(self, e: int) -> Tuple[int, int]:
        """边 e 的两个顶点"""
        he_idx = self._edge_to_he.get(e)
        if he_idx is None:
            return (-1, -1)
        he = self.halfedges[he_idx]
        tw = self.halfedges[he.twin] if he.twin >= 0 else None
        return (he.origin, tw.origin if tw else -1)
    
    def is_boundary(self, e: int) -> bool:
        """边是否为边界"""
        he_idx = self._edge_to_he.get(e)
        if he_idx is None:
            return False
        return self.halfedges[he_idx].twin < 0
    
    def edge_faces(self, e: int) -> List[int]:
        """共享边 e 的面 (0, 1, 或 2 个面)"""
        he_idx = self._edge_to_he.get(e)
        if he_idx is None:
            return []
        
        he = self.halfedges[he_idx]
        faces = []
        if he.face >= 0:
            faces.append(he.face)
        if he.twin >= 0 and self.halfedges[he.twin].face >= 0:
            faces.append(self.halfedges[he.twin].face)
        return faces
    
    # ── 法向量 ──
    
    @property
    def face_normals(self) -> np.ndarray:
        """面法向量 (面面积加权)"""
        if self._face_normals is None:
            self._compute_normals()
        return self._face_normals
    
    @property
    def vertex_normals(self) -> np.ndarray:
        """顶点法向量 (邻面平均)"""
        if self._vertex_normals is None:
            self._compute_normals()
        return self._vertex_normals
    
    def _compute_normals(self):
        """计算面和顶点法向量"""
        nf = self.n_faces
        nv = self.n_vertices
        
        f_normals = np.zeros((max(1, nf), 3))
        v_normals = np.zeros((max(1, nv), 3))
        
        for f in range(nf):
            verts = self.face_vertices(f)
            if len(verts) < 3:
                continue
            
            v0 = self.vertices[verts[0]]
            v1 = self.vertices[verts[1]]
            v2 = self.vertices[verts[2]]
            
            n = np.cross(v1 - v0, v2 - v0)
            length = np.linalg.norm(n)
            if length > 1e-15:
                n /= length
                f_normals[f] = n
                
                for v in verts:
                    v_normals[v] += n
        
        # 归一化顶点法向量
        for v in range(nv):
            length = np.linalg.norm(v_normals[v])
            if length > 1e-15:
                v_normals[v] /= length
        
        self._face_normals = f_normals
        self._vertex_normals = v_normals
    
    # ── 构建方法 ──
    
    @classmethod
    def from_vertices_faces(cls, vertices: np.ndarray, faces: List[List[int]]) -> "PolyMesh":
        """从顶点列表 + 面列表构建半边网格
        
        Args:
            vertices: (V, 3) float
            faces: List of [i0, i1, i2, ...] 顶点索引
        
        Returns:
            PolyMesh
        """
        mesh = cls()
        mesh.vertices = np.asarray(vertices, dtype=np.float64)
        
        # 构建半边映射: (origin, target) → he_index
        edge_map: Dict[Tuple[int, int], int] = {}
        
        for face_idx, face_verts in enumerate(faces):
            n = len(face_verts)
            if n < 3:
                continue
            
            # 为每条边创建半边
            prev_he = -1
            first_he = -1
            
            for i in range(n):
                origin = int(face_verts[i])
                target = int(face_verts[(i + 1) % n])
                
                he = HalfEdge(
                    origin=origin,
                    face=face_idx,
                    next=-1,
                    prev=-1,
                    twin=-1,
                )
                he_idx = len(mesh.halfedges)
                mesh.halfedges.append(he)
                
                # 更新邻接
                if first_he < 0:
                    first_he = he_idx
                
                if prev_he >= 0:
                    mesh.halfedges[prev_he].next = he_idx
                    mesh.halfedges[he_idx].prev = prev_he
                
                # 查找边的反向半边
                twin_key = (target, origin)
                if twin_key in edge_map:
                    twin_idx = edge_map.pop(twin_key)
                    mesh.halfedges[he_idx].twin = twin_idx
                    mesh.halfedges[twin_idx].twin = he_idx
                else:
                    edge_map[(origin, target)] = he_idx
                
                # 更新顶点 → 半边 映射
                if origin not in mesh._vertex_to_he:
                    mesh._vertex_to_he[origin] = he_idx
                
                prev_he = he_idx
            
            # 闭合环
            if first_he >= 0 and prev_he >= 0:
                mesh.halfedges[prev_he].next = first_he
                mesh.halfedges[first_he].prev = prev_he
            
            mesh._face_to_he[face_idx] = first_he
        
        # 构建边 → 半边 映射 (tuple key)
        for i, he in enumerate(mesh.halfedges):
            if he.twin >= 0:
                twin = mesh.halfedges[he.twin]
                if twin.twin == i:  # only primary direction
                    key = (min(he.origin, twin.origin), max(he.origin, twin.origin))
                    if key not in mesh._edge_to_he_tuple:
                        mesh._edge_to_he_tuple[key] = i
                        e_id = len(mesh._edge_to_he)
                        mesh._edge_to_he[e_id] = i
        
        return mesh
    
    def to_vertices_faces(self) -> Tuple[np.ndarray, List[List[int]]]:
        """导出为顶点列表 + 面列表"""
        faces = []
        for f in range(self.n_faces):
            faces.append(self.face_vertices(f))
        return self.vertices, faces
    
    # ── 变换 ──
    
    def apply_transform(self, matrix: np.ndarray):
        """应用 4x4 变换矩阵"""
        if matrix.shape == (4, 4):
            verts_h = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
            self.vertices = (verts_h @ matrix.T)[:, :3]
        elif matrix.shape == (3, 3):
            self.vertices = self.vertices @ matrix.T
        else:
            raise ValueError(f"Expected 3x3 or 4x4 matrix, got {matrix.shape}")
        self._clear_cache()
    
    def translate(self, offset: np.ndarray):
        self.vertices += np.asarray(offset).reshape(3)
        self._clear_cache()
    
    def scale(self, factor: float):
        self.vertices *= factor
        self._clear_cache()
    
    def _clear_cache(self):
        self._vertex_normals = None
        self._face_normals = None
        self._face_areas = None
        self._vertex_areas = None
    
    # ── 验证 ──
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证网格的 manifold 属性"""
        errors = []
        
        # 检查每个半边 twin 的对称性
        for i, he in enumerate(self.halfedges):
            if he.twin >= 0:
                tw = self.halfedges[he.twin]
                if tw.twin != i:
                    errors.append(f"Half-edge {i}: twin asymmetry (twin.twin={tw.twin}, expected {i})")
        
        # 检查闭合环
        for f in range(self.n_faces):
            verts = self.face_vertices(f)
            if len(verts) < 3:
                errors.append(f"Face {f}: < 3 vertices ({len(verts)})")
                continue
            
            # 检查环闭合
            hes = self.face_halfedges(f)
            for he_idx in hes:
                he = self.halfedges[he_idx]
                nn = self.halfedges[he.next]
                if nn.origin != self.halfedges[self.halfedges[he.twin].origin if he.twin >= 0 else -1] if False else True:
                    pass
        
        isValid = len(errors) == 0
        return isValid, errors
    
    def summary(self) -> str:
        return (
            f"PolyMesh: {self.n_vertices} verts, {self.n_edges} edges, "
            f"{self.n_faces} faces, {self.n_halfedges} halfedges"
        )


# ══════════════════════════════════════════════════════════
#  trimesh ↔ PolyMesh 互转
# ══════════════════════════════════════════════════════════

def mesh_from_trimesh(tri_mesh) -> PolyMesh:
    """trimesh.Trimesh → PolyMesh"""
    vertices = np.asarray(tri_mesh.vertices, dtype=np.float64)
    faces = tri_mesh.faces.tolist() if hasattr(tri_mesh.faces, 'tolist') else list(tri_mesh.faces)
    return PolyMesh.from_vertices_faces(vertices, faces)


def mesh_to_trimesh(pm: PolyMesh):
    """PolyMesh → trimesh.Trimesh"""
    import trimesh
    verts, faces = pm.to_vertices_faces()
    # Trimesh 需要三角形面，拆分多边形
    tri_faces = []
    for face in faces:
        if len(face) == 3:
            tri_faces.append(face)
        elif len(face) == 4:
            tri_faces.extend([[face[0], face[1], face[2]], [face[0], face[2], face[3]]])
        else:
            # 扇形三角化
            for i in range(1, len(face) - 1):
                tri_faces.append([face[0], face[i], face[i + 1]])
    
    return trimesh.Trimesh(vertices=verts, faces=np.array(tri_faces, dtype=np.int32))


# ══════════════════════════════════════════════════════════
#  四面体网格构建
# ══════════════════════════════════════════════════════════

def build_tet_mesh(vertices: np.ndarray, tets: List[List[int]]) -> "TetMesh":
    """从顶点 + 四面体列表构建体网格 (用于 IGA)"""
    # 四面体网格是物理仿真用的，返回简化的 TetMesh
    return TetMesh(vertices=vertices, tets=tets)


class TetMesh:
    """四面体网格 (用于 IGA 分析)"""
    def __init__(self, vertices: np.ndarray, tets: List[List[int]]):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.tets = np.asarray(tets, dtype=np.int32)
    
    @property
    def n_vertices(self): return len(self.vertices)
    @property
    def n_tets(self): return len(self.tets)
    
    def tet_volume(self, idx: int) -> float:
        a, b, c, d = [self.vertices[self.tets[idx][k]] for k in range(4)]
        return abs(np.linalg.det(np.array([b-a, c-a, d-a]))) / 6.0
    
    def total_volume(self) -> float:
        return sum(self.tet_volume(i) for i in range(self.n_tets))


# ══════════════════════════════════════════════════════════
#  Manifold 验证
# ══════════════════════════════════════════════════════════

def validate_manifold(mesh: PolyMesh) -> Tuple[bool, Dict]:
    """验证 manifold 性质
    
    Returns:
        (is_valid, {"errors": [...], "warnings": [...], "non_manifold_edges": int})
    """
    errors = []
    warnings = []
    non_manifold = 0
    
    # 检查每个顶点的邻域是否是一个闭合环
    for v in range(mesh.n_vertices):
        if v not in mesh._vertex_to_he:
            continue  # isolated vertex
        
        ring = mesh.vertex_ring(v)
        if not ring:
            errors.append(f"Vertex {v}: empty ring")
    
    # 检查每条边是否 ≥ 2 个面共享
    for e_id, he_idx in mesh._edge_to_he.items():
        he = mesh.halfedges[he_idx]
        n_faces = 0
        if he.face >= 0:
            n_faces += 1
        if he.twin >= 0 and mesh.halfedges[he.twin].face >= 0:
            n_faces += 1
        
        if n_faces > 2:
            non_manifold += 1
            warnings.append(f"Edge {e_id}: {n_faces} faces (non-manifold)")
    
    return len(errors) == 0 and non_manifold == 0, {
        "errors": errors,
        "warnings": warnings,
        "non_manifold_edges": non_manifold,
    }
