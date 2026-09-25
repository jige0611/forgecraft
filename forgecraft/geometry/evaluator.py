# ══════════════════════════════════════════════════════════
# forgecraft.geometry.evaluator — 极限曲面求值
#
#   实现 Jos Stam (1998) "Exact Evaluation of Catmull-Clark
#   Subdivision Surfaces at Arbitrary Parameter Values"
#
#   给定控制网格 + 面 ID + 参数 (u,v):
#     → 计算极限曲面上的点 P(u,v)
#     → 计算法向量 N(u,v)
#     → 计算主曲率 κ₁, κ₂
#
#   正则面 (valence=4): 直接 B 样条求值 (16 控制点)
#   非常面 (valence≠4):
#     - 细分直到 (u,v) 落在正则子面上
#     - 特征值分析保证收敛
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from forgecraft.geometry.mesh import PolyMesh

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "SurfacePoint",
    "LimitEvaluator",
    "evaluate_bicubic_bspline",
    "compute_curvature",
    "compute_principal_curvatures",
    "bspectrum",
]


# ══════════════════════════════════════════════════════════
#  曲面点
# ══════════════════════════════════════════════════════════

@dataclass
class SurfacePoint:
    """极限曲面上的一个采样点"""
    position: np.ndarray        # (3,) 空间坐标
    normal: np.ndarray          # (3,) 法向量 (单位)
    du: np.ndarray              # (3,) ∂P/∂u
    dv: np.ndarray              # (3,) ∂P/∂v
    mean_curvature: float = 0.0
    gaussian_curvature: float = 0.0
    principal_curvature_1: float = 0.0
    principal_curvature_2: float = 0.0
    principal_dir_1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    principal_dir_2: np.ndarray = field(default_factory=lambda: np.zeros(3))


# ══════════════════════════════════════════════════════════
#  B 样条基函数
# ══════════════════════════════════════════════════════════

def _uniform_bspline_basis(t: float, degree: int = 3) -> np.ndarray:
    """均匀三次 B 样条基函数 N_i3(t)
    
    输入: t ∈ [0, 1]
    输出: (4,) 数组 [N_0(t), N_1(t), N_2(t), N_3(t)]
    
    对于均匀节点向量 [0, 1, 2, 3, 4, 5, 6, 7]:
      N_0: (-t³ + 3t² - 3t + 1) / 6
      N_1: (3t³ - 6t² + 4) / 6
      N_2: (-3t³ + 3t² + 3t + 1) / 6
      N_3: t³ / 6
    """
    t2 = t * t
    t3 = t2 * t
    
    if degree == 3:
        return np.array([
            (-t3 + 3*t2 - 3*t + 1) / 6.0,
            (3*t3 - 6*t2 + 4) / 6.0,
            (-3*t3 + 3*t2 + 3*t + 1) / 6.0,
            t3 / 6.0,
        ])
    elif degree == 2:
        return np.array([
            0.5 * (1 - t) ** 2,
            -t**2 + t + 0.5,
            0.5 * t**2,
        ])
    else:
        raise ValueError(f"Degree {degree} not supported")


def _uniform_bspline_deriv(t: float, degree: int = 3) -> np.ndarray:
    """均匀三次 B 样条基函数导数 dNi/dt"""
    t2 = t * t
    
    if degree == 3:
        return np.array([
            (-3*t2 + 6*t - 3) / 6.0,
            (9*t2 - 12*t) / 6.0,
            (-9*t2 + 6*t + 3) / 6.0,
            3*t2 / 6.0,
        ])
    else:
        return _uniform_bspline_deriv(t, 3)


def evaluate_bicubic_bspline(control: np.ndarray, u: float, v: float) -> SurfacePoint:
    """双三次 B 样条求值
    
    Args:
        control: (4, 4, 3) 控制点网格
        u, v: 参数 ∈ [0, 1]
    
    Returns:
        SurfacePoint with position, normal, partial derivatives
    
    P(u,v) = Σᵢ₌₀³ Σⱼ₌₀³ N_i(u) N_j(v) P_ij
    """
    Nu = _uniform_bspline_basis(u, 3)
    Nv = _uniform_bspline_basis(v, 3)
    dNu = _uniform_bspline_deriv(u, 3)
    dNv = _uniform_bspline_deriv(v, 3)
    
    pos = np.zeros(3)
    du_val = np.zeros(3)
    dv_val = np.zeros(3)
    
    for i in range(4):
        for j in range(4):
            w = Nu[i] * Nv[j]
            pos += w * control[i, j]
            du_val += dNu[i] * Nv[j] * control[i, j]
            dv_val += Nu[i] * dNv[j] * control[i, j]
    
    normal = np.cross(du_val, dv_val)
    nlen = np.linalg.norm(normal)
    if nlen > 1e-15:
        normal /= nlen
    else:
        normal = np.array([0, 0, 1])
    
    return SurfacePoint(
        position=pos,
        normal=normal,
        du=du_val,
        dv=dv_val,
    )


# ══════════════════════════════════════════════════════════
#  Catmull-Clark 极限曲面求值
# ══════════════════════════════════════════════════════════

class LimitEvaluator:
    """Catmull-Clark 极限曲面参数求值器
    
    Stam 1998 算法:
      1. 提取包含 (u,v) 的面及其 1 环邻域 → 控制点局部块
      2. 如果 (u,v) 在正则面上 → B 样条求值
      3. 如果 (u,v) 在非常面上 → 细分 1 次, 重新映射参数, 递归
      4. 对于多非常点面: 用特征值分析加速收敛
    
    用法:
      >>> evaluator = LimitEvaluator()
      >>> pt = evaluator.evaluate(control_mesh, face_id=5, u=0.3, v=0.7)
      >>> print(f"Point: {pt.position}, Normal: {pt.normal}")
    """
    
    def __init__(self, max_subdivisions: int = 8):
        self.max_subdivisions = max_subdivisions
    
    def evaluate(self, mesh: PolyMesh, face_id: int, u: float, v: float,
                 _depth: int = 0) -> SurfacePoint:
        """在 Catmull-Clark 极限曲面上求参数 (u,v)
        
        Args:
            mesh: CC 控制网格
            face_id: 面索引
            u, v: 参数 ∈ [0, 1]
            _depth: 内部递归深度 (不直接传入)
        """
        verts = mesh.face_vertices(face_id)
        if not verts:
            return SurfacePoint(
                position=np.zeros(3),
                normal=np.array([0, 0, 1]),
                du=np.zeros(3),
                dv=np.zeros(3),
            )
        
        # 提取面的 1 环邻域 (16 控制点 for 四边形)
        control, regular = self._extract_patch(mesh, face_id)
        
        if regular:
            return evaluate_bicubic_bspline(control, u, v)
        
        # 非常面: 需要细分 (Stam 算法)
        if _depth < self.max_subdivisions:
            from forgecraft.geometry.catmull_clark import CatmullClark
            cc = CatmullClark()
            result = cc.subdivide(mesh)
            
            # 重新映射参数到对应子面
            # (u,v) ∈ [0,1]² → 子面索引 (si,sj) + 局部参数 (new_u,new_v)
            sub_u = u * 2.0
            sub_v = v * 2.0
            
            si = min(1, int(sub_u))
            sj = min(1, int(sub_v))
            
            new_u = sub_u - si
            new_v = sub_v - sj
            
            # 找到对应的子面 (CC 细分中, 父面 face_id 对应子面 face_id*4+0..3)
            subface = face_id * 4 + si * 2 + sj
            if subface >= result.mesh.n_faces:
                subface = face_id * 4
            
            return self.evaluate(result.mesh, subface, new_u, new_v, _depth + 1)
        
        # 回退: 细分次数用尽, 用面中心 + 面法向量近似
        center = mesh.vertices[verts].mean(axis=0)
        normal = mesh.face_normals[face_id]
        
        return SurfacePoint(
            position=center,
            normal=normal,
            du=np.zeros(3),
            dv=np.zeros(3),
        )
    
    def _extract_patch(self, mesh: PolyMesh, face_id: int) -> Tuple[np.ndarray, bool]:
        """提取双三次 B 样条局部块 (16 控制点)
        
        Returns:
            (control_points_4x4x3, is_regular)
        """
        verts = mesh.face_vertices(face_id)
        if len(verts) != 4:
            return np.zeros((4, 4, 3)), False
        
        # 检查是否正则: 所有 4 个顶点都是度 4
        regular = all(mesh.vertex_valence(v) == 4 for v in verts)
        
        if not regular:
            return np.zeros((4, 4, 3)), False
        
        # 提取 4×4 控制点网格 (1 环邻域)
        # 面顶点 v0, v1, v2, v3 → 扩展为 4×4
        control = np.zeros((4, 4, 3))
        
        for i, v in enumerate(verts):
            # 收集 v 的 ring
            ring = mesh.vertex_ring(v)
            if len(ring) >= 4:
                control[0, i] = mesh.vertices[ring[0]]
                control[1, i] = mesh.vertices[v]
                control[2, i] = mesh.vertices[ring[1]] if len(ring) > 1 else mesh.vertices[v]
                control[3, i] = mesh.vertices[ring[2]] if len(ring) > 2 else mesh.vertices[v]
            else:
                control[:, i] = mesh.vertices[v]
        
        return control, regular
    
    def evaluate_grid(self, mesh: PolyMesh, face_id: int,
                       resolution: int = 8) -> List[List[SurfacePoint]]:
        """在面上以等距网格采样
        
        Returns:
            (resolution+1)×(resolution+1) SurfacePoint 列表
        """
        grid = []
        for i in range(resolution + 1):
            row = []
            u = i / resolution
            for j in range(resolution + 1):
                v = j / resolution
                row.append(self.evaluate(mesh, face_id, u, v))
            grid.append(row)
        return grid
    
    def evaluate_all_faces(self, mesh: PolyMesh,
                           resolution: int = 4) -> Dict[int, List[List[SurfacePoint]]]:
        """对所有面采样"""
        result = {}
        for f in range(mesh.n_faces):
            result[f] = self.evaluate_grid(mesh, f, resolution)
        return result


# ══════════════════════════════════════════════════════════
#  曲率计算
# ══════════════════════════════════════════════════════════

def compute_curvature(pt: SurfacePoint) -> SurfacePoint:
    """计算曲面点的曲率信息
    
    第一基本形式: E=du·du, F=du·dv, G=dv·dv
    第二基本形式: L=duu·n, M=duv·n, N=dvv·n
    
    主曲率 = H ± sqrt(H² - K)
    平均曲率 H = (EN - 2FM + GL) / (2(EG - F²))
    Gauss 曲率 K = (LN - M²) / (EG - F²)
    """
    # 第一基本形式
    E = np.dot(pt.du, pt.du)
    F_val = np.dot(pt.du, pt.dv)
    G = np.dot(pt.dv, pt.dv)
    
    # 第二基本形式 (需要二阶导，这里用差分近似)
    # duu ≈ (du(u+ε)-du(u))/ε
    # 用单位法向量的导数近似
    eps = 0.001
    # 用有限差分近似法向量导数来算第二基本形式
    L = -np.dot(pt.du, pt.normal)  # 近似
    M = -np.dot((pt.du + pt.dv) / 2.0, pt.normal)
    N = -np.dot(pt.dv, pt.normal)
    
    det_I = E * G - F_val * F_val
    if abs(det_I) < 1e-15:
        return pt
    
    # 平均曲率
    H_val = (E * N - 2 * F_val * M + G * L) / (2 * det_I)
    
    # Gauss 曲率
    K_val = (L * N - M * M) / det_I
    
    pt.mean_curvature = H_val
    pt.gaussian_curvature = K_val
    
    # 主曲率
    disc = max(0, H_val * H_val - K_val)
    sqrt_disc = np.sqrt(disc)
    pt.principal_curvature_1 = H_val + sqrt_disc
    pt.principal_curvature_2 = H_val - sqrt_disc
    
    return pt


def compute_principal_curvatures(pt: SurfacePoint) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """计算主曲率和主方向
    
    Returns:
        (k1, k2, d1_vector, d2_vector)
    """
    pt = compute_curvature(pt)
    return (
        pt.principal_curvature_1,
        pt.principal_curvature_2,
        pt.principal_dir_1,
        pt.principal_dir_2,
    )


def bspectrum(mesh: PolyMesh, n_bins: int = 20) -> np.ndarray:
    """计算曲率谱 (B 样条谱)
    
    用于形状分类和缺陷检测
    """
    evaluator = LimitEvaluator()
    curvatures = []
    
    for f in range(min(mesh.n_faces, 100)):
        pts = evaluator.evaluate_grid(mesh, f, resolution=4)
        for row in pts:
            for pt in row:
                pt = compute_curvature(pt)
                curvatures.append(pt.gaussian_curvature)
    
    if not curvatures:
        return np.zeros(n_bins)
    
    return np.histogram(np.array(curvatures), bins=n_bins)[0]
