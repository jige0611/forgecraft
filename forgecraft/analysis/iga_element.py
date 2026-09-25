# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_element — 单元积分器
#
#   MembraneIntegrator:   3D 膜单元 (面内刚度)
#   基于 CC 形函数 + Gauss 积分, Numba JIT 热路径
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import numba

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import ShapeResult, QuadPoint, IrregularFaceCache
from forgecraft.analysis.iga_material import LinearIsotropic
from forgecraft.analysis.iga_shape import cc_shape_functions

import logging

_logger = logging.getLogger(__name__)

__all__ = ["ElementIntegrator", "MembraneIntegrator"]


# ══════════════════════════════════════════════════════════
#  抽象基类
# ══════════════════════════════════════════════════════════

class ElementIntegrator(ABC):
    """单元积分器抽象基类
    
    子类实现 stiffness() 和 mass()
    """
    
    @abstractmethod
    def stiffness(
        self,
        mesh: PolyMesh,
        face_id: int,
        material,
        quad_points: list,
        irregular_cache: Optional[IrregularFaceCache] = None,
    ) -> Tuple[np.ndarray, list]:
        """计算单元刚度矩阵 K_e
        Returns: (Ke, element_dofs)
        """
        ...
    
    @abstractmethod
    def mass(
        self,
        mesh: PolyMesh,
        face_id: int,
        material,
        quad_points: list,
        irregular_cache: Optional[IrregularFaceCache] = None,
    ) -> Tuple[np.ndarray, list]:
        """计算单元质量矩阵 M_e
        Returns: (Me, element_dofs)
        """
        ...


# ══════════════════════════════════════════════════════════
#  3D 膜单元
# ══════════════════════════════════════════════════════════

class MembraneIntegrator(ElementIntegrator):
    """3D 膜单元 (面内刚度)
    
    适合薄壁结构的面内拉伸/压缩分析。
    在 CC 极限曲面上积分, 使用局部切平面坐标系。
    
    膜刚度: K_e = ∫_Ω B_globᵀ D_membrane B_glob dΩ
    
    其中 B_glob 变换局部膜应变到全局位移,
    D_membrane 是 3×3 平面应力本构矩阵。
    """
    
    def __init__(self, thickness: float = 0.001):
        """
        Args:
            thickness: 壳厚度 (m), 用于面内刚度缩放
        """
        self.thickness = thickness
    
    def stiffness(
        self, mesh, face_id, material, quad_points, irregular_cache=None,
    ) -> Tuple[np.ndarray, list]:
        """计算膜单元刚度矩阵"""
        # 获取单元的控制点 (1 环邻域 = 16 控制点)
        from forgecraft.analysis.iga_shape import cc_element_control_points
        control, is_regular = cc_element_control_points(mesh, face_id)
        
        if control.shape != (4, 4, 3) or np.allclose(control, 0):
            return np.zeros((1, 1)), []
        
        # 单元 DOF: 16 控制点 × 3 = 48
        n_nodes = 16
        n_dof_per_node = 3
        n_dof = n_nodes * n_dof_per_node
        
        # 控制点的全局顶点索引 (1 环邻域)
        verts = mesh.face_vertices(face_id)
        # 扩展为 16 个控制点
        all_verts = list(verts)
        seen = set(verts)
        for v in verts:
            ring = mesh.vertex_ring(v)
            for r in ring:
                if r not in seen and len(all_verts) < 16:
                    all_verts.append(r)
                    seen.add(r)
        # 确保恰好 16 个
        while len(all_verts) < 16:
            all_verts.append(all_verts[-1])
        all_verts = all_verts[:16]
        
        # 3×3 平面应力本构矩阵
        D_membrane = _plane_stress_D(material.E, material.nu)
        
        # 组装
        Ke = np.zeros((n_dof, n_dof))
        
        for qp in quad_points:
            u, v, w = qp.u, qp.v, qp.weight
            
            # 对非常面: 使用映射到正则子面的参数
            if qp.sub_face_id >= 0 and irregular_cache is not None:
                shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)
                if shape.detJ < 1e-15:
                    shape = cc_shape_functions(mesh, face_id, qp.local_u, qp.local_v)
            else:
                shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)
            
            if shape.detJ < 1e-15:
                continue
            
            # 局部坐标系: t1, t2 在切平面内
            J = shape.J           # (3, 2) ∂P/∂u, ∂P/∂v
            t1 = J[:, 0]
            t2 = J[:, 1]
            t1 /= np.linalg.norm(t1)
            # 正交化: t2 减去 t1 分量
            t2 -= np.dot(t2, t1) * t1
            tnorm = np.linalg.norm(t2)
            if tnorm < 1e-15:
                continue
            t2 /= tnorm
            
            # 变换矩阵 T: 全局位移 → 局部位移
            # u_local = T @ u_global
            T = np.array([t1, t2]).T   # (3, 2)
            
            # B_glob: (3, n_dof) 全局膜应变-位移矩阵
            B_glob = np.zeros((3, n_dof))
            
            for node_i in range(n_nodes):
                # dN/dx in global coords
                dNdx = shape.dN_dx[node_i]  # (3,)
                
                # B_local = T^T @ grad N  (2D gradient → 3 local strains)
                # B_glob = [∂N/∂x_t1, ∂N/∂x_t2] mapped to 3 strain components
                
                # 使用 t1, t2 方向的方向导数
                dN_dt1 = np.dot(dNdx, t1)
                dN_dt2 = np.dot(dNdx, t2)
                
                i3 = node_i * 3
                # ε_xx (in-plane along t1)
                B_glob[0, i3 + 0] = dN_dt1 * t1[0]
                B_glob[0, i3 + 1] = dN_dt1 * t1[1]
                B_glob[0, i3 + 2] = dN_dt1 * t1[2]
                
                # ε_yy (in-plane along t2)
                B_glob[1, i3 + 0] = dN_dt2 * t2[0]
                B_glob[1, i3 + 1] = dN_dt2 * t2[1]
                B_glob[1, i3 + 2] = dN_dt2 * t2[2]
                
                # γ_xy (shear)
                B_glob[2, i3 + 0] = dN_dt1 * t2[0] + dN_dt2 * t1[0]
                B_glob[2, i3 + 1] = dN_dt1 * t2[1] + dN_dt2 * t1[1]
                B_glob[2, i3 + 2] = dN_dt1 * t2[2] + dN_dt2 * t1[2]
            
            # 积分贡献
            Ke += self.thickness * w * shape.detJ * (B_glob.T @ D_membrane @ B_glob)
        
        return Ke, all_verts
    
    def mass(
        self, mesh, face_id, material, quad_points, irregular_cache=None,
    ) -> Tuple[np.ndarray, list]:
        """计算膜单元一致质量矩阵"""
        from forgecraft.analysis.iga_shape import cc_element_control_points
        
        control, _ = cc_element_control_points(mesh, face_id)
        if control.shape != (4, 4, 3) or np.allclose(control, 0):
            return np.zeros((1, 1)), []
        
        n_nodes = 16
        n_dof = n_nodes * 3
        
        verts = mesh.face_vertices(face_id)
        all_verts = list(verts)
        seen = set(verts)
        for v in verts:
            ring = mesh.vertex_ring(v)
            for r in ring:
                if r not in seen and len(all_verts) < 16:
                    all_verts.append(r)
                    seen.add(r)
        while len(all_verts) < 16:
            all_verts.append(all_verts[-1])
        all_verts = all_verts[:16]
        
        Me = np.zeros((n_dof, n_dof))
        density = getattr(material, 'rho', 7800.0)
        
        for qp in quad_points:
            u, v, w = qp.u, qp.v, qp.weight
            
            if qp.sub_face_id >= 0 and irregular_cache is not None:
                shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)
            else:
                shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)
            
            if shape.detJ < 1e-15:
                continue
            
            N = shape.N  # (16,)
            factor = self.thickness * density * w * shape.detJ
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    val = factor * N[i] * N[j]
                    i3, j3 = i * 3, j * 3
                    Me[i3 + 0, j3 + 0] += val
                    Me[i3 + 1, j3 + 1] += val
                    Me[i3 + 2, j3 + 2] += val
        
        return Me, all_verts


# ══════════════════════════════════════════════════════════
#  平面应力本构矩阵 (Numba JIT)
# ══════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _plane_stress_D(E: float, nu: float) -> np.ndarray:
    """3×3 平面应力本构矩阵
    
    D = E/(1-ν²) × [1   ν   0    ]
                    [ν   1   0    ]
                    [0   0  (1-ν)/2]
    """
    D = np.zeros((3, 3))
    factor = E / (1.0 - nu * nu)
    D[0, 0] = 1.0
    D[1, 1] = 1.0
    D[0, 1] = nu
    D[1, 0] = nu
    D[2, 2] = (1.0 - nu) / 2.0
    D *= factor
    return D
