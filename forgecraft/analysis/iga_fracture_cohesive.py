# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_fracture_cohesive — 内聚力界面单元
#
#   CohesiveInterfaceIntegrator:  界面单元插入 + 刚度/质量矩阵
#   cohesive_stiffness:           界面单元刚度矩阵
#   cohesive_traction:            界面牵引力
#   insert_cohesive_elements:     在指定面间插入界面单元
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import DOFMap, ShapeResult, IrregularFaceCache
from forgecraft.analysis.iga_contact_types import CohesiveLaw

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "CohesiveInterfaceIntegrator",
    "cohesive_stiffness",
    "cohesive_traction",
    "insert_cohesive_elements",
    "compute_effective_separation",
]


# ══════════════════════════════════════════════════════════
#  内聚力界面积分器
# ══════════════════════════════════════════════════════════

class CohesiveInterfaceIntegrator:
    """内聚力界面单元积分器
    
    在 CC 面对间插入界面单元，计算界面刚度和牵引力。
    
    界面单元使用 3D 分离向量 δ = [δ_n, δ_t1, δ_t2]:
      - δ_n: 法向分离 (正值 = 张开)
      - δ_t1, δ_t2: 切向滑移
    
    牵引力-分离律通过 CohesiveLaw 定义。
    
    用法:
        ci = CohesiveInterfaceIntegrator(cohesive_law)
        K_coh, f_coh = ci.interface_stiffness(
            mesh, interface_id, u, damage_history
        )
    """
    
    def __init__(
        self,
        cohesive_law: Optional[CohesiveLaw] = None,
    ):
        self.law = cohesive_law or CohesiveLaw(
            sigma_max=1e6, Gc=100.0, delta_0=1e-6, delta_f=1e-4,
        )
        
        # 界面损伤历史 (积分点级别)
        self.damage_history: List[List[float]] = []  # [interface_id][quad_point_idx]
        self.separation_history: List[List[float]] = []
    
    def interface_stiffness(
        self,
        mesh: PolyMesh,
        face_id: int,
        displacement: np.ndarray,
        dof_map: DOFMap,
        quadrature_order: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """返回界面单元的刚度矩阵和内力向量
        
        K_coh = ∫_Γ R^T · (1-D)K₀ · R dΓ
        f_coh = ∫_Γ R^T · t(δ) dΓ
        
        Args:
            mesh: CC 控制网格
            face_id: 界面面 ID
            displacement: (n_dof,) 全局位移向量
            dof_map: DOF 映射
            quadrature_order: 积分阶数
        
        Returns:
            (K_coh, f_coh) 48×48 刚度矩阵和内力向量
        """
        verts = mesh.face_vertices(face_id)
        if len(verts) != 4:
            return np.zeros((1, 1)), np.zeros(1)
        
        # 控制点 patch (16 个控制点)
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
        
        # 构建单元 DOF
        u_e = np.zeros(48)
        for i, vi in enumerate(all_verts):
            dx, dy, dz = dof_map.vertex_dofs(vi)
            u_e[3*i + 0] = displacement[dx]
            u_e[3*i + 1] = displacement[dy]
            u_e[3*i + 2] = displacement[dz]
        
        from forgecraft.analysis.iga_shape import cc_shape_functions
        from forgecraft.analysis.iga_quadrature import gauss_legendre_2d
        
        gauss_pts, gauss_wts = gauss_legendre_2d(quadrature_order)
        
        Ke = np.zeros((48, 48))
        fe = np.zeros(48)
        
        # 面中心法向 (近似)
        p0 = mesh.vertices[verts[0]]
        p1 = mesh.vertices[verts[1]]
        p2 = mesh.vertices[verts[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        n_mag = np.linalg.norm(normal)
        if n_mag < 1e-15:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / n_mag
        
        # 切向方向
        t1 = p1 - p0
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1)
        
        # R 矩阵: 从全局位移到局部分离
        R_local = np.column_stack([normal, t1, t2])  # (3, 3)
        
        for k in range(len(gauss_wts)):
            u_q, v_q = gauss_pts[k, 0], gauss_pts[k, 1]
            shape = cc_shape_functions(mesh, face_id, u_q, v_q)
            
            if shape.detJ < 1e-15:
                continue
            
            # 积分点处分离向量 δ = R_local^T · Δu
            du_q = np.zeros(3)
            for node_i in range(16):
                N_i = shape.N[node_i]
                du_q += N_i * u_e[3*node_i : 3*node_i+3]
            
            delta_local = R_local.T @ du_q
            
            # 损伤和有效分离量
            D, delta_eff = self.law.damage(delta_local, 0.0)
            
            # 牵引力和切线刚度
            t_local, D_t_local = self.law.traction(delta_local, D)
            
            # 回到全局坐标
            t_global = R_local @ t_local
            D_t_global = R_local @ D_t_local @ R_local.T
            
            # 积分
            factor = gauss_wts[k] * shape.detJ
            
            for i in range(16):
                N_i = shape.N[i]
                i3 = i * 3
                
                # 内力
                fe[i3:i3+3] -= factor * N_i * t_global
                
                for j in range(16):
                    N_j = shape.N[j]
                    j3 = j * 3
                    
                    # 切线刚度
                    Ke[i3:i3+3, j3:j3+3] += factor * N_i * N_j * D_t_global
        
        return Ke, fe
    
    def update_damage(
        self,
        interface_id: int,
        separation: np.ndarray,  # (n_quad, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """更新积分点损伤状态
        
        Args:
            interface_id: 界面 ID
            separation: (n_quad, 3) 各积分点分离向量
        
        Returns:
            (D_array, delta_eff_array) 损伤和有效分离量
        """
        # 确保历史数组足够大
        while interface_id >= len(self.damage_history):
            n_q = separation.shape[0]
            self.damage_history.append([0.0] * n_q)
            self.separation_history.append([0.0] * n_q)
        
        n_q = separation.shape[0]
        D = np.zeros(n_q)
        delta_eff = np.zeros(n_q)
        
        for q in range(n_q):
            D[q], delta_eff[q] = self.law.damage(
                separation[q],
                self.separation_history[interface_id][q],
            )
            self.damage_history[interface_id][q] = D[q]
            self.separation_history[interface_id][q] = delta_eff[q]
        
        return D, delta_eff


# ══════════════════════════════════════════════════════════
#  全局内聚力刚度
# ══════════════════════════════════════════════════════════

def cohesive_stiffness(
    mesh: PolyMesh,
    interface_faces: List[int],
    displacement: np.ndarray,
    dof_map: DOFMap,
    cohesive_law: CohesiveLaw,
) -> Tuple[csr_matrix, np.ndarray]:
    """组装全局内聚力刚度矩阵和内力向量
    
    Args:
        mesh: CC 控制网格
        interface_faces: 界面面 ID 列表
        displacement: (n_dof,) 全局位移
        dof_map: DOF 映射
        cohesive_law: 牵引力-分离律
    
    Returns:
        (K_coh, f_coh) 全局 CSR 稀疏矩阵和力向量
    """
    n_dof = dof_map.n_dof
    
    ci = CohesiveInterfaceIntegrator(cohesive_law)
    
    rows = []
    cols = []
    vals_K = []
    f_global = np.zeros(n_dof)
    
    for face_id in interface_faces:
        Ke, fe = ci.interface_stiffness(mesh, face_id, displacement, dof_map)
        
        if Ke.shape[0] <= 1:
            continue
        
        # 面控制点 DOF
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
        
        element_dofs = []
        for vi in all_verts:
            dx, dy, dz = dof_map.vertex_dofs(vi)
            element_dofs.extend([dx, dy, dz])
        
        for i in range(48):
            for j in range(48):
                if abs(Ke[i, j]) > 1e-15:
                    rows.append(element_dofs[i])
                    cols.append(element_dofs[j])
                    vals_K.append(Ke[i, j])
            f_global[element_dofs[i]] += fe[i]
    
    if rows:
        K_coh = coo_matrix(
            (vals_K, (rows, cols)), shape=(n_dof, n_dof),
        ).tocsr()
    else:
        K_coh = csr_matrix((n_dof, n_dof))
    
    return K_coh, f_global


def cohesive_traction(
    separation: np.ndarray,
    cohesive_law: CohesiveLaw,
    damage: float = 0.0,
) -> np.ndarray:
    """计算单个积分点的界面牵引力
    
    Args:
        separation: (3,) 分离向量
        cohesive_law: 牵引力-分离律
        damage: 初始损伤
    
    Returns:
        traction: (3,) 牵引力向量
    """
    D, _ = cohesive_law.damage(separation, 0.0)
    t, _ = cohesive_law.traction(separation, max(D, damage))
    return t


def insert_cohesive_elements(
    mesh: PolyMesh,
    interface_faces: List[int],
) -> List[int]:
    """标记内聚力界面面
    
    在 IGA 中，界面单元不需要修改网格拓扑——
    我们直接在原有面对之间计算分离量。
    这个函数保持接口兼容性。
    
    Args:
        mesh: CC 控制网格
        interface_faces: 界面面列表
    
    Returns:
        interface_faces: 返回相同的接口面 ID 列表
    """
    return interface_faces


def compute_effective_separation(delta: np.ndarray) -> float:
    """计算混合模式有效分离量
    
    Args:
        delta: (3,) 分离向量 [δ_n, δ_t1, δ_t2]
    
    Returns:
        δ_eff = sqrt(⟨δ_n⟩² + δ_t1² + δ_t2²)
    """
    delta_n = max(0.0, delta[0])  # 仅拉伸有效
    delta_t = np.sqrt(delta[1]**2 + delta[2]**2)
    return np.sqrt(delta_n**2 + delta_t**2)
