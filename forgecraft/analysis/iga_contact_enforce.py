# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_contact_enforce — 接触约束施加
#
#   assemble_contact_penalty:      罚函数法
#   solve_penalty_contact:         罚函数接触求解
#   solve_augmented_lagrange:      增广拉格朗日 (Uzawa 算法)
#   assemble_friction_contribution: Coulomb 摩擦贡献
#   contact_force_residual:        接触力残差评估
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.geometry.evaluator import LimitEvaluator
from forgecraft.analysis._iga_base import DOFMap, IrregularFaceCache
from forgecraft.analysis.iga_contact_types import (
    ContactSettings, ContactPair, GapResult, FrictionModel,
)
from forgecraft.analysis.iga_contact_search import (
    find_contact_pairs, build_contact_pairs, AABBTree,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "assemble_contact_penalty",
    "assemble_friction_contribution",
    "solve_penalty_contact",
    "solve_augmented_lagrange",
    "contact_force_residual",
    "compute_contact_energy",
]


# ══════════════════════════════════════════════════════════
#  罚函数法组装
# ══════════════════════════════════════════════════════════

def assemble_contact_penalty(
    mesh: PolyMesh,
    contact_pairs: List[ContactPair],
    dof_map: DOFMap,
    settings: ContactSettings,
    friction: Optional[FrictionModel] = None,
) -> Tuple[csr_matrix, np.ndarray]:
    """罚函数法组装接触刚度矩阵和力向量
    
    对每个活跃接触对:
      K_contact += ε_N · N^T · n⊗n^T · N · w · detJ
      f_contact += ε_N · gap · N^T · n · w · detJ
    
    外加摩擦贡献:
      K_contact += K_friction
      f_contact += f_friction
    
    Args:
        mesh: CC 控制网格 (通常是 slave mesh)
        contact_pairs: 活跃接触对列表
        dof_map: DOF 映射
        settings: 接触设置
        friction: 摩擦模型 (None = 无摩擦)
    
    Returns:
        (K_contact, f_contact) 增量接触贡献
    """
    n_dof = dof_map.n_dof
    
    # 使用 triplet 格式积累
    rows = []
    cols = []
    vals_K = []
    f_contact = np.zeros(n_dof)
    
    from forgecraft.analysis.iga_shape import cc_shape_functions
    from forgecraft.analysis.iga_quadrature import gauss_legendre_2d
    
    gauss_pts, gauss_wts = gauss_legendre_2d(settings.quadrature_order_contact)
    
    for pair in contact_pairs:
        if not pair.active:
            continue
        
        s_face = pair.slave_face
        
        # 提取从面控制顶点 (16 个)
        s_verts = mesh.face_vertices(s_face)
        all_verts = list(s_verts)
        seen = set(s_verts)
        for v in s_verts:
            ring = mesh.vertex_ring(v)
            for r in ring:
                if r not in seen and len(all_verts) < 16:
                    all_verts.append(r)
                    seen.add(r)
        while len(all_verts) < 16:
            all_verts.append(all_verts[-1])
        all_verts = all_verts[:16]
        
        # DOF 编号 (48 DOF: 16 顶点 × 3)
        element_dofs = []
        for vi in all_verts:
            dx, dy, dz = dof_map.vertex_dofs(vi)
            element_dofs.extend([dx, dy, dz])
        
        n = pair.normal  # 主面法向量 (单位)
        u_s, v_s = pair.slave_uv[0], pair.slave_uv[1]
        
        # 从面形函数求值
        shape = cc_shape_functions(mesh, s_face, u_s, v_s)
        if shape.detJ < 1e-15:
            continue
        
        # 法向罚刚度贡献: K_n = ε_N · (N^T n) · (N^T n)^T · w · detJ
        Nn = np.zeros(48)  # N^T · n (每个 DOF 的法向分量)
        for node_i in range(16):
            N_i = shape.N[node_i]
            i3 = node_i * 3
            Nn[i3 + 0] = N_i * n[0]
            Nn[i3 + 1] = N_i * n[1]
            Nn[i3 + 2] = N_i * n[2]
        
        weight = settings.penalty_normal * shape.detJ
        # 简化: 使用 Gauss 积分权重 (积分点在 params 处固定)
        # 更精确的: 在 contact_pairs 构建时就已经知道积分点位置
        
        # 法向接触力
        f_n = -settings.penalty_normal * pair.gap  # 正 gap → 负力 (推回)
        if f_n > 0:  # 间隙为正 → 无接触力
            f_n = 0.0
        
        # 简化: 直接加在相关 DOF 的非对角块
        for i in range(48):
            for j in range(48):
                k_val = Nn[i] * Nn[j]
                if abs(k_val) > 1e-15:
                    rows.append(element_dofs[i])
                    cols.append(element_dofs[j])
                    vals_K.append(weight * k_val)
            
            f_contact[element_dofs[i]] += f_n * Nn[i]
        
        # 摩擦贡献 (切向)
        if friction is not None and friction.mu > 0.0:
            K_T, f_T = _assemble_friction_local(
                shape, element_dofs, pair, friction, settings, n,
            )
            # 将局部刚度添加到 triplet
            for i in range(48):
                for j in range(48):
                    if abs(K_T[i, j]) > 1e-15:
                        rows.append(element_dofs[i])
                        cols.append(element_dofs[j])
                        vals_K.append(K_T[i, j])
                f_contact[element_dofs[i]] += f_T[i]
    
    # 构建 CSR 矩阵
    from scipy.sparse import coo_matrix
    if rows:
        K_contact = coo_matrix(
            (vals_K, (rows, cols)),
            shape=(n_dof, n_dof),
        ).tocsr()
    else:
        K_contact = csr_matrix((n_dof, n_dof))
    
    return K_contact, f_contact


def _assemble_friction_local(
    shape,
    element_dofs: List[int],
    pair: ContactPair,
    friction: FrictionModel,
    settings: ContactSettings,
    normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """局部切向摩擦贡献
    
    切向间隙 (增量):  Δg_T = (I - n⊗n^T) · (Δu_s - Δu_m^*)
    试探牵引力:       t_T^trial = ε_T · Δg_T
    Coulomb 条件:     Φ = ||t_T^trial|| - μ·p_N ≤ 0
    
    Returns:
        (K_T_48x48, f_T_48) 切向刚度和力
    """
    eps_T = settings.penalty_tangential
    mu = friction.mu
    
    # 接触压力 (法向)
    p_n = max(0.0, -settings.penalty_normal * pair.gap)
    
    if p_n < 1e-15:
        return np.zeros((48, 48)), np.zeros(48)
    
    # 切向投影矩阵 P_T = I₃ - n⊗n^T
    P_T = np.eye(3) - np.outer(normal, normal)
    
    # 构建 N^T · P_T · N  (48×48 切向刚度)
    K_T = np.zeros((48, 48))
    f_T = np.zeros(48)
    
    for i in range(16):
        N_i = shape.N[i]
        i3 = i * 3
        
        # 当前滑移增量 (简化: 从 pair.slip 估计)
        slip_vec = np.zeros(3)
        if pair.slip is not None and len(pair.slip) >= 2:
            slip_vec = pair.slip[0] * normal + pair.slip[1] * normal
            # 简化处理
        
        for j in range(16):
            N_j = shape.N[j]
            j3 = j * 3
            
            # K_T_{ij} = ε_T · N_i · P_T · N_j
            k_ij = eps_T * N_i * N_j
            K_T[i3:i3+3, j3:j3+3] = k_ij * P_T
        
        # 切向力
        f_t_local = min(eps_T, mu * p_n / (np.linalg.norm(slip_vec) + 1e-15))
        f_T[i3:i3+3] = f_t_local * N_i * np.dot(P_T, slip_vec)
    
    return K_T, f_T


# ══════════════════════════════════════════════════════════
#  罚函数接触求解
# ══════════════════════════════════════════════════════════

def solve_penalty_contact(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    iga_settings,
    contact_settings: ContactSettings,
    dirichlet_bcs,
    neumann_bcs=None,
    point_loads=None,
    contact_mesh: Optional[PolyMesh] = None,
    irregular_cache=None,
) -> Tuple[np.ndarray, List[ContactPair]]:
    """罚函数接触求解器
    
    Ku = f_ext + f_contact(u)
    
    自接触 (self-contact): 当 contact_mesh is None
    双体接触:             当 contact_mesh 是另一个网格
    
    Args:
        mesh: 主网格
        contact_mesh: 接触网格 (None = 自体接触)
    
    Returns:
        (u, contact_pairs) 位移解和接触对
    """
    from forgecraft.analysis.iga_assembly import (
        assemble_stiffness, assemble_force_vector,
    )
    from forgecraft.analysis.iga_boundary import (
        apply_dirichlet_penalty, apply_dirichlet_elimination,
    )
    
    n_dof = dof_map.n_dof
    
    # 1. 组装基础刚度矩阵
    K = assemble_stiffness(
        mesh, integrator, material, dof_map, iga_settings, irregular_cache,
    )
    F = assemble_force_vector(
        mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
    )
    
    # 2. 接触检测
    if contact_mesh is None:
        # 自体接触
        tree = AABBTree(mesh)
        contact_pairs = find_contact_pairs(
            mesh, mesh, tree, contact_settings,
        )
    else:
        # 双体接触
        contact_pairs = build_contact_pairs(
            mesh, contact_mesh, contact_settings,
        )
    
    # 3. 施加接触约束 (Newton 迭代)
    u = np.zeros(n_dof)
    max_newton = 20
    
    for newton_step in range(max_newton):
        # 接触贡献
        K_contact, f_contact = assemble_contact_penalty(
            mesh, contact_pairs, dof_map, contact_settings,
        )
        
        K_total = K.copy()
        F_total = F.copy()
        
        # 将接触贡献加到全局系统
        if K_contact.nnz > 0:
            K_total = K_total + K_contact
        F_total = F_total + f_contact
        
        # 施加 Dirichlet BC
        if dirichlet_bcs:
            for bc in dirichlet_bcs:
                if iga_settings.solver == "direct":
                    u = np.zeros(n_dof)
                    apply_dirichlet_elimination(K_total.toarray() if hasattr(K_total, 'toarray') else K_total, F_total, bc, u)
                else:
                    apply_dirichlet_penalty(K_total.toarray() if hasattr(K_total, 'toarray') else K_total, F_total, bc)
        
        # 求解
        from forgecraft.analysis.iga_solver import solve_linear
        du, _, converged = solve_linear(K_total, F_total, iga_settings)
        u_new = u + du
        u_new = np.maximum(np.minimum(u_new, 1e-3), -1e-3)  # 位移限制
        
        # 收敛检查
        du_norm = np.linalg.norm(du) / max(np.linalg.norm(u_new), 1e-15)
        u = u_new
        
        if du_norm < 1e-6:
            break
    
    return u, contact_pairs


# ══════════════════════════════════════════════════════════
#  增广拉格朗日法
# ══════════════════════════════════════════════════════════

def solve_augmented_lagrange(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    iga_settings,
    contact_settings: ContactSettings,
    dirichlet_bcs,
    neumann_bcs=None,
    point_loads=None,
    contact_mesh: Optional[PolyMesh] = None,
    irregular_cache=None,
) -> Tuple[np.ndarray, List[ContactPair], bool]:
    """增广拉格朗日法 (Uzawa 算法)
    
    for k in 1..max_uzawa:
        1. 固定 λ，求解位移 → u^{k+1}
        2. 更新乘子: λ^{k+1} = max(0, λ^k + ε_N · gap(u^{k+1}))
        3. 检查: ||λ^{k+1} - λ^k|| < tolerance
    
    优势: 精确满足约束 + 罚因子可以较小 (1e6~1e8)
    
    Returns:
        (u, contact_pairs, converged)
    """
    from forgecraft.analysis.iga_assembly import (
        assemble_stiffness, assemble_force_vector,
    )
    from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty
    from forgecraft.analysis.iga_solver import solve_linear
    
    n_dof = dof_map.n_dof
    
    # 基础刚度矩阵和力向量 (不变)
    K = assemble_stiffness(
        mesh, integrator, material, dof_map, iga_settings, irregular_cache,
    ).toarray()
    F = assemble_force_vector(
        mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
    )
    
    # 施加 Dirichlet BC 的罚修正
    if dirichlet_bcs:
        for bc in dirichlet_bcs:
            apply_dirichlet_penalty(K, F, bc)
    
    # Uzawa 迭代
    u = np.zeros(n_dof)
    contact_pairs = []
    uzawa_converged = False
    
    eps_N = contact_settings.penalty_normal
    tol_uzawa = contact_settings.augmented_tolerance
    
    for k_uzawa in range(contact_settings.augmented_max_iter):
        # 检测接触对 (用当前位移)
        if contact_mesh is None:
            tree = AABBTree(mesh)
            contact_pairs = find_contact_pairs(mesh, mesh, tree, contact_settings)
        else:
            contact_pairs = build_contact_pairs(
                mesh, contact_mesh, contact_settings,
            )
        
        if not contact_pairs:
            # 无接触 → 直接求解
            u = np.linalg.solve(K, F)
            uzawa_converged = True
            break
        
        # 更新法向力 (乘子)
        lambda_old = np.array([pair.lambda_n for pair in contact_pairs])
        
        for pair in contact_pairs:
            pair.lambda_n = max(0.0, pair.lambda_n + eps_N * pair.gap)
        
        lambda_new = np.array([pair.lambda_n for pair in contact_pairs])
        
        # 组装接触力 (乘子贡献)
        F_aug = F.copy()
        for pair in contact_pairs:
            if not pair.active or pair.gap >= contact_settings.gap_tolerance:
                continue
            
            # 等效节点力 = λ · N^T · n
            from forgecraft.analysis.iga_shape import cc_shape_functions
            
            s_verts = mesh.face_vertices(pair.slave_face)
            all_verts = list(s_verts)
            seen = set(s_verts)
            for v in s_verts:
                ring = mesh.vertex_ring(v)
                for r in ring:
                    if r not in seen and len(all_verts) < 16:
                        all_verts.append(r)
                        seen.add(r)
            while len(all_verts) < 16:
                all_verts.append(all_verts[-1])
            all_verts = all_verts[:16]
            
            shape = cc_shape_functions(
                mesh, pair.slave_face, pair.slave_uv[0], pair.slave_uv[1],
            )
            
            for node_i in range(16):
                N_i = shape.N[node_i]
                dx, dy, dz = dof_map.vertex_dofs(all_verts[node_i])
                f_val = pair.lambda_n * N_i
                F_aug[dx] += f_val * pair.normal[0]
                F_aug[dy] += f_val * pair.normal[1]
                F_aug[dz] += f_val * pair.normal[2]
        
        # 求解位移
        u_new = np.linalg.solve(K, F_aug)
        u = u_new
        
        # Uzawa 收敛
        if np.linalg.norm(lambda_new - lambda_old) < tol_uzawa * max(1.0, np.linalg.norm(lambda_new)):
            uzawa_converged = True
            break
    
    return u, contact_pairs, uzawa_converged


# ══════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════

def contact_force_residual(
    mesh: PolyMesh,
    contact_pairs: List[ContactPair],
    dof_map: DOFMap,
    settings: ContactSettings,
) -> float:
    """接触力残差 (用于收敛检查)
    
    ‖f_contact(u)‖ / ‖f_ext‖
    """
    _, f_contact = assemble_contact_penalty(mesh, contact_pairs, dof_map, settings)
    return float(np.linalg.norm(f_contact))


def compute_contact_energy(
    mesh: PolyMesh,
    contact_pairs: List[ContactPair],
    dof_map: DOFMap,
    settings: ContactSettings,
) -> float:
    """接触能量 (罚函数形式)
    
    E_contact = ½ ε_N · Σ (gap² · w · detJ)
    """
    energy = 0.0
    for pair in contact_pairs:
        if pair.active and pair.gap < settings.gap_tolerance:
            energy += 0.5 * settings.penalty_normal * pair.gap * pair.gap
    
    return energy


def assemble_friction_contribution(
    contact_pairs: List[ContactPair],
    friction: FrictionModel,
    penalty_tangential: float,
) -> Tuple[csr_matrix, np.ndarray]:
    """独立的摩擦贡献 (可添加到现有系统)
    
    Args:
        contact_pairs: 接触对列表
        friction: 摩擦模型
        penalty_tangential: 切向罚因子
    
    Returns:
        (K_friction, f_friction)
    """
    # 简化: 零摩擦 (在 assemble_contact_penalty 内部处理)
    n_total = 0  # 需要外部传入 dof_map
    return csr_matrix((n_total, n_total)), np.zeros(n_total)
