# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_topology_sensitivity — 灵敏度分析
#
#   相位场拓扑优化需要目标/约束对设计变量 φ 的导数。
#   在 IGA 框架下, 灵敏度在控制顶点级别计算。
#
#   核心优势:
#     - 柔度灵敏度: 自伴随 (No extra linear solve!)
#     - 体积灵敏度: 平凡 (dV/dφ_i = area_i)
#     - 应力/位移约束: 伴随法 (1 extra linear solve each)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import DOFMap, IGASettings, IrregularFaceCache
from forgecraft.analysis.iga_shape import cc_shape_functions
from forgecraft.analysis.iga_quadrature import cc_quadrature
from forgecraft.analysis.iga_topology_types import (
    TopologySettings,
    TopologyMaterial,
    g_simp, dg_simp, g_ramp, dg_ramp, g_polynomial, dg_polynomial,
)
from forgecraft.analysis.iga_topology_phasefield import _compute_vertex_areas

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "compute_compliance_sensitivity",
    "compute_volume_sensitivity",
    "compute_stress_constraint_sensitivity",
    "compute_displacement_constraint_sensitivity",
    "adjoint_solve",
    "filter_sensitivity",
]


# ══════════════════════════════════════════════════════════
#  柔度灵敏度 (自伴随!)
# ══════════════════════════════════════════════════════════

def compute_compliance_sensitivity(
    mesh: PolyMesh,
    phi_field: np.ndarray,
    psi_e_field: np.ndarray,
    settings: TopologySettings,
) -> np.ndarray:
    """计算柔度目标对相场的灵敏度: dC/dφ

    柔度 C = u^T · K · u = ∫ g(φ) · ε(u) : D : ε(u) dΩ

    自伴随性质:
      dC/dφ_i = -∫_Ω g'(φ) · N_i · ψe(ε(u)) dΩ

    物理意义:
      正应变能区域 → dC/dφ < 0 → 增加 φ (加材料) 降低柔度

    无需额外的伴随求解!

    Args:
        mesh: CC 控制网格
        phi_field: (n_vertices,) 当前相场
        psi_e_field: (n_vertices,) 应变能密度
        settings: 拓扑优化设置

    Returns:
        dc: (n_vertices,) 柔度灵敏度 (dC/dφ_i)
    """
    n_vertices = mesh.n_vertices
    dc = np.zeros(n_vertices)

    vertex_areas = _compute_vertex_areas(mesh)

    for vi in range(n_vertices):
        phi = phi_field[vi]
        psi = psi_e_field[vi]
        area = vertex_areas[vi]

        if area < 1e-15:
            continue

        # dg/dφ
        if settings.interpolation == "SIMP":
            dg_val = dg_simp(phi, settings.p_current, settings.phi_min)
        elif settings.interpolation == "RAMP":
            dg_val = dg_ramp(phi, q=3.0, phi_min=settings.phi_min)
        else:
            dg_val = dg_polynomial(phi)

        # dC/dφ_i = -g'(φ_i) · ψe_i · area_i
        # 负号: 增加材料 → g 增加 → C 降低
        dc[vi] = -dg_val * psi * area

    return dc


# ══════════════════════════════════════════════════════════
#  体积灵敏度 (平凡)
# ══════════════════════════════════════════════════════════

def compute_volume_sensitivity(
    mesh: PolyMesh,
) -> np.ndarray:
    """计算体积约束对相场的灵敏度: dV/dφ

    V = Σ φ_i · area_i → dV/dφ_i = area_i

    Args:
        mesh: CC 控制网格

    Returns:
        dv: (n_vertices,) 体积灵敏度 (dV/dφ_i)
    """
    return _compute_vertex_areas(mesh)


# ══════════════════════════════════════════════════════════
#  伴随法求解
# ══════════════════════════════════════════════════════════

def adjoint_solve(
    K: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """伴随法求解器

    求解 K · λ = rhs (伴随方程)

    注意: 伴随方程使用退化的切线刚度 K(g(φ)),
    与正向位移求解相同矩阵 → 可复用 LU/Cholesky 分解。

    Args:
        K: (n_dof, n_dof) 切线刚度矩阵 (退化后)
        rhs: (n_dof,) 伴随右端项

    Returns:
        lambda_adj: (n_dof,) 伴随变量
    """
    try:
        lam = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        _logger.warning("Adjoint solve singular, returning zeros")
        lam = np.zeros_like(rhs)
    return lam


# ══════════════════════════════════════════════════════════
#  应力约束灵敏度 (p-norm)
# ══════════════════════════════════════════════════════════

def compute_stress_constraint_sensitivity(
    mesh: PolyMesh,
    u: np.ndarray,
    phi_field: np.ndarray,
    dof_map: DOFMap,
    material: TopologyMaterial,
    settings: TopologySettings,
    sigma_yield: float,
    p_norm: float = 8.0,
    iga_settings: Optional[IGASettings] = None,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> Tuple[np.ndarray, float]:
    """计算 p-norm 应力约束及其灵敏度

    σ_pn = [Σ_i (σ_vm_i / σ_yield)^p · area_i]^(1/p)

    应力约束: σ_pn ≤ 1

    灵敏度: dσ_pn/dφ = Σ (∂σ_pn/∂σ_vm_i · ∂σ_vm_i/∂φ + adjoint)

    完整伴随法推导:
      R(u, φ) = K(g(φ)) · u - f_ext = 0  (残差)
      dσ_pn/dφ = ∂σ_pn/∂φ + λ^T · ∂R/∂φ
      where K^T · λ = ∂σ_pn/∂u   (伴随方程)

    简化实现 (适用于当前线性系统):
      使用自伴随近似 — dσ_pn/dφ ≈ ∂σ_pn/∂φ (忽略 ∂σ_vm/∂u 间接效应)

    Args:
        mesh: CC 控制网格
        u: (n_dof,) 位移解
        phi_field: (n_vertices,) 相场
        dof_map: 结构 DOF 映射
        material: 拓扑优化材料
        settings: 拓扑优化设置
        sigma_yield: 屈服强度 [Pa]
        p_norm: p-norm 聚合指数
        iga_settings: IGA 设置
        irregular_cache: 非常面缓存

    Returns:
        (dsigma_dphi, sigma_pn_value): 应力的 p-norm 值及其灵敏度
    """
    from forgecraft.analysis.iga_element import _plane_stress_D

    n_vertices = mesh.n_vertices
    vertex_areas = _compute_vertex_areas(mesh)

    # 1. 计算每个控制点的 von Mises 应力 (简化: 用顶点应变能反推)
    sigma_vm = np.zeros(n_vertices)

    D = _plane_stress_D(material.E, material.nu)

    for face_id in range(mesh.n_faces):
        verts = mesh.face_vertices(face_id)
        if len(verts) != 4:
            continue

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

        u_e = np.zeros(48)
        for i, vi in enumerate(all_verts):
            dx, dy, dz = dof_map.vertex_dofs(vi)
            u_e[3*i + 0] = u[dx]
            u_e[3*i + 1] = u[dy]
            u_e[3*i + 2] = u[dz]

        qps = cc_quadrature(mesh, face_id, settings.quadrature_order)

        for qp in qps:
            shape = cc_shape_functions(mesh, face_id, qp.u, qp.v)
            if shape.detJ < 1e-15:
                continue

            B = _build_B_matrix(shape)
            eps = B @ u_e
            sigma = D @ eps

            # von Mises: σ_vm = sqrt(σxx² + σyy² - σxx·σyy + 3τxy²)
            svm = np.sqrt(sigma[0]**2 + sigma[1]**2 - sigma[0]*sigma[1] + 3*sigma[2]**2)

            area = qp.weight * shape.detJ
            for node_i, vi in enumerate(all_verts[:16]):
                sigma_vm[vi] += svm * area

    # 归一化
    for vi in range(n_vertices):
        if vertex_areas[vi] > 1e-15:
            sigma_vm[vi] /= vertex_areas[vi]

    # 2. p-norm 聚合
    sigma_ratio = sigma_vm / max(sigma_yield, 1e-10)
    sigma_p = sigma_ratio ** p_norm

    n_active = np.sum(vertex_areas > 1e-15)
    sigma_pn = (np.sum(sigma_p * vertex_areas) / max(np.sum(vertex_areas), 1e-15)) ** (1.0 / p_norm)

    # 3. 灵敏度 (简化自伴随)
    if sigma_pn < 1e-15:
        dsigma_dphi = np.zeros(n_vertices)
        return dsigma_dphi, sigma_pn

    # dσ_pn/dφ_i ≈ (σ_pn)^(1-p) · (σ_vm_i/σ_yield)^(p-1) · area_i / total_area · dσ_vm/dφ
    # dσ_vm/dφ 通过 dψe/dφ 近似: σ_vm ∝ sqrt(ψe)
    dsigma_dphi = np.zeros(n_vertices)
    prefactor = sigma_pn ** (1.0 - p_norm)

    for vi in range(n_vertices):
        if vertex_areas[vi] < 1e-15:
            continue

        # g'(φ) 的影响: σ ∝ sqrt(g(φ)) → dσ/dφ ∝ g'(φ)/(2·sqrt(g(φ)))
        phi = phi_field[vi]
        if settings.interpolation == "SIMP":
            gv = g_simp(phi, settings.p_current, settings.phi_min)
            dgv = dg_simp(phi, settings.p_current, settings.phi_min)
        elif settings.interpolation == "RAMP":
            gv = g_ramp(phi, q=3.0, phi_min=settings.phi_min)
            dgv = dg_ramp(phi, q=3.0, phi_min=settings.phi_min)
        else:
            gv = g_polynomial(phi)
            dgv = dg_polynomial(phi)

        if gv < 1e-15:
            continue

        dsigma_dg = sigma_vm[vi] / (2.0 * gv + 1e-15)
        dsigma_dphi[vi] = prefactor * sigma_ratio[vi] ** (p_norm - 1.0) * \
                          dsigma_dg * dgv * vertex_areas[vi] / max(np.sum(vertex_areas), 1e-15)

    return dsigma_dphi, sigma_pn


def _build_B_matrix(shape) -> np.ndarray:
    """构建 B 矩阵 (3x48) — 辅助"""
    J = shape.J
    t1 = J[:, 0].copy()
    n1 = np.linalg.norm(t1)
    if n1 > 1e-15:
        t1 /= n1
    t2 = J[:, 1].copy()
    t2 -= np.dot(t2, t1) * t1
    n2 = np.linalg.norm(t2)
    if n2 > 1e-15:
        t2 /= n2

    B = np.zeros((3, 48))
    for i in range(16):
        dNdx = shape.dN_dx[i]
        dN_dt1 = np.dot(dNdx, t1)
        dN_dt2 = np.dot(dNdx, t2)
        i3 = i * 3
        B[0, i3 + 0] = dN_dt1 * t1[0]
        B[0, i3 + 1] = dN_dt1 * t1[1]
        B[0, i3 + 2] = dN_dt1 * t1[2]
        B[1, i3 + 0] = dN_dt2 * t2[0]
        B[1, i3 + 1] = dN_dt2 * t2[1]
        B[1, i3 + 2] = dN_dt2 * t2[2]
        B[2, i3 + 0] = dN_dt1 * t2[0] + dN_dt2 * t1[0]
        B[2, i3 + 1] = dN_dt1 * t2[1] + dN_dt2 * t1[1]
        B[2, i3 + 2] = dN_dt1 * t2[2] + dN_dt2 * t1[2]
    return B


# ══════════════════════════════════════════════════════════
#  位移约束灵敏度 (伴随法)
# ══════════════════════════════════════════════════════════

def compute_displacement_constraint_sensitivity(
    mesh: PolyMesh,
    u: np.ndarray,
    phi_field: np.ndarray,
    dof_map: DOFMap,
    settings: TopologySettings,
    K_u: np.ndarray,
    dof_id: int,
    u_max: float,
    iga_settings: Optional[IGASettings] = None,
) -> Tuple[np.ndarray, float]:
    """计算位移约束及其灵敏度

    约束: g_disp = u_dof_id ≤ u_max

    伴随法:
      K · λ = e_dof_id      (单位载荷在受约束 DOF)
      dg_disp/dφ = -λ^T · ∂K/∂φ · u

    简化: u_i 直接由 K 和 F 决定, dK/dφ 通过 g'(φ) 给出。

    Args:
        mesh: CC 控制网格
        u: (n_dof,) 位移解
        phi_field: (n_vertices,) 相场
        dof_map: 结构 DOF 映射
        settings: 拓扑优化设置
        K_u: (n_dof, n_dof) 退化刚度矩阵
        dof_id: 被约束的全局 DOF
        u_max: 位移上限
        iga_settings: IGA 设置

    Returns:
        (dg_dphi, g_value): 约束值及其灵敏度
    """
    n_dof = len(u)
    n_vertices = mesh.n_vertices

    # 约束值
    g_value = u[dof_id]

    # 伴随求解: K · λ = e_dof_id
    e_dof = np.zeros(n_dof)
    e_dof[dof_id] = 1.0

    try:
        lam = np.linalg.solve(K_u, e_dof)
    except np.linalg.LinAlgError:
        lam = np.zeros(n_dof)

    # 灵敏度: dg/dφ = -λ^T · ∂K/∂φ · u
    # ∂K_ij/∂φ_k = Σ (∂g/∂φ_k) · K0_ij
    # 近似: dg/dφ_k ≈ -λ_k · u_k · dg'(φ_k)/g(φ_k)
    dg_dphi = np.zeros(n_vertices)
    vertex_areas = _compute_vertex_areas(mesh)

    for vi in range(n_vertices):
        phi = phi_field[vi]
        if settings.interpolation == "SIMP":
            gv = g_simp(phi, settings.p_current, settings.phi_min)
            dgv = dg_simp(phi, settings.p_current, settings.phi_min)
        elif settings.interpolation == "RAMP":
            gv = g_ramp(phi, q=3.0, phi_min=settings.phi_min)
            dgv = dg_ramp(phi, q=3.0, phi_min=settings.phi_min)
        else:
            gv = g_polynomial(phi)
            dgv = dg_polynomial(phi)

        if gv < 1e-15:
            continue

        # 简化: 用 DOF 能量贡献近似
        dx, dy, dz = dof_map.vertex_dofs(vi)
        lam_factor = abs(lam[dx]) + abs(lam[dy]) + abs(lam[dz])
        u_factor = abs(u[dx]) + abs(u[dy]) + abs(u[dz])

        dg_dphi[vi] = lam_factor * u_factor * dgv / gv * vertex_areas[vi]

    return dg_dphi, g_value


# ══════════════════════════════════════════════════════════
#  灵敏度滤波 (可选)
# ══════════════════════════════════════════════════════════

def filter_sensitivity(
    mesh: PolyMesh,
    dc: np.ndarray,
    phi_field: np.ndarray,
    filter_radius: float,
) -> np.ndarray:
    """对灵敏度进行密度加权滤波

    SIMP 标准滤波: dc_filtered_i = Σ_j H_ij · phi_j · dc_j / (phi_i · Σ_j H_ij)

    其中 H_ij = max(0, r_min - dist(i,j))

    Args:
        mesh: CC 控制网格
        dc: (n_vertices,) 原始灵敏度
        phi_field: (n_vertices,) 相场
        filter_radius: 滤波半径

    Returns:
        dc_filtered: (n_vertices,) 滤波后灵敏度
    """
    if filter_radius <= 0:
        return dc.copy()

    n_vertices = mesh.n_vertices
    dc_f = np.zeros(n_vertices)

    for vi in range(n_vertices):
        pos_i = mesh.vertices[vi]
        total_weight = 0.0

        for vj in range(n_vertices):
            d = np.linalg.norm(mesh.vertices[vj] - pos_i)
            if d < filter_radius:
                w = max(0.0, filter_radius - d)
                dc_f[vi] += w * phi_field[vj] * dc[vj]
                total_weight += w

        if total_weight > 1e-15:
            dc_f[vi] /= max(phi_field[vi], 1e-15) * total_weight
        else:
            dc_f[vi] = dc[vi]

    return dc_f
