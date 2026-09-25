# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_topology_phasefield — 相位场 PDE 组装与求解
#
#   在 CC 细分曲面上求解 Allen-Cahn 型相场方程:
#     [κε²·K_laplace + κ·diag(w''(φ)·area)] · φ = g'(φ)·ψe
#
#   同时负责:
#     - 用 g(φ) 退化结构刚度矩阵
#     - 计算应变能密度 ψe 场
#     - 相场箱约束和连续方案
#
#   组装模式复用: K_laplace 和热传导的 assemble_thermal_conductivity
#   相同结构 (标量 Laplace 算子, 每顶点 1 DOF)。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IrregularFaceCache, IGASettings,
)
from forgecraft.analysis.iga_shape import cc_shape_functions
from forgecraft.analysis.iga_quadrature import cc_quadrature_points
from forgecraft.analysis.iga_assembly import StreamingCSRAssembler
from forgecraft.analysis.iga_multiphysics_types import ScalarDOFMap
from forgecraft.analysis.iga_topology_types import (
    TopologySettings,
    g_simp, dg_simp, g_ramp, dg_ramp, g_polynomial, dg_polynomial,
    w_double_well, dw_double_well, d2w_double_well,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "assemble_phasefield_laplacian",
    "assemble_phasefield_stiffness",
    "assemble_phasefield_driving_force",
    "assemble_degraded_stiffness_topology",
    "compute_strain_energy_density",
    "solve_phasefield_field",
    "compute_volume_fraction",
    "apply_density_filter",
]


# ══════════════════════════════════════════════════════════
#  标量 Laplace 算子组装 (相场梯度项)
# ══════════════════════════════════════════════════════════

def assemble_phasefield_laplacian(
    mesh: PolyMesh,
    sdof: ScalarDOFMap,
    settings: TopologySettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装标量 Laplace 刚度矩阵 K_laplace

    K_laplace[i,j] = ∫_Ω ∇N_i · ∇N_j dΩ

    与热传导完全相同的结构, 复用 StreamingCSRAssembler + cc_shape_functions。

    Args:
        mesh: CC 控制网格
        sdof: 标量 DOF 映射
        settings: 拓扑优化设置
        irregular_cache: 非常面缓存

    Returns:
        K_laplace: (n_vertices, n_vertices) CSR 标量 Laplace 矩阵
    """
    n_dof = sdof.n_dof
    n_faces = mesh.n_faces

    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

    assembler = StreamingCSRAssembler(n_dof)

    for face_id in range(n_faces):
        verts = list(mesh.face_vertices(face_id))
        if len(verts) != 4:
            continue

        # 16 控制点 (1 环邻域)
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

        element_dofs = sdof.element_dofs(all_verts)

        qps = quad_list[face_id][1]
        Ke = np.zeros((16, 16))

        for qp in qps:
            u, v, w = qp.u, qp.v, qp.weight
            shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)

            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            dN = shape.dN_dx  # (16, 3)
            factor = w * shape.detJ
            Ke += factor * (dN @ dN.T)

        if Ke.shape[0] > 1:
            assembler.add_element(element_dofs.tolist(), Ke)

    return assembler.to_csr()


# ══════════════════════════════════════════════════════════
#  相场刚度组装
# ══════════════════════════════════════════════════════════

def assemble_phasefield_stiffness(
    mesh: PolyMesh,
    phi_field: np.ndarray,
    sdof: ScalarDOFMap,
    settings: TopologySettings,
    K_laplace: Optional[csr_matrix] = None,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装相场刚度矩阵 (凸分裂)

    使用凸分裂 (Convex Splitting) 保证 K_φ 正定:
      w(φ) = w_c(φ) + w_e(φ)
      w_c(φ) = φ²        →  w_c''(φ) = 2 (恒正, 隐式)
      w_e(φ) = φ⁴ - 2φ³  →  显式处理 (移到 RHS)

    K_φ = κε² · K_laplace + 2κ · diag(area_i)

    双阱势 w(φ) = φ²(1-φ)² 在 φ∈(0.21, 0.79) 有负曲率,
    直接线性化导致 K_φ 非正定 → 凸分裂保证稳定性。

    Args:
        mesh: CC 控制网格
        phi_field: (n_vertices,) 当前相场
        sdof: 标量 DOF 映射
        settings: 拓扑优化设置
        K_laplace: 预组装的 Laplace 矩阵 (避免重复计算)
        irregular_cache: 非常面缓存

    Returns:
        K_phi: (n_vertices, n_vertices) CSR 相场刚度矩阵 (正定)
    """
    n_vertices = mesh.n_vertices

    # 1. 梯度项: κε² · K_laplace
    if K_laplace is None:
        K_laplace = assemble_phasefield_laplacian(
            mesh, sdof, settings, irregular_cache,
        )

    kappa = settings.kappa
    eps = settings.epsilon
    K_phi = (kappa * eps * eps) * K_laplace

    # 2. 凸分裂对角: 2κ · area_i (w_c'' = 2, 恒正)
    K_dense = K_phi.toarray() if hasattr(K_phi, 'toarray') else K_phi.copy()

    vertex_areas = _compute_vertex_areas(mesh)

    for vi in range(n_vertices):
        K_dense[vi, vi] += 2.0 * kappa * vertex_areas[vi]

    return csr_matrix(K_dense)


def _compute_vertex_areas(mesh: PolyMesh) -> np.ndarray:
    """计算每个控制顶点的归属面积 (Voronoi-like)

    通过累加相邻面的面积并均分给面顶点。
    """
    n_vertices = mesh.n_vertices
    areas = np.zeros(n_vertices)

    for f in range(mesh.n_faces):
        verts = mesh.face_vertices(f)
        if len(verts) < 3:
            continue

        # 多边形面积 (分成两个三角形近似)
        p0 = mesh.vertices[verts[0]]
        p1 = mesh.vertices[verts[1]]
        p2 = mesh.vertices[verts[2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

        if len(verts) == 4:
            p3 = mesh.vertices[verts[3]]
            area += 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0))

        # 均分给面顶点
        area_per = area / len(verts)
        for v in verts:
            areas[v] += area_per

    return areas


# ══════════════════════════════════════════════════════════
#  相场驱动力组装
# ══════════════════════════════════════════════════════════

def assemble_phasefield_driving_force(
    mesh: PolyMesh,
    phi_field: np.ndarray,
    psi_e_field: np.ndarray,
    settings: TopologySettings,
) -> np.ndarray:
    """组装相场驱动力向量 (含凸分裂显式项)

    完整 RHS:
      f_i = g'(φ_i) · ψe_i · area_i + κ · w_e'(φ_i) · area_i

    其中:
      - g'(φ)·ψe·area: 应变能驱动力 → 推向高能区 (固体)
      - w_e'(φ)·area:  凸分裂显式双阱势 → 驱动 φ→0 或 φ→1
        w_e(φ) = φ⁴ - 2φ³, w_e'(φ) = 4φ³ - 6φ²

    Args:
        mesh: CC 控制网格
        phi_field: (n_vertices,) 当前相场
        psi_e_field: (n_vertices,) 应变能密度
        settings: 拓扑优化设置

    Returns:
        f_drive: (n_vertices,) 驱动力向量
    """
    n_vertices = mesh.n_vertices
    f_drive = np.zeros(n_vertices)

    vertex_areas = _compute_vertex_areas(mesh)
    kappa = settings.kappa

    for vi in range(n_vertices):
        phi = phi_field[vi]
        psi = psi_e_field[vi]
        area = vertex_areas[vi]

        if area < 1e-15:
            continue

        # 1. 应变能驱动: g'(φ) · ψe · area
        if settings.interpolation == "SIMP":
            dg = dg_simp(phi, settings.p_current, settings.phi_min)
        elif settings.interpolation == "RAMP":
            dg = dg_ramp(phi, q=3.0, phi_min=settings.phi_min)
        else:
            dg = dg_polynomial(phi)

        # 2. 凸分裂显式双阱势: w_e'(φ) = 4φ³ - 6φ²
        w_e_prime = 4.0 * phi**3 - 6.0 * phi**2

        f_drive[vi] = (dg * psi + kappa * w_e_prime) * area

    return f_drive


# ══════════════════════════════════════════════════════════
#  退化结构刚度组装
# ══════════════════════════════════════════════════════════

def assemble_degraded_stiffness_topology(
    K0: np.ndarray,
    phi_field: np.ndarray,
    mesh: PolyMesh,
    dof_map: DOFMap,
    settings: TopologySettings,
) -> np.ndarray:
    """根据相场 φ 退化结构刚度矩阵

    K_degraded[i,j] = sqrt(g(φ_v_i)) * sqrt(g(φ_v_j)) * K0[i,j]

    退化因子基于控制顶点的 g(φ):
      - φ → 1 (实体):    g(φ) → 1, 刚度不变
      - φ → φ_min (空):  g(φ) → φ_min, 刚度接近零

    与断裂相场不同: 这里是材料插值 (g→1 是实体),
    而非退化函数 (g→0 是裂纹)。

    Args:
        K0: (n_dof, n_dof) 完整 (未退化) 刚度矩阵
        phi_field: (n_vertices,) 相场
        mesh: CC 控制网格
        dof_map: 结构 DOF 映射
        settings: 拓扑优化设置

    Returns:
        K_deg: (n_dof, n_dof) 退化后的刚度矩阵
    """
    n_vertices = mesh.n_vertices
    K_deg = K0.copy()

    # 计算退化因子
    g_arr = np.ones(n_vertices)
    for vi in range(n_vertices):
        if settings.interpolation == "SIMP":
            g_arr[vi] = g_simp(phi_field[vi], settings.p_current, settings.phi_min)
        elif settings.interpolation == "RAMP":
            g_arr[vi] = g_ramp(phi_field[vi], q=3.0, phi_min=settings.phi_min)
        else:
            g_arr[vi] = g_polynomial(phi_field[vi])

    # 对称退化 (退化整个 3x3 块, 含耦合项 dx-dy, dx-dz 等)
    for vi in range(n_vertices):
        sqrt_gi = np.sqrt(max(g_arr[vi], 1e-15))

        for vj in range(n_vertices):
            sqrt_gj = np.sqrt(max(g_arr[vj], 1e-15))
            factor = sqrt_gi * sqrt_gj

            for local_i in range(3):
                i_glob = dof_map.vertex_dofs(vi)[local_i]
                for local_j in range(3):
                    j_glob = dof_map.vertex_dofs(vj)[local_j]
                    K_deg[i_glob, j_glob] *= factor

    return K_deg


# ══════════════════════════════════════════════════════════
#  应变能密度计算
# ══════════════════════════════════════════════════════════

def compute_strain_energy_density(
    mesh: PolyMesh,
    integrator,
    material,
    u: np.ndarray,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """计算每个控制顶点的弹性应变能密度 ψe

    用于相场拓扑优化的驱动力: 高 ψe 区域驱动 φ → 1 (实体)。
    与断裂相场不同: 使用完整 ψe (非仅正应变能)。

    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator
        material: 材料模型 (LinearIsotropic)
        u: (n_dof,) 位移解
        dof_map: 结构 DOF 映射
        iga_settings: IGA 设置
        irregular_cache: 非常面缓存

    Returns:
        psi_e: (n_vertices,) 顶点应变能密度
    """
    from forgecraft.analysis.iga_quadrature import cc_quadrature

    n_vertices = mesh.n_vertices

    psi_accum = np.zeros(n_vertices)
    weight_accum = np.zeros(n_vertices)

    # 平面应力本构
    from forgecraft.analysis.iga_element import _plane_stress_D
    D_membrane = _plane_stress_D(material.E, material.nu)

    for face_id in range(mesh.n_faces):
        verts = mesh.face_vertices(face_id)
        if len(verts) != 4:
            continue

        # 16 控制点
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

        # 单元位移
        u_e = np.zeros(48)
        for i, vi in enumerate(all_verts):
            dx, dy, dz = dof_map.vertex_dofs(vi)
            u_e[3*i + 0] = u[dx]
            u_e[3*i + 1] = u[dy]
            u_e[3*i + 2] = u[dz]

        # 积分点
        qps = cc_quadrature(mesh, face_id, iga_settings.quadrature_order, irregular_cache)

        for qp in qps:
            shape = cc_shape_functions(mesh, face_id, qp.u, qp.v, irregular_cache)

            if shape.detJ < 1e-15:
                continue

            # 局部坐标系
            J = shape.J
            t1 = J[:, 0].copy()
            tn1 = np.linalg.norm(t1)
            if tn1 < 1e-15:
                continue
            t1 /= tn1
            t2 = J[:, 1].copy()
            t2 -= np.dot(t2, t1) * t1
            tn2 = np.linalg.norm(t2)
            if tn2 < 1e-15:
                continue
            t2 /= tn2

            # B 矩阵
            B_glob = _build_membrane_B_local(shape, t1, t2)

            # 应变和应力
            eps = B_glob @ u_e
            sigma = D_membrane @ eps

            # 完整应变能密度
            psi_q = 0.5 * np.dot(sigma, eps)
            psi_q = max(psi_q, 0.0)

            # 面积加权分配
            area = qp.weight * shape.detJ
            for node_i in range(16):
                vi = all_verts[node_i]
                psi_accum[vi] += psi_q * area
                weight_accum[vi] += area

    # 归一化
    psi_e = np.zeros(n_vertices)
    for vi in range(n_vertices):
        if weight_accum[vi] > 1e-15:
            psi_e[vi] = psi_accum[vi] / weight_accum[vi]

    return psi_e


def _build_membrane_B_local(shape, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """构建局部切线坐标系 B 矩阵 (3x48)

    参考断裂相场 compute_elastic_energy_density 中的实现。
    """
    B_glob = np.zeros((3, 48))
    for node_i in range(16):
        dNdx = shape.dN_dx[node_i]
        dN_dt1 = np.dot(dNdx, t1)
        dN_dt2 = np.dot(dNdx, t2)

        i3 = node_i * 3
        B_glob[0, i3 + 0] = dN_dt1 * t1[0]
        B_glob[0, i3 + 1] = dN_dt1 * t1[1]
        B_glob[0, i3 + 2] = dN_dt1 * t1[2]

        B_glob[1, i3 + 0] = dN_dt2 * t2[0]
        B_glob[1, i3 + 1] = dN_dt2 * t2[1]
        B_glob[1, i3 + 2] = dN_dt2 * t2[2]

        B_glob[2, i3 + 0] = dN_dt1 * t2[0] + dN_dt2 * t1[0]
        B_glob[2, i3 + 1] = dN_dt1 * t2[1] + dN_dt2 * t1[1]
        B_glob[2, i3 + 2] = dN_dt1 * t2[2] + dN_dt2 * t1[2]

    return B_glob


# ══════════════════════════════════════════════════════════
#  相场求解
# ══════════════════════════════════════════════════════════

def solve_phasefield_field(
    mesh: PolyMesh,
    phi_old: np.ndarray,
    psi_e_field: np.ndarray,
    sdof: ScalarDOFMap,
    settings: TopologySettings,
    K_phi: Optional[csr_matrix] = None,
    lagrange_multiplier: float = 0.0,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """求解相场更新

    求解线性化方程:
      K_φ · φ_new = f_drive - λ · area

    然后应用箱约束: φ_min ≤ φ ≤ 1.0

    Args:
        mesh: CC 控制网格
        phi_old: (n_vertices,) 当前相场
        psi_e_field: (n_vertices,) 应变能密度
        sdof: 标量 DOF 映射
        settings: 拓扑优化设置
        K_phi: 预组装的相场刚度 (可选, 避免重复)
        lagrange_multiplier: 体积约束拉格朗日乘子 λ
        irregular_cache: 非常面缓存

    Returns:
        phi_new: (n_vertices,) 新相场
    """
    n_vertices = mesh.n_vertices

    # 1. 组装刚度
    if K_phi is None:
        K_phi = assemble_phasefield_stiffness(
            mesh, phi_old, sdof, settings, None, irregular_cache,
        )

    # 2. 组装驱动力
    f_drive = assemble_phasefield_driving_force(
        mesh, phi_old, psi_e_field, settings,
    )

    # 3. 体积约束: f_phi = f_drive - λ · area
    vertex_areas = _compute_vertex_areas(mesh)
    f_phi = f_drive - lagrange_multiplier * vertex_areas

    # 4. 求解 (直接法)
    try:
        K_dense = K_phi.toarray() if hasattr(K_phi, 'toarray') else np.asarray(K_phi.todense())
        phi_new = np.linalg.solve(K_dense, f_phi)
    except np.linalg.LinAlgError:
        _logger.warning("Phase-field solve singular, using CG fallback")
        from scipy.sparse.linalg import cg
        phi_new, info = cg(K_phi, f_phi, maxiter=settings.max_iter_inner)
        if info != 0:
            _logger.warning(f"CG did not converge (info={info}), using phi_old")
            phi_new = phi_old.copy()

    # 5. 箱约束
    phi_new = np.clip(phi_new, settings.phi_min, 1.0)

    return phi_new


# ══════════════════════════════════════════════════════════
#  体积分数计算
# ══════════════════════════════════════════════════════════

def compute_volume_fraction(
    mesh: PolyMesh,
    phi_field: np.ndarray,
) -> float:
    """计算当前体积分数 V/V0

    V = Σ φ_i · area_i / Σ area_i

    Args:
        mesh: CC 控制网格
        phi_field: (n_vertices,) 相场

    Returns:
        V_fract: 体积分数 ∈ [φ_min, 1]
    """
    areas = _compute_vertex_areas(mesh)
    total_area = areas.sum()

    if total_area < 1e-15:
        return 1.0

    weighted = np.dot(phi_field, areas)
    return weighted / total_area


# ══════════════════════════════════════════════════════════
#  密度滤波 (可选增强)
# ══════════════════════════════════════════════════════════

def apply_density_filter(
    mesh: PolyMesh,
    phi_field: np.ndarray,
    filter_radius: float,
) -> np.ndarray:
    """可选的 Helmholtz 型密度滤波

    用于超小网格或需要额外平滑时。正常情况下 ε 正则化已足够。

    Args:
        mesh: CC 控制网格
        phi_field: (n_vertices,) 原始相场
        filter_radius: 滤波半径 (>0 启用)

    Returns:
        phi_filtered: (n_vertices,) 滤波后相场
    """
    if filter_radius <= 0:
        return phi_field.copy()

    n_vertices = mesh.n_vertices
    phi_f = phi_field.copy()

    # 简单距离加权平均
    r2 = filter_radius * filter_radius

    for vi in range(n_vertices):
        pos_i = mesh.vertices[vi]
        total_weight = 1.0
        total_value = phi_field[vi]

        # 查找滤波半径内的邻域
        for vj in range(n_vertices):
            if vj == vi:
                continue
            dist2 = np.sum((mesh.vertices[vj] - pos_i) ** 2)
            if dist2 < r2:
                w = max(0.0, 1.0 - np.sqrt(dist2) / filter_radius)
                total_weight += w
                total_value += w * phi_field[vj]

        phi_f[vi] = total_value / total_weight

    return phi_f
