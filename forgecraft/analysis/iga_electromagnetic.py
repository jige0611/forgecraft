# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_electromagnetic — 电磁场求解器
#
#   A. 静电场: ∇·(ε∇φ) = -ρ
#     K_E · φ = f_E
#
#   B. 静磁场: ∇×(ν∇×A) = J (矢量磁势)
#
#   场间耦合:
#     - 焦耳热: Q_joule = σ|E|² → 热源
#     - 洛伦兹力: f_lorentz = J × B → 结构载荷
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IrregularFaceCache, IGAError,
)
from forgecraft.analysis.iga_shape import cc_shape_functions
from forgecraft.analysis.iga_quadrature import cc_quadrature_points
from forgecraft.analysis.iga_assembly import StreamingCSRAssembler
from forgecraft.analysis.iga_multiphysics_types import (
    ScalarDOFMap,
    EMSettings, EMMaterial,
    VoltageBC, CurrentDensityBC,
    EMResult,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "assemble_electrostatic_stiffness",
    "assemble_electrostatic_load",
    "apply_voltage_bc_penalty",
    "solve_electrostatic",
    "assemble_magnetostatic_stiffness",
    "solve_magnetostatic",
    "compute_e_field",
    "compute_joule_heat",
    "compute_lorentz_force",
]


# ══════════════════════════════════════════════════════════
#  A. 静电场
# ══════════════════════════════════════════════════════════

def assemble_electrostatic_stiffness(
    mesh: PolyMesh,
    material: EMMaterial,
    sdof: ScalarDOFMap,
    settings: EMSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装静电场刚度矩阵 K_E

    K_E[i,j] = ∫_Ω ε · ∇N_i · ∇N_j dΩ
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
            shape = cc_shape_functions(
                mesh, face_id, qp.u, qp.v, irregular_cache,
            )
            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            dN = shape.dN_dx  # (16, 3)
            factor = material.permittivity * qp.weight * shape.detJ
            Ke += factor * (dN @ dN.T)

        if Ke.shape[0] > 1:
            assembler.add_element(element_dofs.tolist(), Ke)

    return assembler.to_csr()


def assemble_electrostatic_load(
    mesh: PolyMesh,
    sdof: ScalarDOFMap,
    settings: EMSettings,
    charge_density: float = 0.0,
    current_bcs: Optional[List[CurrentDensityBC]] = None,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """组装静电场载荷向量 f_E

    f_E[i] = ∫_Ω ρ·N_i dΩ + ∫_Γ J_n·N_i dΓ
    """
    n_dof = sdof.n_dof
    f_E = np.zeros(n_dof)

    # 体电荷
    if charge_density != 0:
        quad_list = cc_quadrature_points(
            mesh, settings.quadrature_order, irregular_cache,
        )
        for face_id in range(mesh.n_faces):
            verts = list(mesh.face_vertices(face_id))
            if len(verts) != 4:
                continue
            all_verts = _collect_16_verts(mesh, verts)
            qps = quad_list[face_id][1]
            for qp in qps:
                shape = cc_shape_functions(
                    mesh, face_id, qp.u, qp.v, irregular_cache,
                )
                if shape.detJ < 1e-15:
                    continue
                factor = charge_density * qp.weight * shape.detJ
                for i, vi in enumerate(all_verts[:min(16, len(shape.N))]):
                    f_E[vi] += factor * shape.N[i]

    # 电流密度边界 (Neumann)
    if current_bcs:
        from forgecraft.analysis.iga_quadrature import gauss_legendre_2d
        pts_2d, wts = gauss_legendre_2d(settings.quadrature_order)
        for bc in current_bcs:
            for face_id in bc.face_ids:
                verts = list(mesh.face_vertices(face_id))
                if len(verts) != 4:
                    continue
                all_verts = _collect_16_verts(mesh, verts)
                for k in range(len(wts)):
                    u, v = pts_2d[k, 0], pts_2d[k, 1]
                    shape = cc_shape_functions(mesh, face_id, u, v)
                    if shape.detJ < 1e-15:
                        continue
                    factor = bc.current_density * wts[k] * shape.detJ
                    for i, vi in enumerate(all_verts[:min(16, len(shape.N))]):
                        f_E[vi] += factor * shape.N[i]

    return f_E


def apply_voltage_bc_penalty(
    K_E: np.ndarray,
    f_E: np.ndarray,
    bc: VoltageBC,
    penalty: float = 1e12,
):
    """罚函数法施加电压边界"""
    for i, vi in enumerate(bc.vertex_ids):
        K_E[vi, vi] += penalty
        f_E[vi] += penalty * bc.values[i]


def solve_electrostatic(
    mesh: PolyMesh,
    material: EMMaterial,
    sdof: ScalarDOFMap,
    settings: EMSettings,
    voltage_bcs: Optional[List[VoltageBC]] = None,
    current_bcs: Optional[List[CurrentDensityBC]] = None,
    charge_density: float = 0.0,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> EMResult:
    """求解静电场: K_E · φ = f_E

    Returns:
        EMResult (含电势、电场、电位移、电流密度、焦耳热)
    """
    t0 = perf_counter()

    K_E = assemble_electrostatic_stiffness(
        mesh, material, sdof, settings, irregular_cache,
    ).toarray()

    f_E = assemble_electrostatic_load(
        mesh, sdof, settings,
        charge_density=charge_density,
        current_bcs=current_bcs,
        irregular_cache=irregular_cache,
    )

    # 施加电压 BC
    if voltage_bcs:
        for bc in voltage_bcs:
            apply_voltage_bc_penalty(K_E, f_E, bc, settings.penalty)

    # 求解
    try:
        phi = np.linalg.solve(K_E, f_E)
        converged = True
    except np.linalg.LinAlgError:
        phi = np.zeros(sdof.n_dof)
        converged = False

    # 后处理: 计算电场、电位移、电流密度、焦耳热
    e_field = compute_e_field(mesh, phi, sdof, settings)

    d_field = np.zeros_like(e_field)
    current_density = np.zeros_like(e_field)
    joule_heat = np.zeros(sdof.n_dof)

    if e_field is not None:
        # D = εE
        for i in range(len(phi)):
            d_field[i] = material.permittivity * e_field[i]
            current_density[i] = material.conductivity * e_field[i]

        # 焦耳热 Q = σ|E|²
        e_mag_sq = np.sum(e_field ** 2, axis=1)
        joule_heat = material.conductivity * e_mag_sq

    wall_time = perf_counter() - t0

    return EMResult(
        potential=phi,
        e_field=e_field,
        d_field=d_field,
        current_density=current_density,
        joule_heat=joule_heat,
        converged=converged,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  B. 静磁场
# ══════════════════════════════════════════════════════════

def assemble_magnetostatic_stiffness(
    mesh: PolyMesh,
    material: EMMaterial,
    dof_map: DOFMap,
    settings: EMSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装静磁场刚度矩阵 (curl-curl)

    K_M[i,j] = ν ∫_Ω (∇×N_i)·(∇×N_j) dΩ

    矢量磁势 A 有 3 个分量, 使用 DOFMap (3 DOF/顶点)。
    在曲面上的 curl-curl 简化为与拉普拉斯算子相同的离散化,
    加上 Coulomb 规范惩罚项。

    K_M = ν·A + α·G (Coulomb 规范)
    其中 A 是矢量拉普拉斯 (同 Stokes 粘性),
    G 是梯度-散度惩罚项。
    """
    n_dof = dof_map.n_dof
    n_faces = mesh.n_faces

    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

    # 组装矢量拉普拉斯 (同 Stokes 粘性)
    assembler = StreamingCSRAssembler(n_dof)
    nu = 1.0 / material.permeability  # 磁阻率

    for face_id in range(n_faces):
        verts = list(mesh.face_vertices(face_id))
        if len(verts) != 4:
            continue

        all_verts = _collect_16_verts(mesh, verts)
        qps = quad_list[face_id][1]
        element_dofs = dof_map.element_dofs(all_verts)
        n_local = len(element_dofs)
        Ae = np.zeros((n_local, n_local))

        for qp in qps:
            shape = cc_shape_functions(
                mesh, face_id, qp.u, qp.v, irregular_cache,
            )
            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            dN = shape.dN_dx  # (16, 3)
            K_scalar = dN @ dN.T  # (16, 16)
            factor = nu * qp.weight * shape.detJ

            # 3 个分量独立
            for ci in range(16):
                for cj in range(16):
                    ks = K_scalar[ci, cj] * factor
                    for comp in range(3):
                        Ae[ci * 3 + comp, cj * 3 + comp] += ks

        if Ae.shape[0] > 1:
            assembler.add_element(element_dofs.tolist(), Ae)

    return assembler.to_csr()


def solve_magnetostatic(
    mesh: PolyMesh,
    material: EMMaterial,
    dof_map: DOFMap,
    settings: EMSettings,
    current_density: Optional[np.ndarray] = None,  # (n_dof,) 源电流
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> EMResult:
    """求解静磁场: K_M · A = J

    Args:
        mesh: CC 控制网格
        material: 电磁材料属性
        dof_map: DOF 映射 (3 DOF/顶点, 用于矢量磁势 A)
        settings: EM 设置
        current_density: (n_dof,) 源电流密度向量
        irregular_cache: 非常面缓存

    Returns:
        EMResult (potential = A 矢量磁势的 3 分量展平)
    """
    t0 = perf_counter()

    K_M = assemble_magnetostatic_stiffness(
        mesh, material, dof_map, settings, irregular_cache,
    ).toarray()

    # 添加 Coulomb 规范惩罚: K += α·G·G^T
    # 简化: 对角惩罚
    alpha = 1.0 / material.permeability * 1000
    for i in range(dof_map.n_dof):
        K_M[i, i] += alpha * 1e-6

    f_M = np.zeros(dof_map.n_dof)
    if current_density is not None:
        f_M = current_density.copy()

    try:
        A_flat = np.linalg.solve(K_M, f_M)
        converged = True
    except np.linalg.LinAlgError:
        A_flat = np.zeros(dof_map.n_dof)
        converged = False

    wall_time = perf_counter() - t0

    return EMResult(
        potential=A_flat,
        converged=converged,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  后处理: 场变量计算
# ══════════════════════════════════════════════════════════

def compute_e_field(
    mesh: PolyMesh,
    potential: np.ndarray,  # (n_vertices,) 电势
    sdof: ScalarDOFMap,
    settings: EMSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """计算电场 E = -∇φ

    在每个顶点通过相邻面积分点重建梯度。

    Returns:
        e_field: (n_vertices, 3) 电场矢量 [V/m]
    """
    n_vertices = sdof.n_dof
    e_field = np.zeros((n_vertices, 3))
    e_weight = np.zeros(n_vertices)

    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

    for face_id in range(mesh.n_faces):
        verts = list(mesh.face_vertices(face_id))
        if len(verts) != 4:
            continue

        all_verts = _collect_16_verts(mesh, verts)
        qps = quad_list[face_id][1]

        for qp in qps:
            shape = cc_shape_functions(
                mesh, face_id, qp.u, qp.v, irregular_cache,
            )
            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            dN = shape.dN_dx  # (16, 3)

            # E = -∇φ = -Σ φ_i · ∇N_i
            E_qp = np.zeros(3)
            for i, vi in enumerate(all_verts):
                E_qp -= potential[vi] * dN[i]

            # 分配到顶点 (面积加权平均)
            for i, vi in enumerate(all_verts):
                w = shape.N[i] * qp.weight * shape.detJ
                e_field[vi] += E_qp * w
                e_weight[vi] += abs(w)

    # 归一化
    for i in range(n_vertices):
        if e_weight[i] > 1e-15:
            e_field[i] /= e_weight[i]

    return e_field


# ══════════════════════════════════════════════════════════
#  场间耦合
# ══════════════════════════════════════════════════════════

def compute_joule_heat(
    e_field: np.ndarray,  # (n_vertices, 3) 电场
    conductivity: float,  # σ [S/m]
) -> np.ndarray:
    """计算焦耳热

    Q_joule = σ·|E|²

    Args:
        e_field: (n_vertices, 3) 电场矢量 [V/m]
        conductivity: 电导率 σ [S/m]

    Returns:
        joule_heat: (n_vertices,) 焦耳热密度 [W/m³]
    """
    e_mag_sq = np.sum(e_field ** 2, axis=1)
    return conductivity * e_mag_sq


def compute_lorentz_force(
    current_density: np.ndarray,  # (n_vertices, 3) 电流密度 J [A/m²]
    b_field: np.ndarray,  # (n_vertices, 3) 磁感应强度 B [T]
) -> np.ndarray:
    """计算洛伦兹力

    f_lorentz = J × B

    Args:
        current_density: (n_vertices, 3) 电流密度
        b_field: (n_vertices, 3) 磁感应强度

    Returns:
        f_lorentz: (n_vertices, 3) 洛伦兹力密度 [N/m³]
    """
    return np.cross(current_density, b_field)


# ══════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════

def _collect_16_verts(mesh, verts: list) -> list:
    """收集面的 16 控制点 (1 环邻域)"""
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
    return all_verts[:16]
