# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_fluid — 流体求解器
#
#   两种层次:
#     A. 薄膜流 (Reynolds 方程):  标量压力 p, 同热场模式
#        ∇·(h³∇p) = 6μU·∇h + 12μ·∂h/∂t
#
#     B. 体积 Stokes 流 (混合格式):  速度 u + 压力 p
#        -μ∇²v + ∇p = f,   ∇·v = 0
#
#   CC 曲面上的薄膜流利用 IGA 天然参数化,
#   体积流通过法向挤出薄层实现。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IrregularFaceCache, IGAError,
)
from forgecraft.analysis.iga_shape import cc_shape_functions
from forgecraft.analysis.iga_quadrature import cc_quadrature_points, gauss_legendre_2d
from forgecraft.analysis.iga_assembly import StreamingCSRAssembler
from forgecraft.analysis.iga_multiphysics_types import (
    ScalarDOFMap,
    FluidSettings, FluidMaterial,
    VelocityBC, PressureBC,
    FluidResult,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "assemble_reynolds_stiffness",
    "assemble_reynolds_load",
    "solve_thin_film",
    "assemble_stokes_viscous",
    "assemble_stokes_pressure_gradient",
    "assemble_stokes_divergence",
    "solve_stokes",
    "compute_fsi_force",
]


# ══════════════════════════════════════════════════════════
#  层次 A: 薄膜流 — Reynolds 方程
# ══════════════════════════════════════════════════════════

def assemble_reynolds_stiffness(
    mesh: PolyMesh,
    material: FluidMaterial,
    sdof: ScalarDOFMap,
    settings: FluidSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装 Reynolds 方程刚度矩阵

    Reynolds: ∇·(h³∇p) = 6μU·∇h + 12μ·∂h/∂t

    K_R[i,j] = ∫_Ω h³/(12μ) · ∇N_i · ∇N_j dΩ

    与热导率矩阵完全相同的模式，只是系数变为 h³/(12μ)。

    Args:
        mesh: CC 控制网格
        material: 流体材料 (μ, ρ)
        sdof: 标量 DOF 映射
        settings: 流体设置 (h, U)
        irregular_cache: 非常面缓存

    Returns:
        K_R: (n_dof, n_dof) CSR Reynolds 矩阵
    """
    n_dof = sdof.n_dof
    n_faces = mesh.n_faces

    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

    h = settings.film_thickness
    coeff = h ** 3 / (12.0 * material.viscosity)

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
            factor = coeff * qp.weight * shape.detJ

            Ke += factor * (dN @ dN.T)

        if Ke.shape[0] > 1:
            assembler.add_element(element_dofs.tolist(), Ke)

    return assembler.to_csr()


def assemble_reynolds_load(
    mesh: PolyMesh,
    material: FluidMaterial,
    sdof: ScalarDOFMap,
    settings: FluidSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """组装 Reynolds 方程载荷向量 (楔形项)

    f_R[i] = ∫ 6μU · ∂h/∂x · N_i dΩ

    简化: 假设沿 x 方向滑移，h 为常数。
    完整实现需要传入膜厚场 h(x)。

    Args:
        mesh: CC 控制网格
        material: 流体材料
        sdof: 标量 DOF 映射
        settings: 流体设置
        irregular_cache: 非常面缓存

    Returns:
        f_R: (n_dof,) 载荷向量
    """
    n_dof = sdof.n_dof
    f_R = np.zeros(n_dof)

    # 当 h 为常数且无挤压 (∂h/∂t=0) 时，f_R = 0
    # 仅当膜厚变化时非零
    return f_R


# ══════════════════════════════════════════════════════════
#  薄膜流求解
# ══════════════════════════════════════════════════════════

def solve_thin_film(
    mesh: PolyMesh,
    material: FluidMaterial,
    sdof: ScalarDOFMap,
    settings: FluidSettings,
    pressure_bcs: Optional[List[PressureBC]] = None,
    velocity_bcs: Optional[List[VelocityBC]] = None,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> FluidResult:
    """求解薄膜流 Reynolds 方程

    K_R · p = f_R

    Args:
        mesh: CC 控制网格
        material: 流体材料
        sdof: 标量 DOF 映射
        settings: 流体设置
        pressure_bcs: 压力边界列表
        velocity_bcs: 速度边界 (用于压力梯度 B.C.)
        irregular_cache: 非常面缓存

    Returns:
        FluidResult
    """
    t0 = perf_counter()

    # 组装
    K_R = assemble_reynolds_stiffness(
        mesh, material, sdof, settings, irregular_cache,
    ).toarray()

    f_R = assemble_reynolds_load(
        mesh, material, sdof, settings, irregular_cache,
    )

    # 施加压力 BC (Dirichlet)
    if pressure_bcs:
        for bc in pressure_bcs:
            for face_id in bc.face_ids:
                verts = list(mesh.face_vertices(face_id))
                if len(verts) != 4:
                    continue
                for vi in verts:
                    K_R[vi, vi] += settings.penalty
                    f_R[vi] += settings.penalty * bc.pressure

    # 求解
    try:
        p = np.linalg.solve(K_R, f_R)
        converged = True
    except np.linalg.LinAlgError:
        p = np.zeros(sdof.n_dof)
        converged = False

    wall_time = perf_counter() - t0

    return FluidResult(
        pressure=p,
        converged=converged,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  层次 B: Stokes 流 (混合格式)
# ══════════════════════════════════════════════════════════

def assemble_stokes_viscous(
    mesh: PolyMesh,
    material: FluidMaterial,
    dof_map: DOFMap,
    settings: FluidSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装 Stokes 粘性矩阵 A

    A[i,j] = μ ∫_Ω ∇N_i : ∇N_j dΩ

    每个速度分量独立, 与热导率模式相同, 但每个顶点 ×3 分量。

    拉普拉斯算子离散:
      A_ij = μ ∫ (dN_i/dx·dN_j/dx + dN_i/dy·dN_j/dy + dN_i/dz·dN_j/dz) dΩ

    对于 3 个速度分量, 组装 3 个独立的块。

    Args:
        mesh: CC 控制网格
        material: 流体材料
        dof_map: 结构 DOF 映射 (3 DOF/顶点, 用于速度)
        settings: 流体设置
        irregular_cache: 非常面缓存

    Returns:
        A: (n_dof, n_dof) CSR 粘性矩阵
    """
    n_dof = dof_map.n_dof
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

        qps = quad_list[face_id][1]

        # 单元 DOF: 16 顶点 × 3 = 48
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
            # 标量拉普拉斯: K_scalar[i,j] = dN[i]·dN[j]
            # 对 3 个速度分量, 每个分量独立
            K_scalar = dN @ dN.T  # (16, 16)

            factor = material.viscosity * qp.weight * shape.detJ

            # 将 16×16 标量块扩展为 48×48 速度块
            for ci in range(16):
                for cj in range(16):
                    ks = K_scalar[ci, cj] * factor
                    for comp in range(3):
                        i_row = ci * 3 + comp
                        i_col = cj * 3 + comp
                        Ae[i_row, i_col] += ks

        if Ae.shape[0] > 1:
            assembler.add_element(element_dofs.tolist(), Ae)

    return assembler.to_csr()


def assemble_stokes_pressure_gradient(
    mesh: PolyMesh,
    sdof_p: ScalarDOFMap,
    dof_map_v: DOFMap,
    settings: FluidSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装 Stokes 压力梯度矩阵 B

    B[i,j] = -∫_Ω N_p_j · (∇N_v_i) dΩ

    B 矩阵将压力耦合到动量方程: -∇p 项
    尺寸: (n_dof_v × n_dof_p)

    Args:
        mesh: CC 控制网格
        sdof_p: 压力标量 DOF 映射
        dof_map_v: 速度 DOF 映射
        settings: 流体设置
        irregular_cache: 非常面缓存

    Returns:
        B: (n_dof_v, n_dof_p) CSR 压力梯度矩阵
    """
    n_dof_v = dof_map_v.n_dof
    n_dof_p = sdof_p.n_dof
    n_faces = mesh.n_faces

    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

    # 使用 COO 直接构建
    rows = []
    cols = []
    vals = []

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

        qps = quad_list[face_id][1]

        for qp in qps:
            shape = cc_shape_functions(
                mesh, face_id, qp.u, qp.v, irregular_cache,
            )

            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            dN = shape.dN_dx  # (16, 3) 速度形函数梯度
            Np = shape.N[:16]  # (16,)  压力形函数值 (同族)

            factor = qp.weight * shape.detJ

            for ci in range(16):  # 速度分量 i
                vi_v = all_verts[ci]
                for comp in range(3):
                    i_row = dof_map_v.vertex_dofs(vi_v)[comp]

                    for cj in range(16):  # 压力分量 j
                        j_col = sdof_p.vertex_dof(all_verts[cj])

                        # B[3vi+comp, vj] = -∫ Np_j · dN_i/dx_comp dΩ
                        val = -Np[cj] * dN[ci, comp] * factor
                        rows.append(i_row)
                        cols.append(j_col)
                        vals.append(val)

    from scipy.sparse import coo_matrix
    B = coo_matrix(
        (vals, (rows, cols)),
        shape=(n_dof_v, n_dof_p),
    ).tocsr()

    return B


def assemble_stokes_divergence(
    mesh: PolyMesh,
    sdof_p: ScalarDOFMap,
    dof_map_v: DOFMap,
    settings: FluidSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装 Stokes 散度矩阵 B^T

    不可压缩约束: ∇·v = 0
    B^T[j,i] = -∫_Ω N_p_j · (∇N_v_i) dΩ

    实际上 B^T = (pressure_gradient)^T
    直接复用压力梯度的转置。

    Returns:
        B_T: (n_dof_p, n_dof_v) CSR 散度矩阵
    """
    B = assemble_stokes_pressure_gradient(
        mesh, sdof_p, dof_map_v, settings, irregular_cache,
    )
    return B.T.tocsr()


def solve_stokes(
    mesh: PolyMesh,
    material: FluidMaterial,
    dof_map_v: DOFMap,
    sdof_p: ScalarDOFMap,
    settings: FluidSettings,
    velocity_bcs: Optional[List[VelocityBC]] = None,
    pressure_bcs: Optional[List[PressureBC]] = None,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> FluidResult:
    """求解 Stokes 方程 (混合格式)

    [A    B] [v]   [f_v]
    [B^T  0] [p] = [0  ]

    使用罚函数稳定化: [A  B; B^T  -ε·M_p]

    Args:
        mesh: CC 控制网格
        material: 流体材料
        dof_map_v: 速度 DOF 映射
        sdof_p: 压力 DOF 映射
        settings: 流体设置
        velocity_bcs: 速度边界
        pressure_bcs: 压力边界
        irregular_cache: 非常面缓存

    Returns:
        FluidResult
    """
    t0 = perf_counter()

    # 组装子矩阵
    A = assemble_stokes_viscous(
        mesh, material, dof_map_v, settings, irregular_cache,
    ).toarray()

    B = assemble_stokes_pressure_gradient(
        mesh, sdof_p, dof_map_v, settings, irregular_cache,
    ).toarray()

    B_T = B.T

    n_v = dof_map_v.n_dof
    n_p = sdof_p.n_dof

    # 构造鞍点系统
    # | A         B  | | v |   | f_v |
    # | B^T  -ε·M_p | | p | = | 0   |
    # 小罚参数稳定化压力
    eps = 1e-10
    M_p = np.eye(n_p) * eps

    K_saddle = np.block([
        [A, B],
        [B_T, -M_p],
    ])

    f_saddle = np.zeros(n_v + n_p)

    # 施加速度 BC
    if velocity_bcs:
        penalty = settings.penalty
        for bc in velocity_bcs:
            for i, vi in enumerate(bc.vertex_ids):
                for comp in range(3):
                    dof = dof_map_v.vertex_dofs(vi)[comp]
                    K_saddle[dof, dof] += penalty
                    f_saddle[dof] += penalty * bc.values[i, comp]

    # 施加压力 BC
    if pressure_bcs:
        penalty = settings.penalty
        for bc in pressure_bcs:
            for face_id in bc.face_ids:
                verts = list(mesh.face_vertices(face_id))
                if len(verts) != 4:
                    continue
                for vi in verts:
                    dof_p = n_v + vi
                    K_saddle[dof_p, dof_p] += penalty
                    f_saddle[dof_p] += penalty * bc.pressure

    # 求解
    try:
        solution = np.linalg.solve(K_saddle, f_saddle)
        converged = True
    except np.linalg.LinAlgError:
        solution = np.zeros(n_v + n_p)
        converged = False

    v = solution[:n_v].reshape(-1, 3)
    p = solution[n_v:]

    wall_time = perf_counter() - t0

    return FluidResult(
        velocity=v,
        pressure=p,
        converged=converged,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  流固耦合 (FSI)
# ══════════════════════════════════════════════════════════

def compute_fsi_force(
    mesh: PolyMesh,
    pressure: np.ndarray,  # (n_vertices,) 流体压力
    dof_map: DOFMap,
    quadrature_order: int = 3,
) -> np.ndarray:
    """计算流体压力对结构的等效节点力

    f_fsi[i] = ∫_Γ p · n · N_i dΓ

    将标量压力场 p(x) 在每个面积分点上:
      f_i += p(x) · n(x) · N_i · w · detJ

    Args:
        mesh: CC 控制网格
        pressure: (n_vertices,) 流体压力场 [Pa]
        dof_map: 结构 DOF 映射
        quadrature_order: 积分阶数

    Returns:
        f_fsi: (n_dof,) 流固耦合力向量
    """
    from forgecraft.analysis.iga_shape import cc_shape_functions
    from forgecraft.analysis.iga_quadrature import cc_quadrature_points

    n_dof = dof_map.n_dof
    f_fsi = np.zeros(n_dof)

    quad_list = cc_quadrature_points(mesh, quadrature_order)

    for face_id in range(mesh.n_faces):
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

        qps = quad_list[face_id][1]
        element_dofs = dof_map.element_dofs(all_verts)

        for qp in qps:
            shape = cc_shape_functions(mesh, face_id, qp.u, qp.v)

            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            # 法向量: J = [dP/du, dP/dv], n = (dP/du × dP/dv) / |...|
            J = shape.J
            n = np.cross(J[:, 0], J[:, 1])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm

            # 积分点压力 (插值)
            p_qp = 0.0
            for i, vi in enumerate(all_verts):
                p_qp += shape.N[i] * pressure[vi]

            # 等效节点力
            factor = p_qp * qp.weight * shape.detJ
            for i, vi in enumerate(all_verts):
                f_vec = factor * shape.N[i] * n
                dx, dy, dz = dof_map.vertex_dofs(vi)
                f_fsi[dx] += f_vec[0]
                f_fsi[dy] += f_vec[1]
                f_fsi[dz] += f_vec[2]

    return f_fsi
