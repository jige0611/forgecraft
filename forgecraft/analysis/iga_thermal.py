# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_thermal — 热传导求解器
#
#   在 CC 细分曲面上求解热传导方程:
#     ρ·cp·∂T/∂t = ∇·(k∇T) + Q
#
#   稳态: K_T · T = f_T
#   瞬态: C_T · Ṫ + K_T · T = f_T  (隐式 Euler / Crank-Nicolson)
#
#   组装模式复用 Structural 的 StreamingCSRAssembler 和
#   cc_shape_functions。标量场 (T) 使用 ScalarDOFMap。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    IrregularFaceCache, ShapeResult, IGAError,
)
from forgecraft.analysis.iga_shape import cc_shape_functions
from forgecraft.analysis.iga_quadrature import cc_quadrature_points
from forgecraft.analysis.iga_assembly import StreamingCSRAssembler
from forgecraft.analysis.iga_multiphysics_types import (
    ScalarDOFMap,
    ThermalSettings, ThermalMaterial,
    TemperatureBC, HeatFluxBC, ConvectionBC, RadiationBC,
    ThermalResult,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "assemble_thermal_conductivity",
    "assemble_thermal_capacity",
    "assemble_thermal_load",
    "apply_temperature_bc_penalty",
    "solve_thermal_steady",
    "solve_thermal_transient",
    "compute_thermal_strain",
    "compute_thermal_expansion_force",
]


# ══════════════════════════════════════════════════════════
#  组装: 热导率矩阵 K_T
# ══════════════════════════════════════════════════════════

def assemble_thermal_conductivity(
    mesh: PolyMesh,
    material: ThermalMaterial,
    sdof: ScalarDOFMap,
    settings: ThermalSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装热导率矩阵 K_T

    K_T[i,j] = ∫_Ω k · ∇N_i · ∇N_j dΩ

    每个积分点: Ke = k · w · detJ · (dN_dx)^T · (dN_dx)
    其中 dN_dx 是 (16, 3) 的物理空间梯度

    Args:
        mesh: CC 控制网格
        material: 热材料属性
        sdof: 标量 DOF 映射
        settings: 热求解设置
        irregular_cache: 非常面缓存

    Returns:
        K_T: (n_dof, n_dof) CSR 热导率矩阵
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

        # 收集 16 控制点 (1 环邻域)
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

        # 标量 DOF 编号
        element_dofs = sdof.element_dofs(all_verts)

        qps = quad_list[face_id][1]
        Ke = np.zeros((16, 16))

        for qp in qps:
            u, v, w = qp.u, qp.v, qp.weight
            shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)

            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            # 热导率标量 dN_dx 的每个分量 (16, 3) → 只用对角贡献
            # 在曲面上: ∇N_i^T · ∇N_j = dN_dx[i,0]*dN_dx[j,0] +
            #                           dN_dx[i,1]*dN_dx[j,1] +
            #                           dN_dx[i,2]*dN_dx[j,2]
            dN = shape.dN_dx  # (16, 3)
            factor = material.conductivity * w * shape.detJ

            Ke += factor * (dN @ dN.T)

        if Ke.shape[0] > 1:
            assembler.add_element(element_dofs.tolist(), Ke)

    return assembler.to_csr()


# ══════════════════════════════════════════════════════════
#  组装: 热容矩阵 C_T
# ══════════════════════════════════════════════════════════

def assemble_thermal_capacity(
    mesh: PolyMesh,
    material: ThermalMaterial,
    sdof: ScalarDOFMap,
    settings: ThermalSettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> csr_matrix:
    """组装热容矩阵 C_T

    C_T[i,j] = ∫_Ω ρ·cp · N_i·N_j dΩ

    使用集总 (lumped) 质量: 对角集中, 易于显式时间积分。

    Args:
        mesh: CC 控制网格
        material: 热材料属性
        sdof: 标量 DOF 映射
        settings: 热求解设置
        irregular_cache: 非常面缓存

    Returns:
        C_T: (n_dof, n_dof) CSR 热容矩阵 (对角集总)
    """
    n_dof = sdof.n_dof
    n_faces = mesh.n_faces

    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

    # 对角集总
    C_lumped = np.zeros(n_dof)

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
            u, v, w = qp.u, qp.v, qp.weight
            shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)

            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            # 一致性质量: M_ij = ρ·cp · N_i·N_j · w · detJ
            # 集总: M_ii = sum_j M_ij = ρ·cp · N_i · w · detJ
            rho_cp = material.density * material.specific_heat
            if rho_cp <= 0:
                continue

            N = shape.N[:16]  # 基函数值
            factor = rho_cp * w * shape.detJ

            for i, vi in enumerate(all_verts):
                C_lumped[vi] += factor * N[i]

    # 转为稀疏对角矩阵
    from scipy.sparse import diags
    return diags(C_lumped)


# ══════════════════════════════════════════════════════════
#  组装: 热载荷向量 f_T
# ══════════════════════════════════════════════════════════

def assemble_thermal_load(
    mesh: PolyMesh,
    sdof: ScalarDOFMap,
    settings: ThermalSettings,
    heat_flux_bcs: Optional[List[HeatFluxBC]] = None,
    convection_bcs: Optional[List[ConvectionBC]] = None,
    radiation_bcs: Optional[List[RadiationBC]] = None,
    internal_heat_source: float = 0.0,  # 体热源 Q [W/m³]
    current_temperature: Optional[np.ndarray] = None,  # 用于对流/辐射线性化
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """组装热载荷向量 f_T

    包含:
      - 内部热源: Q ∫ N_i dΩ
      - 热流边界: q ∫_Γ N_i dΓ
      - 对流边界: h·T_inf ∫_Γ N_i dΓ (线性部分, 矩阵部分见 apply_convection)
      - 辐射边界: ε·σ·T_inf⁴ ∫_Γ N_i dΓ (线性化)

    Args:
        mesh: CC 控制网格
        sdof: 标量 DOF 映射
        settings: 热求解设置
        heat_flux_bcs: 热流边界条件列表
        convection_bcs: 对流边界条件列表
        radiation_bcs: 辐射边界条件列表
        internal_heat_source: 体热源密度 [W/m³]
        current_temperature: 当前温度场 (对流/辐射线性化用)
        irregular_cache: 非常面缓存

    Returns:
        f_T: (n_dof,) 热载荷向量
    """
    n_dof = sdof.n_dof
    f_T = np.zeros(n_dof)

    n_faces = mesh.n_faces
    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

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
            u, v, w = qp.u, qp.v, qp.weight
            shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)

            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            N = shape.N[:16]
            factor = w * shape.detJ

            # 内部热源
            if internal_heat_source != 0:
                for i, vi in enumerate(all_verts):
                    f_T[vi] += internal_heat_source * N[i] * factor

    # 面边界贡献
    # 热流 Neumann
    if heat_flux_bcs:
        _apply_heat_flux_bcs(mesh, sdof, heat_flux_bcs, f_T, settings.quadrature_order)

    # 对流 Robin (载荷部分)
    if convection_bcs:
        _apply_convection_load(
            mesh, sdof, convection_bcs, f_T,
            current_temperature, settings.quadrature_order,
        )

    # 辐射 Robin (载荷部分, 线性化)
    if radiation_bcs and current_temperature is not None:
        _apply_radiation_load(
            mesh, sdof, radiation_bcs, f_T,
            current_temperature, settings.quadrature_order,
        )

    return f_T


def _apply_heat_flux_bcs(
    mesh, sdof, bcs: List[HeatFluxBC], f_T: np.ndarray, quad_order: int,
):
    """施加热流边界 (Neumann)"""
    from forgecraft.analysis.iga_quadrature import gauss_legendre_2d

    pts_2d, wts = gauss_legendre_2d(quad_order)

    for bc in bcs:
        for face_id in bc.face_ids:
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

            for k in range(len(wts)):
                u, v = pts_2d[k, 0], pts_2d[k, 1]
                shape = cc_shape_functions(mesh, face_id, u, v)

                if shape.detJ < 1e-15:
                    continue

                factor = bc.flux * wts[k] * shape.detJ
                for i, vi in enumerate(all_verts[:min(16, len(shape.N))]):
                    f_T[vi] += factor * shape.N[i]


def _apply_convection_load(
    mesh, sdof, bcs: List[ConvectionBC], f_T: np.ndarray,
    current_T: Optional[np.ndarray], quad_order: int,
):
    """施加对流边界载荷部分: f_i += h·T_inf ∫ N_i dΓ"""
    from forgecraft.analysis.iga_quadrature import gauss_legendre_2d

    pts_2d, wts = gauss_legendre_2d(quad_order)

    for bc in bcs:
        for face_id in bc.face_ids:
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

            for k in range(len(wts)):
                u, v = pts_2d[k, 0], pts_2d[k, 1]
                shape = cc_shape_functions(mesh, face_id, u, v)

                if shape.detJ < 1e-15:
                    continue

                factor = bc.h * bc.T_inf * wts[k] * shape.detJ
                for i, vi in enumerate(all_verts[:min(16, len(shape.N))]):
                    f_T[vi] += factor * shape.N[i]


def _apply_radiation_load(
    mesh, sdof, bcs: List[RadiationBC], f_T: np.ndarray,
    current_T: np.ndarray, quad_order: int,
):
    """施加辐射边界载荷部分 (线性化)

    线性化: q_rad ≈ ε·σ·T_inf⁴ - ε·σ·4·T_inf³·(T - T_inf)
    载荷部分: +ε·σ·T_inf⁴ ∫ N_i dΓ
    """
    from forgecraft.analysis.iga_quadrature import gauss_legendre_2d

    pts_2d, wts = gauss_legendre_2d(quad_order)
    sigma = 5.67e-8

    for bc in bcs:
        for face_id in bc.face_ids:
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

            for k in range(len(wts)):
                u, v = pts_2d[k, 0], pts_2d[k, 1]
                shape = cc_shape_functions(mesh, face_id, u, v)

                if shape.detJ < 1e-15:
                    continue

                T_inf3 = bc.T_inf ** 3
                q_rad = bc.emissivity * sigma * T_inf3 * bc.T_inf
                factor = q_rad * wts[k] * shape.detJ

                for i, vi in enumerate(all_verts[:min(16, len(shape.N))]):
                    f_T[vi] += factor * shape.N[i]


def apply_convection_stiffness(
    mesh, sdof, bcs: List[ConvectionBC], K_T: np.ndarray, quad_order: int,
):
    """将对流贡献加到导热矩阵 (原地修改)

    K_T[i,j] += h ∫_Γ N_i·N_j dΓ (加到对角和相关项)
    """
    from forgecraft.analysis.iga_quadrature import gauss_legendre_2d

    pts_2d, wts = gauss_legendre_2d(quad_order)

    for bc in bcs:
        for face_id in bc.face_ids:
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

            for k in range(len(wts)):
                u, v = pts_2d[k, 0], pts_2d[k, 1]
                shape = cc_shape_functions(mesh, face_id, u, v)

                if shape.detJ < 1e-15:
                    continue

                factor = bc.h * wts[k] * shape.detJ
                N = shape.N[:min(16, len(shape.N))]

                for i, vi in enumerate(all_verts):
                    K_T[vi, vi] += factor * N[i] * N[i]


def apply_radiation_stiffness(
    mesh, sdof, bcs: List[RadiationBC], K_T: np.ndarray,
    current_T: np.ndarray, quad_order: int,
):
    """将辐射贡献加到导热矩阵 (原地修改)

    K_T[i,j] += ε·σ·4·T³_avg ∫ N_i·N_j dΓ
    """
    from forgecraft.analysis.iga_quadrature import gauss_legendre_2d

    pts_2d, wts = gauss_legendre_2d(quad_order)
    sigma = 5.67e-8

    for bc in bcs:
        for face_id in bc.face_ids:
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

            for k in range(len(wts)):
                u, v = pts_2d[k, 0], pts_2d[k, 1]
                shape = cc_shape_functions(mesh, face_id, u, v)

                if shape.detJ < 1e-15:
                    continue

                # 积分点平均温度
                T_qp = 0.0
                N = shape.N[:min(16, len(shape.N))]
                for i, vi in enumerate(all_verts):
                    T_qp += N[i] * current_T[vi]

                T_cube = T_qp ** 3
                h_rad = bc.emissivity * sigma * 4.0 * T_cube
                factor = h_rad * wts[k] * shape.detJ

                for i, vi in enumerate(all_verts):
                    K_T[vi, vi] += factor * N[i] * N[i]


# ══════════════════════════════════════════════════════════
#  Dirichlet BC 施加
# ══════════════════════════════════════════════════════════

def apply_temperature_bc_penalty(
    K_T: np.ndarray,
    f_T: np.ndarray,
    bc: TemperatureBC,
    penalty: float = 1e12,
):
    """罚函数法施加温度边界

    K_T[j,j] += penalty
    f_T[j]   += penalty * T_bar_j

    Args:
        K_T: 热导率矩阵 (原地修改)
        f_T: 热载荷向量 (原地修改)
        bc: 温度边界条件
        penalty: 罚因子
    """
    for i, vi in enumerate(bc.vertex_ids):
        K_T[vi, vi] += penalty
        f_T[vi] += penalty * bc.values[i]


# ══════════════════════════════════════════════════════════
#  稳态热求解
# ══════════════════════════════════════════════════════════

def solve_thermal_steady(
    mesh: PolyMesh,
    material: ThermalMaterial,
    sdof: ScalarDOFMap,
    settings: ThermalSettings,
    temperature_bcs: Optional[List[TemperatureBC]] = None,
    heat_flux_bcs: Optional[List[HeatFluxBC]] = None,
    convection_bcs: Optional[List[ConvectionBC]] = None,
    radiation_bcs: Optional[List[RadiationBC]] = None,
    internal_heat_source: float = 0.0,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> ThermalResult:
    """稳态热传导求解: K_T · T = f_T

    Args:
        mesh: CC 控制网格
        material: 热材料属性
        sdof: 标量 DOF 映射
        settings: 热求解设置
        temperature_bcs: 温度边界列表
        heat_flux_bcs: 热流边界列表
        convection_bcs: 对流边界列表
        radiation_bcs: 辐射边界列表
        internal_heat_source: 体热源密度 [W/m³]
        irregular_cache: 非常面缓存

    Returns:
        ThermalResult
    """
    t0 = perf_counter()
    n_dof = sdof.n_dof

    if settings.verbosity >= 1:
        _logger.info(f"Assembling thermal conductivity ({n_dof} DOF)...")

    # 1. 组装热导率矩阵
    K_T = assemble_thermal_conductivity(
        mesh, material, sdof, settings, irregular_cache,
    ).toarray()

    # 2. 初始温度猜测 (室温)
    T_current = np.full(n_dof, 300.0)

    # 非线性求解 (辐射 / 温度相关材料)
    max_iter = settings.max_iter if settings.nonlinear or settings.radiation else 1
    iterations = 0
    converged = True

    for iteration in range(max_iter):
        # 组装载荷向量
        f_T = assemble_thermal_load(
            mesh, sdof, settings,
            heat_flux_bcs=heat_flux_bcs,
            convection_bcs=convection_bcs,
            radiation_bcs=radiation_bcs,
            internal_heat_source=internal_heat_source,
            current_temperature=T_current,
            irregular_cache=irregular_cache,
        )

        # 施加对流贡献到 K_T
        K_iter = K_T.copy()
        if convection_bcs:
            apply_convection_stiffness(
                mesh, sdof, convection_bcs, K_iter, settings.quadrature_order,
            )

        # 施加辐射贡献到 K_T
        if radiation_bcs and settings.radiation:
            apply_radiation_stiffness(
                mesh, sdof, radiation_bcs, K_iter, T_current,
                settings.quadrature_order,
            )

        # 施加温度 BC
        if temperature_bcs:
            for bc in temperature_bcs:
                apply_temperature_bc_penalty(K_iter, f_T, bc, settings.penalty)

        # 求解
        try:
            if settings.solver == "cg":
                K_csr = csr_matrix(K_iter)
                from scipy.sparse.linalg import cg
                T_new, info = cg(
                    K_csr, f_T,
                    tol=settings.tolerance,
                    maxiter=settings.max_iter,
                )
                converged = info == 0
            else:
                T_new = np.linalg.solve(K_iter, f_T)
        except np.linalg.LinAlgError:
            T_new = T_current
            converged = False

        iterations = iteration + 1

        # 非线性收敛检查
        if max_iter > 1:
            delta = np.linalg.norm(T_new - T_current) / max(
                np.linalg.norm(T_new), 1e-10
            )
            T_current = T_new
            if delta < settings.tolerance:
                break
        else:
            T_current = T_new

    wall_time = perf_counter() - t0

    if settings.verbosity >= 1:
        _logger.info(
            f"Thermal solved in {wall_time:.2f}s, "
            f"T ∈ [{T_current.min():.2f}, {T_current.max():.2f}] K"
        )

    return ThermalResult(
        temperature=T_current,
        iterations=iterations,
        converged=converged,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  瞬态热求解
# ══════════════════════════════════════════════════════════

def solve_thermal_transient(
    mesh: PolyMesh,
    material: ThermalMaterial,
    sdof: ScalarDOFMap,
    settings: ThermalSettings,
    temperature_bcs: Optional[List[TemperatureBC]] = None,
    heat_flux_bcs: Optional[List[HeatFluxBC]] = None,
    convection_bcs: Optional[List[ConvectionBC]] = None,
    initial_temperature: float = 300.0,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> ThermalResult:
    """瞬态热传导: C_T·Ṫ + K_T·T = f_T

    使用隐式 Euler (θ=1) 或 Crank-Nicolson (θ=0.5) 时间积分。

    Args:
        mesh: CC 控制网格
        material: 热材料属性 (需 density + specific_heat)
        sdof: 标量 DOF 映射
        settings: 热求解设置
        temperature_bcs: 温度边界列表
        heat_flux_bcs: 热流边界列表
        convection_bcs: 对流边界列表
        initial_temperature: 初始温度 [K]
        irregular_cache: 非常面缓存

    Returns:
        ThermalResult (含最终温度场)
    """
    t0 = perf_counter()
    n_dof = sdof.n_dof

    # 组装矩阵 (稳态部分不变)
    K_T = assemble_thermal_conductivity(
        mesh, material, sdof, settings, irregular_cache,
    ).toarray()

    C_T = assemble_thermal_capacity(
        mesh, material, sdof, settings, irregular_cache,
    )

    # 初始温度
    T = np.full(n_dof, initial_temperature, dtype=np.float64)

    # 施加 Dirichlet BC 初始值
    if temperature_bcs:
        for bc in temperature_bcs:
            for i, vi in enumerate(bc.vertex_ids):
                T[vi] = bc.values[i]

    dt = settings.time_step
    theta = settings.theta

    for step in range(settings.n_steps):
        # 载荷向量
        f_T = assemble_thermal_load(
            mesh, sdof, settings,
            heat_flux_bcs=heat_flux_bcs,
            convection_bcs=convection_bcs,
            internal_heat_source=0.0,
            current_temperature=T,
            irregular_cache=irregular_cache,
        )

        # 对流贡献
        K_step = K_T.copy()
        if convection_bcs:
            apply_convection_stiffness(
                mesh, sdof, convection_bcs, K_step,
                settings.quadrature_order,
            )

        # θ 方法: [C/Δt + θ·K] T_new = [C/Δt - (1-θ)·K] T_old + f
        C_diag = C_T.diagonal()
        C_over_dt = C_diag / dt

        K_eff = K_step.copy()
        np.fill_diagonal(K_eff, K_eff.diagonal() + C_over_dt * theta)

        f_eff = f_T.copy()
        f_eff += (C_over_dt - (1.0 - theta) * K_step.diagonal()) * T

        # 施加温度 BC
        if temperature_bcs:
            K_bc = K_eff.copy()
            f_bc = f_eff.copy()
            for bc in temperature_bcs:
                apply_temperature_bc_penalty(K_bc, f_bc, bc, settings.penalty)
            K_eff, f_eff = K_bc, f_bc

        try:
            T_new = np.linalg.solve(K_eff, f_eff)
        except np.linalg.LinAlgError:
            T_new = T

        T = T_new

    wall_time = perf_counter() - t0

    return ThermalResult(
        temperature=T,
        converged=True,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  热-结构耦合
# ══════════════════════════════════════════════════════════

def compute_thermal_strain(
    temperature: np.ndarray,
    T_ref: float,
    alpha: float,
) -> np.ndarray:
    """计算热应变

    ε_th = α·(T - T_ref)·[1, 1, 1, 0, 0, 0]^T  (Voigt 记法)

    Args:
        temperature: (n_vertices,) 温度场 [K]
        T_ref: 参考温度 [K]
        alpha: 热膨胀系数 [1/K]

    Returns:
        epsilon_th: (n_vertices, 6) 热应变 Voigt 分量
    """
    n = len(temperature)
    eps_th = np.zeros((n, 6))
    delta_T = alpha * (temperature - T_ref)
    eps_th[:, 0] = delta_T  # ε_xx
    eps_th[:, 1] = delta_T  # ε_yy
    eps_th[:, 2] = delta_T  # ε_zz
    return eps_th


def compute_thermal_expansion_force(
    mesh: PolyMesh,
    dof_map,  # DOFMap (3 DOF/vertex)
    material,  # 结构材料 (LinearIsotropic etc.)
    temperature: np.ndarray,
    T_ref: float,
    alpha: float,
    settings,  # IGASettings
    irregular_cache=None,
) -> np.ndarray:
    """计算热膨胀等效节点力

    f_th = ∫_Ω B^T · D · ε_th dΩ

    对每个面积分点:
      f_th_e += B^T · D · [αΔT, αΔT, αΔT, 0, 0, 0] · w · detJ · thickness

    Args:
        mesh: CC 控制网格
        dof_map: 结构 DOF 映射 (3 DOF/顶点)
        material: 结构材料
        temperature: (n_vertices,) 温度场
        T_ref: 参考温度
        alpha: 热膨胀系数
        settings: IGA 设置
        irregular_cache: 非常面缓存

    Returns:
        f_th: (n_dof,) 热膨胀等效节点力向量
    """
    from forgecraft.analysis.iga_shape import cc_shape_functions
    from forgecraft.analysis.iga_quadrature import cc_quadrature_points

    n_dof = dof_map.n_dof
    f_th = np.zeros(n_dof)

    quad_list = cc_quadrature_points(
        mesh, settings.quadrature_order, irregular_cache,
    )

    # 获取本构矩阵 D (3x3 平面应力)
    from forgecraft.analysis.iga_element import _plane_stress_D
    D_ps = _plane_stress_D(material.E, material.nu)

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

        # 积分点平均温度
        for qp in qps:
            u, v, w = qp.u, qp.v, qp.weight
            shape = cc_shape_functions(mesh, face_id, u, v, irregular_cache)

            if shape.detJ < 1e-15 or shape.n_control < 16:
                continue

            # 积分点温度 (插值)
            T_qp = 0.0
            for i, vi in enumerate(all_verts):
                T_qp += shape.N[i] * temperature[vi]

            delta_T = alpha * (T_qp - T_ref)
            eps_th = np.array([delta_T, delta_T, 0.0])  # 平面应力: ε_zz 自由

            # 热应力: σ_th = D · ε_th
            sigma_th = D_ps @ eps_th  # (3,) [σ_xx, σ_yy, τ_xy]

            # B 矩阵需要在每个积分点构建
            # 热膨胀力: 对每个节点 i, f_i = ∫ B_i^T · σ_th dΩ
            B = _build_membrane_B(shape, all_verts, mesh)

            factor = w * shape.detJ
            element_dofs = dof_map.element_dofs(all_verts)

            # f_e = B^T · σ_th · factor
            # B: (3, 48), σ_th: (3,) → f_e: (48,)
            f_e = B.T @ sigma_th * factor

            for i, d in enumerate(element_dofs):
                f_th[d] += f_e[i]

    return f_th


def _build_membrane_B(shape: ShapeResult, all_verts: list, mesh) -> np.ndarray:
    """构建膜单元 B 矩阵 (3x48)

    在局部切线坐标系 (t1, t2):
      B = [dN/dx_t1  0         ]
          [0          dN/dx_t2  ]
          [dN/dx_t2  dN/dx_t1  ]

    Args:
        shape: ShapeResult (含 dN_dx)
        all_verts: 16 控制顶点
        mesh: CC 网格

    Returns:
        B_glob: (3, 48) 全局坐标系的 B 矩阵
    """
    # 构建局部切线坐标系
    J = shape.J  # (3, 2)
    t1 = J[:, 0]
    t2 = J[:, 1]
    t1 = t1 / (np.linalg.norm(t1) + 1e-15)
    n = np.cross(t1, t2)
    n = n / (np.linalg.norm(n) + 1e-15)
    t2 = np.cross(n, t1)

    # 旋转矩阵: 全局 → 局部
    R = np.vstack([t1, t2])  # (2, 3)

    # dN_dx 在全局坐标: (16, 3)
    # dN_dx_local = dN_dx @ R^T → (16, 2)
    dN_local = shape.dN_dx @ R.T  # (16, 2)

    n_ctrl = 16
    B = np.zeros((3, n_ctrl * 3))

    for i in range(n_ctrl):
        dNx = dN_local[i, 0]
        dNy = dN_local[i, 1]

        B[0, i * 3] = dNx
        B[0, i * 3 + 1] = 0.0
        B[0, i * 3 + 2] = 0.0

        B[1, i * 3] = 0.0
        B[1, i * 3 + 1] = dNy
        B[1, i * 3 + 2] = 0.0

        B[2, i * 3] = dNy
        B[2, i * 3 + 1] = dNx
        B[2, i * 3 + 2] = 0.0

    return B
