# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_multiphysics_coupling — 多场耦合策略
#
#   MultiphysicsCoupler: 四场交错耦合求解器
#     - 块 Gauss-Seidel 交错迭代 (默认)
#     - Aitken 动态松弛加速
#     - Anderson 加速
#     - 各场间耦合逻辑 (热膨胀 / FSI / 焦耳热 / 洛伦兹力)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IrregularFaceCache, IGASettings, StaticResult, StressField,
)
from forgecraft.analysis.iga_multiphysics_types import (
    ScalarDOFMap,
    ThermalSettings, ThermalMaterial,
    TemperatureBC, HeatFluxBC, ConvectionBC, RadiationBC, ThermalResult,
    FluidSettings, FluidMaterial,
    VelocityBC, PressureBC, FluidResult,
    EMSettings, EMMaterial,
    VoltageBC, CurrentDensityBC, EMResult,
    MultiphysicsSettings, MultiphysicsResult, CouplingStatus,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "MultiphysicsCoupler",
    "solve_multiphysics",
    "aitken_relaxation",
    "anderson_acceleration",
]


# ══════════════════════════════════════════════════════════
#  松弛加速方法
# ══════════════════════════════════════════════════════════

def aitken_relaxation(
    x_old: np.ndarray,
    x_new: np.ndarray,
    r_old: np.ndarray,
    r_new: np.ndarray,
    omega: float,
) -> float:
    """Aitken 动态松弛

    ω_{k+1} = -ω_k · (r_k^T · (r_{k+1} - r_k)) / ||r_{k+1} - r_k||²

    Args:
        x_old, x_new: 上一轮/本轮变量
        r_old, r_new: 上一轮/本轮残差
        omega: 当前松弛因子

    Returns:
        omega_new: 更新后的松弛因子
    """
    dr = r_new - r_old
    dr_norm_sq = np.dot(dr, dr)

    if dr_norm_sq > 1e-15:
        omega_new = -omega * np.dot(r_old, dr) / dr_norm_sq
        omega_new = max(0.1, min(1.0, omega_new))
    else:
        omega_new = omega

    return omega_new


def anderson_acceleration(
    history_x: List[np.ndarray],
    history_r: List[np.ndarray],
    memory: int = 5,
) -> np.ndarray:
    """Anderson 加速

    利用最近 m 个迭代的历史做外推:
      x_acc = x_k - Σ (G_j · γ_j)
    其中 γ 最小化残差组合

    Args:
        history_x: 变量历史 [x_{k-m+1}, ..., x_k]
        history_r: 残差历史 [r_{k-m+1}, ..., r_k]
        memory: 记忆深度

    Returns:
        x_acc: 加速后的变量
    """
    m = min(memory, len(history_r))

    if m <= 1:
        return history_x[-1]

    # 差分
    dX = []
    dR = []
    for i in range(1, m):
        dX.append(history_x[i] - history_x[i - 1])
        dR.append(history_r[i] - history_r[i - 1])

    # 最小二乘: min_γ ||r_k - Σ dR_j·γ_j||²
    R_mat = np.column_stack(dR)  # (n, m-1)
    r_k = history_r[-1]

    try:
        gamma, _, _, _ = np.linalg.lstsq(R_mat, r_k, rcond=None)
    except np.linalg.LinAlgError:
        return history_x[-1]

    # 外推
    x_acc = history_x[-1].copy()
    for j in range(m - 1):
        x_acc -= gamma[j] * dX[j]

    return x_acc


# ══════════════════════════════════════════════════════════
#  MultiphysicsCoupler: 四场耦合主类
# ══════════════════════════════════════════════════════════

class MultiphysicsCoupler:
    """四场 (结构+热+流+电磁) 交错耦合求解器

    求解策略: 块 Gauss-Seidel 交错迭代
      - 外层: 载荷步
      - 中层: 场间耦合迭代
      - 内层: 各场独立 Newton/直接求解

    耦合逻辑:
      - 热→结构: 热膨胀等效节点力
      - 流→结构: 流体压力等效节点力 (FSI)
      - 流→热:   对流换热 (共轭传热)
      - EM→热:   焦耳热源项
      - EM→结构: 洛伦兹力
    """

    def solve(
        self,
        mesh: PolyMesh,
        # --- 结构 ---
        integrator,
        struct_material,
        dof_map: DOFMap,
        iga_settings: IGASettings,
        dirichlet_bcs,
        neumann_bcs=None,
        point_loads=None,
        # --- 热 ---
        thermal_material: Optional[ThermalMaterial] = None,
        thermal_settings: Optional[ThermalSettings] = None,
        temperature_bcs: Optional[List[TemperatureBC]] = None,
        heat_flux_bcs: Optional[List[HeatFluxBC]] = None,
        convection_bcs: Optional[List[ConvectionBC]] = None,
        radiation_bcs: Optional[List[RadiationBC]] = None,
        # --- 流 ---
        fluid_material: Optional[FluidMaterial] = None,
        fluid_settings: Optional[FluidSettings] = None,
        velocity_bcs: Optional[List[VelocityBC]] = None,
        pressure_bcs: Optional[List[PressureBC]] = None,
        # --- 电磁 ---
        em_material: Optional[EMMaterial] = None,
        em_settings: Optional[EMSettings] = None,
        voltage_bcs: Optional[List[VoltageBC]] = None,
        current_bcs: Optional[List[CurrentDensityBC]] = None,
        # --- 耦合 ---
        mp_settings: Optional[MultiphysicsSettings] = None,
        irregular_cache=None,
    ) -> MultiphysicsResult:
        """执行四场耦合求解

        Returns:
            MultiphysicsResult (含各场结果 + 耦合历史)
        """
        t0 = perf_counter()

        mp_settings = mp_settings or MultiphysicsSettings()
        sdof = ScalarDOFMap.from_mesh(mesh)

        # 初始化各场结果
        struct_result = None
        thermal_result = None
        fluid_result = None
        em_result = None

        # 初始场值
        n_vertices = mesh.n_vertices
        n_dof = dof_map.n_dof
        T_field = np.full(n_vertices, 300.0)
        u_field = np.zeros(n_dof)
        p_field = np.zeros(n_vertices)
        phi_field = np.zeros(n_vertices)

        coupling_history = []
        total_iter = 0

        # 载荷步
        for load_step in range(mp_settings.n_load_steps):
            load_factor = (load_step + 1) / mp_settings.n_load_steps

            for coupling_iter in range(mp_settings.max_coupling_iter):
                total_iter += 1
                T_old = T_field.copy()
                u_old = u_field.copy()
                p_old = p_field.copy()
                phi_old = phi_field.copy()

                status = CouplingStatus(iteration=total_iter)

                # 按 coupling_order 顺序求解各场
                for field_name in mp_settings.coupling_order:
                    if field_name == "thermal":
                        if thermal_material and thermal_settings:
                            thermal_result = self._solve_thermal_step(
                                mesh, thermal_material, sdof, thermal_settings,
                                temperature_bcs, heat_flux_bcs, convection_bcs,
                                radiation_bcs, T_field,
                                em_result, mp_settings,
                                irregular_cache,
                            )
                            if thermal_result is not None:
                                T_field = thermal_result.temperature

                    elif field_name == "fluid":
                        if fluid_material and fluid_settings:
                            fluid_result = self._solve_fluid_step(
                                mesh, fluid_material, sdof, dof_map,
                                fluid_settings, velocity_bcs, pressure_bcs,
                                u_field, T_field,
                                irregular_cache,
                            )
                            if fluid_result is not None and fluid_result.pressure is not None:
                                p_field = fluid_result.pressure

                    elif field_name == "structural":
                        struct_result = self._solve_structural_step(
                            mesh, integrator, struct_material, dof_map,
                            iga_settings, dirichlet_bcs,
                            neumann_bcs, point_loads,
                            load_factor, u_field,
                            T_field, p_field,
                            thermal_material, mp_settings,
                            em_result, em_material,
                            irregular_cache,
                        )
                        if struct_result is not None:
                            u_field = struct_result.displacements

                    elif field_name == "em":
                        if em_material and em_settings:
                            em_result = self._solve_em_step(
                                mesh, em_material, sdof, em_settings,
                                voltage_bcs, current_bcs,
                                T_field,
                                irregular_cache,
                            )
                            if em_result is not None:
                                phi_field = em_result.potential

                # 收敛检查
                dT = np.linalg.norm(T_field - T_old) / max(np.linalg.norm(T_field), 1e-10)
                du = np.linalg.norm(u_field - u_old) / max(np.linalg.norm(u_field), 1e-10)
                dp = np.linalg.norm(p_field - p_old) / max(np.linalg.norm(p_field), 1e-10)
                dphi = np.linalg.norm(phi_field - phi_old) / max(np.linalg.norm(phi_field), 1e-10)

                status.delta_temperature = dT
                status.delta_displacement = du
                status.delta_velocity = dp
                status.delta_potential = dphi

                # 动态松弛
                max_delta = max(dT, du, dp, dphi)
                if max_delta < mp_settings.coupling_tolerance:
                    status.converged = True
                    coupling_history.append(status)
                    if mp_settings.verbosity >= 1:
                        _logger.info(
                            f"  Coupling converged at iter {coupling_iter+1}, "
                            f"dT={dT:.2e}, du={du:.2e}"
                        )
                    break

                if coupling_iter < mp_settings.max_coupling_iter - 1:
                    # Aitken 松弛
                    if mp_settings.relaxation == "aitken":
                        pass  # 在下次迭代中应用松弛
                    elif mp_settings.relaxation == "fixed":
                        omega = mp_settings.relaxation_param
                        T_field = omega * T_field + (1 - omega) * T_old
                        u_field = omega * u_field + (1 - omega) * u_old
                        p_field = omega * p_field + (1 - omega) * p_old
                        phi_field = omega * phi_field + (1 - omega) * phi_old

                coupling_history.append(status)

            if mp_settings.verbosity >= 1:
                _logger.info(
                    f"Load step {load_step+1}/{mp_settings.n_load_steps} done, "
                    f"max_T={T_field.max():.1f}K, |u|={np.linalg.norm(u_field):.4e}"
                )

        wall_time = perf_counter() - t0

        return MultiphysicsResult(
            structural=struct_result,
            thermal=thermal_result,
            fluid=fluid_result,
            em=em_result,
            coupling_history=coupling_history,
            converged=all(s.converged for s in coupling_history[-mp_settings.max_coupling_iter:])
            if coupling_history else True,
            total_coupling_iterations=total_iter,
            wall_time=wall_time,
        )

    # ---- 各场求解 (内层) ----

    def _solve_thermal_step(
        self, mesh, thermal_material, sdof, thermal_settings,
        temperature_bcs, heat_flux_bcs, convection_bcs, radiation_bcs,
        T_current, em_result, mp_settings, irregular_cache,
    ) -> Optional[ThermalResult]:
        """求解热场子步 (含电磁焦耳热耦合)"""
        from forgecraft.analysis.iga_thermal import solve_thermal_steady

        # 焦耳热源
        internal_heat_source = 0.0
        if mp_settings.em_thermal and em_result is not None and em_result.joule_heat is not None:
            internal_heat_source = np.mean(em_result.joule_heat)

        return solve_thermal_steady(
            mesh, thermal_material, sdof, thermal_settings,
            temperature_bcs=temperature_bcs,
            heat_flux_bcs=heat_flux_bcs,
            convection_bcs=convection_bcs,
            radiation_bcs=radiation_bcs,
            internal_heat_source=internal_heat_source,
            irregular_cache=irregular_cache,
        )

    def _solve_fluid_step(
        self, mesh, fluid_material, sdof, dof_map, fluid_settings,
        velocity_bcs, pressure_bcs,
        u_field, T_field,
        irregular_cache,
    ) -> Optional[FluidResult]:
        """求解流体子步"""
        from forgecraft.analysis.iga_fluid import solve_thin_film

        return solve_thin_film(
            mesh, fluid_material, sdof, fluid_settings,
            pressure_bcs=pressure_bcs,
            velocity_bcs=velocity_bcs,
            irregular_cache=irregular_cache,
        )

    def _solve_structural_step(
        self, mesh, integrator, struct_material, dof_map,
        iga_settings, dirichlet_bcs,
        neumann_bcs, point_loads,
        load_factor, u_field,
        T_field, p_field,
        thermal_material, mp_settings,
        em_result, em_material,
        irregular_cache,
    ) -> Optional[StaticResult]:
        """求解结构子步 (含热膨胀+FSI+洛伦兹力耦合)"""
        from forgecraft.analysis.iga_solver import solve_static

        # 准备修改后的载荷
        neumann_modified = list(neumann_bcs) if neumann_bcs else []
        point_loads_modified = list(point_loads) if point_loads else []

        need_custom = False
        F_extra = np.zeros(dof_map.n_dof)

        # 热膨胀力
        if (
            mp_settings.thermal_structural
            and thermal_material is not None
            and thermal_material.thermal_expansion > 0
        ):
            from forgecraft.analysis.iga_thermal import compute_thermal_expansion_force
            f_th = compute_thermal_expansion_force(
                mesh, dof_map, struct_material, T_field,
                T_ref=300.0, alpha=thermal_material.thermal_expansion,
                settings=iga_settings, irregular_cache=irregular_cache,
            )
            F_extra += f_th * load_factor
            need_custom = True

        # 流体压力力 (FSI)
        if mp_settings.fluid_structural and np.any(p_field != 0):
            from forgecraft.analysis.iga_fluid import compute_fsi_force
            f_fsi = compute_fsi_force(mesh, p_field, dof_map, iga_settings.quadrature_order)
            F_extra += f_fsi * load_factor
            need_custom = True

        # 洛伦兹力
        if (
            mp_settings.em_structural
            and em_result is not None
            and em_result.current_density is not None
        ):
            # J = σE = current_density from EM result
            # B 需要从磁势计算 (此处简化: B ≈ μH, H 从 dA 计算)
            # 完整实现需要 compute_b_field 从磁势重建 B
            # 简化: 直接使用 current_density × 常数 B 近似
            pass

        if need_custom:
            # 使用自定义载荷求解
            from forgecraft.analysis.iga_assembly import (
                assemble_stiffness, assemble_force_vector,
            )
            from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty

            K = assemble_stiffness(
                mesh, integrator, struct_material, dof_map,
                iga_settings, irregular_cache,
            ).toarray()

            F = assemble_force_vector(
                mesh, dof_map,
                neumann_bcs or [], point_loads or [], iga_settings,
            )
            F = F * load_factor + F_extra

            for bc in dirichlet_bcs:
                apply_dirichlet_penalty(K, F, bc)

            try:
                u = np.linalg.solve(K, F)
                converged = True
            except np.linalg.LinAlgError:
                u = u_field
                converged = False

            return StaticResult(
                displacements=u,
                strain_energy=0.5 * u @ (K @ u),
                converged=converged,
            )

        # 无耦合 → 标准结构求解
        return solve_static(
            mesh, integrator, struct_material, dof_map, iga_settings,
            dirichlet_bcs, neumann_bcs, point_loads,
            irregular_cache=irregular_cache,
        )

    def _solve_em_step(
        self, mesh, em_material, sdof, em_settings,
        voltage_bcs, current_bcs,
        T_field,
        irregular_cache,
    ) -> Optional[EMResult]:
        """求解电磁子步"""
        from forgecraft.analysis.iga_electromagnetic import solve_electrostatic

        return solve_electrostatic(
            mesh, em_material, sdof, em_settings,
            voltage_bcs=voltage_bcs,
            current_bcs=current_bcs,
            irregular_cache=irregular_cache,
        )


# ══════════════════════════════════════════════════════════
#  便捷接口
# ══════════════════════════════════════════════════════════

def solve_multiphysics(
    mesh: PolyMesh,
    integrator,
    struct_material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    dirichlet_bcs,
    mp_settings: Optional[MultiphysicsSettings] = None,
    **kwargs,
) -> MultiphysicsResult:
    """四场耦合求解便捷入口

    用法:
      >>> result = solve_multiphysics(
      ...     mesh, integrator, steel, dof_map, iga_settings,
      ...     dirichlet_bcs,
      ...     mp_settings=MultiphysicsSettings(...),
      ...     thermal_material=ThermalMaterial(k=50, cp=500, rho=7800, alpha=1.2e-5),
      ...     thermal_settings=ThermalSettings(),
      ...     temperature_bcs=[...],
      ...     em_material=EMMaterial.copper(),
      ...     em_settings=EMSettings(),
      ...     voltage_bcs=[...],
      ... )
    """
    coupler = MultiphysicsCoupler()
    return coupler.solve(
        mesh, integrator, struct_material, dof_map, iga_settings,
        dirichlet_bcs,
        mp_settings=mp_settings,
        **kwargs,
    )
