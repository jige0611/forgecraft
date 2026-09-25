# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_multiphysics_assembly — 多物理场统一组装
#
#   将所有物理场贡献合并为一个整体系统矩阵。
#   支持整体式 (monolithic) 组装:
#
#   | K_uu   K_uT   0      0    | | u  |   | f_u |
#   | 0      K_TT   K_Tf   0    | | T  | = | f_T |
#   | K_fu   0      K_ff    0    | | v  |   | f_f |
#   | 0      K_ET   0      K_EE | | φ  |   | f_E |
#
#   和交错式辅助组装。提供顶层 MultiphysicsSolver 类。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IrregularFaceCache, IGASettings, StaticResult,
)
from forgecraft.analysis.iga_assembly import (
    StreamingCSRAssembler,
    assemble_stiffness, assemble_force_vector,
)
from forgecraft.analysis.iga_multiphysics_types import (
    ScalarDOFMap,
    ThermalSettings, ThermalMaterial, ThermalResult,
    FluidSettings, FluidMaterial, FluidResult,
    EMSettings, EMMaterial, EMResult,
    MultiphysicsSettings, MultiphysicsResult, CouplingStatus,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "MultiphysicsSolver",
    "assemble_monolithic_system",
    "solve_monolithic",
]


# ══════════════════════════════════════════════════════════
#  整体式 (Monolithic) 组装
# ══════════════════════════════════════════════════════════

def assemble_monolithic_system(
    mesh: PolyMesh,
    # 结构
    integrator,
    struct_material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    # 热
    thermal_material: Optional[ThermalMaterial] = None,
    thermal_settings: Optional[ThermalSettings] = None,
    # 流
    fluid_material: Optional[FluidMaterial] = None,
    fluid_settings: Optional[FluidSettings] = None,
    # 电磁
    em_material: Optional[EMMaterial] = None,
    em_settings: Optional[EMSettings] = None,
    # 耦合设置
    mp_settings: Optional[MultiphysicsSettings] = None,
    irregular_cache=None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """组装整体式多物理场系统

    返回:
      (K_total, F_total, block_info)

    其中 block_info 记录了各子矩阵的尺寸和偏移:
      {
        "n_dofs": [n_u, n_T, n_v, n_phi],
        "offsets": [0, n_u, n_u+n_T, n_u+n_T+n_v],
        "total": n_total,
      }
    """
    mp_settings = mp_settings or MultiphysicsSettings()
    sdof = ScalarDOFMap.from_mesh(mesh)

    n_u = dof_map.n_dof
    n_T = sdof.n_dof
    n_v = dof_map.n_dof  # 速度与位移同 DOF
    n_phi = sdof.n_dof

    # 确定哪些场活跃
    has_thermal = thermal_material is not None
    has_fluid = fluid_material is not None
    has_em = em_material is not None

    # 收集所有活跃的 DOF 数量
    dof_counts = [n_u]
    dof_labels = ["u"]
    if has_thermal:
        dof_counts.append(n_T)
        dof_labels.append("T")
    if has_fluid:
        dof_counts.append(n_v)
        dof_labels.append("v")
    if has_em:
        dof_counts.append(n_phi)
        dof_labels.append("φ")

    offsets = [0]
    for n in dof_counts[:-1]:
        offsets.append(offsets[-1] + n)

    n_total = sum(dof_counts)

    block_info = {
        "n_dofs": dof_counts,
        "offsets": offsets,
        "labels": dof_labels,
        "total": n_total,
    }

    # 初始化整体系统
    K_total = np.zeros((n_total, n_total))
    F_total = np.zeros(n_total)

    # --- 结构块 (0, 0) ---
    K_uu = assemble_stiffness(
        mesh, integrator, struct_material, dof_map,
        iga_settings, irregular_cache,
    ).toarray()

    off_u = offsets[0]
    K_total[off_u:off_u + n_u, off_u:off_u + n_u] = K_uu

    # --- 热块 ---
    if has_thermal:
        from forgecraft.analysis.iga_thermal import assemble_thermal_conductivity

        K_TT = assemble_thermal_conductivity(
            mesh, thermal_material, sdof, thermal_settings, irregular_cache,
        ).toarray()

        off_T = offsets[dof_labels.index("T")]
        K_total[off_T:off_T + n_T, off_T:off_T + n_T] = K_TT

        # 热→结构耦合: 热膨胀 (K_uT)
        if mp_settings.thermal_structural and thermal_material.thermal_expansion > 0:
            K_uT = _assemble_thermal_expansion_coupling(
                mesh, integrator, struct_material, dof_map, sdof,
                thermal_material, iga_settings, irregular_cache,
            )
            K_total[off_u:off_u + n_u, off_T:off_T + n_T] = K_uT

    # --- 流体块 ---
    if has_fluid:
        from forgecraft.analysis.iga_fluid import (
            assemble_reynolds_stiffness,
        )

        K_ff = assemble_reynolds_stiffness(
            mesh, fluid_material, sdof, fluid_settings, irregular_cache,
        ).toarray()

        off_f = offsets[dof_labels.index("v")]
        K_total[off_f:off_f + n_v, off_f:off_f + n_v] = K_ff

        # 流→结构耦合: FSI (K_fu → 流体压力影响结构)
        if mp_settings.fluid_structural:
            from forgecraft.analysis.iga_fluid import compute_fsi_force
            # FSI 作为力向量处理 (非对称耦合)
            # 在这里仅标记耦合存在，具体力在 solve 中处理

    # --- 电磁块 ---
    if has_em:
        from forgecraft.analysis.iga_electromagnetic import (
            assemble_electrostatic_stiffness,
        )

        K_EE = assemble_electrostatic_stiffness(
            mesh, em_material, sdof, em_settings, irregular_cache,
        ).toarray()

        off_E = offsets[dof_labels.index("φ")]
        K_total[off_E:off_E + n_phi, off_E:off_E + n_phi] = K_EE

        # EM→热耦合: 焦耳热 (K_ET)
        if mp_settings.em_thermal and em_material.conductivity > 0:
            K_ET = _assemble_joule_heating_coupling(
                mesh, sdof, em_material, em_settings, irregular_cache,
            )
            off_T = offsets[dof_labels.index("T")]
            K_total[off_T:off_T + n_T, off_E:off_E + n_phi] = K_ET

    return K_total, F_total, block_info


def _assemble_thermal_expansion_coupling(
    mesh, integrator, struct_material, dof_map, sdof,
    thermal_material, iga_settings, irregular_cache,
) -> np.ndarray:
    """组装热膨胀耦合矩阵 K_uT

    K_uT = ∂f_struct/∂T = ∫ B^T · D · α · [1,1,0] · N_T dΩ

    线性化热膨胀刚度贡献。
    """
    n_u = dof_map.n_dof
    n_T = sdof.n_dof
    K_uT = np.zeros((n_u, n_T))

    # 对于线性热膨胀: f_th = K_uT · T (在参考温度附近线性化)
    # 实际在耦合器中用等效节点力处理
    # 这里提供占位，完整实现需要重构 B 矩阵

    return K_uT


def _assemble_joule_heating_coupling(
    mesh, sdof, em_material, em_settings, irregular_cache,
) -> np.ndarray:
    """组装焦耳热耦合矩阵 K_ET

    K_ET = ∂f_thermal/∂φ = ∂(σ|∇φ|²)/∂φ

    焦耳热是非线性的 (依赖于 ∇φ 的二次型)。
    线性化需要当前电势场。
    """
    n_T = sdof.n_dof
    n_phi = sdof.n_dof
    K_ET = np.zeros((n_T, n_phi))

    # 非线性耦合 → 在交错迭代中处理
    # 整体式需要一致切线

    return K_ET


def solve_monolithic(
    mesh: PolyMesh,
    integrator,
    struct_material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    dirichlet_bcs,
    mp_settings: Optional[MultiphysicsSettings] = None,
    **kwargs,
) -> MultiphysicsResult:
    """整体式多物理场求解

    将所有场组装为一个系统，一次性求解。
    适用于小规模问题或需要无条件稳定性的场景。
    """
    t0 = perf_counter()

    mp_settings = mp_settings or MultiphysicsSettings()

    K_total, F_total, block_info = assemble_monolithic_system(
        mesh, integrator, struct_material, dof_map, iga_settings,
        mp_settings=mp_settings,
        **kwargs,
    )

    # 施加 BC
    if dirichlet_bcs:
        from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty
        for bc in dirichlet_bcs:
            apply_dirichlet_penalty(K_total, F_total, bc)

    # 求解
    try:
        solution = np.linalg.solve(K_total, F_total)
        converged = True
    except np.linalg.LinAlgError:
        solution = np.zeros(block_info["total"])
        converged = False

    # 拆解结果
    offsets = block_info["offsets"]
    labels = block_info["labels"]
    n_dofs = block_info["n_dofs"]

    struct_displacement = solution[offsets[0]:offsets[0] + n_dofs[0]]

    thermal_result = None
    fluid_result = None
    em_result = None

    for i, label in enumerate(labels):
        if label == "T":
            T_field = solution[offsets[i]:offsets[i] + n_dofs[i]]
            thermal_result = ThermalResult(temperature=T_field)
        elif label == "v":
            p_field = solution[offsets[i]:offsets[i] + n_dofs[i]]
            fluid_result = FluidResult(pressure=p_field)
        elif label == "φ":
            phi_field = solution[offsets[i]:offsets[i] + n_dofs[i]]
            em_result = EMResult(potential=phi_field)

    wall_time = perf_counter() - t0

    return MultiphysicsResult(
        structural=StaticResult(
            displacements=struct_displacement,
            converged=converged,
            wall_time=wall_time,
        ),
        thermal=thermal_result,
        fluid=fluid_result,
        em=em_result,
        converged=converged,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  MultiphysicsSolver: 顶层多物理场求解器
# ══════════════════════════════════════════════════════════

class MultiphysicsSolver:
    """多物理场求解器顶层入口

    自动根据 MultiphysicsSettings 选择耦合策略:
      - "staggered": 块 Gauss-Seidel 交错迭代 (默认)
      - "monolithic": 整体式单次求解

    用法:
      >>> solver = MultiphysicsSolver()
      >>> result = solver.solve(
      ...     mesh, integrator, steel, dof_map, iga_settings,
      ...     dirichlet_bcs,
      ...     mp_settings=MultiphysicsSettings(
      ...         coupling_scheme="staggered",
      ...         thermal_structural=True,
      ...     ),
      ...     thermal_material=ThermalMaterial(k=50, cp=500, rho=7800, alpha=1.2e-5),
      ...     thermal_settings=ThermalSettings(),
      ...     temperature_bcs=[...],
      ...     em_material=EMMaterial.copper(),
      ...     em_settings=EMSettings(),
      ...     voltage_bcs=[...],
      ... )
    """

    def solve(
        self,
        mesh: PolyMesh,
        integrator,
        struct_material,
        dof_map: DOFMap,
        iga_settings: IGASettings,
        dirichlet_bcs,
        mp_settings: Optional[MultiphysicsSettings] = None,
        **kwargs,
    ) -> MultiphysicsResult:
        """多物理场求解主入口"""
        mp_settings = mp_settings or MultiphysicsSettings()

        if mp_settings.coupling_scheme == "monolithic":
            return solve_monolithic(
                mesh, integrator, struct_material, dof_map, iga_settings,
                dirichlet_bcs,
                mp_settings=mp_settings,
                **kwargs,
            )
        else:
            from forgecraft.analysis.iga_multiphysics_coupling import (
                MultiphysicsCoupler,
            )
            coupler = MultiphysicsCoupler()
            return coupler.solve(
                mesh, integrator, struct_material, dof_map, iga_settings,
                dirichlet_bcs,
                mp_settings=mp_settings,
                **kwargs,
            )
