# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_topology_optimizer — 相位场拓扑优化器
#
#   主优化器类: 协调相场求解、弹性分析、灵敏度计算、
#   设计更新和连续方案的完整迭代流程。
#
#   优化算法:
#     - OC (最优性准则): 单约束 (体积), 简单高效
#     - Augmented Lagrangian: 处理体积 + 应力/位移多约束
#
#   连续方案 (Continuation):
#     逐步增大惩罚指数 p (1→3) 和收缩界面 ε,
#     避免陷入局部极小值。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IGASettings, IrregularFaceCache,
)
from forgecraft.analysis.iga_assembly import (
    assemble_stiffness, assemble_force_vector,
)
from forgecraft.analysis.iga_multiphysics_types import ScalarDOFMap
from forgecraft.analysis.iga_topology_types import (
    TopologySettings, TopologyMaterial,
    TopologyConstraint, ConstraintType,
    TopologyOptimizationResult, TopologyIterationData,
    g_simp, dg_simp, g_ramp, dg_ramp, g_polynomial, dg_polynomial,
)
from forgecraft.analysis.iga_topology_phasefield import (
    assemble_phasefield_stiffness,
    assemble_phasefield_driving_force,
    assemble_degraded_stiffness_topology,
    compute_strain_energy_density,
    solve_phasefield_field,
    compute_volume_fraction,
    apply_density_filter,
    _compute_vertex_areas,
)
from forgecraft.analysis.iga_topology_sensitivity import (
    compute_compliance_sensitivity,
    compute_volume_sensitivity,
    filter_sensitivity,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "TopologyOptimizer",
    "topology_optimize",
]


# ══════════════════════════════════════════════════════════
#  主优化器类
# ══════════════════════════════════════════════════════════

class TopologyOptimizer:
    """相位场拓扑优化主求解器

    交错策略 (每个外迭代):
      1. 用 g(φ, p) 退化材料 → 组装 K_u
      2. 求解位移 u = K_u⁻¹ · f_ext
      3. 计算应变能密度 ψe
      4. 求解相场更新 φ_new (Allen-Cahn PDE)
      5. 更新体积约束拉格朗日乘子 λ
      6. 计算灵敏度 dC/dφ, dV/dφ
      7. OC 或增广拉格朗日设计更新
      8. 连续方案更新 p, ε
      9. 收敛检查

    用法:
        settings = TopologySettings(vol_frac=0.3)
        material = TopologyMaterial(E=210e9, nu=0.3)
        optimizer = TopologyOptimizer(settings)
        result = optimizer.optimize(mesh, integrator, material, ...)
    """

    def __init__(self, settings: TopologySettings):
        self.settings = settings

        # 内部状态
        self.phi_field: Optional[np.ndarray] = None
        self.lagrange_multiplier: float = settings.lagrangian_multiplier
        self.penalty_param: float = settings.penalty_param
        self.p_current: float = settings.p_init
        self.epsilon_current: float = settings.epsilon

        # 历史
        self.compliance_history: List[float] = []
        self.volume_history: List[float] = []
        self.iteration_data: List[TopologyIterationData] = []

        # 单次组装的缓存
        self._K0: Optional[np.ndarray] = None      # 完整刚度 (未退化)
        self._F_ref: Optional[np.ndarray] = None    # 参考载荷

    def optimize(
        self,
        mesh: PolyMesh,
        integrator,
        material: TopologyMaterial,
        dof_map: DOFMap,
        iga_settings: IGASettings,
        dirichlet_bcs=None,
        neumann_bcs=None,
        point_loads=None,
        constraints: Optional[List[TopologyConstraint]] = None,
        irregular_cache: Optional[IrregularFaceCache] = None,
        initial_phi: Optional[np.ndarray] = None,
    ) -> TopologyOptimizationResult:
        """执行相位场拓扑优化

        Args:
            mesh: CC 控制网格
            integrator: ElementIntegrator 实例
            material: 拓扑优化材料
            dof_map: 结构 DOF 映射
            iga_settings: IGA 设置
            dirichlet_bcs: Dirichlet 边界条件
            neumann_bcs: Neumann 边界条件
            point_loads: 点载荷
            constraints: 额外约束列表 (体积自动添加)
            irregular_cache: 非常面缓存
            initial_phi: 初始相场 (None = 均匀 V_f)

        Returns:
            TopologyOptimizationResult
        """
        t0 = perf_counter()

        n_vertices = mesh.n_vertices
        n_dof = dof_map.n_dof

        # 标量 DOF 映射 (相场)
        sdof = ScalarDOFMap.from_mesh(mesh)

        # ── 初始化 ──
        # 相场
        if initial_phi is not None:
            self.phi_field = initial_phi.copy()
        else:
            self.phi_field = np.full(n_vertices, self.settings.vol_frac)

        self.lagrange_multiplier = self.settings.lagrangian_multiplier
        self.penalty_param = self.settings.penalty_param
        self.p_current = self.settings.p_init
        self.epsilon_current = self.settings.epsilon

        # 累积历史
        self.compliance_history = []
        self.volume_history = []
        self.iteration_data = []

        # 组装完整刚度 (未退化, 供退化用)
        if iga_settings.verbosity >= 1:
            _logger.info("Assembling full stiffness matrix...")

        self._K0 = assemble_stiffness(
            mesh, integrator, material, dof_map, iga_settings, irregular_cache,
        ).toarray()

        # 组装参考力向量
        self._F_ref = assemble_force_vector(
            mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
        )

        # 施加 Dirichlet BC 到 K0
        from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty
        F_dummy = np.zeros(n_dof)
        if dirichlet_bcs:
            for bc in dirichlet_bcs:
                apply_dirichlet_penalty(self._K0, F_dummy, bc)

        self._F_ref = self._F_ref.astype(np.float64)

        # 合并约束 (体积总是存在)
        all_constraints = list(constraints or [])
        vol_constraint = TopologyConstraint.volume(
            target=self.settings.vol_frac,
            tolerance=self.settings.design_tol,
        )
        # 确保体积约束在最前面
        has_volume = any(c.constraint_type == ConstraintType.VOLUME for c in all_constraints)
        if not has_volume:
            all_constraints.insert(0, vol_constraint)

        u = np.zeros(n_dof)
        iter_count = 0
        converged = False
        termination_reason = "max_iter"

        # ── 外循环 ──
        for iteration in range(self.settings.max_iter):
            iter_count = iteration + 1

            # 连续方案
            if iteration > 0 and iteration % self.settings.continuation_freq == 0:
                self._continuation_step()

            # ---- Step 1: 退化刚度 + 求解位移 ----
            K_u = assemble_degraded_stiffness_topology(
                self._K0, self.phi_field, mesh, dof_map, self.settings,
            )

            # 微小正则化防止 CC 边界控制点不足导致的近奇异
            diag_reg = 1e-12 * np.max(np.abs(K_u.diagonal()))
            K_u_reg = K_u + diag_reg * np.eye(K_u.shape[0])

            try:
                u = np.linalg.solve(K_u_reg, self._F_ref)
            except np.linalg.LinAlgError:
                _logger.warning(f"Iter {iteration}: singular stiffness, stopping")
                termination_reason = "singular"
                break

            # ---- Step 2: 应变能密度 ----
            psi_e = compute_strain_energy_density(
                mesh, integrator, material, u, dof_map, iga_settings, irregular_cache,
            )

            # ---- Step 3: 柔度和体积 ----
            compliance = float(np.dot(u, self._F_ref))
            V_f = compute_volume_fraction(mesh, self.phi_field)

            self.compliance_history.append(compliance)
            self.volume_history.append(V_f)

            # ---- Step 4: 求解相场 ----
            phi_new = solve_phasefield_field(
                mesh, self.phi_field, psi_e, sdof, self.settings,
                K_phi=None,
                lagrange_multiplier=self.lagrange_multiplier,
                irregular_cache=irregular_cache,
            )

            # 可选密度滤波
            if self.settings.density_filter_radius > 0:
                phi_new = apply_density_filter(
                    mesh, phi_new, self.settings.density_filter_radius,
                )

            # ---- Step 5: 灵敏度分析 ----
            dc = compute_compliance_sensitivity(
                mesh, self.phi_field, psi_e, self.settings,
            )
            dv = compute_volume_sensitivity(mesh)

            # 可选灵敏度滤波
            if self.settings.density_filter_radius > 0:
                dc = filter_sensitivity(
                    mesh, dc, self.phi_field, self.settings.density_filter_radius,
                )

            # ---- Step 6: 设计更新 ----
            if self.settings.optimizer == "OC":
                phi_new = self._oc_update(
                    phi_new, dc, dv, self.settings.vol_frac,
                )
            elif self.settings.optimizer == "augmented_lagrangian":
                phi_new = self._augmented_lagrangian_update(
                    phi_new, dc, dv, V_f,
                )

            # ---- Step 7: 更新体积约束乘子 ----
            V_new = compute_volume_fraction(mesh, phi_new)
            self.lagrange_multiplier += self.penalty_param * (V_new - self.settings.vol_frac)
            self.lagrange_multiplier = max(self.lagrange_multiplier, 0.0)  # 保持 ≥ 0
            self.penalty_param = min(
                self.penalty_param * self.settings.penalty_growth,
                self.settings.penalty_max,
            )

            # ---- Step 8: 收敛检查 ----
            phi_change = np.max(np.abs(phi_new - self.phi_field))

            if len(self.compliance_history) >= 2:
                compliance_change = abs(
                    compliance - self.compliance_history[-2]
                ) / max(abs(compliance), 1e-15)
            else:
                compliance_change = 1.0

            # 存储迭代数据
            self.iteration_data.append(TopologyIterationData(
                iteration=iter_count,
                compliance=compliance,
                volume_fraction=V_f,
                phi_change=phi_change,
                compliance_change=compliance_change,
                lagrange_multiplier=self.lagrange_multiplier,
                penalty_param=self.penalty_param,
                sim_p=self.p_current,
                epsilon=self.epsilon_current,
            ))

            # 更新相场
            self.phi_field = phi_new

            if self.settings.verbosity >= 1:
                _logger.info(
                    f"Iter {iter_count:4d}: C={compliance:.4e}, "
                    f"V={V_f:.4f} (target={self.settings.vol_frac:.4f}), "
                    f"Δφ={phi_change:.4e}, ΔC={compliance_change:.4e}, "
                    f"λ={self.lagrange_multiplier:.4e}, p={self.p_current:.2f}"
                )

            # 收敛
            if phi_change < self.settings.design_tol:
                if compliance_change < self.settings.obj_tol:
                    converged = True
                    termination_reason = "converged"
                    break

        # ── 后处理 ──
        wall_time = perf_counter() - t0

        # 计算优化密度 g(φ)
        optimized_density = np.zeros(n_vertices)
        for vi in range(n_vertices):
            if self.settings.interpolation == "SIMP":
                optimized_density[vi] = g_simp(
                    self.phi_field[vi], self.settings.p_current, self.settings.phi_min,
                )
            elif self.settings.interpolation == "RAMP":
                optimized_density[vi] = g_ramp(
                    self.phi_field[vi], q=3.0, phi_min=self.settings.phi_min,
                )
            else:
                optimized_density[vi] = g_polynomial(self.phi_field[vi])

        # 实体区域
        solid_vertices = np.where(self.phi_field >= 0.5)[0]
        solid_volume = float(np.sum(self.phi_field[solid_vertices] * _compute_vertex_areas(mesh)[solid_vertices]))
        total_volume = float(np.sum(_compute_vertex_areas(mesh)))
        solid_mass = solid_volume * material.rho
        total_mass = total_volume * material.rho
        mass_reduction = 1.0 - solid_mass / max(total_mass, 1e-15)

        result = TopologyOptimizationResult(
            phi_field=self.phi_field.copy(),
            optimized_density=optimized_density,
            displacement=u.copy(),
            iterations=iter_count,
            converged=converged,
            termination_reason=termination_reason,
            compliance_history=list(self.compliance_history),
            volume_history=list(self.volume_history),
            iteration_data=list(self.iteration_data),
            solid_vertices=solid_vertices,
            solid_volume=solid_volume,
            solid_mass=solid_mass,
            mass_reduction=mass_reduction,
            wall_time=wall_time,
            n_total_linear_solves=iter_count * 2,  # 位移 + 相场
        )

        if self.settings.verbosity >= 1:
            _logger.info(result.summary())

        return result

    # ══════════════════════════════════════════════════════
    #  OC (最优性准则) 设计更新
    # ══════════════════════════════════════════════════════

    def _oc_update(
        self,
        phi: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        vol_target: float,
    ) -> np.ndarray:
        """最优性准则 (OC) 设计更新

        SIMP 标准 OC 更新:
          B_i = -dc_i / (λ · dv_i)
          φ_new = max(φ_min, φ_i - M)  if φ_i·B_i^η ≤ max(φ_min, φ_i - M)
          φ_new = min(1.0, φ_i + M)     if φ_i·B_i^η ≥ min(1.0, φ_i + M)
          φ_new = φ_i·B_i^η              otherwise

        其中 M = move_limit, η = 0.5 (阻尼)
        λ 通过二分搜索满足体积约束。

        Args:
            phi: (n_vertices,) 当前相场
            dc: (n_vertices,) 柔度灵敏度 dC/dφ
            dv: (n_vertices,) 体积灵敏度 dV/dφ
            vol_target: 目标体积分数

        Returns:
            phi_new: (n_vertices,) 更新后相场
        """
        n = len(phi)
        move = self.settings.move_limit
        eta = 0.5  # 数值阻尼
        phi_min_val = self.settings.phi_min

        # 二分搜索 Lagrange 乘子 λ
        l1 = 0.0
        l2 = 1e9

        # dc 是负值 (加材料降低柔度) → 用 -dc 做正优化
        dcost = -dc  # 取正 → 高灵敏度应加材料

        for _ in range(100):  # 最大二分迭代
            lmid = 0.5 * (l1 + l2)

            phi_new = np.zeros(n)
            for i in range(n):
                if dv[i] < 1e-15:
                    phi_new[i] = phi[i]
                    continue

                B_i = dcost[i] / max(lmid * dv[i], 1e-15)
                phi_oc = phi[i] * (B_i ** eta)

                lower = max(phi_min_val, phi[i] - move)
                upper = min(1.0, phi[i] + move)
                phi_new[i] = np.clip(phi_oc, lower, upper)

            # 体积约束
            V_new = np.dot(phi_new, dv) / max(np.sum(dv), 1e-15)

            if abs(V_new - vol_target) < 1e-6:
                break
            if V_new > vol_target:
                l1 = lmid
            else:
                l2 = lmid

            if l2 - l1 < 1e-12:
                break

        return phi_new

    # ══════════════════════════════════════════════════════
    #  增广拉格朗日设计更新
    # ══════════════════════════════════════════════════════

    def _augmented_lagrangian_update(
        self,
        phi: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        V_f: float,
    ) -> np.ndarray:
        """增广拉格朗日设计更新

        增广拉格朗日函数:
          L(φ) = C(φ) + λ·(V-V_target) + ρ/2·(V-V_target)²

        梯度下降步:
          φ_new = φ - α · dL/dφ
          dL/dφ_i = dc_i + λ·dv_i + ρ·(V-V_target)·dv_i

        然后施加箱约束。

        Args:
            phi: (n_vertices,) 当前相场
            dc: (n_vertices,) 柔度灵敏度
            dv: (n_vertices,) 体积灵敏度
            V_f: 当前体积分数

        Returns:
            phi_new: (n_vertices,) 更新后相场
        """
        n = len(phi)
        vol_target = self.settings.vol_frac
        rho = self.penalty_param
        lam = self.lagrange_multiplier
        move = self.settings.move_limit
        phi_min_val = self.settings.phi_min

        # 增广拉格朗日梯度
        V_error = V_f - vol_target
        dL = dc + lam * dv + rho * V_error * dv

        # 线搜索步长 (自适应)
        dL_max = max(np.max(np.abs(dL)), 1e-15)
        alpha = min(move / dL_max, move * 10.0)

        # 梯度下降 + 箱约束
        phi_new = phi - alpha * dL
        phi_new = np.clip(phi_new, phi_min_val, 1.0)

        # 移动限制
        lower = np.maximum(phi_min_val, phi - move)
        upper = np.minimum(1.0, phi + move)
        phi_new = np.clip(phi_new, lower, upper)

        return phi_new

    # ══════════════════════════════════════════════════════
    #  连续方案
    # ══════════════════════════════════════════════════════

    def _continuation_step(self):
        """连续方案: 递增 p, 收缩 ε"""
        if self.settings.continue_p:
            self.p_current = min(
                self.p_current + self.settings.p_step,
                self.settings.p_max,
            )
        if self.settings.continue_epsilon:
            self.epsilon_current *= self.settings.epsilon_decay
            self.epsilon_current = max(self.epsilon_current, 0.001)

        if self.settings.verbosity >= 1:
            _logger.info(
                f"  Continuation: p={self.p_current:.2f}, ε={self.epsilon_current:.4f}"
            )


# ══════════════════════════════════════════════════════════
#  便捷接口
# ══════════════════════════════════════════════════════════

def topology_optimize(
    mesh: PolyMesh,
    integrator,
    material: TopologyMaterial,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    settings: Optional[TopologySettings] = None,
    dirichlet_bcs=None,
    neumann_bcs=None,
    point_loads=None,
    constraints: Optional[List[TopologyConstraint]] = None,
    **kwargs,
) -> TopologyOptimizationResult:
    """便捷接口: 一键拓扑优化

    用法:
        result = topology_optimize(
            mesh, integrator, material, dof_map, iga_settings,
            dirichlet_bcs=[fixed_left],
            neumann_bcs=[load_right],
        )
        print(result.summary())
    """
    if settings is None:
        settings = TopologySettings()

    optimizer = TopologyOptimizer(settings)
    return optimizer.optimize(
        mesh, integrator, material, dof_map, iga_settings,
        dirichlet_bcs=dirichlet_bcs,
        neumann_bcs=neumann_bcs,
        point_loads=point_loads,
        constraints=constraints,
        **kwargs,
    )
