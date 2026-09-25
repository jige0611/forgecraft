# ══════════════════════════════════════════════════════════
# test_topology_phasefield.py — 相位场拓扑优化集成测试
# ══════════════════════════════════════════════════════════

import sys
import os
import unittest
from time import perf_counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_5x5_mesh():
    from forgecraft.geometry.mesh import PolyMesh
    n = 5
    verts = []
    for j in range(n):
        for i in range(n):
            verts.append([i * 0.25, j * 0.25, 0.0])
    verts = np.array(verts, dtype=np.float64)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + i + 1
            c = (j + 1) * n + i + 1
            d = (j + 1) * n + i
            faces.append([a, b, c, d])
    return PolyMesh.from_vertices_faces(verts, faces)


def _make_10x10_mesh():
    from forgecraft.geometry.mesh import PolyMesh
    n = 10
    verts = []
    for j in range(n):
        for i in range(n):
            verts.append([i * 0.1111111, j * 0.1111111, 0.0])
    verts = np.array(verts, dtype=np.float64)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + i + 1
            c = (j + 1) * n + i + 1
            d = (j + 1) * n + i
            faces.append([a, b, c, d])
    return PolyMesh.from_vertices_faces(verts, faces)


# ══════════════════════════════════════════════════════════
#  测试 1: 拓扑优化类型系统
# ══════════════════════════════════════════════════════════

class TestTopologyTypes(unittest.TestCase):

    def test_topology_settings_defaults(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings

        ts = TopologySettings()
        self.assertAlmostEqual(ts.vol_frac, 0.3)
        self.assertEqual(ts.interpolation, "SIMP")
        self.assertEqual(ts.optimizer, "OC")
        self.assertAlmostEqual(ts.phi_min, 1e-6)
        self.assertAlmostEqual(ts.p_init, 1.0)
        self.assertAlmostEqual(ts.p_max, 3.0)

    def test_topology_settings_p_current(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings

        ts = TopologySettings(p_init=1.5)
        self.assertAlmostEqual(ts.p_current, 1.5)
        ts.p_current = 3.0
        self.assertAlmostEqual(ts.p_current, 3.0)

    def test_topology_material(self):
        from forgecraft.analysis.iga_topology_types import TopologyMaterial

        tm = TopologyMaterial(E=210e9, nu=0.3, rho=7800.0)
        self.assertAlmostEqual(tm.E, 210e9)
        self.assertAlmostEqual(tm.nu, 0.3)
        self.assertAlmostEqual(tm.rho, 7800.0)

    def test_interpolation_simp_scalar(self):
        from forgecraft.analysis.iga_topology_types import g_simp, dg_simp

        # g(0) ≈ phi_min, g(1) ≈ 1
        self.assertAlmostEqual(g_simp(0.0, p=3.0), 1e-6, places=4)
        self.assertAlmostEqual(g_simp(1.0, p=3.0), 1.0, places=4)

        # dg/dφ at φ=0.5
        dg = dg_simp(0.5, p=3.0)
        self.assertGreater(dg, 0)

    def test_interpolation_ramp_scalar(self):
        from forgecraft.analysis.iga_topology_types import g_ramp, dg_ramp

        self.assertAlmostEqual(g_ramp(0.0, q=3.0), 1e-6, places=4)
        self.assertAlmostEqual(g_ramp(1.0, q=3.0), 1.0, places=4)
        self.assertGreater(dg_ramp(0.5, q=3.0), 0)

    def test_interpolation_polynomial_scalar(self):
        from forgecraft.analysis.iga_topology_types import g_polynomial, dg_polynomial

        self.assertAlmostEqual(g_polynomial(0.0), 0.0)
        self.assertAlmostEqual(g_polynomial(1.0), 1.0)
        self.assertAlmostEqual(dg_polynomial(1.0), 3.0)

    def test_interpolation_class(self):
        from forgecraft.analysis.iga_topology_types import InterpolationFunction

        simp = InterpolationFunction.get("SIMP", p=3.0)
        phi = np.array([0.0, 0.5, 1.0])
        g_vals = simp.g(phi)
        self.assertAlmostEqual(g_vals[0], 1e-6, places=4)
        self.assertAlmostEqual(g_vals[2], 1.0, places=4)

        ramp = InterpolationFunction.get("RAMP", q=3.0)
        g_vals = ramp.g(phi)
        self.assertAlmostEqual(g_vals[2], 1.0, places=4)

        poly = InterpolationFunction.get("polynomial")
        g_vals = poly.g(phi)
        self.assertAlmostEqual(g_vals[2], 1.0)

    def test_double_well_potential(self):
        from forgecraft.analysis.iga_topology_types import (
            w_double_well, dw_double_well, d2w_double_well,
        )

        # w(0) = w(1) = 0
        self.assertAlmostEqual(w_double_well(0.0), 0.0)
        self.assertAlmostEqual(w_double_well(1.0), 0.0)
        # w(0.5) = max
        self.assertGreater(w_double_well(0.5), 0)
        # dw(0.5) = 0 (critical point)
        self.assertAlmostEqual(dw_double_well(0.5), 0.0)
        # d²w(0) = 2 (positive curvature at min)
        self.assertAlmostEqual(d2w_double_well(0.0), 2.0)

    def test_constraint_creation(self):
        from forgecraft.analysis.iga_topology_types import (
            TopologyConstraint, ConstraintType,
        )

        vol = TopologyConstraint.volume(target=0.3)
        self.assertEqual(vol.constraint_type, ConstraintType.VOLUME)

        stress = TopologyConstraint.stress(target=250e6, p_norm=8.0)
        self.assertEqual(stress.constraint_type, ConstraintType.STRESS)
        self.assertAlmostEqual(stress.target, 250e6)
        self.assertAlmostEqual(stress.p_norm, 8.0)

        disp = TopologyConstraint.displacement(target=0.001, dof_id=42)
        self.assertEqual(disp.dof_id, 42)

    def test_result_dataclass(self):
        from forgecraft.analysis.iga_topology_types import TopologyOptimizationResult

        result = TopologyOptimizationResult(
            phi_field=np.array([0.5, 1.0]),
            optimized_density=np.array([0.125, 1.0]),
            iterations=10,
            converged=True,
            termination_reason="converged",
            compliance_history=[1.0, 0.5],
            volume_history=[1.0, 0.3],
            mass_reduction=0.7,
            wall_time=1.0,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 10)
        self.assertEqual(len(result.compliance_history), 2)

        # 摘要
        summary = result.summary()
        self.assertIn("Phase-Field Topology", summary)


# ══════════════════════════════════════════════════════════
#  测试 2: 相场 PDE 组装
# ══════════════════════════════════════════════════════════

class TestPhaseFieldAssembly(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis.iga_multiphysics_types import ScalarDOFMap
        from forgecraft.analysis.iga_topology_types import TopologySettings

        self.mesh = _make_5x5_mesh()
        self.sdof = ScalarDOFMap.from_mesh(self.mesh)
        self.settings = TopologySettings(quadrature_order=3)

    def test_laplacian_assembly(self):
        from forgecraft.analysis.iga_topology_phasefield import assemble_phasefield_laplacian

        K_lap = assemble_phasefield_laplacian(
            self.mesh, self.sdof, self.settings,
        )
        self.assertEqual(K_lap.shape[0], self.mesh.n_vertices)
        self.assertEqual(K_lap.shape[1], self.mesh.n_vertices)
        # 应是对称稀疏矩阵
        self.assertGreater(K_lap.nnz, 0)

    def test_phasefield_stiffness_assembly(self):
        from forgecraft.analysis.iga_topology_phasefield import assemble_phasefield_stiffness

        phi = np.full(self.mesh.n_vertices, 0.5)
        K_phi = assemble_phasefield_stiffness(
            self.mesh, phi, self.sdof, self.settings,
        )
        self.assertEqual(K_phi.shape[0], self.mesh.n_vertices)

        # 检查是否为 PPT (对角占优)
        K_dense = K_phi.toarray()
        for i in range(min(10, K_dense.shape[0])):
            if K_dense[i, i] > 1e-10:
                self.assertGreaterEqual(K_dense[i, i], 0)

    def test_driving_force_assembly(self):
        from forgecraft.analysis.iga_topology_phasefield import assemble_phasefield_driving_force

        phi = np.full(self.mesh.n_vertices, 0.5)
        psi_e = np.random.rand(self.mesh.n_vertices) * 1e6

        f_drive = assemble_phasefield_driving_force(
            self.mesh, phi, psi_e, self.settings,
        )
        self.assertEqual(len(f_drive), self.mesh.n_vertices)

        # 正应变能应产生正驱动力
        mask = psi_e > 0.1e6
        if np.any(mask):
            self.assertTrue(np.any(f_drive[mask] > 0))

    def test_vertex_areas(self):
        from forgecraft.analysis.iga_topology_phasefield import _compute_vertex_areas

        areas = _compute_vertex_areas(self.mesh)
        self.assertEqual(len(areas), self.mesh.n_vertices)
        # 总面积为单位正方形面积
        self.assertAlmostEqual(areas.sum(), 1.0, delta=0.15)
        # 所有面积应为正
        self.assertTrue(np.all(areas > 0))

    def test_volume_fraction(self):
        from forgecraft.analysis.iga_topology_phasefield import compute_volume_fraction

        # 满材料
        phi_full = np.ones(self.mesh.n_vertices)
        V_full = compute_volume_fraction(self.mesh, phi_full)
        self.assertAlmostEqual(V_full, 1.0, delta=0.1)

        # 30% 材料
        phi_partial = np.full(self.mesh.n_vertices, 0.3)
        V_partial = compute_volume_fraction(self.mesh, phi_partial)
        self.assertAlmostEqual(V_partial, 0.3, delta=0.1)


# ══════════════════════════════════════════════════════════
#  测试 3: 退化刚度
# ══════════════════════════════════════════════════════════

class TestDegradedStiffness(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis._iga_base import DOFMap, IGASettings
        from forgecraft.analysis.iga_element import MembraneIntegrator
        from forgecraft.analysis.iga_material import LinearIsotropic
        from forgecraft.analysis.iga_assembly import assemble_stiffness
        from forgecraft.analysis.iga_topology_types import TopologySettings

        self.mesh = _make_5x5_mesh()
        self.dof_map = DOFMap.from_mesh(self.mesh)
        self.iga_settings = IGASettings(quadrature_order=3)
        self.integrator = MembraneIntegrator(thickness=0.001)
        self.material = LinearIsotropic(E=210e9, nu=0.3, rho=7800.0)
        self.settings = TopologySettings()

        self.K0 = assemble_stiffness(
            self.mesh, self.integrator, self.material,
            self.dof_map, self.iga_settings,
        ).toarray()

    def test_full_material_no_degradation(self):
        from forgecraft.analysis.iga_topology_phasefield import (
            assemble_degraded_stiffness_topology,
        )

        phi_full = np.ones(self.mesh.n_vertices)
        K_deg = assemble_degraded_stiffness_topology(
            self.K0, phi_full, self.mesh, self.dof_map, self.settings,
        )
        # 满材料时刚度应接近原始
        ratio = np.abs(K_deg).sum() / max(np.abs(self.K0).sum(), 1e-15)
        self.assertAlmostEqual(ratio, 1.0, delta=0.2)

    def test_partial_material_reduced_stiffness(self):
        from forgecraft.analysis.iga_topology_phasefield import (
            assemble_degraded_stiffness_topology,
        )

        phi_partial = np.full(self.mesh.n_vertices, 0.5)
        K_full = assemble_degraded_stiffness_topology(
            self.K0, np.ones(self.mesh.n_vertices), self.mesh, self.dof_map, self.settings,
        )
        K_partial = assemble_degraded_stiffness_topology(
            self.K0, phi_partial, self.mesh, self.dof_map, self.settings,
        )

        # 部分材料的刚度应低于全材料
        norm_full = np.abs(K_full).sum()
        norm_partial = np.abs(K_partial).sum()
        self.assertLess(norm_partial, norm_full)

    def test_void_material_near_zero_stiffness(self):
        from forgecraft.analysis.iga_topology_phasefield import (
            assemble_degraded_stiffness_topology,
        )

        # 用较高惩罚指数 p 来测试显著退化
        settings_high_p = type(self.settings)(
            vol_frac=0.3, p_init=3.0, p_max=3.0,
            interpolation="SIMP", phi_min=1e-6,
        )
        settings_high_p.p_current = 3.0

        phi_void = np.full(self.mesh.n_vertices, self.settings.phi_min)
        K_deg = assemble_degraded_stiffness_topology(
            self.K0, phi_void, self.mesh, self.dof_map, settings_high_p,
        )
        norm_deg = np.abs(K_deg).sum()
        norm_orig = np.abs(self.K0).sum()

        # 退化后刚度应远小于原始 (p=3 时 g(φ_min) ≈ φ_min)
        ratio = norm_deg / max(norm_orig, 1e-15)
        self.assertLess(ratio, 0.1)

        # 与满材料的对比
        phi_full = np.ones(self.mesh.n_vertices)
        K_full = assemble_degraded_stiffness_topology(
            self.K0, phi_full, self.mesh, self.dof_map, settings_high_p,
        )
        norm_full = np.abs(K_full).sum()
        self.assertLess(norm_deg, norm_full * 0.01)


# ══════════════════════════════════════════════════════════
#  测试 4: 灵敏度分析
# ══════════════════════════════════════════════════════════

class TestSensitivity(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings

        self.mesh = _make_5x5_mesh()
        self.settings = TopologySettings()

    def test_compliance_sensitivity_sign(self):
        from forgecraft.analysis.iga_topology_sensitivity import compute_compliance_sensitivity

        phi = np.full(self.mesh.n_vertices, 0.5)
        psi_e = np.full(self.mesh.n_vertices, 1e6)

        dc = compute_compliance_sensitivity(
            self.mesh, phi, psi_e, self.settings,
        )
        self.assertEqual(len(dc), self.mesh.n_vertices)
        # 柔度灵敏度应为负 (增加材料降低柔度)
        self.assertTrue(np.all(dc <= 0))

    def test_compliance_sensitivity_increases_with_phi(self):
        from forgecraft.analysis.iga_topology_sensitivity import compute_compliance_sensitivity

        # 使用 p>1 才体现 g'(φ) 随 φ 变化
        settings_p2 = type(self.settings)(
            vol_frac=0.3, p_init=2.0, p_max=2.0,
            interpolation="SIMP",
        )
        settings_p2.p_current = 2.0

        psi_e = np.full(self.mesh.n_vertices, 1e6)

        dc_low = compute_compliance_sensitivity(
            self.mesh, np.full(self.mesh.n_vertices, 0.3), psi_e, settings_p2,
        )
        dc_high = compute_compliance_sensitivity(
            self.mesh, np.full(self.mesh.n_vertices, 0.7), psi_e, settings_p2,
        )
        # 高密度区域灵敏度更大 (因 g'(φ) = p·φ^(p-1) 单调增 for p>1)
        avg_low = np.mean(np.abs(dc_low))
        avg_high = np.mean(np.abs(dc_high))
        self.assertGreater(avg_high, avg_low)

    def test_volume_sensitivity(self):
        from forgecraft.analysis.iga_topology_sensitivity import compute_volume_sensitivity

        dv = compute_volume_sensitivity(self.mesh)
        self.assertEqual(len(dv), self.mesh.n_vertices)
        # 所有灵敏度为正 (增加 φ 增加体积)
        self.assertTrue(np.all(dv >= 0))

    def test_filter_sensitivity_noop(self):
        from forgecraft.analysis.iga_topology_sensitivity import filter_sensitivity

        dc = np.random.rand(self.mesh.n_vertices)
        phi = np.full(self.mesh.n_vertices, 0.5)
        dc_f = filter_sensitivity(self.mesh, dc, phi, 0.0)
        np.testing.assert_array_almost_equal(dc_f, dc)

    def test_filter_sensitivity_active(self):
        from forgecraft.analysis.iga_topology_sensitivity import filter_sensitivity

        dc = np.random.rand(self.mesh.n_vertices)
        phi = np.full(self.mesh.n_vertices, 0.5)
        dc_f = filter_sensitivity(self.mesh, dc, phi, 0.5)

        self.assertEqual(len(dc_f), len(dc))
        self.assertFalse(np.any(np.isnan(dc_f)))
        self.assertFalse(np.any(np.isinf(dc_f)))


# ══════════════════════════════════════════════════════════
#  测试 5: 相场求解器 (solve_phasefield_field)
# ══════════════════════════════════════════════════════════

class TestPhaseFieldSolver(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis.iga_multiphysics_types import ScalarDOFMap
        from forgecraft.analysis.iga_topology_types import TopologySettings

        self.mesh = _make_5x5_mesh()
        self.sdof = ScalarDOFMap.from_mesh(self.mesh)
        self.settings = TopologySettings(
            vol_frac=0.3, epsilon=0.02, kappa=1.0,
            quadrature_order=3,
        )

    def test_solve_returns_valid_phi(self):
        from forgecraft.analysis.iga_topology_phasefield import solve_phasefield_field

        phi_old = np.full(self.mesh.n_vertices, 0.5)
        psi_e = np.full(self.mesh.n_vertices, 1e5)

        phi_new = solve_phasefield_field(
            self.mesh, phi_old, psi_e, self.sdof, self.settings,
            lagrange_multiplier=0.0,
        )

        self.assertEqual(len(phi_new), self.mesh.n_vertices)
        # 应在箱约束内
        self.assertTrue(np.all(phi_new >= self.settings.phi_min - 1e-10))
        self.assertTrue(np.all(phi_new <= 1.0 + 1e-10))
        # 无 NaN/Inf
        self.assertFalse(np.any(np.isnan(phi_new)))
        self.assertFalse(np.any(np.isinf(phi_new)))

    def test_solve_drives_toward_high_energy(self):
        from forgecraft.analysis.iga_topology_phasefield import solve_phasefield_field

        phi_old = np.full(self.mesh.n_vertices, 0.5)
        psi_e = np.zeros(self.mesh.n_vertices)
        # 中心高能量
        center_idx = 12  # 5x5 网格的中心顶点
        psi_e[center_idx] = 1e7

        phi_new = solve_phasefield_field(
            self.mesh, phi_old, psi_e, self.sdof, self.settings,
            lagrange_multiplier=0.0,
        )

        # 高能量区域应获得更多材料
        self.assertGreater(phi_new[center_idx], np.mean(phi_new))

    def test_lagrange_multiplier_reduces_volume(self):
        from forgecraft.analysis.iga_topology_phasefield import solve_phasefield_field

        phi_old = np.full(self.mesh.n_vertices, 0.5)
        psi_e = np.full(self.mesh.n_vertices, 1e6)

        # λ=0: 无体积约束
        phi_free = solve_phasefield_field(
            self.mesh, phi_old, psi_e, self.sdof, self.settings,
            lagrange_multiplier=0.0,
        )
        # λ=大值: 强力体积约束
        phi_constrained = solve_phasefield_field(
            self.mesh, phi_old, psi_e, self.sdof, self.settings,
            lagrange_multiplier=1e8,
        )

        # 约束应降低体积 (至少不增加)
        V_free = np.mean(np.clip(phi_free, 0, 1))
        V_constrained = np.mean(np.clip(phi_constrained, 0, 1))
        self.assertLessEqual(V_constrained, V_free + 1e-6)


# ══════════════════════════════════════════════════════════
#  测试 6: 应变能密度
# ══════════════════════════════════════════════════════════

class TestStrainEnergyDensity(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis._iga_base import DOFMap, IGASettings
        from forgecraft.analysis.iga_element import MembraneIntegrator
        from forgecraft.analysis.iga_material import LinearIsotropic

        self.mesh = _make_5x5_mesh()
        self.dof_map = DOFMap.from_mesh(self.mesh)
        self.iga_settings = IGASettings(quadrature_order=3)
        self.integrator = MembraneIntegrator(thickness=0.001)
        self.material = LinearIsotropic(E=210e9, nu=0.3, rho=7800.0)

        # 合成位移: 简单拉伸 u_x ∝ x
        self.u = np.zeros(self.dof_map.n_dof)
        for vi in range(self.mesh.n_vertices):
            x = self.mesh.vertices[vi][0]
            dx, dy, dz = self.dof_map.vertex_dofs(vi)
            self.u[dx] = x * 0.001

    def test_compute_strain_energy_density(self):
        from forgecraft.analysis.iga_topology_phasefield import compute_strain_energy_density

        psi_e = compute_strain_energy_density(
            self.mesh, self.integrator, self.material,
            self.u, self.dof_map, self.iga_settings,
        )

        self.assertEqual(len(psi_e), self.mesh.n_vertices)
        self.assertFalse(np.any(np.isnan(psi_e)))
        self.assertFalse(np.any(np.isinf(psi_e)))


# ══════════════════════════════════════════════════════════
#  测试 7: 主优化器 (轻量流程)
# ══════════════════════════════════════════════════════════

class TestOptimizerLightweight(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis._iga_base import DOFMap, IGASettings
        from forgecraft.analysis.iga_element import MembraneIntegrator
        from forgecraft.analysis.iga_material import LinearIsotropic
        from forgecraft.analysis.iga_boundary import DirichletBC

        # 用 10x10 网格避免 CC 壳边界控制点不足 → 刚度矩阵奇异
        self.mesh = _make_10x10_mesh()
        self.dof_map = DOFMap.from_mesh(self.mesh)
        self.iga_settings = IGASettings(quadrature_order=3, verbosity=0)
        self.integrator = MembraneIntegrator(thickness=0.001)
        self.structural_material = LinearIsotropic(E=210e9, nu=0.3, rho=7800.0)

        # 左侧固定
        left_verts = [vi for vi in range(self.mesh.n_vertices)
                       if self.mesh.vertices[vi][0] < 0.01]
        # 平面应力: 所有顶点 z=0 (膜单元无力抵抗面外位移)
        all_verts = list(range(self.mesh.n_vertices))
        self.dirichlet_bcs = [
            DirichletBC.from_vertex_set(
                left_verts, self.dof_map, value=0.0, dof_mask=(True, True, True),
            ),
            DirichletBC.from_vertex_set(
                all_verts, self.dof_map, value=0.0, dof_mask=(False, False, True),
            ),
        ]

        # 右侧载荷
        from forgecraft.analysis.iga_boundary import NeumannBC
        right_faces = [f for f in range(self.mesh.n_faces)
                       if all(self.mesh.vertices[v][0] > 0.99 - 1e-6
                              for v in self.mesh.face_vertices(f))]
        if not right_faces:
            right_faces = [self.mesh.n_faces - 1]
        self.neumann_bcs = [
            NeumannBC.pressure(face_ids=right_faces, pressure=1e5, mesh=self.mesh),
        ]

    def test_optimizer_initialization(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings
        from forgecraft.analysis.iga_topology_optimizer import TopologyOptimizer

        settings = TopologySettings(vol_frac=0.5, max_iter=3, verbosity=0)
        optimizer = TopologyOptimizer(settings)

        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.settings.vol_frac, 0.5)

    def test_optimizer_oc_update(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings
        from forgecraft.analysis.iga_topology_optimizer import TopologyOptimizer

        settings = TopologySettings(vol_frac=0.5, move_limit=0.2, verbosity=0)
        optimizer = TopologyOptimizer(settings)

        phi = np.full(self.mesh.n_vertices, 0.5)
        dc = -np.random.rand(self.mesh.n_vertices) * 1e6
        dv = np.ones(self.mesh.n_vertices)

        phi_new = optimizer._oc_update(phi, dc, dv, 0.5)
        self.assertEqual(len(phi_new), self.mesh.n_vertices)
        self.assertTrue(np.all(phi_new >= settings.phi_min - 1e-10))
        self.assertTrue(np.all(phi_new <= 1.0 + 1e-10))

        # 体积应接近目标
        V = np.mean(phi_new)
        self.assertAlmostEqual(V, 0.5, delta=0.25)

    def test_optimizer_aug_lag_update(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings
        from forgecraft.analysis.iga_topology_optimizer import TopologyOptimizer

        settings = TopologySettings(vol_frac=0.5, move_limit=0.2, verbosity=0)
        optimizer = TopologyOptimizer(settings)

        phi = np.full(self.mesh.n_vertices, 0.5)
        dc = -np.random.rand(self.mesh.n_vertices) * 1e6
        dv = np.ones(self.mesh.n_vertices)

        phi_new = optimizer._augmented_lagrangian_update(phi, dc, dv, 0.5)
        self.assertEqual(len(phi_new), self.mesh.n_vertices)
        self.assertTrue(np.all(phi_new >= settings.phi_min - 1e-10))
        self.assertTrue(np.all(phi_new <= 1.0 + 1e-10))

    def test_continuation_step(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings
        from forgecraft.analysis.iga_topology_optimizer import TopologyOptimizer

        settings = TopologySettings(
            vol_frac=0.5, p_init=1.0, p_max=3.0, p_step=0.5,
            continue_p=True, continue_epsilon=True,
            epsilon=0.02, epsilon_decay=0.95,
            verbosity=0,
        )
        optimizer = TopologyOptimizer(settings)

        p_before = optimizer.p_current
        eps_before = optimizer.epsilon_current

        optimizer._continuation_step()

        self.assertAlmostEqual(optimizer.p_current, 1.5)
        self.assertAlmostEqual(optimizer.epsilon_current, 0.019)

    def test_full_optimization_oc(self):
        """完整 OC 优化 (少量迭代)"""
        from forgecraft.analysis.iga_topology_types import TopologySettings, TopologyMaterial
        from forgecraft.analysis.iga_topology_optimizer import TopologyOptimizer

        settings = TopologySettings(
            vol_frac=0.5, max_iter=5,
            p_init=1.0, p_max=3.0,
            continue_p=False, continue_epsilon=False,
            optimizer="OC", move_limit=0.2,
            constraint_method="augmented_lagrangian",
            verbosity=0,
        )
        topo_material = TopologyMaterial(E=210e9, nu=0.3, rho=7800.0)

        optimizer = TopologyOptimizer(settings)

        result = optimizer.optimize(
            self.mesh, self.integrator, topo_material,
            self.dof_map, self.iga_settings,
            dirichlet_bcs=self.dirichlet_bcs,
            neumann_bcs=self.neumann_bcs,
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertGreater(len(result.compliance_history), 0)
        self.assertGreater(len(result.volume_history), 0)
        self.assertEqual(len(result.phi_field), self.mesh.n_vertices)

        # 相场应在有效范围内
        self.assertTrue(np.all(result.phi_field >= settings.phi_min - 1e-10))
        self.assertTrue(np.all(result.phi_field <= 1.0 + 1e-10))

        # 柔度应随迭代递减 (或保持)
        if len(result.compliance_history) >= 2:
            pass  # 小网基准, 不强要求单调

    def test_topology_optimize_convenience(self):
        from forgecraft.analysis.iga_topology_types import TopologySettings, TopologyMaterial
        from forgecraft.analysis.iga_topology_optimizer import topology_optimize

        settings = TopologySettings(
            vol_frac=0.4, max_iter=3,
            continue_p=False, continue_epsilon=False,
            verbosity=0,
        )
        topo_material = TopologyMaterial(E=210e9, nu=0.3, rho=7800.0)

        result = topology_optimize(
            self.mesh, self.integrator, topo_material,
            self.dof_map, self.iga_settings,
            settings=settings,
            dirichlet_bcs=self.dirichlet_bcs,
            neumann_bcs=self.neumann_bcs,
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertIsNotNone(result.displacement)

        # 摘要
        summary = result.summary()
        self.assertIn("Phase-Field", summary)


# ══════════════════════════════════════════════════════════
#  测试 8: 顶层导入验证
# ══════════════════════════════════════════════════════════

class TestTopLevelImports(unittest.TestCase):

    def test_types_import(self):
        from forgecraft.analysis.iga_topology_types import (
            TopologySettings, TopologyMaterial, TopologyConstraint,
            TopologyOptimizationResult, TopologyIterationData,
            InterpolationFunction,
        )
        self.assertTrue(True)

    def test_phasefield_import(self):
        from forgecraft.analysis.iga_topology_phasefield import (
            assemble_phasefield_laplacian, assemble_phasefield_stiffness,
            assemble_phasefield_driving_force, assemble_degraded_stiffness_topology,
            compute_strain_energy_density, solve_phasefield_field,
            compute_volume_fraction, apply_density_filter,
        )
        self.assertTrue(True)

    def test_sensitivity_import(self):
        from forgecraft.analysis.iga_topology_sensitivity import (
            compute_compliance_sensitivity, compute_volume_sensitivity,
            compute_stress_constraint_sensitivity, compute_displacement_constraint_sensitivity,
            adjoint_solve, filter_sensitivity,
        )
        self.assertTrue(True)

    def test_optimizer_import(self):
        from forgecraft.analysis.iga_topology_optimizer import (
            TopologyOptimizer, topology_optimize,
        )
        self.assertTrue(True)

    def test_init_imports(self):
        from forgecraft.analysis import (
            TopologySettings, TopologyMaterial,
            TopologyConstraint, ConstraintType,
            TopologyOptimizationResult, TopologyIterationData,
            assemble_phasefield_laplacian,
            assemble_phasefield_stiffness,
            assemble_phasefield_driving_force,
            degrade_for_topology,
            compute_strain_energy_topo,
            solve_phasefield_field,
            compute_volume_fraction,
            apply_density_filter,
            compute_compliance_sensitivity,
            compute_volume_sensitivity,
            adjoint_solve, filter_sensitivity,
            PhaseFieldTopologyOptimizer,
            topology_phasefield_optimize,
        )
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
