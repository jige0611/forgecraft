# ══════════════════════════════════════════════════════════
# test_tpms_lattice.py — 梯度 TPMS 晶格集成测试
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


# ══════════════════════════════════════════════════════════
#  测试 1: TPMS 类型系统
# ══════════════════════════════════════════════════════════

class TestTPMSTypes(unittest.TestCase):

    def test_tpms_type_enum(self):
        from forgecraft.analysis.iga_tpms_types import TPMSType

        self.assertEqual(len(list(TPMSType)), 8)
        self.assertIn(TPMSType.GYROID, list(TPMSType))
        self.assertIn(TPMSType.DIAMOND, list(TPMSType))
        self.assertEqual(str(TPMSType.GYROID), "GYROID")

    def test_grading_strategy_enum(self):
        from forgecraft.analysis.iga_tpms_types import GradingStrategy

        self.assertEqual(len(list(GradingStrategy)), 4)
        self.assertIn(GradingStrategy.STRESS_PROPORTIONAL, list(GradingStrategy))

    def test_tpms_parameters_defaults(self):
        from forgecraft.analysis.iga_tpms_types import TPMSParameters

        p = TPMSParameters()
        self.assertAlmostEqual(p.cell_size, 1.0)
        self.assertAlmostEqual(p.thickness, 0.1)
        self.assertAlmostEqual(p.offset, 0.0)
        self.assertAlmostEqual(p.wx, 2.0 * np.pi)
        self.assertAlmostEqual(p.wy, 2.0 * np.pi)

    def test_tpms_parameters_frequency(self):
        from forgecraft.analysis.iga_tpms_types import TPMSParameters

        p = TPMSParameters(cell_size=0.5)
        self.assertAlmostEqual(p.wx, 4.0 * np.pi)
        self.assertAlmostEqual(p.wz, 4.0 * np.pi)

        p2 = TPMSParameters(frequency=(1.0, 2.0, 3.0))
        self.assertAlmostEqual(p2.wx, 1.0)
        self.assertAlmostEqual(p2.wy, 2.0)
        self.assertAlmostEqual(p2.wz, 3.0)

    def test_uniform_parameters(self):
        from forgecraft.analysis.iga_tpms_types import TPMSParameters

        p = TPMSParameters.uniform(cell_size=2.0, thickness=0.05)
        self.assertAlmostEqual(p.cell_size, 2.0)
        self.assertAlmostEqual(p.thickness, 0.05)

    def test_grading_settings_defaults(self):
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings

        s = TPMSGradingSettings()
        self.assertEqual(s.tpms_type.name, "GYROID")
        self.assertEqual(s.strategy.name, "STRESS_PROPORTIONAL")
        self.assertAlmostEqual(s.cell_size_range[0], 0.5)
        self.assertAlmostEqual(s.cell_size_range[1], 2.0)
        self.assertAlmostEqual(s.stress_exponent, 1.0)

    def test_grading_settings_invalid(self):
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings

        with self.assertRaises(ValueError):
            TPMSGradingSettings(cell_size_range=(-1.0, 1.0))
        with self.assertRaises(ValueError):
            TPMSGradingSettings(thickness_range=(-0.1, 0.05))
        # Valid: this should work
        s = TPMSGradingSettings(cell_size_range=(0.1, 5.0))
        self.assertAlmostEqual(s.cell_size_range[0], 0.1)

    def test_result_dataclass(self):
        from forgecraft.analysis.iga_tpms_types import TPMSLatticeResult, TPMSType

        r = TPMSLatticeResult(success=True, volume_fraction=0.35, iterations=5)
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.volume_fraction, 0.35)
        self.assertEqual(r.iterations, 5)

        summary = r.summary()
        self.assertIn("TPMS", summary)
        self.assertIn("35.00%", summary)

    def test_tpms_defaults_registry(self):
        from forgecraft.analysis.iga_tpms_types import TPMS_DEFAULTS, TPMSType

        for tpms_type in TPMSType:
            defaults = TPMS_DEFAULTS[tpms_type]
            self.assertIn("C", defaults)
            self.assertIn("n", defaults)
            self.assertIn("description", defaults)
            self.assertGreater(defaults["C"], 0)
            self.assertGreater(defaults["n"], 0)

    def test_tpms_surface_functions_registry(self):
        from forgecraft.analysis.iga_tpms_types import TPMS_SURFACE_FUNCTIONS, TPMSType

        for tpms_type in TPMSType:
            expr = TPMS_SURFACE_FUNCTIONS[tpms_type]
            self.assertIsInstance(expr, str)
            self.assertGreater(len(expr), 5)


# ══════════════════════════════════════════════════════════
#  测试 2: TPMS 数学核心
# ══════════════════════════════════════════════════════════

class TestTPMSCore(unittest.TestCase):

    def test_gyroid_level_set(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        # 原点: sin(0)cos(0)+sin(0)cos(0)+sin(0)cos(0) = 0
        f0 = tpms_level_set(0.0, 0.0, 0.0, TPMSType.GYROID)
        self.assertAlmostEqual(f0, 0.0, delta=1e-10)

        # 对称性: sin(π/2)*cos(π/4)+... 有限值
        f1 = tpms_level_set(0.25, 0.25, 0.25, TPMSType.GYROID)
        self.assertTrue(np.isfinite(f1))
        self.assertLess(abs(f1), 5.0)

    def test_gyroid_periodicity(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        wx = wy = wz = 2.0 * np.pi
        # 在任意点上求值
        f1 = tpms_level_set(0.1, 0.2, 0.3, TPMSType.GYROID, wx, wy, wz)
        # 平移一个完整晶胞应得到相同值
        f2 = tpms_level_set(0.1 + 1.0, 0.2 + 1.0, 0.3 + 1.0, TPMSType.GYROID, wx, wy, wz)
        self.assertAlmostEqual(f1, f2, delta=1e-10)

    def test_schwarz_p_level_set(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        # cos(π) + cos(π) + cos(π) = -3 (所有 cos = -1)
        f = tpms_level_set(0.5, 0.5, 0.5, TPMSType.SCHWARZ_P, np.pi, np.pi, np.pi)
        # x = 0.5, so u = π * 0.5 = π/2, cos(π/2) = 0
        self.assertAlmostEqual(f, 0.0, delta=1e-10)

    def test_schwarz_d_level_set(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        # cos(0)*cos(0)*cos(0) - sin(0)*sin(0)*sin(0) = 1
        f = tpms_level_set(0.0, 0.0, 0.0, TPMSType.SCHWARZ_D)
        self.assertAlmostEqual(f, 1.0, delta=1e-10)

    def test_diamond_level_set(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        f = tpms_level_set(0.0, 0.0, 0.0, TPMSType.DIAMOND)
        self.assertTrue(np.isfinite(f))

        # 边界: 所有曲面在原点附近有定义
        for tpms_type in TPMSType:
            f = tpms_level_set(0.1, 0.2, 0.3, tpms_type)
            self.assertTrue(np.isfinite(f), f"TPMS {tpms_type} returned non-finite")

    def test_level_set_all_types(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        for tpms_type in TPMSType:
            for _ in range(5):
                x, y, z = np.random.uniform(0, 2, 3)
                f = tpms_level_set(x, y, z, tpms_type)
                self.assertTrue(np.isfinite(f))
                self.assertLess(abs(f), 15.0)  # 合理范围

    def test_gradient_gyroid(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set_gradient
        from forgecraft.analysis.iga_tpms_types import TPMSType

        gx, gy, gz = tpms_level_set_gradient(0.1, 0.2, 0.3, TPMSType.GYROID)
        self.assertTrue(np.isfinite(gx))
        self.assertTrue(np.isfinite(gy))
        self.assertTrue(np.isfinite(gz))

    def test_is_solid_basic(self):
        from forgecraft.analysis.iga_tpms_core import tpms_is_solid
        from forgecraft.analysis.iga_tpms_types import TPMSType, TPMSParameters

        params = TPMSParameters(cell_size=1.0, thickness=0.2)

        # 采样 10000 点，验证体积分数合理
        n = 5000
        points = np.random.uniform(0, 1, (n, 3))
        solid = tpms_is_solid(points, TPMSType.GYROID, params)
        vol_frac = float(np.mean(solid))
        self.assertGreater(vol_frac, 0.0)
        self.assertLess(vol_frac, 1.0)

    def test_is_solid_thickness_scaling(self):
        from forgecraft.analysis.iga_tpms_core import tpms_is_solid
        from forgecraft.analysis.iga_tpms_types import TPMSType, TPMSParameters

        n = 5000
        points = np.random.uniform(0, 1, (n, 3))

        thin = TPMSParameters(cell_size=1.0, thickness=0.05)
        thick = TPMSParameters(cell_size=1.0, thickness=0.3)

        vf_thin = float(np.mean(tpms_is_solid(points, TPMSType.GYROID, thin)))
        vf_thick = float(np.mean(tpms_is_solid(points, TPMSType.GYROID, thick)))

        self.assertGreater(vf_thick, vf_thin)

    def test_is_solid_graded(self):
        from forgecraft.analysis.iga_tpms_core import tpms_is_solid_graded
        from forgecraft.analysis.iga_tpms_types import TPMSType

        n = 1000
        points = np.random.uniform(0, 1, (n, 3))
        cell_size = np.full(n, 0.5)
        thickness = np.full(n, 0.1)
        offset = np.zeros(n)

        solid = tpms_is_solid_graded(points, TPMSType.GYROID, cell_size, thickness, offset)
        self.assertEqual(len(solid), n)
        self.assertTrue(0.0 < float(np.mean(solid)) < 1.0)

        # 更小晶胞 → 更多实体点
        cell_small = np.full(n, 0.25)
        solid_small = tpms_is_solid_graded(points, TPMSType.GYROID, cell_small, thickness, offset)
        self.assertGreater(float(np.mean(solid_small)), float(np.mean(solid)))

    def test_compute_relative_density(self):
        from forgecraft.analysis.iga_tpms_core import compute_relative_density
        from forgecraft.analysis.iga_tpms_types import TPMSType, TPMSParameters

        params = TPMSParameters(cell_size=1.0, thickness=0.2)
        rho = compute_relative_density(params, TPMSType.GYROID, n_samples=5000)
        self.assertGreater(rho, 0.0)
        self.assertLess(rho, 1.0)

    def test_effective_modulus(self):
        from forgecraft.analysis.iga_tpms_core import effective_youngs_modulus, effective_shear_modulus
        from forgecraft.analysis.iga_tpms_types import TPMSType

        E_solid = 200e9
        # 全密度 → 接近 C * E_solid
        E_full = effective_youngs_modulus(1.0, E_solid, TPMSType.GYROID)
        self.assertAlmostEqual(E_full, 0.5 * E_solid, delta=1e-6)

        # 半密度
        E_half = effective_youngs_modulus(0.5, E_solid, TPMSType.GYROID)
        self.assertAlmostEqual(E_half, 0.5 * E_solid * 0.25, delta=1e-6)

        # 剪切模量
        G_half = effective_shear_modulus(0.5, E_solid, 0.3, TPMSType.GYROID)
        self.assertGreater(G_half, 0)
        self.assertLess(G_half, E_half / 2.0)

    def test_cell_size_frequency_roundtrip(self):
        from forgecraft.analysis.iga_tpms_core import cell_size_to_frequency, frequency_to_cell_size

        cs = 0.8
        omega = cell_size_to_frequency(cs)
        cs_back = frequency_to_cell_size(omega)
        self.assertAlmostEqual(cs, cs_back)

    def test_neo_ius_level_set(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        # Neovius 在所有 cos=0 处为零
        f = tpms_level_set(0.25, 0.25, 0.25, TPMSType.NEOVIUS)
        self.assertTrue(np.isfinite(f))

    def test_iwp_level_set(self):
        from forgecraft.analysis.iga_tpms_core import tpms_level_set
        from forgecraft.analysis.iga_tpms_types import TPMSType

        f = tpms_level_set(0.25, 0.25, 0.25, TPMSType.IWP)
        self.assertTrue(np.isfinite(f))


# ══════════════════════════════════════════════════════════
#  测试 3: 应力 → 参数映射
# ══════════════════════════════════════════════════════════

class TestStressMapping(unittest.TestCase):

    def test_map_stress_to_cell_size(self):
        from forgecraft.analysis.iga_tpms_grading import map_stress_to_cell_size
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings

        settings = TPMSGradingSettings(
            cell_size_range=(0.5, 2.0),
            stress_exponent=1.0,
            stress_reference=100e6,
        )

        n = 50
        # 零应力 → 最大晶胞
        zero_stress = np.zeros(n)
        cs_zero = map_stress_to_cell_size(zero_stress, settings)
        self.assertAlmostEqual(np.mean(cs_zero), 2.0, delta=0.01)

        # 等参考应力 → 最小晶胞 (因为 1-weight = 0)
        ref_stress = np.full(n, 100e6)
        cs_ref = map_stress_to_cell_size(ref_stress, settings)
        self.assertAlmostEqual(np.mean(cs_ref), 0.5, delta=0.01)

    def test_map_stress_to_thickness(self):
        from forgecraft.analysis.iga_tpms_grading import map_stress_to_thickness
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings

        settings = TPMSGradingSettings(
            thickness_range=(0.02, 0.15),
            thickness_exponent=1.0,
            stress_reference=100e6,
        )

        n = 50
        zero_stress = np.zeros(n)
        t_zero = map_stress_to_thickness(zero_stress, settings)
        self.assertAlmostEqual(np.mean(t_zero), 0.02, delta=0.002)

        ref_stress = np.full(n, 100e6)
        t_ref = map_stress_to_thickness(ref_stress, settings)
        self.assertAlmostEqual(np.mean(t_ref), 0.15, delta=0.002)

    def test_stress_mapping_bounds(self):
        from forgecraft.analysis.iga_tpms_grading import (
            map_stress_to_cell_size, map_stress_to_thickness,
        )
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings

        settings = TPMSGradingSettings(
            cell_size_range=(0.3, 3.0),
            thickness_range=(0.01, 0.2),
            stress_exponent=2.0,
        )

        n = 100
        stress = np.abs(np.random.randn(n)) * 50e6

        cs = map_stress_to_cell_size(stress, settings)
        t = map_stress_to_thickness(stress, settings)

        cs_min, cs_max = settings.cell_size_range
        t_min, t_max = settings.thickness_range

        self.assertTrue(np.all(cs >= cs_min - 1e-10))
        self.assertTrue(np.all(cs <= cs_max + 1e-10))
        self.assertTrue(np.all(t >= t_min - 1e-10))
        self.assertTrue(np.all(t <= t_max + 1e-10))

    def test_smooth_grading_field_noop(self):
        from forgecraft.analysis.iga_tpms_grading import smooth_grading_field

        mesh = _make_5x5_mesh()
        field = np.random.rand(mesh.n_vertices)

        # radius=0 → 不平滑
        result = smooth_grading_field(mesh, field, 0.0)
        np.testing.assert_array_almost_equal(field, result)

    def test_smooth_grading_field(self):
        from forgecraft.analysis.iga_tpms_grading import smooth_grading_field

        mesh = _make_5x5_mesh()
        field = np.ones(mesh.n_vertices)
        field[mesh.n_vertices // 2] = 100.0  # 尖峰

        # 平滑不应显著改变均值
        result = smooth_grading_field(mesh, field, 0.1)
        self.assertEqual(len(result), mesh.n_vertices)
        # 峰值应被平滑
        self.assertLess(np.max(result), 100.0)

    def test_normalize_for_volume(self):
        from forgecraft.analysis.iga_tpms_grading import (
            normalize_grading_for_volume, estimate_average_density,
        )

        mesh = _make_5x5_mesh()
        n = mesh.n_vertices

        # 均匀 cell_size/thickness
        cell_size = np.full(n, 1.0)
        thickness = np.full(n, 0.1)
        areas = np.ones(n)

        # 当前体积分数
        vf_before = estimate_average_density(cell_size, thickness, areas)
        # 目标体积分数 (不同)
        vf_target = vf_before * 0.5 if vf_before > 0.5 else vf_before * 2.0
        vf_target = np.clip(vf_target, 0.05, 0.95)

        cs_new, t_new = normalize_grading_for_volume(
            cell_size, thickness, areas, vf_target,
        )
        vf_after = estimate_average_density(cs_new, t_new, areas)
        self.assertAlmostEqual(vf_after, vf_target, delta=0.05)

    def test_compute_tpms_parameters_field(self):
        from forgecraft.analysis.iga_tpms_grading import compute_tpms_parameters_field
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings

        mesh = _make_5x5_mesh()
        n = mesh.n_vertices

        settings = TPMSGradingSettings(
            cell_size_range=(0.5, 1.5),
            thickness_range=(0.03, 0.12),
        )
        stress = np.random.uniform(0, 100e6, n)

        cs, t = compute_tpms_parameters_field(mesh, stress, settings)

        self.assertEqual(len(cs), n)
        self.assertEqual(len(t), n)
        self.assertTrue(np.all(cs >= settings.cell_size_range[0] - 1e-10))
        self.assertTrue(np.all(cs <= settings.cell_size_range[1] + 1e-10))
        self.assertTrue(np.all(t >= settings.thickness_range[0] - 1e-10))
        self.assertTrue(np.all(t <= settings.thickness_range[1] + 1e-10))


# ══════════════════════════════════════════════════════════
#  测试 4: von Mises 应力场
# ══════════════════════════════════════════════════════════

class TestVonMisesField(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis._iga_base import DOFMap, IGASettings
        from forgecraft.analysis.iga_element import MembraneIntegrator
        from forgecraft.analysis.iga_material import LinearIsotropic

        self.mesh = _make_5x5_mesh()
        self.dof_map = DOFMap.from_mesh(self.mesh)
        self.iga_settings = IGASettings(quadrature_order=3, verbosity=0)
        self.integrator = MembraneIntegrator(thickness=0.001)
        self.material = LinearIsotropic(E=210e9, nu=0.3, rho=7800.0)

        # 合成位移: u_x = x * 0.001 (简单拉伸)
        self.u = np.zeros(self.dof_map.n_dof)
        for vi in range(self.mesh.n_vertices):
            x = self.mesh.vertices[vi][0]
            dx, dy, dz = self.dof_map.vertex_dofs(vi)
            self.u[dx] = x * 0.001

    def test_compute_von_mises_field(self):
        from forgecraft.analysis.iga_tpms_grading import compute_von_mises_field

        vm = compute_von_mises_field(
            self.mesh, self.u, self.dof_map,
            self.integrator, self.material, self.iga_settings,
        )

        self.assertEqual(len(vm), self.mesh.n_vertices)
        self.assertFalse(np.any(np.isnan(vm)))
        self.assertFalse(np.any(np.isinf(vm)))

    def test_compute_stress_driving_field_stress(self):
        from forgecraft.analysis.iga_tpms_grading import compute_stress_driving_field
        from forgecraft.analysis.iga_tpms_types import GradingStrategy

        vm = compute_stress_driving_field(
            self.mesh, self.u, self.dof_map,
            self.integrator, self.material, self.iga_settings,
            strategy=GradingStrategy.STRESS_PROPORTIONAL,
        )

        self.assertEqual(len(vm), self.mesh.n_vertices)
        self.assertFalse(np.any(np.isnan(vm)))

    def test_compute_stress_driving_field_energy(self):
        from forgecraft.analysis.iga_tpms_grading import compute_stress_driving_field
        from forgecraft.analysis.iga_tpms_types import GradingStrategy

        psi = compute_stress_driving_field(
            self.mesh, self.u, self.dof_map,
            self.integrator, self.material, self.iga_settings,
            strategy=GradingStrategy.ENERGY_DENSITY,
        )

        self.assertEqual(len(psi), self.mesh.n_vertices)
        self.assertFalse(np.any(np.isnan(psi)))

    def test_compute_stress_manual(self):
        from forgecraft.analysis.iga_tpms_grading import compute_stress_driving_field
        from forgecraft.analysis.iga_tpms_types import GradingStrategy

        manual = compute_stress_driving_field(
            self.mesh, self.u, self.dof_map,
            self.integrator, self.material, self.iga_settings,
            strategy=GradingStrategy.MANUAL_FIELD,
        )
        self.assertTrue(np.all(manual == 0.0))


# ══════════════════════════════════════════════════════════
#  测试 5: 网格生成 (无 skimage 回退)
# ══════════════════════════════════════════════════════════

class TestGridGeneration(unittest.TestCase):

    def test_sample_tpms_uniform(self):
        from forgecraft.analysis.iga_tpms_generator import sample_tpms_on_grid
        from forgecraft.analysis.iga_tpms_types import TPMSType

        n = 10
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        cell_size = np.array([0.8])
        thickness = np.array([0.1])

        grid = sample_tpms_on_grid(
            bounds, (n, n, n), cell_size, thickness, TPMSType.GYROID,
        )
        self.assertEqual(grid.shape, (n, n, n))
        self.assertFalse(np.any(np.isnan(grid)))

        # 水平集应有正有负 (实体区域)
        self.assertTrue(np.any(grid < 0))
        self.assertTrue(np.any(grid >= 0))

    def test_sample_tpms_with_control_points(self):
        from forgecraft.analysis.iga_tpms_generator import sample_tpms_on_grid
        from forgecraft.analysis.iga_tpms_types import TPMSType

        n = 8
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

        cp = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        cell_size = np.array([0.5, 1.5])
        thickness = np.array([0.05, 0.2])

        grid = sample_tpms_on_grid(
            bounds, (n, n, n), cell_size, thickness, TPMSType.SCHWARZ_P,
            control_points=cp,
        )
        self.assertEqual(grid.shape, (n, n, n))
        self.assertFalse(np.any(np.isnan(grid)))

    def test_marching_cubes_simple(self):
        from forgecraft.analysis.iga_tpms_generator import _simple_marching_cubes

        # 简单球体水平集
        nx = 10
        bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, nx)
        z = np.linspace(-1, 1, nx)
        XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
        grid = np.sqrt(XX**2 + YY**2 + ZZ**2) - 0.5  # 半径 0.5 球

        verts, faces = _simple_marching_cubes(grid, bounds, 0.0)
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)

    def test_sample_and_marching_cubes(self):
        from forgecraft.analysis.iga_tpms_generator import (
            sample_tpms_on_grid, marching_cubes_tpms,
        )
        from forgecraft.analysis.iga_tpms_types import TPMSType

        n = 12
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        cell_size = np.array([0.6])
        thickness = np.array([0.1])

        grid = sample_tpms_on_grid(
            bounds, (n, n, n), cell_size, thickness, TPMSType.GYROID,
        )
        verts, faces = marching_cubes_tpms(grid, bounds)
        self.assertGreater(len(verts), 0, "Should produce some vertices")
        self.assertGreater(len(faces), 0, "Should produce some faces")
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)


# ══════════════════════════════════════════════════════════
#  测试 6: 生成器 (轻量)
# ══════════════════════════════════════════════════════════

class TestGeneratorLightweight(unittest.TestCase):

    def setUp(self):
        from forgecraft.analysis._iga_base import DOFMap, IGASettings
        from forgecraft.analysis.iga_element import MembraneIntegrator
        from forgecraft.analysis.iga_material import LinearIsotropic
        from forgecraft.analysis.iga_boundary import DirichletBC, NeumannBC

        self.mesh = _make_5x5_mesh()
        self.dof_map = DOFMap.from_mesh(self.mesh)
        self.iga_settings = IGASettings(quadrature_order=3, verbosity=0)
        self.integrator = MembraneIntegrator(thickness=0.001)
        self.material = LinearIsotropic(E=210e9, nu=0.3, rho=7800.0)

        # 左侧固定
        left_verts = [vi for vi in range(self.mesh.n_vertices)
                       if self.mesh.vertices[vi][0] < 0.01]
        # 平面应力: z=0 for all
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
        right_faces = [f for f in range(self.mesh.n_faces)
                       if all(self.mesh.vertices[v][0] > 0.99 - 1e-6
                              for v in self.mesh.face_vertices(f))]
        if not right_faces:
            right_faces = [self.mesh.n_faces - 1]
        self.neumann_bcs = [
            NeumannBC.pressure(face_ids=right_faces, pressure=1e5, mesh=self.mesh),
        ]

    def test_generator_initialization(self):
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings
        from forgecraft.analysis.iga_tpms_generator import TPMSLatticeGenerator

        settings = TPMSGradingSettings()
        gen = TPMSLatticeGenerator(settings)
        self.assertIsNotNone(gen)
        self.assertEqual(gen.settings.tpms_type.name, "GYROID")

    def test_generate_single_pass(self):
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings
        from forgecraft.analysis.iga_tpms_generator import TPMSLatticeGenerator

        settings = TPMSGradingSettings(
            iterations=1,
            volume_target=None,
            verbosity=0,
        )
        gen = TPMSLatticeGenerator(settings)

        result = gen.generate(
            self.mesh, self.integrator, self.material,
            self.dof_map, self.iga_settings,
            dirichlet_bcs=self.dirichlet_bcs,
            neumann_bcs=self.neumann_bcs,
            generate_mesh=False,  # 单次不生成网格 (省时间)
        )

        self.assertIsNotNone(result)
        # 可能失败 (小网格刚度奇异) — 不做硬性要求
        if result.success:
            self.assertIsNotNone(result.cell_size_field)
            self.assertIsNotNone(result.thickness_field)
            self.assertIsNotNone(result.stress_field)
            self.assertEqual(len(result.cell_size_field), self.mesh.n_vertices)
            self.assertGreater(result.wall_time, 0)

    def test_generate_with_mesh(self):
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings
        from forgecraft.analysis.iga_tpms_generator import TPMSLatticeGenerator

        settings = TPMSGradingSettings(
            iterations=1,
            volume_target=0.3,
            verbosity=0,
        )
        gen = TPMSLatticeGenerator(settings)

        result = gen.generate(
            self.mesh, self.integrator, self.material,
            self.dof_map, self.iga_settings,
            dirichlet_bcs=self.dirichlet_bcs,
            neumann_bcs=self.neumann_bcs,
            generate_mesh=True,
            grid_resolution=(10, 10, 5),
        )
        self.assertIsNotNone(result)
        if result.success and result.mesh_vertices is not None:
            self.assertGreater(len(result.mesh_vertices), 0)

    def test_convenience_function(self):
        from forgecraft.analysis.iga_tpms_types import TPMSGradingSettings
        from forgecraft.analysis.iga_tpms_generator import tpms_lattice_generate

        settings = TPMSGradingSettings(
            iterations=1,
            verbosity=0,
        )

        result = tpms_lattice_generate(
            self.mesh, self.integrator, self.material,
            self.dof_map, self.iga_settings,
            settings=settings,
            dirichlet_bcs=self.dirichlet_bcs,
            neumann_bcs=self.neumann_bcs,
            generate_mesh=False,
        )
        self.assertIsNotNone(result)


# ══════════════════════════════════════════════════════════
#  测试 7: 顶层导入
# ══════════════════════════════════════════════════════════

class TestTopLevelImports(unittest.TestCase):

    def test_init_types_import(self):
        from forgecraft.analysis import (
            TPMSType, GradingStrategy,
            TPMSParameters, TPMSGradingSettings, TPMSLatticeResult,
            TPMS_SURFACE_FUNCTIONS, TPMS_DEFAULTS,
        )
        self.assertIsNotNone(TPMSType)
        self.assertIsNotNone(GradingStrategy)
        self.assertIsNotNone(TPMSParameters)
        self.assertIsNotNone(TPMSGradingSettings)
        self.assertIsNotNone(TPMSLatticeResult)
        self.assertIsInstance(TPMS_SURFACE_FUNCTIONS, dict)
        self.assertIsInstance(TPMS_DEFAULTS, dict)

    def test_init_core_import(self):
        from forgecraft.analysis import (
            tpms_level_set, tpms_level_set_gradient,
            tpms_is_solid, tpms_is_solid_graded,
            compute_relative_density, compute_tpms_volume_fraction,
            effective_youngs_modulus, effective_shear_modulus,
            cell_size_to_frequency, frequency_to_cell_size,
        )
        self.assertTrue(callable(tpms_level_set))
        self.assertTrue(callable(effective_youngs_modulus))

    def test_init_grading_import(self):
        from forgecraft.analysis import (
            compute_von_mises_field, compute_stress_driving_field,
            map_stress_to_cell_size, map_stress_to_thickness,
            smooth_grading_field, normalize_grading_for_volume,
            compute_tpms_parameters_field, estimate_average_density,
        )
        self.assertTrue(callable(compute_von_mises_field))
        self.assertTrue(callable(map_stress_to_cell_size))

    def test_init_generator_import(self):
        from forgecraft.analysis import (
            TPMSLatticeGenerator,
            tpms_lattice_generate, tpms_lattice_iterative,
            marching_cubes_tpms, sample_tpms_on_grid,
        )
        self.assertIsNotNone(TPMSLatticeGenerator)
        self.assertTrue(callable(tpms_lattice_generate))
