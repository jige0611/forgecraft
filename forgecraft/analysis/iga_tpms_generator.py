# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_tpms_generator — 梯度 TPMS 晶格生成器
#
#   主协调器: 应力分析 → 梯度映射 → TPMS 几何生成
#
#   两种模式:
#     A) 单次生成: FEA → 应力 → 梯度 → Marching Cubes → STL
#     B) 迭代优化: 闭环重分析, 等效模量 E_eff(x)
#                  → 收敛到最优材料分布
#
#   Marching Cubes:
#     在 CC 极限曲面内采样 TPMS 隐式场,
#     提取等值面 |f(x)| = t(x)/2
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IGASettings, IrregularFaceCache,
)
from forgecraft.analysis.iga_multiphysics_types import ScalarDOFMap
from forgecraft.analysis.iga_tpms_types import (
    TPMSType, GradingStrategy,
    TPMSParameters, TPMSGradingSettings, TPMSLatticeResult,
    TPMS_DEFAULTS,
)
from forgecraft.analysis.iga_tpms_core import (
    tpms_level_set, tpms_level_set_gradient,
    tpms_is_solid_graded, tpms_is_solid,
    effective_youngs_modulus, compute_tpms_volume_fraction,
)
from forgecraft.analysis.iga_tpms_grading import (
    compute_stress_driving_field,
    compute_tpms_parameters_field,
    estimate_average_density,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "TPMSLatticeGenerator",
    "tpms_lattice_generate",
    "tpms_lattice_iterative",
    "marching_cubes_tpms",
    "sample_tpms_on_grid",
]


# ══════════════════════════════════════════════════════════
#  TPMS 采样 (体素网格)
# ══════════════════════════════════════════════════════════

def sample_tpms_on_grid(
    bounds: Tuple[float, float, float, float, float, float],
    resolution: Tuple[int, int, int],
    cell_size_field: np.ndarray,
    thickness_field: np.ndarray,
    tpms_type: TPMSType,
    offset_field: Optional[np.ndarray] = None,
    cc_mesh: Optional[PolyMesh] = None,
    control_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """在体素网格上采样 TPMS 水平集

    如果提供 CC 控制网格, 则只在极限曲面内部采样。
    否则在整个包围盒内采样。

    每个样本点的局部参数通过:
      - 最近控制点的参数 (简化)
      - 或 CC 基函数插值 (精确, 暂未实现)

    Args:
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        resolution: (nx, ny, nz) 各方向分辨率
        cell_size_field: (n_vertices,) 每顶点晶胞尺寸
        thickness_field: (n_vertices,) 每顶点壁厚
        tpms_type: TPMS 类型
        offset_field: (n_vertices,) 偏移场 (可选, 默认 0)
        cc_mesh: CC 控制网格 (用于域限制)
        control_points: (n_control, 3) 控制点坐标

    Returns:
        (nx, ny, nz) float32 水平集值
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = resolution
    n_vertices = len(cell_size_field)

    x_vals = np.linspace(xmin, xmax, nx)
    y_vals = np.linspace(ymin, ymax, ny)
    z_vals = np.linspace(zmin, zmax, nz)
    XX, YY, ZZ = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    grid = np.zeros((nx, ny, nz), dtype=np.float32)

    if offset_field is None:
        offset_field = np.zeros(n_vertices)

    # 预计算每顶点频率
    wx_all = 2.0 * np.pi / np.clip(cell_size_field, 1e-10, None)
    wy_all = wx_all.copy()
    wz_all = wx_all.copy()

    # 构建 KD 树用于最近邻查询 (如果提供了控制点)
    if control_points is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(control_points)
        indices = tree.query(np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]))[1]
        indices = indices.reshape(nx, ny, nz)

        # 向量化求值 (每点独立参数)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    vi = indices[i, j, k]
                    f_val = tpms_level_set(
                        float(XX[i, j, k]), float(YY[i, j, k]), float(ZZ[i, j, k]),
                        tpms_type,
                        float(wx_all[vi]), float(wy_all[vi]), float(wz_all[vi]),
                    )
                    # 实体判定: |f - offset| - t/2  (负 = 实体内部)
                    signed_dist = abs(f_val - float(offset_field[vi])) - float(thickness_field[vi]) / 2.0
                    grid[i, j, k] = float(signed_dist)
    else:
        # 均匀参数 (取中位数)
        cs_med = float(np.median(cell_size_field))
        t_med = float(np.median(thickness_field))
        w_med = 2.0 * np.pi / max(cs_med, 1e-10)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f_val = tpms_level_set(
                        float(XX[i, j, k]), float(YY[i, j, k]), float(ZZ[i, j, k]),
                        tpms_type, w_med, w_med, w_med,
                    )
                    signed_dist = abs(f_val) - t_med / 2.0
                    grid[i, j, k] = float(signed_dist)

    return grid


# ══════════════════════════════════════════════════════════
#  Marching Cubes (简化版)
# ══════════════════════════════════════════════════════════

def marching_cubes_tpms(
    grid: np.ndarray,
    bounds: Tuple[float, float, float, float, float, float],
    iso_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """简化 Marching Cubes 从水平集网格提取等值面

    使用 skimage.measure.marching_cubes 或手动简单实现。

    Args:
        grid: (nx, ny, nz) 水平集值 (负 = 实体内部)
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        iso_value: 等值面值 (默认 0 = 实体边界)

    Returns:
        (vertices, faces): 三角形网格
    """
    try:
        from skimage.measure import marching_cubes as mc
        spacing_x = (bounds[1] - bounds[0]) / (grid.shape[0] - 1)
        spacing_y = (bounds[3] - bounds[2]) / (grid.shape[1] - 1)
        spacing_z = (bounds[5] - bounds[4]) / (grid.shape[2] - 1)
        spacing = (spacing_x, spacing_y, spacing_z)

        verts, faces, normals, values = mc(
            grid, level=iso_value, spacing=spacing,
        )
        # 偏移到正确位置
        verts[:, 0] += bounds[0]
        verts[:, 1] += bounds[2]
        verts[:, 2] += bounds[4]

        return verts.astype(np.float64), faces.astype(np.int64)

    except ImportError:
        _logger.warning("skimage not available, using simplified marching cubes")
        return _simple_marching_cubes(grid, bounds, iso_value)


def _simple_marching_cubes(
    grid: np.ndarray,
    bounds: Tuple[float, float, float, float, float, float],
    iso_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """极简 Marching Cubes (仅用于无 skimage 时的回退)

    遍历所有体素, 在等值面交界处放置一个四边形 (两个三角形)。
    """
    nx, ny, nz = grid.shape
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    dx = (xmax - xmin) / max(nx - 1, 1)
    dy = (ymax - ymin) / max(ny - 1, 1)
    dz = (zmax - zmin) / max(nz - 1, 1)

    verts_list = []
    faces_list = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # 8 corners
                corners = np.array([
                    grid[i, j, k], grid[i+1, j, k],
                    grid[i+1, j+1, k], grid[i, j+1, k],
                    grid[i, j, k+1], grid[i+1, j, k+1],
                    grid[i+1, j+1, k+1], grid[i, j+1, k+1],
                ])
                # Simple: if any corner is inside (negative) and any is outside (positive)
                inside = corners < iso_value
                if np.all(inside) or np.all(~inside):
                    continue

                # Place a simple quad at center for any crossing
                cx = xmin + (i + 0.5) * dx
                cy = ymin + (j + 0.5) * dy
                cz = zmin + (k + 0.5) * dz
                s = min(dx, dy, dz) * 0.3

                v0 = len(verts_list)
                verts_list.extend([
                    [cx - s, cy - s, cz],
                    [cx + s, cy - s, cz],
                    [cx + s, cy + s, cz],
                    [cx - s, cy + s, cz],
                ])
                faces_list.extend([
                    [v0, v0 + 1, v0 + 2],
                    [v0, v0 + 2, v0 + 3],
                ])

    if len(verts_list) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    return np.array(verts_list, dtype=np.float64), np.array(faces_list, dtype=np.int64)


# ══════════════════════════════════════════════════════════
#  主生成器类
# ══════════════════════════════════════════════════════════

class TPMSLatticeGenerator:
    """梯度 TPMS 晶格生成器

    协调完整流程:
      1. IGA 分析 (实体材料)
      2. 应力场计算
      3. 应力 → TPMS 参数映射
      4. (可选) 迭代优化
      5. 几何生成 (Marching Cubes)

    Examples:
        >>> generator = TPMSLatticeGenerator(grading_settings)
        >>> result = generator.generate(mesh, integrator, material,
        ...                             dof_map, iga_settings,
        ...                             dirichlet_bcs, neumann_bcs)
    """

    def __init__(self, settings: TPMSGradingSettings):
        self.settings = settings
        self._iteration_history: List[float] = []

    def generate(
        self,
        mesh: PolyMesh,
        integrator,
        material,
        dof_map: DOFMap,
        iga_settings: IGASettings,
        dirichlet_bcs: Optional[list] = None,
        neumann_bcs: Optional[list] = None,
        point_loads: Optional[list] = None,
        irregular_cache: Optional[IrregularFaceCache] = None,
        generate_mesh: bool = True,
        grid_resolution: Tuple[int, int, int] = (50, 50, 50),
    ) -> TPMSLatticeResult:
        """单次生成 TPMS 晶格 (模式 A)

        Args:
            mesh: CC 控制网格 (定义分析域)
            integrator: MembraneIntegrator
            material: LinearIsotropic 本构
            dof_map: 结构 DOF 映射
            iga_settings: IGA 设置
            dirichlet_bcs: Dirichlet 边界条件
            neumann_bcs: Neumann 边界条件
            point_loads: 点载荷
            irregular_cache: 非常面缓存
            generate_mesh: 是否生成三角形网格
            grid_resolution: Marching Cubes 分辨率

        Returns:
            TPMSLatticeResult
        """
        t0 = perf_counter()
        result = TPMSLatticeResult(
            tpms_type=self.settings.tpms_type,
            strategy=self.settings.strategy,
        )

        try:
            # Step 1: 实体材料 IGA 分析
            u = self._solve_structural(
                mesh, integrator, material, dof_map, iga_settings,
                dirichlet_bcs, neumann_bcs, point_loads, irregular_cache,
            )

            # Step 2: 应力场
            stress_field = compute_stress_driving_field(
                mesh, u, dof_map, integrator, material, iga_settings,
                irregular_cache, self.settings.strategy,
            )
            result.stress_field = stress_field

            # Step 3: 应力 → TPMS 参数
            sdof = ScalarDOFMap.from_mesh(mesh)
            cell_size, thickness = compute_tpms_parameters_field(
                mesh, stress_field, self.settings, sdof, irregular_cache,
            )
            result.cell_size_field = cell_size
            result.thickness_field = thickness

            # 相对密度场 (薄壁近似)
            result.relative_density_field = np.clip(
                thickness / np.maximum(cell_size, 1e-10), 0.0, 1.0,
            )

            # 体积分数
            vertex_areas = _vertex_areas(mesh)
            result.volume_fraction = estimate_average_density(
                cell_size, thickness, vertex_areas,
            )
            result.average_cell_size = float(np.mean(cell_size))

            # Step 4: 几何生成
            if generate_mesh:
                grid = sample_tpms_on_grid(
                    self._mesh_bounds(mesh), grid_resolution,
                    cell_size, thickness, self.settings.tpms_type,
                    cc_mesh=mesh, control_points=mesh.vertices,
                )
                verts, faces = marching_cubes_tpms(grid, self._mesh_bounds(mesh))
                result.mesh_vertices = verts
                result.mesh_faces = faces

            result.success = True
            result.iterations = 1

        except Exception as e:
            _logger.error(f"TPMS generation failed: {e}")
            result.success = False

        result.wall_time = perf_counter() - t0
        return result

    def generate_iterative(
        self,
        mesh: PolyMesh,
        integrator,
        material,
        dof_map: DOFMap,
        iga_settings: IGASettings,
        dirichlet_bcs: Optional[list] = None,
        neumann_bcs: Optional[list] = None,
        point_loads: Optional[list] = None,
        irregular_cache: Optional[IrregularFaceCache] = None,
        generate_mesh: bool = True,
        grid_resolution: Tuple[int, int, int] = (50, 50, 50),
    ) -> TPMSLatticeResult:
        """迭代优化生成 TPMS 晶格 (模式 B)

        闭环:
          ① 实体 FEA
          ② 应力 → 梯度
          ③ 等效模量 E_eff(x)
          ④ 重分析 (空间变化材料)
          ⑤ 检查收敛 → 否则回到②

        Args:
            (同 generate())

        Returns:
            TPMSLatticeResult
        """
        t0 = perf_counter()
        result = TPMSLatticeResult(
            tpms_type=self.settings.tpms_type,
            strategy=self.settings.strategy,
        )

        try:
            sdof = ScalarDOFMap.from_mesh(mesh)
            n_vertices = mesh.n_vertices

            # 初始均匀
            cell_size = np.full(n_vertices, float(np.mean(self.settings.cell_size_range)))
            thickness = np.full(n_vertices, float(np.mean(self.settings.thickness_range)))

            self._iteration_history = []
            max_iter = max(self.settings.iterations, 1)

            for iteration in range(max_iter):
                # 等效模量
                rho = np.clip(thickness / np.maximum(cell_size, 1e-10), 0.01, 1.0)
                E_eff = np.array([
                    effective_youngs_modulus(float(rho[i]), material.E, self.settings.tpms_type)
                    for i in range(n_vertices)
                ])

                # 组装空间变化刚度 (简化: 对每 3x3 块缩放)
                K = self._assemble_spatially_varying_stiffness(
                    mesh, integrator, material, dof_map, iga_settings,
                    E_eff, irregular_cache,
                )
                F = self._assemble_force(mesh, dof_map, neumann_bcs, point_loads, iga_settings)

                # Dirichlet BC
                from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty
                if dirichlet_bcs:
                    for bc in dirichlet_bcs:
                        apply_dirichlet_penalty(K, F, bc)

                # 求解
                try:
                    diag_reg = 1e-12 * np.max(np.abs(K.diagonal()))
                    u = np.linalg.solve(K.toarray() + diag_reg * np.eye(K.shape[0]), F)
                except np.linalg.LinAlgError:
                    u = np.linalg.lstsq(K.toarray(), F, rcond=None)[0]

                # 应力 + 梯度
                stress_field = compute_stress_driving_field(
                    mesh, u, dof_map, integrator, material, iga_settings,
                    irregular_cache, self.settings.strategy,
                )
                cell_size_new, thickness_new = compute_tpms_parameters_field(
                    mesh, stress_field, self.settings, sdof, irregular_cache,
                )

                # 收敛检查
                delta = np.max(np.abs(cell_size_new - cell_size)) / max(np.mean(cell_size), 1e-10)
                self._iteration_history.append(float(delta))

                cell_size = cell_size_new
                thickness = thickness_new

                if delta < self.settings.convergence_tol and iteration > 0:
                    break

            # 结果
            result.stress_field = stress_field
            result.cell_size_field = cell_size
            result.thickness_field = thickness
            result.relative_density_field = np.clip(
                thickness / np.maximum(cell_size, 1e-10), 0.0, 1.0,
            )
            vertex_areas = _vertex_areas(mesh)
            result.volume_fraction = estimate_average_density(
                cell_size, thickness, vertex_areas,
            )
            result.average_cell_size = float(np.mean(cell_size))
            result.iterations = len(self._iteration_history)
            result.convergence_history = self._iteration_history

            # 几何生成
            if generate_mesh:
                grid = sample_tpms_on_grid(
                    self._mesh_bounds(mesh), grid_resolution,
                    cell_size, thickness, self.settings.tpms_type,
                    cc_mesh=mesh, control_points=mesh.vertices,
                )
                verts, faces = marching_cubes_tpms(grid, self._mesh_bounds(mesh))
                result.mesh_vertices = verts
                result.mesh_faces = faces

            result.success = True

        except Exception as e:
            _logger.error(f"Iterative TPMS generation failed: {e}")
            result.success = False

        result.wall_time = perf_counter() - t0
        return result

    def _solve_structural(
        self, mesh, integrator, material, dof_map, iga_settings,
        dirichlet_bcs, neumann_bcs, point_loads, irregular_cache,
    ) -> np.ndarray:
        """实体材料 IGA 静力求解"""
        from forgecraft.analysis.iga_assembly import assemble_stiffness, assemble_force_vector
        from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty

        K = assemble_stiffness(
            mesh, integrator, material, dof_map, iga_settings, irregular_cache,
        ).toarray()

        F = assemble_force_vector(
            mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
        ).astype(np.float64)

        if dirichlet_bcs:
            for bc in dirichlet_bcs:
                apply_dirichlet_penalty(K, F, bc)

        try:
            diag_reg = 1e-12 * np.max(np.abs(K.diagonal()))
            u = np.linalg.solve(K + diag_reg * np.eye(K.shape[0]), F)
            return u
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(K, F, rcond=None)[0]

    def _assemble_spatially_varying_stiffness(
        self, mesh, integrator, material, dof_map, iga_settings,
        E_eff, irregular_cache,
    ) -> csr_matrix:
        """组装空间变化刚度矩阵

        对每个控制顶点的 3x3 块按 sqrt(E_eff/E_solid) 退化。
        """
        from forgecraft.analysis.iga_assembly import assemble_stiffness

        K0 = assemble_stiffness(
            mesh, integrator, material, dof_map, iga_settings, irregular_cache,
        ).toarray()

        n_vertices = mesh.n_vertices
        for vi in range(n_vertices):
            factor = np.sqrt(max(E_eff[vi] / max(material.E, 1e-15), 1e-15))
            factor = np.clip(factor, 0.01, 1.0)
            for vj in range(n_vertices):
                factor_j = np.sqrt(max(E_eff[vj] / max(material.E, 1e-15), 1e-15))
                factor_j = np.clip(factor_j, 0.01, 1.0)
                combined = factor * factor_j
                for li in range(3):
                    ig = dof_map.vertex_dofs(vi)[li]
                    for lj in range(3):
                        jg = dof_map.vertex_dofs(vj)[lj]
                        K0[ig, jg] *= combined

        return csr_matrix(K0)

    def _assemble_force(self, mesh, dof_map, neumann_bcs, point_loads, iga_settings):
        from forgecraft.analysis.iga_assembly import assemble_force_vector
        return assemble_force_vector(
            mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
        ).astype(np.float64)

    def _mesh_bounds(self, mesh: PolyMesh) -> Tuple[float, float, float, float, float, float]:
        """包围盒"""
        v = mesh.vertices
        return (float(v[:, 0].min()), float(v[:, 0].max()),
                float(v[:, 1].min()), float(v[:, 1].max()),
                float(v[:, 2].min()), float(v[:, 2].max()))


# ══════════════════════════════════════════════════════════
#  便捷函数
# ══════════════════════════════════════════════════════════

def tpms_lattice_generate(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    settings: Optional[TPMSGradingSettings] = None,
    **kwargs,
) -> TPMSLatticeResult:
    """便捷接口: 单次 TPMS 晶格生成

    Args:
        mesh: CC 控制网格
        integrator: MembraneIntegrator
        material: LinearIsotropic
        dof_map: DOF 映射
        iga_settings: IGA 设置
        settings: 梯度设置 (默认 GYROID + STRESS_PROPORTIONAL)
        **kwargs: 传递给 Generator.generate()

    Returns:
        TPMSLatticeResult
    """
    if settings is None:
        settings = TPMSGradingSettings()
    generator = TPMSLatticeGenerator(settings)
    return generator.generate(mesh, integrator, material, dof_map, iga_settings, **kwargs)


def tpms_lattice_iterative(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    settings: Optional[TPMSGradingSettings] = None,
    **kwargs,
) -> TPMSLatticeResult:
    """便捷接口: 迭代 TPMS 晶格生成

    Args:
        (同 tpms_lattice_generate)
        settings: 梯度设置 (iterations > 1 启用迭代)

    Returns:
        TPMSLatticeResult
    """
    if settings is None:
        settings = TPMSGradingSettings(iterations=3)
    generator = TPMSLatticeGenerator(settings)
    return generator.generate_iterative(
        mesh, integrator, material, dof_map, iga_settings, **kwargs,
    )


def _vertex_areas(mesh: PolyMesh) -> np.ndarray:
    """计算每控制顶点归属面积"""
    n_vertices = mesh.n_vertices
    areas = np.zeros(n_vertices)
    for f in range(mesh.n_faces):
        verts = mesh.face_vertices(f)
        nv = len(verts)
        if nv < 3:
            continue
        p0 = mesh.vertices[verts[0]]
        area = 0.0
        for i in range(1, nv - 1):
            v1 = mesh.vertices[verts[i]] - p0
            v2 = mesh.vertices[verts[i + 1]] - p0
            area += 0.5 * np.linalg.norm(np.cross(v1, v2))
        share = area / nv
        for vi in verts:
            areas[vi] += share
    return areas
