# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_tpms_grading — 应力自适应梯度映射
#
#   从 IGA 位移场计算应力, 映射到 TPMS 晶格参数:
#     σ_vm(x) → cell_size(x), thickness(x)
#
#   管道:
#     IGA 位移 → 应变能密度 → von Mises → 幂律映射 → Helmholtz 平滑
#
#   复用现有基础设施:
#     - assemble_phasefield_laplacian (Helmholtz 滤波)
#     - compute_strain_energy_density (应变能)
#     - compute_von_mises (后处理)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import DOFMap, IGASettings, IrregularFaceCache
from forgecraft.analysis.iga_multiphysics_types import ScalarDOFMap
from forgecraft.analysis.iga_tpms_types import (
    TPMSType, GradingStrategy,
    TPMSParameters, TPMSGradingSettings,
)
from forgecraft.analysis.iga_post import compute_von_mises

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "compute_von_mises_field",
    "compute_stress_driving_field",
    "map_stress_to_cell_size",
    "map_stress_to_thickness",
    "smooth_grading_field",
    "normalize_grading_for_volume",
    "compute_tpms_parameters_field",
    "estimate_average_density",
]


# ══════════════════════════════════════════════════════════
#  应力场计算
# ══════════════════════════════════════════════════════════

def compute_stress_driving_field(
    mesh: PolyMesh,
    u: np.ndarray,
    dof_map: DOFMap,
    integrator,
    material,
    iga_settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
    strategy: GradingStrategy = GradingStrategy.STRESS_PROPORTIONAL,
) -> np.ndarray:
    """计算应力驱动场 (每控制顶点标量)

    根据策略返回:
      - STRESS_PROPORTIONAL: von Mises 应力 (积分点 → 顶点面积加权)
      - ENERGY_DENSITY: 应变能密度 (复用 compute_strain_energy_density)
      - VON_MISES_THRESHOLD: 同 STRESS_PROPORTIONAL
      - MANUAL_FIELD: 返回零场 (调用方自己提供)

    实现: 跟随 compute_strain_energy_density 的组装模式
    (_plane_stress_D + _build_membrane_B_local)。

    Args:
        mesh: CC 控制网格
        u: (n_dof,) 位移向量
        dof_map: 结构 DOF 映射
        integrator: MembraneIntegrator
        material: LinearIsotropic 本构
        iga_settings: IGA 设置
        irregular_cache: 非常面缓存
        strategy: 驱动场策略

    Returns:
        (n_vertices,) 标量应力驱动场
    """
    n_vertices = mesh.n_vertices
    n_dof = dof_map.n_dof

    if strategy == GradingStrategy.ENERGY_DENSITY:
        from forgecraft.analysis.iga_topology_phasefield import compute_strain_energy_density
        return compute_strain_energy_density(
            mesh, integrator, material, u, dof_map, iga_settings, irregular_cache,
        )
    elif strategy == GradingStrategy.MANUAL_FIELD:
        return np.zeros(n_vertices)

    # STRESS_PROPORTIONAL / VON_MISES_THRESHOLD:
    # 在积分点计算 VM, 面积加权平均到控制顶点
    from forgecraft.analysis.iga_quadrature import cc_quadrature
    from forgecraft.analysis.iga_shape import cc_shape_functions
    from forgecraft.analysis.iga_element import _plane_stress_D
    from forgecraft.analysis.iga_topology_phasefield import _build_membrane_B_local

    D_membrane = _plane_stress_D(material.E, material.nu)

    vertex_vm_sum = np.zeros(n_vertices)
    vertex_weight_sum = np.zeros(n_vertices)

    for face_id in range(mesh.n_faces):
        verts = list(mesh.face_vertices(face_id))
        if len(verts) != 4:
            continue

        # 16 控制点 (1 环邻域)
        all_verts = list(verts)
        seen = set(verts)
        for v in verts:
            for r in mesh.vertex_ring(v):
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

            # 局部切线坐标系
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

            # B 矩阵 + 应变 + 应力
            B_glob = _build_membrane_B_local(shape, t1, t2)
            eps = B_glob @ u_e
            sigma = D_membrane @ eps

            # von Mises (平面应力)
            sxx, syy, txy = float(sigma[0]), float(sigma[1]), float(sigma[2])
            svm = np.sqrt(max(sxx**2 + syy**2 - sxx * syy + 3.0 * txy**2, 0.0))

            # 面积加权分配
            area = qp.weight * shape.detJ
            for k, vi in enumerate(all_verts):
                Nk = shape.N[k]
                contribution = Nk * area
                vertex_vm_sum[vi] += svm * contribution
                vertex_weight_sum[vi] += contribution

    # 归一化
    vm_field = np.zeros(n_vertices)
    for vi in range(n_vertices):
        if vertex_weight_sum[vi] > 1e-15:
            vm_field[vi] = vertex_vm_sum[vi] / vertex_weight_sum[vi]

    return vm_field


def compute_von_mises_field(
    mesh: PolyMesh,
    u: np.ndarray,
    dof_map: DOFMap,
    integrator,
    material,
    iga_settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """计算每控制顶点 von Mises 应力 (便捷接口)

    等价于 compute_stress_driving_field(..., strategy=STRESS_PROPORTIONAL)
    """
    return compute_stress_driving_field(
        mesh, u, dof_map, integrator, material, iga_settings,
        irregular_cache, strategy=GradingStrategy.STRESS_PROPORTIONAL,
    )


# ══════════════════════════════════════════════════════════
#  应力 → TPMS 参数映射
# ══════════════════════════════════════════════════════════

def map_stress_to_cell_size(
    stress_field: np.ndarray,
    settings: TPMSGradingSettings,
) -> np.ndarray:
    """应力场 → 晶胞尺寸场

    cell_size(x) = CS_min + (CS_max - CS_min) · (σ(x)/σ_ref)^exponent

    高应力 → 小晶胞 (高频 → 更密)
    低应力 → 大晶胞 (低频 → 更疏)

    Args:
        stress_field: (n_vertices,) 应力驱动场
        settings: 梯度设置

    Returns:
        (n_vertices,) 晶胞尺寸
    """
    cs_min, cs_max = settings.cell_size_range
    sigma = np.copy(stress_field).astype(np.float64)

    # 处理零/负应力
    sigma = np.maximum(sigma, 1e-15)

    # 参考应力
    sigma_ref = settings.stress_reference
    if sigma_ref is None:
        sigma_ref = float(np.max(sigma))
    if sigma_ref < 1e-15:
        sigma_ref = 1.0

    # 归一化 + 幂律
    sigma_norm = np.clip(sigma / sigma_ref, 0.0, 1.0)
    # 注意: 高应力 → 小 cell_size, 所以用 (1 - sigma_norm)
    weight = np.power(sigma_norm, settings.stress_exponent)

    cell_size = cs_min + (cs_max - cs_min) * (1.0 - weight)
    return np.clip(cell_size, cs_min, cs_max)


def map_stress_to_thickness(
    stress_field: np.ndarray,
    settings: TPMSGradingSettings,
) -> np.ndarray:
    """应力场 → 壁厚场

    thickness(x) = T_min + (T_max - T_min) · (σ(x)/σ_ref)^thickness_exponent

    高应力 → 厚壁
    低应力 → 薄壁

    Args:
        stress_field: (n_vertices,) 应力驱动场
        settings: 梯度设置

    Returns:
        (n_vertices,) 壁厚
    """
    t_min, t_max = settings.thickness_range
    sigma = np.maximum(np.copy(stress_field).astype(np.float64), 1e-15)

    sigma_ref = settings.stress_reference
    if sigma_ref is None:
        sigma_ref = float(np.max(sigma))
    if sigma_ref < 1e-15:
        sigma_ref = 1.0

    sigma_norm = np.clip(sigma / sigma_ref, 0.0, 1.0)
    weight = np.power(sigma_norm, settings.thickness_exponent)

    thickness = t_min + (t_max - t_min) * weight
    return np.clip(thickness, t_min, t_max)


# ══════════════════════════════════════════════════════════
#  Helmholtz 平滑滤波
# ══════════════════════════════════════════════════════════

def smooth_grading_field(
    mesh: PolyMesh,
    field: np.ndarray,
    radius: float,
    sdof: Optional[ScalarDOFMap] = None,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """Helmholtz PDE 平滑梯度场

    求解: (I - R²∇²) φ_smooth = φ_raw

    使用标量 Laplace 矩阵的组装模式 (复用 iga_topology_phasefield 的
    assemble_phasefield_laplacian 实现)。

    Args:
        mesh: CC 控制网格
        field: (n_vertices,) 原始场
        radius: 平滑半径 R (0 = 不平滑)
        sdof: 标量 DOF 映射 (可选, 自动创建)
        irregular_cache: 非常面缓存

    Returns:
        (n_vertices,) 平滑后的场
    """
    if radius <= 1e-15:
        return field.copy()

    n = mesh.n_vertices
    if sdof is None:
        sdof = ScalarDOFMap.from_mesh(mesh)

    # 组装标量 Laplace 矩阵
    from forgecraft.analysis.iga_topology_phasefield import assemble_phasefield_laplacian
    from forgecraft.analysis.iga_topology_types import TopologySettings

    # 使用临时 settings 仅作为 Laplace 组装的参数容器
    tmp_settings = TopologySettings(quadrature_order=3)
    K_lap = assemble_phasefield_laplacian(
        mesh, sdof, tmp_settings, irregular_cache,
    )

    # 构建 Helmholtz 系统: (I + R²·K_lap) · φ_s = φ_raw
    # 注意: K_lap 是半正定的 Laplace 矩阵 (∇N·∇N)
    # Helmholtz: (M + R²·K) φ = M·φ_raw
    # 简化: 用 lumped 单位质量矩阵 diag(area_i)
    vertex_areas = _compute_vertex_areas(mesh)
    M_lumped = diags(vertex_areas, format='csr')
    A = M_lumped + (radius ** 2) * K_lap
    rhs = M_lumped @ field

    phi_smooth = spsolve(A, rhs)
    return np.asarray(phi_smooth).ravel()


def _compute_vertex_areas(mesh: PolyMesh) -> np.ndarray:
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


# ══════════════════════════════════════════════════════════
#  体积归一化
# ══════════════════════════════════════════════════════════

def normalize_grading_for_volume(
    cell_size: np.ndarray,
    thickness: np.ndarray,
    vertex_areas: np.ndarray,
    volume_target: float,
    max_iter: int = 50,
    tol: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """全局缩放厚度场以满足目标体积分数

    使用二分搜索全局缩放因子 α:
      thickness_scaled = α · thickness
      使得 V(thickness_scaled) ≈ V_target

    每顶点相对密度近似:
      ρ_i ≈ t_i / cs_i   (薄壁近似)

    体积分数:
      V = Σ ρ_i · area_i / Σ area_i

    Args:
        cell_size: (n_vertices,) 晶胞尺寸
        thickness: (n_vertices,) 壁厚
        vertex_areas: (n_vertices,) 顶点面积
        volume_target: 目标体积分数 [0, 1]
        max_iter: 最大二分迭代
        tol: 收敛容差

    Returns:
        (cell_size, thickness_normalized)
    """
    total_area = float(np.sum(vertex_areas))
    if total_area < 1e-15:
        return cell_size, thickness

    # 当前体积分数 (薄壁近似)
    def compute_vol(scale):
        t_scaled = scale * thickness
        rho = np.clip(t_scaled / np.maximum(cell_size, 1e-10), 0.0, 1.0)
        return float(np.sum(rho * vertex_areas) / total_area)

    current_vol = compute_vol(1.0)
    if current_vol < 1e-15 or abs(current_vol - volume_target) < tol:
        return cell_size, thickness

    # 二分搜索
    alpha_lo = 0.01
    alpha_hi = 10.0
    # 扩展上界直到包含目标
    while compute_vol(alpha_hi) < volume_target and alpha_hi < 1000:
        alpha_hi *= 2.0

    for _ in range(max_iter):
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        vol_mid = compute_vol(alpha_mid)

        if abs(vol_mid - volume_target) < tol:
            break
        elif vol_mid > volume_target:
            alpha_hi = alpha_mid
        else:
            alpha_lo = alpha_mid

    alpha = 0.5 * (alpha_lo + alpha_hi)
    thickness_normalized = np.clip(alpha * thickness, 0.0, None)

    return cell_size, thickness_normalized


# ══════════════════════════════════════════════════════════
#  完整梯度管线
# ══════════════════════════════════════════════════════════

def compute_tpms_parameters_field(
    mesh: PolyMesh,
    stress_field: np.ndarray,
    settings: TPMSGradingSettings,
    sdof: Optional[ScalarDOFMap] = None,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """从应力场计算 TPMS 参数场 (完整管道)

    1. 应力 → cell_size (幂律映射)
    2. 应力 → thickness (幂律映射)
    3. Helmholtz 平滑 (可选)
    4. 体积归一化 (可选)

    Args:
        mesh: CC 控制网格
        stress_field: (n_vertices,) 应力驱动场
        settings: 梯度设置
        sdof: 标量 DOF 映射
        irregular_cache: 非常面缓存

    Returns:
        (cell_size_field, thickness_field): 各 (n_vertices,)
    """
    if sdof is None:
        sdof = ScalarDOFMap.from_mesh(mesh)

    # Step 1-2: 应力 → 参数
    cell_size = map_stress_to_cell_size(stress_field, settings)
    thickness = map_stress_to_thickness(stress_field, settings)

    # Step 3: 平滑
    if settings.smoothing_radius > 0:
        cell_size = smooth_grading_field(
            mesh, cell_size, settings.smoothing_radius, sdof, irregular_cache,
        )
        thickness = smooth_grading_field(
            mesh, thickness, settings.smoothing_radius, sdof, irregular_cache,
        )

    # Step 4: 体积归一化
    if settings.volume_target is not None:
        vertex_areas = _compute_vertex_areas(mesh)
        cell_size, thickness = normalize_grading_for_volume(
            cell_size, thickness, vertex_areas, settings.volume_target,
        )

    return cell_size, thickness


def estimate_average_density(
    cell_size: np.ndarray,
    thickness: np.ndarray,
    vertex_areas: np.ndarray,
) -> float:
    """估算梯度 TPMS 体积分数 (薄壁近似)

    V ≈ Σ (t_i / cs_i) · area_i / Σ area_i

    Args:
        cell_size: (n_vertices,) 晶胞尺寸
        thickness: (n_vertices,) 壁厚
        vertex_areas: (n_vertices,) 顶点面积

    Returns:
        体积分数 [0, 1]
    """
    total_area = float(np.sum(vertex_areas))
    if total_area < 1e-15:
        return 0.0
    rho = np.clip(thickness / np.maximum(cell_size, 1e-10), 0.0, 1.0)
    return float(np.sum(rho * vertex_areas) / total_area)
