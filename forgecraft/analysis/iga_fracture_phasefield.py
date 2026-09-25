# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_fracture_phasefield — 相场断裂
#
#   PhaseFieldSolver:           交错求解器
#   solve_phasefield:           主入口
#   assemble_phasefield_system: 组装相场刚度/力向量
#   compute_elastic_energy:    弹性应变能密度
#   extract_crack_geometry:    裂纹几何提取 (后处理)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.linalg import spsolve

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IrregularFaceCache, IGASettings,
)
from forgecraft.analysis.iga_contact_types import (
    PhaseFieldSettings, CrackInfo,
)
from forgecraft.analysis.iga_assembly import assemble_stiffness, StreamingCSRAssembler

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "PhaseFieldSolver",
    "solve_phasefield",
    "assemble_phasefield_system",
    "compute_elastic_energy_density",
    "assemble_degraded_stiffness",
    "extract_crack_geometry",
]


# ══════════════════════════════════════════════════════════
#  相场求解器
# ══════════════════════════════════════════════════════════

class PhaseFieldSolver:
    """相场断裂交错求解器
    
    AT2 模型: g(d) = (1-d)² + η
    
    交错策略:
      - 固定 d，求解 u:  K_u(g(d)) · u = f_ext
      - 固定 u，求解 d:  K_d · d = f_d(ψ_e)
      - 不可逆约束:   d ≥ d_history
    
    用法:
        solver = PhaseFieldSolver(settings)
        result = solver.solve(mesh, integrator, material, dof_map, ...)
    """
    
    def __init__(self, pf_settings: PhaseFieldSettings):
        self.pf = pf_settings
        
        # 内部状态
        self.d_field: Optional[np.ndarray] = None       # (n_vertices,) 相场
        self.d_history: Optional[np.ndarray] = None     # 历史最大损伤
        self.psi_e_history: Optional[np.ndarray] = None # 历史应变能
        self.load_factor: float = 0.0
    
    def solve(
        self,
        mesh: PolyMesh,
        integrator,
        material,
        dof_map: DOFMap,
        iga_settings: IGASettings,
        dirichlet_bcs=None,
        neumann_bcs=None,
        point_loads=None,
        load_steps: int = 10,
        max_load: float = 1.0,
        irregular_cache=None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """交错求解相场断裂问题
        
        Args:
            mesh: CC 控制网格
            integrator: ElementIntegrator 实例
            material: 材料模型
            dof_map: DOF 映射
            iga_settings: IGA 设置
            dirichlet_bcs: Dirichlet BC
            neumann_bcs: Neumann BC (可用载荷步缩放)
            point_loads: 点载荷
            load_steps: 载荷步数
            max_load: 最大载荷因子
            irregular_cache: 非常面缓存
        
        Returns:
            (u_history, d_history, final_u) 位移历史、相场历史、最终位移
        """
        t0 = perf_counter()
        
        n_vertices = mesh.n_vertices
        n_dof = dof_map.n_dof
        
        # 初始化相场
        if self.d_field is None:
            self.d_field = np.zeros(n_vertices)
            self.d_history = np.zeros(n_vertices)
            self.psi_e_history = np.zeros(n_vertices)
        
        # 组装基础刚度矩阵 (完整，未退化)
        K0 = assemble_stiffness(
            mesh, integrator, material, dof_map, iga_settings, irregular_cache,
        ).toarray()
        
        # 组装力向量 (基础)
        from forgecraft.analysis.iga_assembly import assemble_force_vector
        F_ref = assemble_force_vector(
            mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
        )
        
        # 施加 Dirichlet BC (add identity row on fixed DOFs for phase-field)
        if dirichlet_bcs:
            from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty
            F_dummy = np.zeros(n_dof)
            for bc in dirichlet_bcs:
                apply_dirichlet_penalty(K0, F_dummy, bc)
        
        u = np.zeros(n_dof)
        u_history = [u.copy()]
        d_history = [self.d_field.copy()]
        
        # 载荷步循环
        for step in range(1, load_steps + 1):
            self.load_factor = step / load_steps * max_load
            F_step = self.load_factor * F_ref
            
            stagger_converged = False
            
            for stagger_iter in range(self.pf.stagger_max_iter):
                d_old = self.d_field.copy()
                
                # ---- 位移子问题 (固定 d) ----
                K_u = assemble_degraded_stiffness(
                    K0, self.d_field, mesh, dof_map, self.pf,
                )
                
                try:
                    u_new = np.linalg.solve(K_u, F_step)
                except np.linalg.LinAlgError:
                    u_new = u
                
                u = u_new
                
                # ---- 应变能计算 ----
                psi_e = compute_elastic_energy_density(
                    mesh, integrator, material, u, dof_map, iga_settings,
                    irregular_cache,
                )
                
                # 更新历史应变能 (用于驱动相场)
                if self.pf.irreversibility == "history":
                    self.psi_e_history = np.maximum(self.psi_e_history, psi_e)
                    driving_force = self.psi_e_history
                else:
                    driving_force = np.maximum(psi_e, self.psi_e_history)
                    self.psi_e_history = driving_force
                
                # ---- 相场子问题 (固定 u) ----
                K_d, f_d = assemble_phasefield_system(
                    mesh, driving_force, self.pf, dof_map,
                )
                
                # 不可逆约束: 下界 d ≥ d_history
                # 用半光滑 Newton: 对 d_i < d_history_i 的 DOF 施加 dirichlet
                d_new = spsolve(K_d, f_d)
                d_new = np.maximum(d_new, self.d_history)
                d_new = np.minimum(d_new, 1.0)  # 上界
                
                self.d_field = d_new
                
                # 交错收敛
                d_change = np.linalg.norm(self.d_field - d_old) / max(
                    np.linalg.norm(self.d_field), 1e-15
                )
                
                if d_change < self.pf.stagger_tolerance:
                    stagger_converged = True
                    break
            
            # 更新历史损伤
            self.d_history = np.maximum(self.d_history, self.d_field)
            
            # 存储历史
            u_history.append(u.copy())
            d_history.append(self.d_field.copy())
            
            if iga_settings.verbosity >= 1:
                max_d = np.max(self.d_field)
                _logger.info(
                    f"Step {step}/{load_steps}: "
                    f"max_d={max_d:.4f}, stagger_iters={stagger_iter + 1}"
                )
        
        wall_time = perf_counter() - t0
        
        if _logger.isEnabledFor(20):
            _logger.info(f"Phase-field solved in {wall_time:.2f}s, {load_steps} steps")
        
        return u_history, d_history, u


def solve_phasefield(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    pf_settings: PhaseFieldSettings,
    dirichlet_bcs=None,
    neumann_bcs=None,
    point_loads=None,
    **kwargs,
):
    """便捷接口"""
    solver = PhaseFieldSolver(pf_settings)
    return solver.solve(
        mesh, integrator, material, dof_map, iga_settings,
        dirichlet_bcs=dirichlet_bcs,
        neumann_bcs=neumann_bcs,
        point_loads=point_loads,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════
#  退化刚度组装
# ══════════════════════════════════════════════════════════

def assemble_degraded_stiffness(
    K0: np.ndarray,                 # (n_dof, n_dof) 完整刚度矩阵
    d_field: np.ndarray,            # (n_vertices,) 相场
    mesh: PolyMesh,
    dof_map: DOFMap,
    pf_settings: PhaseFieldSettings,
) -> np.ndarray:
    """根据相场退化完整刚度矩阵
    
    K_degraded = (1-d)² · K0 (逐 DOF 退化)
    
    退化策略:
      对每个 DOF 所属顶点的相场值 d_v:
        K[i, j] *= (1-d_v_i) * (1-d_v_j)   (对称退化)
    
    简化: 用顶点损伤的平均值退化对应的 DOF 块
    """
    K_deg = K0.copy()
    
    # 为每个控制顶点计算退化因子
    n_vertices = mesh.n_vertices
    g = np.ones(n_vertices)
    for vi in range(n_vertices):
        g[vi] = pf_settings.g(d_field[vi])
    
    # 退化: 对每个 DOF 对，乘以 sqrt(g_i * g_j)
    for vi in range(n_vertices):
        dx, dy, dz = dof_map.vertex_dofs(vi)
        sqrt_gi = np.sqrt(g[vi])
        
        for vj in range(n_vertices):
            jx, jy, jz = dof_map.vertex_dofs(vj)
            sqrt_gj = np.sqrt(g[vj])
            
            factor = sqrt_gi * sqrt_gj
            
            # 退化 3×3 块
            for n_local in range(3):
                i_glob = dof_map.vertex_dofs(vi)[n_local]
                j_glob = dof_map.vertex_dofs(vj)[n_local]
                K_deg[i_glob, j_glob] *= factor
    
    return K_deg


# ══════════════════════════════════════════════════════════
#  弹性应变能密度
# ══════════════════════════════════════════════════════════

def compute_elastic_energy_density(
    mesh: PolyMesh,
    integrator,
    material,
    u: np.ndarray,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> np.ndarray:
    """计算每个控制顶点的弹性应变能密度
    
    用于相场断裂的驱动力: ψ_e⁺ (仅拉伸正应变贡献)
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator
        material: 材料模型
        u: (n_dof,) 位移解
        dof_map: DOF 映射
        iga_settings: IGA 设置
    
    Returns:
        psi_e: (n_vertices,) 顶点应变能密度
    """
    from forgecraft.analysis.iga_shape import cc_shape_functions
    from forgecraft.analysis.iga_quadrature import cc_quadrature
    
    n_vertices = mesh.n_vertices
    
    psi_accum = np.zeros(n_vertices)
    weight_accum = np.zeros(n_vertices)
    
    D_membrane = _plane_stress_D_numba(material.E, material.nu)
    
    for face_id in range(mesh.n_faces):
        verts = mesh.face_vertices(face_id)
        if len(verts) != 4:
            continue
        
        # 面控制点
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
        
        # 提取单元位移
        u_e = np.zeros(48)
        for i, vi in enumerate(all_verts):
            dx, dy, dz = dof_map.vertex_dofs(vi)
            u_e[3*i + 0] = u[dx]
            u_e[3*i + 1] = u[dy]
            u_e[3*i + 2] = u[dz]
        
        # 积分点循环
        qps = cc_quadrature(mesh, face_id, iga_settings.quadrature_order, irregular_cache)
        
        for qp in qps:
            shape = cc_shape_functions(mesh, face_id, qp.u, qp.v, irregular_cache)
            
            if shape.detJ < 1e-15:
                continue
            
            # 局部坐标系
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
            
            # B 矩阵 (全局应变-位移)
            B_glob = np.zeros((3, 48))
            for node_i in range(16):
                dNdx = shape.dN_dx[node_i]
                dN_dt1 = np.dot(dNdx, t1)
                dN_dt2 = np.dot(dNdx, t2)
                
                i3 = node_i * 3
                B_glob[0, i3 + 0] = dN_dt1 * t1[0]
                B_glob[0, i3 + 1] = dN_dt1 * t1[1]
                B_glob[0, i3 + 2] = dN_dt1 * t1[2]
                
                B_glob[1, i3 + 0] = dN_dt2 * t2[0]
                B_glob[1, i3 + 1] = dN_dt2 * t2[1]
                B_glob[1, i3 + 2] = dN_dt2 * t2[2]
                
                B_glob[2, i3 + 0] = dN_dt1 * t2[0] + dN_dt2 * t1[0]
                B_glob[2, i3 + 1] = dN_dt1 * t2[1] + dN_dt2 * t1[1]
                B_glob[2, i3 + 2] = dN_dt1 * t2[2] + dN_dt2 * t1[2]
            
            # 膜应变和应力
            eps = B_glob @ u_e
            sigma = D_membrane @ eps
            
            # 应变能密度 ψ_e = ½ σ·ε = ½ σ^T ε
            psi_q = 0.5 * np.dot(sigma, eps)
            psi_q = max(0.0, psi_q)  # 仅正应变能驱动裂纹（简化）
            
            # 面积加权分配
            area = qp.weight * shape.detJ
            for node_i in range(16):
                vi = all_verts[node_i]
                psi_accum[vi] += psi_q * area
                weight_accum[vi] += area
    
    # 归一化
    psi_e = np.zeros(n_vertices)
    for vi in range(n_vertices):
        if weight_accum[vi] > 1e-15:
            psi_e[vi] = psi_accum[vi] / weight_accum[vi]
    
    return psi_e


# ══════════════════════════════════════════════════════════
#  相场系统组装
# ══════════════════════════════════════════════════════════

def assemble_phasefield_system(
    mesh: PolyMesh,
    driving_force: np.ndarray,          # (n_vertices,) 历史应变能 ψ_e⁺(x)
    pf_settings: PhaseFieldSettings,
    dof_map: DOFMap,
    irregular_cache: Optional[IrregularFaceCache] = None,
) -> Tuple[csr_matrix, np.ndarray]:
    """组装相场刚度矩阵和力向量
    
    K_d = ∫_Ω [ (Gc/ℓ + 2ψ_e⁺) N^T N + Gc·ℓ B^T B ] dΩ
    f_d = ∫_Ω 2ψ_e⁺ N^T dΩ
    
    Args:
        mesh: CC 控制网格
        driving_force: (n_vertices,) 驱动应变能
        pf_settings: 相场设置
        dof_map: DOF 映射
    
    Returns:
        (K_d, f_d) 相场 CSR 刚度矩阵和力向量
    """
    n_vertices = mesh.n_vertices
    Gc = pf_settings.Gc
    ell = pf_settings.length_scale
    
    # 简化: 用集总方法 (每顶点一个相场 DOF)
    K_d = np.zeros((n_vertices, n_vertices))
    f_d = np.zeros(n_vertices)
    
    for face_id in range(mesh.n_faces):
        verts = mesh.face_vertices(face_id)
        if len(verts) != 4:
            continue
        
        # 面顶点: 在相场中是 4 个节点
        for qi in range(4):
            vi = verts[qi]
            # 近似面积: 四边形的面积 / 4
            p0 = mesh.vertices[verts[0]]
            p1 = mesh.vertices[verts[1]]
            p2 = mesh.vertices[verts[2]]
            p3 = mesh.vertices[verts[3]]
            area = 0.5 * (
                np.linalg.norm(np.cross(p1 - p0, p2 - p0)) +
                np.linalg.norm(np.cross(p2 - p0, p3 - p0))
            )
            area_per_vertex = area / 4.0
            
            # 源项
            psi_e_i = driving_force[vi]
            
            # K_d[i, i] += (Gc/ℓ + 2ψ_e) * area
            K_d[vi, vi] += (Gc / ell + 2.0 * psi_e_i) * area_per_vertex
            
            # f_d[i] += 2ψ_e * area
            f_d[vi] += 2.0 * psi_e_i * area_per_vertex
            
            # 梯度项 (相邻顶点的贡献)
            for qj in range(4):
                if qi >= qj:
                    continue
                vj = verts[qj]
                
                # 近似长度 (顶点间距离)
                dist = np.linalg.norm(
                    mesh.vertices[vi] - mesh.vertices[vj]
                )
                if dist < 1e-10:
                    continue
                
                # 梯度刚度: Gc·ℓ / (2·dist²) · area
                grad_stiff = Gc * ell / (2.0 * dist * dist) * area_per_vertex
                
                K_d[vi, vi] += grad_stiff
                K_d[vj, vj] += grad_stiff
                K_d[vi, vj] -= grad_stiff
                K_d[vj, vi] -= grad_stiff
    
    # 转换为 CSR
    K_csr = csr_matrix(K_d)
    return K_csr, f_d


# ══════════════════════════════════════════════════════════
#  裂纹几何提取
# ══════════════════════════════════════════════════════════

def extract_crack_geometry(
    mesh: PolyMesh,
    d_field: np.ndarray,
    threshold: float = 0.8,
) -> CrackInfo:
    """从相场提取裂纹几何信息
    
    d(x) ≥ threshold → 裂纹位置
    
    Args:
        mesh: CC 控制网格
        d_field: (n_vertices,) 相场值
        threshold: 裂纹阈值
    
    Returns:
        CrackInfo
    """
    n_vertices = mesh.n_vertices
    crack_vertices = np.where(d_field >= threshold)[0]
    
    # 裂纹顶点位置
    positions = mesh.vertices[crack_vertices].copy()
    
    # 近似法向量 (平均面法向)
    normals = np.zeros((len(crack_vertices), 3))
    for i, vi in enumerate(crack_vertices):
        faces = mesh.vertex_faces(vi)
        avg_normal = np.zeros(3)
        for f in faces:
            verts = mesh.face_vertices(f)
            if len(verts) >= 3:
                p0 = mesh.vertices[verts[0]]
                p1 = mesh.vertices[verts[1]]
                p2 = mesh.vertices[verts[2]]
                fn = np.cross(p1 - p0, p2 - p0)
                if np.linalg.norm(fn) > 1e-15:
                    avg_normal += fn / np.linalg.norm(fn)
        n_mag = np.linalg.norm(avg_normal)
        if n_mag > 1e-15:
            normals[i] = avg_normal / n_mag
    
    # 裂纹面积 (近似)
    crack_area = 0.0
    for f in range(mesh.n_faces):
        verts = mesh.face_vertices(f)
        d_avg = np.mean([d_field[v] for v in verts])
        if d_avg >= threshold:
            p0 = mesh.vertices[verts[0]]
            p1 = mesh.vertices[verts[1]]
            p2 = mesh.vertices[verts[2]]
            p3 = mesh.vertices[verts[3]]
            area = 0.5 * (
                np.linalg.norm(np.cross(p1 - p0, p2 - p0)) +
                np.linalg.norm(np.cross(p2 - p0, p3 - p0))
            )
            crack_area += area
    
    crack_length = np.sqrt(crack_area) if crack_area > 0 else 0.0
    
    return CrackInfo(
        positions=positions,
        normals=normals,
        damage_field=d_field,
        crack_area=crack_area,
        crack_length=crack_length,
    )


def _plane_stress_D_numba(E: float, nu: float) -> np.ndarray:
    """3×3 平面应力本构矩阵"""
    D = np.zeros((3, 3))
    factor = E / (1.0 - nu * nu)
    D[0, 0] = 1.0
    D[1, 1] = 1.0
    D[0, 1] = nu
    D[1, 0] = nu
    D[2, 2] = (1.0 - nu) / 2.0
    D *= factor
    return D
