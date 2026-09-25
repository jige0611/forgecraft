# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_contact_assembly — 接触+断裂联合组装
#
#   将接触刚度/力和断裂退化刚度合并到全局系统。
#   统一的组装入口，供求解器调用。
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IrregularFaceCache, IGASettings,
)
from forgecraft.analysis.iga_contact_types import (
    ContactSettings, ContactPair, FrictionModel,
    PhaseFieldSettings, CohesiveLaw,
)
from forgecraft.analysis.iga_contact_search import (
    build_contact_pairs, find_contact_pairs, AABBTree,
)
from forgecraft.analysis.iga_contact_enforce import (
    assemble_contact_penalty,
)
from forgecraft.analysis.iga_fracture_phasefield import (
    assemble_degraded_stiffness,
)
from forgecraft.analysis.iga_fracture_cohesive import (
    cohesive_stiffness,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "assemble_with_contact_fracture",
    "ContactFractureSolver",
    "solve_contact_fracture",
    "assemble_total_system",
]


# ══════════════════════════════════════════════════════════
#  联合组装
# ══════════════════════════════════════════════════════════

def assemble_with_contact_fracture(
    mesh: PolyMesh,
    K0: np.ndarray,                     # 基础刚度矩阵 (dense)
    d_field: np.ndarray,                # 相场 (n_vertices,)
    contact_pairs: List[ContactPair],
    dof_map: DOFMap,
    pf_settings: Optional[PhaseFieldSettings],
    contact_settings: ContactSettings,
    friction: Optional[FrictionModel] = None,
    interface_faces: Optional[List[int]] = None,
    cohesive_law: Optional[CohesiveLaw] = None,
    displacement: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """联合组装: 断裂退化 + 接触 + 内聚力
    
    K_total = g(d)·K0 + K_contact + K_cohesive
    f_total = f_ext + f_contact + f_cohesive
    
    Args:
        mesh: CC 控制网格
        K0: (n_dof, n_dof) 未退化刚度矩阵
        d_field: (n_vertices,) 相场 (None = 无退化)
        contact_pairs: 接触对列表
        dof_map: DOF 映射
        pf_settings: 相场设置 (None = 无断裂)
        contact_settings: 接触设置
        friction: 摩擦模型
        interface_faces: 内聚力界面面
        cohesive_law: 内聚力律
        displacement: 当前位移 (用于内聚力内力)
    
    Returns:
        (K_total, f_total) 全局刚度矩阵和力向量
    """
    n_dof = dof_map.n_dof
    
    # 1. 退化刚度 (相场)
    if d_field is not None and pf_settings is not None:
        K = assemble_degraded_stiffness(K0, d_field, mesh, dof_map, pf_settings)
    else:
        K = K0.copy()
    
    F = np.zeros(n_dof)
    
    # 2. 接触贡献
    if contact_pairs:
        K_contact, f_contact = assemble_contact_penalty(
            mesh, contact_pairs, dof_map, contact_settings, friction,
        )
        if K_contact.nnz > 0:
            K += K_contact.toarray()
        F += f_contact
    
    # 3. 内聚力贡献 (界面断裂)
    if interface_faces and cohesive_law and displacement is not None:
        K_coh, f_coh = cohesive_stiffness(
            mesh, interface_faces, displacement, dof_map, cohesive_law,
        )
        if K_coh.nnz > 0:
            K += K_coh.toarray()
        F += f_coh
    
    return K, F


def assemble_total_system(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    dirichlet_bcs,
    neumann_bcs=None,
    point_loads=None,
    d_field: Optional[np.ndarray] = None,
    pf_settings: Optional[PhaseFieldSettings] = None,
    contact_pairs: Optional[List[ContactPair]] = None,
    contact_settings: Optional[ContactSettings] = None,
    interface_faces: Optional[List[int]] = None,
    cohesive_law: Optional[CohesiveLaw] = None,
    displacement: Optional[np.ndarray] = None,
    irregular_cache=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """完整的系统矩阵组装
    
    包含: 退化刚度 + Dirichlet BC + 接触 + 内聚力 + 外部载荷
    
    Args:
        mesh: CC 控制网格
        (标准 IGA 参数...)
        d_field: 相场值
        pf_settings: 相场设置
        contact_pairs: 接触对
        contact_settings: 接触设置
        interface_faces: 内聚力面
        cohesive_law: 内聚力律
        displacement: 当前位移
    
    Returns:
        (K_total, F_total)
    """
    from forgecraft.analysis.iga_assembly import (
        assemble_stiffness, assemble_force_vector,
    )
    from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty
    
    # 基础刚度矩阵
    K0 = assemble_stiffness(
        mesh, integrator, material, dof_map, iga_settings, irregular_cache,
    ).toarray()
    
    # 基础力向量
    F = assemble_force_vector(
        mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
    )
    
    # 联合组装 (退化 + 接触 + 内聚力)
    K, F = assemble_with_contact_fracture(
        mesh, K0, d_field if d_field is not None else np.zeros(mesh.n_vertices),
        contact_pairs or [],
        dof_map,
        pf_settings,
        contact_settings or ContactSettings(),
        interface_faces=interface_faces,
        cohesive_law=cohesive_law,
        displacement=displacement,
    )
    
    # 施加 Dirichlet BC
    if dirichlet_bcs:
        for bc in dirichlet_bcs:
            apply_dirichlet_penalty(K, F, bc)
    
    return K, F


# ══════════════════════════════════════════════════════════
#  接触+断裂联合求解器
# ══════════════════════════════════════════════════════════

class ContactFractureSolver:
    """接触+断裂联合求解器
    
    处理裂纹表面自接触的双体相互作用:
      1. 相场裂纹演化 → 新自由面
      2. 自由面之间的接触检测
      3. 罚函数/增广拉格朗日施加接触约束
    
    求解策略: 交错迭代
      - 外层: 载荷步
      - 中层: 断裂 + 接触交错
      - 内层: Newton-Raphson (仅位移)
    """
    
    def solve(
        self,
        mesh: PolyMesh,
        integrator,
        material,
        dof_map: DOFMap,
        iga_settings: IGASettings,
        pf_settings: PhaseFieldSettings,
        contact_settings: ContactSettings,
        dirichlet_bcs,
        neumann_bcs=None,
        point_loads=None,
        load_steps: int = 10,
        irregular_cache=None,
    ) -> dict:
        """联合求解接触+断裂问题
        
        Returns:
            dict with keys: displacements, d_field, contact_pairs, crack_info
        """
        from forgecraft.analysis.iga_fracture_phasefield import (
            PhaseFieldSolver, extract_crack_geometry,
        )
        from forgecraft.analysis.iga_contact_enforce import (
            assemble_contact_penalty, build_contact_pairs,
        )
        from forgecraft.analysis.iga_boundary import apply_dirichlet_penalty
        
        n_dof = dof_map.n_dof
        n_vertices = mesh.n_vertices
        
        # 基础刚度矩阵
        from forgecraft.analysis.iga_assembly import (
            assemble_stiffness, assemble_force_vector,
        )
        K0 = assemble_stiffness(
            mesh, integrator, material, dof_map, iga_settings, irregular_cache,
        ).toarray()
        F_ref = assemble_force_vector(
            mesh, dof_map, neumann_bcs or [], point_loads or [], iga_settings,
        )
        
        # 施加 BC
        if dirichlet_bcs:
            for bc in dirichlet_bcs:
                apply_dirichlet_penalty(K0, F_ref, bc)
        
        # 状态变量
        u = np.zeros(n_dof)
        d_field = np.zeros(n_vertices)
        d_history = np.zeros(n_vertices)
        
        displacement_history = [u.copy()]
        d_field_history = [d_field.copy()]
        contact_history = []
        
        for step in range(1, load_steps + 1):
            load_factor = step / load_steps
            F_step = load_factor * F_ref
            
            for stagger_iter in range(pf_settings.stagger_max_iter):
                d_old = d_field.copy()
                
                # ---- 位移子问题 (含接触) ----
                # 检测接触对 (当前位移)
                # 自体接触: 使用同一个 mesh + 排除拓扑相邻面
                tree = AABBTree(mesh, expansion=0.01)
                contact_pairs = find_contact_pairs(
                    mesh, mesh, tree, contact_settings,
                )
                
                # 组装: 退化 + 接触
                K_deg = assemble_degraded_stiffness(
                    K0, d_field, mesh, dof_map, pf_settings,
                )
                K_contact, f_contact = assemble_contact_penalty(
                    mesh, contact_pairs, dof_map, contact_settings,
                )
                
                K_total = K_deg + K_contact.toarray()
                F_total = F_step + f_contact
                
                try:
                    u_new = np.linalg.solve(K_total, F_total)
                except np.linalg.LinAlgError:
                    u_new = u
                
                u = u_new
                
                # ---- 断裂子问题 ----
                from forgecraft.analysis.iga_fracture_phasefield import (
                    compute_elastic_energy_density, assemble_phasefield_system,
                )
                from scipy.sparse.linalg import spsolve
                
                psi_e = compute_elastic_energy_density(
                    mesh, integrator, material, u, dof_map, iga_settings,
                    irregular_cache,
                )
                
                K_d, f_d = assemble_phasefield_system(
                    mesh, psi_e, pf_settings, dof_map,
                )
                
                d_new = spsolve(K_d, f_d)
                d_new = np.maximum(d_new, d_history)
                d_new = np.minimum(d_new, 1.0)
                
                d_field = d_new
                
                # 交错收敛
                if np.linalg.norm(d_field - d_old) < pf_settings.stagger_tolerance:
                    break
            
            d_history = np.maximum(d_history, d_field)
            displacement_history.append(u.copy())
            d_field_history.append(d_field.copy())
            contact_history.append(contact_pairs if stagger_iter == 0 else [])
            
            if iga_settings.verbosity >= 1:
                _logger.info(
                    f"Step {step}/{load_steps}: "
                    f"|u|={np.linalg.norm(u):.4e}, max_d={np.max(d_field):.3f}"
                )
        
        crack_info = extract_crack_geometry(mesh, d_field)
        
        return {
            "displacements": displacement_history,
            "d_field": d_field_history,
            "contact_pairs": contact_history,
            "crack_info": crack_info,
            "final_displacement": u,
            "final_damage": d_field,
        }


# ══════════════════════════════════════════════════════════
#  便捷接口
# ══════════════════════════════════════════════════════════

def solve_contact_fracture(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    iga_settings: IGASettings,
    pf_settings: PhaseFieldSettings,
    contact_settings: ContactSettings,
    dirichlet_bcs,
    **kwargs,
) -> dict:
    """接触+断裂联合求解便捷入口"""
    solver = ContactFractureSolver()
    return solver.solve(
        mesh, integrator, material, dof_map, iga_settings,
        pf_settings, contact_settings, dirichlet_bcs, **kwargs,
    )
