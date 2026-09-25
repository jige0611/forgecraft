# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_solver — 求解器
#
#   solve_static:   静力求解 Ku = F
#     - direct:     scipy.sparse.linalg.spsolve (UMFPACK)
#     - cg:         Conjugate Gradient (带 Jacobi 预条件)
#     - minres:     MINRES (对称不定)
#   solve_linear:   线性方程组求解 (通用接口)
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, minres, spsolve, LinearOperator

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IGASettings, StaticResult, StressField, IGAError,
)
from forgecraft.analysis.iga_assembly import (
    assemble_stiffness, assemble_force_vector, assemble_stiffness_parallel,
)
from forgecraft.analysis.iga_boundary import (
    DirichletBC, NeumannBC, PointLoad,
    apply_dirichlet_penalty, apply_dirichlet_elimination,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "solve_static",
    "solve_linear",
    "jacobi_preconditioner",
    "compute_reactions",
]


# ══════════════════════════════════════════════════════════
#  静力求解
# ══════════════════════════════════════════════════════════

def solve_static(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    settings: IGASettings,
    dirichlet_bcs: List[DirichletBC],
    neumann_bcs: Optional[List[NeumannBC]] = None,
    point_loads: Optional[List[PointLoad]] = None,
    irregular_cache=None,
) -> StaticResult:
    """静力求解主入口
    
    Ku = F  →  u = K^{-1} F
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator 实例
        material: 材料模型
        dof_map: DOF 映射
        settings: IGA 设置
        dirichlet_bcs: Dirichlet BC 列表
        neumann_bcs: Neumann BC 列表
        point_loads: 点载荷列表
        irregular_cache: 非常面缓存
    
    Returns:
        StaticResult (displacements, reactions, strain_energy, ...)
    """
    t0 = perf_counter()
    
    # 1. 组装刚度矩阵
    if settings.verbosity >= 1:
        _logger.info(f"Assembling stiffness matrix ({dof_map.n_dof} DOF)...")
    
    if settings.parallel:
        K = assemble_stiffness_parallel(
            mesh, integrator, material, dof_map, settings, irregular_cache,
        )
    else:
        K = assemble_stiffness(
            mesh, integrator, material, dof_map, settings, irregular_cache,
        )
    
    if settings.verbosity >= 2:
        _logger.info(f"  K: {K.shape}, nnz={K.nnz}")
    
    # 2. 组装力向量
    if settings.verbosity >= 1:
        _logger.info("Assembling force vector...")
    
    F = assemble_force_vector(
        mesh, dof_map,
        neumann_bcs or [], point_loads or [],
        settings,
    )
    
    # 3. 施加 Dirichlet BC
    K_dense = K.toarray() if settings.solver == "direct" else K
    
    if settings.verbosity >= 1:
        _logger.info(f"Applying {len(dirichlet_bcs)} Dirichlet BC(s)...")
    
    if settings.solver == "direct":
        u = np.zeros(dof_map.n_dof)
        free_dofs = []
        for bc in dirichlet_bcs:
            free_dofs = apply_dirichlet_elimination(K_dense, F, bc, u)
        for bc in dirichlet_bcs:
            apply_dirichlet_penalty(K_dense, F, bc, penalty=1e12)
    else:
        u = np.zeros(dof_map.n_dof)
        for bc in dirichlet_bcs:
            apply_dirichlet_penalty(K_dense, F, bc, penalty=1e12)
    
    # 4. 求解
    if settings.verbosity >= 1:
        _logger.info(f"Solving linear system (solver={settings.solver})...")
    
    u_solution, iterations, converged = solve_linear(
        K_dense, F, settings,
    )
    
    u += u_solution
    
    # 5. 应变能
    strain_energy = 0.5 * np.dot(u, K_dense.dot(u)) if isinstance(K_dense, np.ndarray) else 0.5 * u @ (K @ u)
    
    wall_time = perf_counter() - t0
    
    if settings.verbosity >= 1:
        _logger.info(f"Solved in {wall_time:.2f}s, strain energy = {strain_energy:.6e}")
    
    return StaticResult(
        displacements=u,
        strain_energy=strain_energy,
        iterations=iterations,
        converged=converged,
        wall_time=wall_time,
    )


# ══════════════════════════════════════════════════════════
#  线性求解
# ══════════════════════════════════════════════════════════

def solve_linear(
    K,                # np.ndarray or csr_matrix
    F: np.ndarray,
    settings: IGASettings,
) -> Tuple[np.ndarray, int, bool]:
    """求解 Ku = F
    
    Args:
        K: 刚度矩阵 (dense ndarray 或 sparse csr_matrix)
        F: 力向量
        settings: IGA 设置
    
    Returns:
        (u, iterations, converged)
    """
    solver = settings.solver.lower()
    
    if solver == "direct":
        try:
            if isinstance(K, np.ndarray):
                u = np.linalg.solve(K, F)
            else:
                u = spsolve(K.tocsr(), F)
            return u, 0, True
        except np.linalg.LinAlgError as e:
            raise IGAError(f"Direct solver failed: {e}")
    
    elif solver == "cg":
        K_csr = K.tocsr() if not isinstance(K, np.ndarray) else csr_matrix(K)
        M = jacobi_preconditioner(K_csr)
        u, info = cg(
            K_csr, F,
            tol=settings.tolerance,
            maxiter=settings.max_iter,
            M=M,
        )
        converged = info == 0
        if not converged:
            _logger.warning(f"CG did not converge (info={info})")
        return u, info if converged else settings.max_iter, converged
    
    elif solver == "minres":
        K_csr = K.tocsr() if not isinstance(K, np.ndarray) else csr_matrix(K)
        M = jacobi_preconditioner(K_csr)
        u, info = minres(
            K_csr, F,
            tol=settings.tolerance,
            maxiter=settings.max_iter,
            M=M,
        )
        converged = info == 0
        if not converged:
            _logger.warning(f"MINRES did not converge (info={info})")
        return u, info if converged else settings.max_iter, converged
    
    else:
        raise IGAError(f"Unknown solver: {solver}")


# ══════════════════════════════════════════════════════════
#  预条件器
# ══════════════════════════════════════════════════════════

def jacobi_preconditioner(K: csr_matrix) -> LinearOperator:
    """Jacobi (对角) 预条件器
    
    M = diag(K)^-1
    """
    n = K.shape[0]
    diag = K.diagonal()
    
    # 避免除零
    diag_inv = np.where(np.abs(diag) > 1e-15, 1.0 / diag, 0.0)
    
    def matvec(x):
        return diag_inv * x
    
    return LinearOperator((n, n), matvec=matvec, dtype=np.float64)


# ══════════════════════════════════════════════════════════
#  支反力计算
# ══════════════════════════════════════════════════════════

def compute_reactions(
    K,                  # np.ndarray or csr_matrix
    u: np.ndarray,
    bc: DirichletBC,
) -> np.ndarray:
    """计算 Dirichlet BC 约束点的支反力
    
    R = K @ u - F_external
    在约束 DOF 处: R_j = sum_i K_ji u_i - F_j
    
    Args:
        K: 刚度矩阵
        u: 位移解
        bc: Dirichlet BC
    
    Returns:
        reactions: (len(bc.dof_ids),) 支反力
    """
    if isinstance(K, np.ndarray):
        f_int = K @ u
    else:
        f_int = K.dot(u)
    
    reactions = f_int[bc.dof_ids]
    return reactions
