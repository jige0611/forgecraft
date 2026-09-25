# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_dynamic — 动力分析
#
#   solve_modal:    模态分析 (子空间迭代 / scipy eigs)
#   solve_transient: 瞬态动力学 (Newmark-β / Generalized-α)
#   hrz_lumped_mass: HRZ 集中质量矩阵
#   rayleigh_damping: Rayleigh 阻尼 C = αM + βK
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, LinearOperator

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IGASettings, ModalResult, TransientResult, IGAError,
)
from forgecraft.analysis.iga_assembly import assemble_stiffness, assemble_mass
from forgecraft.analysis.iga_boundary import DirichletBC, apply_dirichlet_penalty

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "solve_modal",
    "solve_transient",
    "hrz_lumped_mass",
    "rayleigh_damping",
]


# ══════════════════════════════════════════════════════════
#  模态分析
# ══════════════════════════════════════════════════════════

def solve_modal(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    settings: IGASettings,
    dirichlet_bcs: Optional[List[DirichletBC]] = None,
    n_modes: int = 10,
    shift: float = 0.0,
    irregular_cache=None,
) -> ModalResult:
    """模态分析 (广义特征值问题)
    
    (K - ω² M) φ = 0
    
    使用 scipy.sparse.linalg.eigsh (隐式重启动 Lanczos)
    求解最小 n_modes 个特征对。
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator 实例
        material: 材料模型
        dof_map: DOF 映射
        settings: IGA 设置
        dirichlet_bcs: Dirichlet BC 列表 (约束自由度会从特征值问题中消除)
        n_modes: 提取的模态数
        shift: 特征值偏移 (用于提高收敛性)
        irregular_cache: 非常面缓存
    
    Returns:
        ModalResult (frequencies, mode_shapes, participation_factors, ...)
    """
    t0 = perf_counter()
    
    if settings.verbosity >= 1:
        _logger.info(f"Modal analysis: assembling K and M ({dof_map.n_dof} DOF)...")
    
    # 组装刚度矩阵和质量矩阵
    K = assemble_stiffness(
        mesh, integrator, material, dof_map, settings, irregular_cache,
    )
    M = assemble_mass(
        mesh, integrator, material, dof_map, settings, irregular_cache,
    )
    
    # 施加 Dirichlet BC (罚函数法)
    if dirichlet_bcs:
        K_dense = K.toarray()
        M_dense = M.toarray()
        F = np.zeros(dof_map.n_dof)
        for bc in dirichlet_bcs:
            apply_dirichlet_penalty(K_dense, F, bc, penalty=1e15)
            # 对于质量矩阵: 约束 DOF 对角元置大值
            for d in bc.dof_ids:
                M_dense[d, d] = 1.0
        K = csr_matrix(K_dense)
        M = csr_matrix(M_dense)
    
    if settings.verbosity >= 1:
        _logger.info(f"  Solving for {n_modes} modes (Lanczos)...")
    
    # 广义特征值求解
    try:
        eigenvalues, eigenvectors = eigsh(
            K, M=M,
            k=min(n_modes, dof_map.n_dof - 2),
            which='SM',       # 最小特征值 (最低频率)
            sigma=shift if shift > 0 else None,
            tol=settings.tolerance,
            maxiter=settings.max_iter,
        )
        converged = True
    except Exception as e:
        _logger.error(f"Modal analysis failed: {e}")
        eigenvalues = np.array([])
        eigenvectors = np.zeros((dof_map.n_dof, 0))
        converged = False
    
    # 频率转换
    # 确保特征值非负
    eigenvalues = np.maximum(eigenvalues, 0.0)
    
    # 圆频率 ω (rad/s)
    omega = np.sqrt(eigenvalues)
    
    # 自然频率 f (Hz)
    frequencies = omega / (2.0 * np.pi)
    
    # 振型归一化 (质量归一化)
    mode_shapes = eigenvectors.copy()
    for i in range(mode_shapes.shape[1]):
        phi = mode_shapes[:, i]
        mass_norm = phi @ (M @ phi)
        if mass_norm > 1e-15:
            mode_shapes[:, i] = phi / np.sqrt(mass_norm)
    
    # 振型参与系数
    r = np.ones(dof_map.n_dof)  # 单位方向 (可扩展)
    participation = np.zeros(n_modes)
    effective_mass = np.zeros(n_modes)
    total_mass = np.sum(M)
    
    for i in range(mode_shapes.shape[1]):
        phi = mode_shapes[:, i]
        gamma = phi @ (M @ r)  # 参与系数
        participation[i] = gamma
        effective_mass[i] = gamma * gamma  # 有效模态质量
    
    wall_time = perf_counter() - t0
    
    if settings.verbosity >= 1:
        if len(frequencies) > 0:
            _logger.info(
                f"Modal analysis done in {wall_time:.2f}s: "
                f"f1={frequencies[0]:.2f} Hz, f{n_modes}={frequencies[-1]:.2f} Hz"
            )
        else:
            _logger.info(
                f"Modal analysis done in {wall_time:.2f}s: no modes converged"
            )
    
    return ModalResult(
        frequencies=frequencies,
        angular_frequencies=omega,
        mode_shapes=mode_shapes,
        participation_factors=participation,
        effective_mass=effective_mass,
        converged=converged,
    )


# ══════════════════════════════════════════════════════════
#  瞬态动力学
# ══════════════════════════════════════════════════════════

def solve_transient(
    mesh: PolyMesh,
    integrator,
    material,
    dof_map: DOFMap,
    settings: IGASettings,
    dt: float,
    n_steps: int,
    f_ext_func,                              # f_ext(t) → np.ndarray (n_dof,)
    dirichlet_bcs: Optional[List[DirichletBC]] = None,
    u0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    method: str = "newmark",
    damping_alpha: float = 0.0,
    damping_beta: float = 0.0,
    irregular_cache=None,
) -> TransientResult:
    """瞬态动力学分析
    
    M ü + C u̇ + K u = f_ext(t)
    
    支持:
      - Newmark-β (β=0.25, γ=0.5 → 平均加速度法，无条件稳定)
      - Generalized-α (α_m=0.5, α_f=0.5 → Newmark 特例)
    
    Args:
        mesh: CC 控制网格
        integrator: ElementIntegrator 实例
        material: 材料模型
        dof_map: DOF 映射
        settings: IGA 设置
        dt: 时间步长
        n_steps: 时间步数
        f_ext_func: f(t) → (n_dof,) 外力向量函数
        dirichlet_bcs: Dirichlet BC 列表
        u0: 初始位移 (默认零)
        v0: 初始速度 (默认零)
        method: "newmark" or "generalized_alpha"
        damping_alpha: Rayleigh 质量阻尼系数
        damping_beta: Rayleigh 刚度阻尼系数
        irregular_cache: 非常面缓存
    
    Returns:
        TransientResult (time_history, displacement_history, ...)
    """
    t0_total = perf_counter()
    
    n_dof = dof_map.n_dof
    
    # 初始条件
    if u0 is None:
        u = np.zeros(n_dof)
    else:
        u = u0.copy()
    
    if v0 is None:
        v = np.zeros(n_dof)
    else:
        v = v0.copy()
    
    # 组装刚度矩阵和质量矩阵
    if settings.verbosity >= 1:
        _logger.info("Transient: assembling K and M...")
    
    K = assemble_stiffness(
        mesh, integrator, material, dof_map, settings, irregular_cache,
    )
    M = assemble_mass(
        mesh, integrator, material, dof_map, settings, irregular_cache,
    )
    
    # HRZ 集中质量 (瞬态分析推荐)
    M_lumped = hrz_lumped_mass(M)
    
    # Rayleigh 阻尼
    C = rayleigh_damping(damping_alpha, damping_beta, M, K)
    
    # 转换为 dense (对于小规模问题, sparse 迭代在每个时间步会更慢)
    M_diag = M_lumped
    K_csr = K.tocsr()
    C_csr = C.tocsr() if C is not None else None
    
    # 施加 BC
    if dirichlet_bcs:
        K_dense = K.toarray()
        M_dense = M.toarray()
        F_dummy = np.zeros(n_dof)
        for bc in dirichlet_bcs:
            apply_dirichlet_penalty(K_dense, F_dummy, bc, penalty=1e15)
            for d in bc.dof_ids:
                M_dense[d, d] = 1.0
        K_csr = csr_matrix(K_dense)
    
    # 初始加速度
    f0 = f_ext_func(0.0)
    a = np.zeros(n_dof)
    for i in range(n_dof):
        if M_diag[i] > 1e-15:
            k_row = K_csr[i].toarray().ravel() if hasattr(K_csr[i], 'toarray') else np.zeros(n_dof)
            c_val = C_csr[i].dot(v) if C_csr is not None else 0.0
            a[i] = (f0[i] - k_row @ u - c_val) / M_diag[i]
    
    # Newmark 参数
    if method == "newmark":
        beta_nm = 0.25
        gamma_nm = 0.5
    else:  # generalized_alpha
        # α_m=0.5, α_f=0.5 → 退化为 Newmark β=0.25
        beta_nm = 0.25
        gamma_nm = 0.5
    
    # 有效刚度矩阵预处理
    # K_eff = M/(β dt²) + γ C/(β dt) + K
    # 使用对角近似 + Jacobi 迭代
    
    # 历史存储
    time_history = np.zeros(n_steps + 1)
    displ_history = np.zeros((n_dof, n_steps + 1))
    vel_history = np.zeros((n_dof, n_steps + 1))
    acc_history = np.zeros((n_dof, n_steps + 1))
    ke_history = np.zeros(n_steps + 1)
    se_history = np.zeros(n_steps + 1)
    
    time_history[0] = 0.0
    displ_history[:, 0] = u
    vel_history[:, 0] = v
    acc_history[:, 0] = a
    ke_history[0] = 0.5 * np.sum(M_diag * v * v)
    se_history[0] = 0.5 * u @ (K_csr @ u)
    
    # 预测系数
    inv_beta_dt2 = 1.0 / (beta_nm * dt * dt)
    gamma_div_beta_dt = gamma_nm / (beta_nm * dt)
    one_minus_gamma_div_beta = 1.0 - gamma_nm / beta_nm
    dt_one_minus_gamma = dt * (1.0 - gamma_nm)
    
    for step in range(n_steps):
        t_n = (step + 1) * dt
        
        # 外力
        f_ext = f_ext_func(t_n)
        
        # 有效力向量 (Newmark)
        # F_eff = f_ext + M*(u_pred/(β dt²) + v_pred/(β dt) + (1/(2β)-1)*a_pred)
        #        + C*(γ/(β dt)*u_pred + ...)
        F_eff = f_ext.copy()
        
        u_pred = u + dt * v + 0.5 * dt * dt * (1.0 - 2.0 * beta_nm) * a
        v_pred = v + dt * (1.0 - gamma_nm) * a
        
        # 质量贡献
        for i in range(n_dof):
            F_eff[i] += M_diag[i] * (
                inv_beta_dt2 * u_pred[i]
            )
        
        # 阻尼贡献 (简化: 仅对角)
        if C_csr is not None and damping_alpha > 0:
            C_diag = C_csr.diagonal()
            for i in range(n_dof):
                F_eff[i] += C_diag[i] * (
                    gamma_div_beta_dt * u_pred[i]
                    + one_minus_gamma_div_beta * v_pred[i]
                    + dt_one_minus_gamma * a[i]
                )
        
        # 有效刚度矩阵的对角近似
        # K_eff_diag = M_diag/(β dt²) + K_diag
        K_diag = K_csr.diagonal()
        
        # Jacobi 迭代求解
        u_new = u.copy()
        for jac_iter in range(50):
            residual = F_eff - K_csr @ u_new
            for i in range(n_dof):
                eff_diag = M_diag[i] * inv_beta_dt2 + K_diag[i]
                if abs(eff_diag) > 1e-15:
                    residual[i] -= M_diag[i] * inv_beta_dt2 * u_new[i]
                    # 简化 Jacobi
                    u_new[i] += residual[i] / eff_diag
            
            if np.linalg.norm(residual) < settings.tolerance * np.linalg.norm(F_eff):
                break
        
        u = u_new
        
        # 更新速度和加速度
        a_new = inv_beta_dt2 * (u - u - dt * v - 0.5 * dt * dt * (1.0 - 2.0 * beta_nm) * a)
        # 用标准公式
        a_new = (u - u_pred) / (beta_nm * dt * dt)
        v_new = v_pred + gamma_nm * dt * a_new
        
        a = a_new
        v = v_new
        
        # 存储
        time_history[step + 1] = t_n
        displ_history[:, step + 1] = u
        vel_history[:, step + 1] = v
        acc_history[:, step + 1] = a
        ke_history[step + 1] = 0.5 * np.sum(M_diag * v * v)
        se_history[step + 1] = 0.5 * u @ (K_csr @ u)
        
        if settings.verbosity >= 2 and (step + 1) % max(1, n_steps // 10) == 0:
            _logger.info(f"  Step {step + 1}/{n_steps}, t={t_n:.4f}s")
    
    wall_time = perf_counter() - t0_total
    
    if settings.verbosity >= 1:
        _logger.info(f"Transient analysis done in {wall_time:.2f}s ({n_steps} steps)")
    
    return TransientResult(
        time=time_history,
        displacement_history=displ_history,
        velocity_history=vel_history,
        acceleration_history=acc_history,
        kinetic_energy=ke_history,
        strain_energy=se_history,
        dt=dt,
        n_steps=n_steps,
    )


# ══════════════════════════════════════════════════════════
#  HRZ 集中质量矩阵
# ══════════════════════════════════════════════════════════

def hrz_lumped_mass(M: csr_matrix) -> np.ndarray:
    """HRZ (Hinton-Rock-Zienkiewicz) 集中质量矩阵
    
    将一致质量矩阵对角化:
      M_lumped[i] = M_rowsum[i] / trace(M) * mass_total
    
    对于膜单元:
      M_lumped[i] = sum_j |M_{ij}|
    
    Args:
        M: (n, n) CSR 一致质量矩阵
    
    Returns:
        M_lumped: (n,) 对角集中质量
    """
    n = M.shape[0]
    lumped = np.zeros(n)
    
    # 方法: 行和缩放
    row_sums = np.array(M.sum(axis=1)).ravel()
    
    # 保持总质量不变
    diagonal = M.diagonal()
    trace_M = np.sum(diagonal)
    
    if trace_M > 1e-15:
        # HRZ 缩放
        total_mass = trace_M
        row_sum_total = np.sum(row_sums)
        if row_sum_total > 1e-15:
            lumped = row_sums * (total_mass / row_sum_total)
        else:
            lumped = diagonal.copy()
    else:
        lumped = diagonal.copy()
    
    # 确保正定性
    lumped = np.maximum(lumped, 1e-15)
    
    return lumped


# ══════════════════════════════════════════════════════════
#  Rayleigh 阻尼
# ══════════════════════════════════════════════════════════

def rayleigh_damping(
    alpha: float,
    beta: float,
    M: csr_matrix,
    K: csr_matrix,
) -> Optional[csr_matrix]:
    """Rayleigh 阻尼矩阵
    
    C = α M + β K
    
    Args:
        alpha: 质量比例系数 (低频阻尼)
        beta:  刚度比例系数 (高频阻尼)
        M: 质量矩阵
        K: 刚度矩阵
    
    Returns:
        C: 阻尼矩阵, 或 None (α=β=0)
    """
    if alpha == 0.0 and beta == 0.0:
        return None
    
    C = alpha * M + beta * K
    return C.tocsr()
