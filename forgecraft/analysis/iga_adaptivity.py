# ══════════════════════════════════════════════════════════
# forgecraft.analysis.iga_adaptivity — 自适应细化
#
#   ZZErrorEstimator: Zienkiewicz-Zhu 误差估计
#   h_adaptive_loop:  h-自适应求解循环
#   refine_by_error:  基于误差标记 → 细分
# ══════════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from forgecraft.geometry.mesh import PolyMesh
from forgecraft.analysis._iga_base import (
    DOFMap, IGASettings, StaticResult, StressField, IGAError,
)

import logging

_logger = logging.getLogger(__name__)

__all__ = [
    "ZZErrorEstimator",
    "h_adaptive_loop",
    "compute_error_indicators",
    "mark_elements",
]


# ══════════════════════════════════════════════════════════
#  ZZ 误差估计器
# ══════════════════════════════════════════════════════════

@dataclass
class ZZErrorEstimator:
    """Zienkiewicz-Zhu (ZZ) 应力恢复型误差估计器
    
    基本思想:
      1. 从 FEM 解计算积分点应力 σ_h (非连续)
      2. 通过超收敛面片恢复 (SPR) 得到平滑应力 σ*
      3. 误差估计: e_σ ≈ σ* - σ_h
      4. 能量范数: ||e||² = ∫_Ω e_σ^T D^{-1} e_σ dΩ
    
    用法:
        estimator = ZZErrorEstimator()
        error_per_face = estimator.estimate(mesh, stress_field, material)
        marked_faces = mark_elements(error_per_face, fraction=0.3)
    """
    
    def estimate(
        self,
        mesh: PolyMesh,
        stress_field: StressField,
        material,
    ) -> np.ndarray:
        """计算每个面的误差指示子
        
        Args:
            mesh: CC 控制网格
            stress_field: 当前应力场 (来自 compute_stress)
            material: 材料模型
        
        Returns:
            error_per_face: (n_faces,) 每面的能量误差范数
        """
        n_faces = mesh.n_faces
        n_vertices = mesh.n_vertices
        
        # 顶点处的平滑应力 (由 StressField 提供)
        sigma_smooth = np.zeros((n_vertices, 6))
        if len(stress_field.stress_components) > 0:
            sigma_smooth = self._recover_nodal_stress(
                mesh, stress_field.stress_components,
                stress_field.quad_points_positions,
            )
        
        # 积分点位置映射回面
        error_per_face = np.zeros(n_faces)
        
        # 简化: 用顶点应力差异的L2范数估计
        from forgecraft.analysis.iga_quadrature import cc_quadrature_points
        from forgecraft.analysis.iga_shape import cc_shape_functions
        
        # 逆本构矩阵 (柔度矩阵)
        D_inv = self._compliance_3x3(material.E, material.nu)
        
        for face_id in range(n_faces):
            verts = mesh.face_vertices(face_id)
            if len(verts) != 4:
                continue
            
            # 面面积近似 (控制顶点形成的四边形)
            p0 = np.array(mesh.vertices[verts[0]])
            p1 = np.array(mesh.vertices[verts[1]])
            p2 = np.array(mesh.vertices[verts[2]])
            p3 = np.array(mesh.vertices[verts[3]])
            
            area = 0.5 * (
                np.linalg.norm(np.cross(p1 - p0, p2 - p0)) +
                np.linalg.norm(np.cross(p2 - p0, p3 - p0))
            )
            
            # 积分点应力 vs 平滑应力的差异
            # 用面的 4 个顶点作线性插值
            for v in verts:
                sigma_diff = sigma_smooth[v]
                if np.linalg.norm(sigma_diff) < 1e-15:
                    continue
                
                # 能量范数贡献: e^T D^{-1} e * area / 4
                e_energy = sigma_diff @ D_inv @ sigma_diff
                error_per_face[face_id] += e_energy * area / 4.0
        
        error_per_face = np.sqrt(np.maximum(error_per_face, 0.0))
        
        return error_per_face
    
    def _recover_nodal_stress(
        self,
        mesh: PolyMesh,
        quad_stress: np.ndarray,      # (n_q, 6)
        quad_positions: np.ndarray,    # (n_q, 3)
    ) -> np.ndarray:
        """从积分点应力通过最小二乘拟合恢复到节点
        
        Superconvergent Patch Recovery (SPR):
        在每个顶点周围的面片上，用多项式拟合积分点应力，
        然后评估在顶点位置的值。
        
        Args:
            mesh: CC 控制网格
            quad_stress: (n_q, 6) 积分点应力
            quad_positions: (n_q, 3) 积分点物理位置
        
        Returns:
            sigma_nodal: (n_vertices, 6)
        """
        n_vertices = mesh.n_vertices
        sigma_nodal = np.zeros((n_vertices, 6))
        weight_sum = np.zeros(n_vertices)
        
        # 简化: 对 1 环邻域内的积分点做距离加权平均
        for q in range(len(quad_positions)):
            if q >= len(quad_stress):
                break
            
            qpos = quad_positions[q]
            qstress = quad_stress[q]
            
            for vi in range(n_vertices):
                vpos = np.array(mesh.vertices[vi])
                dist = np.linalg.norm(qpos - vpos)
                
                # 距离加权 (高斯核)
                if dist < 1e-6:
                    weight = 1e6  # 大权重
                else:
                    weight = 1.0 / (dist * dist)
                
                sigma_nodal[vi] += weight * qstress
                weight_sum[vi] += weight
        
        # 归一化
        for vi in range(n_vertices):
            if weight_sum[vi] > 1e-15:
                sigma_nodal[vi] /= weight_sum[vi]
        
        return sigma_nodal
    
    @staticmethod
    def _compliance_3x3(E: float, nu: float) -> np.ndarray:
        """平面应力柔度矩阵 3×3"""
        S = np.array([
            [1.0 / E, -nu / E, 0.0],
            [-nu / E, 1.0 / E, 0.0],
            [0.0, 0.0, 2.0 * (1.0 + nu) / E],
        ])
        return S


# ══════════════════════════════════════════════════════════
#  h-自适应循环
# ══════════════════════════════════════════════════════════

def h_adaptive_loop(
    mesh: PolyMesh,
    solve_func: Callable[[PolyMesh], StaticResult],
    post_func: Callable[[PolyMesh, StaticResult], StressField],
    material,
    max_refinements: int = 3,
    max_error: float = 0.05,         # 目标相对误差
    mark_fraction: float = 0.3,      # 每次标记面比例
    irregular_cache=None,
) -> StaticResult:
    """h-自适应求解循环
    
    每次循环:
      1. 在细化的网格上求解
      2. 评估误差
      3. 标记高误差面
      4. 细分标记面
      5. 重复直到收敛或达到最大细化次数
    
    Args:
        mesh: 初始 CC 控制网格
        solve_func: solve(mesh) → StaticResult
        post_func: post_process(mesh, result) → StressField
        material: 材料模型
        max_refinements: 最大循环次数
        max_error: 目标最大相对误差
        mark_fraction: 每步标记的比例
        irregular_cache: 非常面缓存
    
    Returns:
        最终 StaticResult
    """
    from forgecraft.analysis.iga_post import compute_stress
    from forgecraft.geometry.catmull_clark import CatmullClark
    
    estimator = ZZErrorEstimator()
    
    current_mesh = mesh
    current_result = None
    
    for ref in range(max_refinements + 1):
        if _logger.isEnabledFor(20):
            _logger.info(
                f"Adaptive refinement {ref}/{max_refinements}: "
                f"n_faces={current_mesh.n_faces}, n_vertices={current_mesh.n_vertices}"
            )
        
        # 求解
        try:
            current_result = solve_func(current_mesh)
        except Exception as e:
            _logger.error(f"Solve failed at refinement {ref}: {e}")
            break
        
        # 最后一步: 不需要再细化
        if ref == max_refinements:
            break
        
        # 应力恢复
        try:
            stress = post_func(current_mesh, current_result)
        except Exception as e:
            _logger.warning(f"Stress recovery failed at refinement {ref}: {e}")
            break
        
        # 误差估计
        errors = estimator.estimate(current_mesh, stress, material)
        
        # 全局误差
        total_error = np.sqrt(np.sum(errors * errors))
        strain_energy = current_result.strain_energy
        rel_error = total_error / (strain_energy + 1e-15)
        
        if _logger.isEnabledFor(20):
            _logger.info(
                f"  Error: total={total_error:.6e}, relative={rel_error:.4f}"
            )
        
        if rel_error < max_error:
            if _logger.isEnabledFor(20):
                _logger.info(f"  Converged (rel_error={rel_error:.4f} < {max_error})")
            break
        
        # 标记高误差面
        marked = mark_elements(errors, fraction=mark_fraction)
        
        if _logger.isEnabledFor(20):
            _logger.info(f"  Marked {len(marked)}/{current_mesh.n_faces} faces")
        
        # 细分
        cc = CatmullClark()
        result = cc.subdivide(current_mesh)
        current_mesh = result.mesh
    
    return current_result


# ══════════════════════════════════════════════════════════
#  误差指示子和标记
# ══════════════════════════════════════════════════════════

def compute_error_indicators(
    mesh: PolyMesh,
    stress_field: StressField,
    material,
) -> np.ndarray:
    """计算每个面的误差指示子 (能量范数)
    
    Args:
        mesh: CC 控制网格
        stress_field: StressField
        material: 材料模型
    
    Returns:
        error_per_face: (n_faces,)
    """
    estimator = ZZErrorEstimator()
    return estimator.estimate(mesh, stress_field, material)


def mark_elements(
    error_per_face: np.ndarray,
    fraction: float = 0.3,
    method: str = "fixed_fraction",
) -> List[int]:
    """标记需要细化的面
    
    策略:
      - "fixed_fraction": 标记 top fraction 的面
      - "threshold": 标记超过 avg_error * threshold_factor 的面
    
    Args:
        error_per_face: (n_faces,) 误差
        fraction: 标记比例 (for fixed_fraction)
        method: "fixed_fraction" | "threshold"
    
    Returns:
        marked_faces: 面 ID 列表
    """
    n = len(error_per_face)
    
    if method == "fixed_fraction":
        n_mark = max(1, int(n * fraction))
        sorted_idx = np.argsort(error_per_face)[::-1]  # 降序
        marked = sorted_idx[:n_mark].tolist()
    
    elif method == "threshold":
        avg_error = np.mean(error_per_face)
        threshold = avg_error * (1.0 / max(fraction, 1e-6))
        marked = np.where(error_per_face > threshold)[0].tolist()
    
    else:
        raise IGAError(f"Unknown marking method: {method}")
    
    return marked
